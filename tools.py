import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric import transforms
from torch_geometric import utils
from torch_geometric.loader import DataLoader
from accelerate import Accelerator
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, to_undirected, to_dense_adj, scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes
from scipy.sparse.linalg import eigsh
import accelerate
import wandb
import time
import cupy as cp
# import networkx as nx
# import pickle

def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps.

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, source, dim=0, dim_size=num_nodes, reduce='sum')  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing

def compute_clustering_coefficient(edge_index, max_num_nodes):
    adj = to_dense_adj(edge_index, max_num_nodes=max_num_nodes).squeeze(0)

    deg = adj.sum(dim=1)
    triangle = torch.mm(adj, torch.mm(adj, adj))
    clustering = triangle.diag() / (deg * (deg - 1))

    # Handling NaN values for nodes with degree 1 or 0
    clustering[deg <= 1] = 0.0
    return clustering

def compute_laplacian(graph, k, N, dev):
    edge_i, edge_w = get_laplacian(graph.edge_index, normalization="sym", num_nodes=N)
    dense_L = to_dense_adj(edge_index=edge_i, edge_attr=edge_w).squeeze()
    L = cp.asarray(dense_L)
    eigval, eigvec = cp.linalg.eigh(L)
    lapse = torch.tensor(eigvec[:, :k], device=dev)
    length = lapse.size(1)
    if length < k: ## zero padding if k > length
        lapse = torch.concat([lapse, torch.zeros((N, k - depth), device=dev)], dim=-1)
        
#     L = to_scipy_sparse_matrix(*get_laplacian(graph.edge_index, normalization="sym", num_nodes=N))
#     eigval_sparse, eigvec_sparse = eigsh(L, k=k)
#     print(np.sum(cp.asnumpy(eigval[:k]) - eigval_sparse))
    return lapse

def feature_augmentation(datasetlst, features, dev, k=4):
    ## features=True if it already has features.
    for graph in datasetlst:
#         groupA = time.time()
        num_nodes = graph.x.shape[0] if features else graph.num_nodes
        concat_edges = torch.cat([graph.edge_index[0], graph.edge_index[1]], dim=-1)
        degrees = utils.degree(concat_edges, num_nodes=num_nodes).view(-1, 1) ## degrees
        constant = torch.ones((num_nodes, 1), device=dev) ## constants
        clustering = compute_clustering_coefficient(graph.edge_index, num_nodes).view(-1, 1) # clustering 
        
#         groupB = time.time()
#         print("hey, how are you")
        lapse = compute_laplacian(graph, k=k, N=num_nodes, dev=dev) ## Lapse
#         lapse_time = time.time()
        RWSE = get_rw_landing_probs(ksteps=range(1, 17), edge_index=graph.edge_index, num_nodes=graph.x.size(0))
#         RWSE_time = time.time()
        
#         print(groupB - groupA)
#         print(lapse_time - groupB)
#         print(RWSE_time - lapse_time)
#         print('^^')
#         lapse = lapse.to(dev)
        if features:
            cpu = torch.device("cpu")
            graph.x = torch.cat([graph.x, constant, degrees, clustering, lapse, RWSE], dim=-1).to(cpu)
#             graph.x = torch.cat([graph.x, constant, degrees, clustering, RWSE], dim=-1).to(dev)
        else:
            graph.x = torch.cat([constant, degrees, clustering,], dim=-1).to(dev)

def prepare_data_regression(dataset, og_dataset=None, train_size=0.7, val_size=0.3, test_size=0):
    ## for multiple graphs.
    ## returns train, val, test split graphs.
    train_size = int(len(dataset) * train_size)
    val_size = int(len(dataset) * val_size)
    train = dataset[:train_size]
    val = dataset[train_size:train_size + val_size]
    test = dataset[train_size + val_size:]
    if og_dataset is not None:
        dataset = og_dataset
    print(f"length of dataset: {len(dataset)}")
    print(f"Num of feature dimensions: {dataset[0].num_features}")
    return train, val, test

def set_seed(seed=999):
    random.set_seed(seed)
    np.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_graphs(m, optimizer, train_loader, val_loader, args, accelerator):
    m, optimizer, train_loader, val_loader = accelerator.prepare(m, optimizer, train_loader, val_loader)
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    start = time.time()
    for epoch in range(args["epochs"]):
        train_loss = 0
        correct = 0
        total = 0
        m.train()
        for data in train_loader:
            data = data
            optimizer.zero_grad(set_to_none=True)
            out = m(data)
#             loss = F.cross_entropy(out, data.y)
            loss = F.mse_loss(out, data.y.view(out.size(0), 2))
#             for i, (int_out, int_out2) in enumerate(zip(intermediate_out, intermediate_out2)):
#                 loss += F.cross_entropy(int_out, dense_adj) ## intermediate losses
#                 loss += 0.05 * F.mse_loss(int_out, dense_adj) ## intermediate losses
#                 loss += F.mse_loss(int_out2, data.y) if args["multireg"] else F.mse_loss(int_out2, data.y.unsqueeze(1))
#             loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            train_loss += loss.item() 

#             pred_prob = F.softmax(out, dim=-1)
#             pred = torch.argmax(pred_prob, dim=-1)
            total += data.batch[-1].item()+1 ## get the last batch_id and +1 becuz id starts from 0
        train_loss /= len(train_loader)
#         train_mse = correct / total

        if epoch % 10 == 0 or epoch == args["epochs"] - 1:
            m.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                val_loss = 0
                for data in val_loader:
                    data = data
                    out = m(data)
                    loss = F.mse_loss(out, data.y.view(out.size(0), 2))

#                     pred_prob = F.softmax(out, dim=-1)
#                     pred = torch.argmax(pred_prob, dim=-1)
#                     sparse_y = torch.argmax(data.y, dim=-1)
#                     correct += (pred == data.y).sum().item() # this would be tensor
                    total += data.batch[-1].item()+1 ## get the last batch_id and +1 becuz id starts from 0
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            val_acc = correct / total
            
            train_losses += [train_loss]
            val_losses += [val_loss]
#             train_accs += [train_acc]
#             val_accs += [val_acc]
            accelerator.print(f"Epoch {epoch}:\t train_loss: {train_loss:.4f}\t val_loss: {val_loss:.4f}")
            accelerator.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
            })
    end = time.time()
    accelerator.print(end-start)
    accelerator.log({"total_train_time": end-start})
    accelerator.end_training()
    return train_losses, val_losses

