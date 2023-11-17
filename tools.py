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
from accelerate import Accelerator
import accelerate
import wandb
# import networkx as nx
# import pickle
    
def compute_clustering_coefficient(edge_index, max_num_nodes):
    adj = to_dense_adj(edge_index, max_num_nodes=max_num_nodes).squeeze(0)

    deg = adj.sum(dim=1)
    triangle = torch.mm(adj, torch.mm(adj, adj))
    clustering = triangle.diag() / (deg * (deg - 1))

    # Handling NaN values for nodes with degree 1 or 0
    clustering[deg <= 1] = 0.0
    return clustering

# def compute_laplacian(graph, k):
#     edge_index = graph.edge_index
#     lap_edges, lap_weights = utils.get_laplacian(edge_index, normalization="sym")
#     N = graph.x.size(0)
#     L = to_dense_adj(edge_index=lap_edges, edge_attr=lap_weights, max_num_nodes=N).squeeze()#[48, 0]
# #         eigval, eigvec  = torch.linalg.eigh(L)
#     eigval, eigvec = eigsh(L.to("cpu").numpy(), k=k)
#     lapse = eigvec[:, :k]
#     lapse = torch.tensor(eigvec[:, :k])
#     depth = lapse.size(1)
#     if depth != k:
#         lapse = torch.concat([lapse, torch.zeros((N, k - depth))], dim=-1)
#     return lapse

def feature_augmentation(datasetlst, features, dev, k=5):
    for graph in datasetlst:
        num_nodes = graph.x.shape[0] if features else graph.num_nodes.to(dev)
        concat_edges = torch.cat([graph.edge_index[0], graph.edge_index[1]], dim=-1).to(dev)
        degrees = utils.degree(concat_edges, num_nodes=num_nodes).view(-1, 1).to(dev) ## degrees
        constant = torch.ones((num_nodes, 1)).to(dev) #.to(args['dev']) ## constants
#         lapse = compute_laplacian(graph, k=k).to(args["dev"]) ## Lapse

        clustering = compute_clustering_coefficient(graph.edge_index, num_nodes).view(-1, 1).to(dev) # clustering coefficient
        if features:
            graph.x = torch.cat([graph.x, constant, degrees, clustering,], dim=-1)
        else:
            graph.x = torch.cat([constant, degrees, clustering,], dim=-1)

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

def train_graphs(m, optimizer, train_loader, val_loader, args):
    ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=False) # , broadcast_buffers=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], log_with="wandb")
    accelerator.init_trackers(project_name="PMT")
    m, optimizer, train_loader, val_loader = accelerator.prepare(m, optimizer, train_loader, val_loader)
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
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
                "batch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })
    wandb.finish()
    return train_losses, val_losses


