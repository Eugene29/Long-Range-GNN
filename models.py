import numpy as np
import pickle
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric import transforms
from torch_geometric import utils
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data
from torch_geometric.nn import Linear, global_mean_pool, global_add_pool
from torch_geometric.nn import GCNConv, GATv2Conv

class PMTGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2, num_hops=2):
        super().__init__()
        self.num_hops = num_hops

        def create_ffnn(input_dim=hidden_dim, hidden_dim=2*hidden_dim, output_dim=hidden_dim):
            ffnn = nn.Sequential(
                Linear(input_dim, hidden_dim),
                nn.GELU(),
                Linear(hidden_dim, output_dim)
            )
            return ffnn

        self.preprocess = create_ffnn(input_dim=input_dim)
#         self.deepsets = create_ffnn(input_dim=input_dim)
        self.ffnns = nn.ModuleList([create_ffnn() for _ in range(num_hops)]) # 1 for pre_process
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_hops)])
#         self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_hops)])
        self.postprocess = create_ffnn(output_dim=output_dim)
        self.bns = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(num_hops))
#         self.lns = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_hops))
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()
        
        x = self.preprocess(x)
        for i in range(self.num_hops):
            c = self.convs[i](x, edge_index) 
            x = self.bns[i](F.dropout(c, p=self.dropout) + x)
#             x = self.lns[i](x)
            x = F.dropout(self.ffnns[i](x), p=self.dropout) + x 
#             x = self.bns2?[i](F.dropout(c, p=self.dropout) + x)
            
        x = global_mean_pool(x, batch)
        x = self.postprocess(x)
        return x

class PMTGNN_VN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2, num_hops=2):
        super().__init__()
        self.num_hops = num_hops

        def create_ffnn(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim):
            ffnn = nn.Sequential(
                Linear(input_dim, hidden_dim),
#                 nn.ReLU(),
                nn.GELU(),
                Linear(hidden_dim, output_dim)
            )
            return ffnn

        self.vn = torch.nn.Parameter(torch.randn(1, hidden_dim)) ## VN embedding
        self.update_vn = nn.ModuleList([create_ffnn(hidden_dim, 2*hidden_dim) for _ in range(num_hops)])
#             self.propogate_vn = nn.ModuleList([Linear(4*hidden_dim, hidden_dim) for _ in range(num_hops)])
        self.preprocess = create_ffnn(input_dim=input_dim)
        self.ffnns = nn.ModuleList([create_ffnn() for _ in range(num_hops)]) # 1 for pre_process
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_hops)])
#         self.convs = nn.ModuleList([GATv2Conv(hidden_dim, hidden_dim//8, heads=8) for _ in range(num_hops)])
        self.postprocess = create_ffnn(output_dim=output_dim)
        self.bns = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(num_hops))
        self.bns2 = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(num_hops))
#         self.lns = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_hops))
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()
        
        x = self.preprocess(x)
        for i in range(self.num_hops):
            ## adds the mean and ffnn to send out
            vn = self.vn + self.update_vn[i](global_mean_pool(x, batch)) ## broadcasting here?
            vn = vn[batch] ## reversing it to 2D
#             outvn = self.propogate_vn[i](vn) # [B, H]
            conv = self.convs[i](F.dropout(vn) + x, edge_index)
            x = self.bns[i](F.dropout(conv, p=self.dropout) + x)
            x = F.dropout(self.ffnns[i](x), p=self.dropout) + x
            x = self.bns2[i](x)
#             x = self.lns[i](x)
#             else:
#                 x = self.ffnns[i](x) + x
#         x = F.dropout(x, p=self.dropout)
            
        x = global_mean_pool(x, batch)
        x = self.postprocess(x)
        return x

class PMTGNN_GT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=8, dropout=0.1, num_blocks=5, exponential_dim=True):
        super().__init__()
        self.num_blocks = num_blocks

        def create_ffnn(input_dim=hidden_dim, output_dim=hidden_dim, hid_dim=2*hidden_dim):
            ffnn = nn.Sequential(
                Linear(input_dim, hid_dim),
                nn.GELU(),
                Linear(hid_dim, output_dim)
            )
            return ffnn
        hid_dims = [hidden_dim for _ in range(num_blocks)]
        
        self.preprocess = create_ffnn(input_dim=input_dim)
        if exponential_dim:
            self.convs = nn.ModuleList([GCNConv(i, i) for i in hid_dims])
#             self.convs = nn.ModuleList([GATv2Conv(i, i//heads, heads) for i in hid_dims])
#             self.Q = nn.ModuleList([create_ffnn(i, i) for i in qkv_dim])
#             self.K = nn.ModuleList([create_ffnn(i, i) for i in qkv_dim])
#             self.V = nn.ModuleList([create_ffnn(i, i) for i in qkv_dim])
#             self.Q = nn.ModuleList([Linear(i, i) for i in hid_dims])
#             self.K = nn.ModuleList([Linear(i, i) for i in hid_dims])
#             self.V = nn.ModuleList([Linear(i, i) for i in hid_dims])
            self.self_attn = torch.nn.MultiheadAttention(hidden_dim, heads=heads, dropout=0.5, batch_first=True)
            self.bns_mpnn = nn.ModuleList(nn.BatchNorm1d(i) for i in hid_dims)
            self.bns_former = nn.ModuleList(nn.BatchNorm1d(i) for i in hid_dims)
            self.bns = nn.ModuleList(nn.BatchNorm1d(i) for i in hid_dims)
            self.ffnns = nn.ModuleList([create_ffnn() for i in hid_dims])
            self.postprocess = create_ffnn(output_dim=output_dim)
        self.dropout = dropout

    def forward(self, data):
#         x, edge_index, batch, edge_weight = data.x, data.edge_index, data.batch, data.edge_weight
        x, edge_index, batch,  = data.x, data.edge_index, data.batch,
        x = x.float()
        intermediate_Adj_losses, intermediate_out = [], []
        x, mask = to_dense_adj(x, batch=batch)
        
        x = self.preprocess(x)
        for i in range(self.num_blocks):
            c = self.convs[i](x, edge_index)
            c = self.bns_mpnn[i](F.dropout(c, p=self.dropout) + x) 
            ## mask lets attention mechanism ignore padded parts (from batching)
            t = self.self_attn[i](x, x, x, key_padding_mask=~mask, need_weights=False)[0]
            t = self.bns_former[i](t + x) ## dropout already applied
            x = self.ffnns[i](c + t)
            x = self.bns[i](x)
        x = x[mask] ## back to 2D
        x = global_mean_pool(x, batch)
        x = self.postprocess(x)
        return x, intermediate_Adj_losses, intermediate_out


