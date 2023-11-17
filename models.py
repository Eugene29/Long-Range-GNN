import numpy as np
import pickle
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric import transforms
from torch_geometric import utils
from torch_geometric.data import Data
from torch_geometric.nn import Linear, global_mean_pool, global_add_pool
from torch_geometric.nn import GCNConv, GATv2Conv

class PMTGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2, num_hops=2):
        super().__init__()
        self.num_hops = num_hops

        def create_ffnn(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim):
            ffnn = nn.Sequential(
                Linear(input_dim, hidden_dim),
                nn.GELU(),
                Linear(hidden_dim, output_dim)
            )
            return ffnn

        self.preprocess = create_ffnn(input_dim=input_dim)
        self.ffnns = nn.ModuleList([create_ffnn() for _ in range(num_hops)]) # 1 for pre_process
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_hops)])
        self.postprocess = create_ffnn(output_dim=output_dim)
        self.bns = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(num_hops))
#         self.lns = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_hops))
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()
        
        x = self.preprocess(x)
        for i in range(self.num_hops):
            x = self.convs[i](x, edge_index) + x
            x = self.bns[i](x)
#             x = self.lns[i](x)
            x = F.dropout(x, p=self.dropout) ## pre-ffnn dropout
            x = self.ffnns[i](x) + x
            x = F.dropout(x, p=self.dropout) ## post-ffnn dropout
            
        x = global_mean_pool(x, batch)
        x = self.postprocess(x)
        return x

class PMTGCN_VN(nn.Module):
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

        self.preprocess = create_ffnn(input_dim=input_dim)
        self.ffnns = nn.ModuleList([create_ffnn() for _ in range(num_hops)]) # 1 for pre_process
#         self.vn = torch.nn.Parameter(torch.randn(1, 2 * hidden_dim))
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_hops)])
#         self.convs = nn.ModuleList([GATv2Conv(hidden_dim, hidden_dim//8, heads=8) for _ in range(num_hops)])
        self.postprocess = create_ffnn(output_dim=output_dim)
        self.bns = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(num_hops))
#         self.lns = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_hops))
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()
        
        x = self.preprocess(x)
        for i in range(self.num_hops):
            x = self.convs[i](x, edge_index) + x
            ## Try two different messages method? (one before vn aggregating) ##
#             x = self.vn[i](global_mean_pool(x, batch)) + x
#             if i == self.num_hops // 2:
            x = self.bns[i](x)
#             x = self.lns[i](x)
            x = self.ffnns[i](x) + x
            x = F.dropout(x, p=self.dropout) ## post-ffnn dropout
#         x = F.dropout(x, p=self.dropout)
            
        x = global_mean_pool(x, batch)
        x = self.postprocess(x)
        return x











