import torch
import numpy as np
import pandas as pd
import networkx as nx
import pickle
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric import transforms
from torch_geometric import utils
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
## .py imports ##
from models import PMTGNN, PMTGNN_VN, PMTGNN_GT
from tools import prepare_data_regression, train_graphs
import wandb
import accelerate
from accelerate import Accelerator
import random
import sys
import os

args = {
    "epochs": 200, ## fixed
    "batch_size": 32,
    "dropout": 0.3,
    "lr": 0.0003,
    "num_hops": 15,
    "graph": True, ## fixed
    "gnn": "PMTGNN",
    "dim_h": 64,
}

if len(sys.argv) < 8:
    print("you need these args: gnn, batch_size, num_hops, lr, dim_h, dropout")
else:
    args["gnn"] = sys.argv[1]
    args["batch_size"] = int(sys.argv[2])
    args["num_hops"] = int(sys.argv[3])
    args["lr"] = float(sys.argv[4])
    args["dim_h"] = int(sys.argv[5])
    args["dropout"] = float(sys.argv[6])
#     args["wandb"] = bool(sys.argv[7])
    os.environ["WANDB_DISABLED"] = sys.argv[7]

# torch.cuda.empty_cache()
ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=False, broadcast_buffers=False)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], log_with="wandb")
accelerator.init_trackers(project_name="PMT", config=args)

device = torch.device('cpu')
# datasetlst = torch.load('datasetlst_v2.pt', map_location=torch.device('cpu'))
datasetlst = torch.load('knn8_datasetlst_[1].pt', map_location=device)
# datasetlst = torch.load("tiny_dataset.pt")
train, val, test = prepare_data_regression(datasetlst)

## batch train, val
batch_size = args["batch_size"]
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size)

## Build Model
input_dim = datasetlst[0].num_features ## 
hidden_dim = args["dim_h"] # 4 * input_dim
output_dim = datasetlst[0].y.size(0)
if args["gnn"] == "PMTGNN":
    m = PMTGNN(input_dim, hidden_dim, output_dim, dropout=args["dropout"], num_hops = args["num_hops"])
elif args["gnn"] == "PMTGNN_VN":
    m = PMTGNN_VN(input_dim, hidden_dim, output_dim, dropout=args["dropout"], num_hops = args["num_hops"])
elif args["gnn"] == "PMTGNN_GT":
    m = PMTGNN_GT(input_dim, hidden_dim, output_dim, dropout=args["dropout"], num_hops = args["num_hops"])
else:
    raise IndexError
learnable = [p.numel() for p in m.parameters() if p.requires_grad]
accelerator.print(sum(learnable))

optim = torch.optim.Adam(m.parameters(), lr=args["lr"], weight_decay = 5e-5)
train_losses, val_losses = train_graphs(m, optim, train_loader, val_loader, args, accelerator)
