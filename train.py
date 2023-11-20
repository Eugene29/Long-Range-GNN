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
from models import PMTGCN, PMTGCN_VN
from tools import prepare_data_regression, train_graphs
import wandb
import accelerate
from accelerate import Accelerator
import random

args = {
    "epochs": 800,
    "batch_size": 32,
    "dropout": 0.3,
    "lr": 0.0003,
    "num_hops": 15,
    "vn": True,
    "graph": True,
}

torch.cuda.empty_cache()
ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=False, broadcast_buffers=False)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], log_with="wandb")
accelerator.init_trackers(project_name="PMT", config=args)

device = torch.device('cpu')
# datasetlst = torch.load('datasetlst_v2.pt', map_location=torch.device('cpu'))
datasetlst = torch.load('knn6_datasetlst_v2.pt', map_location=device)
# datasetlst = torch.load("tiny_dataset.pt")
train, val, test = prepare_data_regression(datasetlst)

## batch train, val
batch_size = args["batch_size"]
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size)

## Build Model
input_dim = datasetlst[0].num_features ## 3 augmented
hidden_dim = 64 # 4 * input_dim
output_dim = datasetlst[0].y.size(0)
m = PMTGCN_VN(input_dim, hidden_dim, output_dim, dropout=args["dropout"], num_hops = args["num_hops"], virtual=args["vn"])
learnable = [p.numel() for p in m.parameters() if p.requires_grad]
accelerator.print(sum(learnable))

optim = torch.optim.Adam(m.parameters(), lr=args["lr"], weight_decay = 5e-4)
train_losses, val_losses = train_graphs(m, optim, train_loader, val_loader, args, accelerator)