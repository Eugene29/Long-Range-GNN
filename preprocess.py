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
## .py imports ##
from models import PMTGCN
from read_point_cloud import get_pmtxyz, load_graph, load_tensor
from tools import feature_augmentation
import sys 

## How do I remind myself what the args are? 
print(f'args provided: {sys.argv}')
dev = "cuda" if torch.cuda.is_available() else "cpu"
dev

pmtxyz = get_pmtxyz('pmt_xyz.dat')
pmtsize = 2126
datasetlst = []

versions = [2]
if sys.argv[1] == "graph":
    datasetlst = load_graph(versions, pmtxyz, dev)
    print(len(datasetlst))
    print(f"edge_index: {datasetlst[0].edge_index.shape}")
    ## graph augmentation (extra features)
    feature_augmentation(datasetlst, features=True, dev=dev)
else:
    datasetlst = load_tensor(versions, pmtxyz, dev)
    print(datasetlst.shape)

torch.save(datasetlst, "datasetlst_v2.pt")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(*pmtxyz.T)
plt.savefig("pmt_sensors.png")



