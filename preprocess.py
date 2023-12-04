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
from read_point_cloud import get_pmtxyz, load_graph, load_tensor
from tools import feature_augmentation
import sys
import time

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pmtxyz = get_pmtxyz('pmt_xyz.dat')
pmtsize = 2126
datasetlst = []

print(f'args provided: {sys.argv}')
if len(sys.argv) < 4:
    raise ValueError("you need to pass args: \n graph:{graph, tensor}, k: (int: knn), versions: {1, 2, 3, 12, ..., 123})")

k = int(sys.argv[2])
versions = [int(x) for x in sys.argv[3]]
# versions = [2]
if sys.argv[1] == "graph":
    datasetlst = load_graph(versions, pmtxyz, dev=dev, k=k)
    print(len(datasetlst))
    print(f"edge_index: {datasetlst[0].edge_index.shape}")
    ## graph augmentation (extra features)
    start = time.time()
    feature_augmentation(datasetlst, features=True, dev=dev)
    end = time.time()
    print(end - start)
else:
    datasetlst = load_tensor(versions, pmtxyz, dev)
    print(datasetlst.shape)

# torch.save(datasetlst, "datasetlst_v2.pt")
torch.save(datasetlst, f"knn{k}_datasetlst_{str(versions)}.pt")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(*pmtxyz.T)
plt.savefig("pmt_sensors.png")
