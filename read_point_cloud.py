import numpy as np
import pickle
import torch
import pandas as pd
from torch_geometric import transforms
from torch_geometric import utils
from torch_geometric.data import Data

## not used
# def load_data(version):
#     fname = f"PointNet{version}.pickle"
#     with open(fname, 'rb') as f:
#         while True:
#             try: 
#                 loaded_dictionary = pickle.load(f, encoding="latin1")
#             except pickle.UnpicklingError:
#                 print("Reached the end of file!")
#                 break
#             except:
#                 print("only grabbing the first one")
#                 break
#     return loaded_dictionary

## load pmt pos into a tensor
def get_pmtxyz(fname):
    df = pd.read_csv(fname, delim_whitespace=True, header=None)
    pmtpos = torch.tensor(df.values)
    pmtxyz = pmtpos[:, 1:] # get rid of redundant id column
    print(f"print pmtpos shape: {pmtxyz.shape}") # [2126, 3]
    return pmtxyz

def turn_dict_to_tensor(loaded_dict, pmtxyz, dev):
    ## create node features
    pmtsize = 2126
    nodes = torch.zeros((pmtsize, 2)).float()
    
    hitids = loaded_dict["hitid"]
    nodes[hitids] = torch.tensor([loaded_dict["hitcharge"], loaded_dict["hittime"]], dtype=torch.float).T
    nodes = torch.cat([nodes, pmtxyz], dim=-1) ## add positions to the node features
    return nodes

def turn_dictionary_to_geom_data(loaded_dict, pmtxyz, dev, k=3):
    ### Turns loaded_dict into a torch_geom data ###
    
    ## create node features
    pmtsize = 2126
    nodes = torch.zeros((pmtsize, 2)).float()
    
    hitids = loaded_dict["hitid"]
    nodes[hitids] = torch.tensor([loaded_dict["hitcharge"], loaded_dict["hittime"]], dtype=torch.float).T
#     nodes = torch.cat([nodes, noidpmt], dim=-1) ## add position to the node features
    
    ## Create a graph using radius graph
    raw_data = Data(x=nodes, pos=pmtxyz).to(dev)
#     transform = transforms.RadiusGraph(r=100)
    transform = transforms.KNNGraph(k=k)
    data = transform(raw_data)
    data.edge_index = utils.to_undirected(data.edge_index)
    
    ## virtual node
#     vn = torch.randn(nodes.size(-1)).to(dev)
#     vn[0] = torch.tensor(loaded_dict["energy"])
#     data.x = torch.cat([data.x, vn.view(1, -1)], dim=0)
#     new_edge_idx = torch.tensor([[2126, i] for i in range(2126)]).T.to(dev)
#     data.edge_index = torch.cat([data.edge_index, new_edge_idx], dim=-1)
    
    zpos = loaded_dict["zpos"]
    vertex = loaded_dict["vertex"]
    data.y = torch.tensor([zpos, vertex])
    return data

## two main ways to laod data (graph or tensor(for MLP or Transformers))
def load_graph(versions, pmtxyz, k, dev):
    datasetlst = []
    for v in versions:
        fname = f"PointNet{v}.pickle"
        with open(fname, 'rb') as f:
            print(f"opening file version {v}")
            while True:
#             for _ in range(100): # test
                try: 
                    loaded_dictionary = pickle.load(f, encoding="latin1")
                    datasetlst += [turn_dictionary_to_geom_data(loaded_dictionary, pmtxyz, dev, k)]
                except pickle.UnpicklingError:
                    print(f"finished reading file version {v}")
                    break
                except Exception as e:
                    print(e)
                    break
    return datasetlst

def load_tensor(versions, pmtxyz, dev):
    datasetlst = []
    for v in versions:
        fname = f"PointNet{v}.pickle"
        with open(fname, 'rb') as f:
            print(f"opening file version {v}")
            while True:
                try: 
                    loaded_dictionary = pickle.load(f, encoding="latin1")
                    datasetlst += [turn_dict_to_tensor(loaded_dictionary, pmtxyz, dev)]
                except pickle.UnpicklingError:
                    print(f"finished reading file version {v}")
                    break
                except Exception as e:
                    print(e)
                    break
    return torch.stack(datasetlst, dim=0)







