import torch
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric.datasets as da
from torch_geometric.transforms import NormalizeFeatures
from collections import Counter

# Alias for torch_geometric.dataset.CitationFull with name="cora"
dataset_1 = da.CoraFull(root='../dataset/CoraFull', transform=None, pre_transform=None)
dataset_2 = da.CitationFull(root='../dataset/DBLP', name='DBLP', transform=None, pre_transform=None)
dataset_3 = da.Amazon(root='../dataset/Amazon', name='computers', transform=None, pre_transform=None)
dataset_4 = da.CitationFull(root='../dataset/DBLP', name='PubMed', transform=None, pre_transform=None)
dataset_5 = da.Coauthor(root='../dataset/Coauthor', name='Physics', transform=None, pre_transform=None)
# dataset_6 = da.Flickr(root='../dataset/Flickr', transform=None, pre_transform=None)
dataset_7 = da.GEDDataset(root='../dataset/AIDS700nef', name='AIDS700nef', train=True, transform=None, pre_transform=None, pre_filter=None)
#dataset_8 = da.DBP15K(root='../dataset/DBP15K', pair='en_zh', transform=None, pre_transform=None)
dataset_ori = da.Planetoid(root='../dataset/Planetoid', name='Cora', transform=NormalizeFeatures())

print(f'Dataset: {dataset_1}:')
print('======================')
print(f'Number of graphs: {len(dataset_1)}')
print(f'Number of features: {dataset_1.num_features}')

print()
print(f'Dataset: {dataset_2}:')
print('======================')
print(f'Number of graphs: {len(dataset_2)}')
print(f'Number of features: {dataset_2.num_features}')

print()
print(f'Dataset: {dataset_3}:')
print('======================')
print(f'Number of graphs: {len(dataset_3)}')
print(f'Number of features: {dataset_3.num_features}')

print()
print(f'Dataset: {dataset_4}:')
print('======================')
print(f'Number of graphs: {len(dataset_4)}')
print(f'Number of features: {dataset_4.num_features}')

print()
print(f'Dataset: {dataset_5}:')
print('======================')
print(f'Number of graphs: {len(dataset_5)}')
print(f'Number of features: {dataset_5.num_features}')

print()
#print(f'Dataset: {dataset_6}:')
#print('======================')
#print(f'Number of graphs: {len(dataset_6)}')
#print(f'Number of features: {dataset_6.num_features}')

print()
print(f'Dataset: {dataset_7}:')
print('======================')
print(f'Number of graphs: {len(dataset_7)}')
print(f'Number of features: {dataset_7.num_features}')

#print()
#print(f'Dataset: {dataset_8}:')
#print('======================')
#print(f'Number of graphs: {len(dataset_8)}')
#print(f'Number of features: {dataset_8.num_features}')

print()
print(f'Dataset: {dataset_ori}:')
print('======================')
print(f'Number of graphs: {len(dataset_ori)}')
print(f'Number of features: {dataset_ori.num_features}')


data_corafull = dataset_1[0]
data_dblpfull = dataset_2[0]
data_amazon = dataset_3[0]
data_pubmedfull = dataset_4[0]
data_coauthorphysics = dataset_5[0]
#data_flickr = dataset_6[0]
data_aids700nef = dataset_7[0]
#data_dbp15k = dataset_8[0]
data_cora = dataset_ori[0]

print('======================')
print(data_corafull, data_dblpfull, 
      data_amazon, data_pubmedfull, 
      data_coauthorphysics, 
      data_aids700nef, data_dblpfull, 
      data_amazon, data_pubmedfull, 
      data_coauthorphysics, 
      data_aids700nef, data_cora)

print("=====================")
# take 3 to see the counter number
print(Counter(data_cora.train_mask.numpy()), 
        Counter(data_cora.val_mask.numpy()), 
        Counter(data_cora.test_mask.numpy()))
