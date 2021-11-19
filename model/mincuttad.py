import neptune.new as neptune

import pandas as pd
import numpy as np
import os
from torch_geometric.nn.dense import dense_mincut_pool, dense_diff_pool
from torch_geometric import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F

#https://github.com/FilippoMB/Spectral-Clustering-with-Graph-Neural-Networks-for-Graph-Pooling/blob/master/Clustering.py
#https://github.com/pyg-team/pytorch_geometric/blob/74245f3a680c1f6fd1944623e47d9e677b43e827/torch_geometric/nn/dense/mincut_pool.py
#https://github.com/convei-lab/toptimize/blob/c4ef429c9174d8819279533ed8aead0fd2973791/toptimize/examples/proteins_mincut_pool.py

'''

X_1 = GraphConvSkip(P['n_channels'],
                    kernel_initializer='he_normal',
                    activation=P['ACTIV'])([X_in, A_in])


## IF MINCUTPOOL
Output:
Reduced node features of shape ([batch], K, n_node_features);
Reduced adjacency matrix of shape ([batch], K, K);
If return_mask=True, the soft clustering matrix of shape ([batch], n_nodes, K).
k: number of output nodes;

#https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/pooling/mincut_pool.py

'''

class MinCutTAD(nn.Module):
    def __init__(self, parameters, n_clust):
        super(MinCutTAD, self).__init__()
        self.n_clust = n_clust
        self.parameters = parameters
        self.conv = GraphConv()
        # TODO
        #We need sth. like: GCSConv - https://graphneural.network/layers/convolution/
        #As far as I am able to say we donÄt have sth. likte this: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GraphConv

    def forward(self, data):

        X, edge_index = data.X, data.edge_index
        X = self.conv(X, edge_index)

        for i in range(0, self.parameters["n_mincut_layer"]):
            X, edge_index, mincut_loss, ortho_loss = dense_mincut_pool(X, edge_index, self.n_clust)  # n_clust muss Tensor werden
        # labels_pred, #NOT SURE WHAT C is
        #X, edge_index, labels_pred, C = dense_diff_pool(X, , edge_index, self.n_clust) #n_clust muss Tensor werden
        #https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/pooling/diff_pool.py

        # TODO Check output
        return mincut_loss
