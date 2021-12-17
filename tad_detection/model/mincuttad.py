import sys
sys.path.insert(1, './tad_detection/')
sys.path.insert(1, './tad_detection/preprocessing/')
sys.path.insert(1, './tad_detection/model/')
sys.path.insert(1, './tad_detection/evaluation/')

import logging
logger = logging.getLogger('training')

import pandas as pd
import numpy as np
import os
from torch_geometric.nn.dense import dense_mincut_pool, dense_diff_pool
from torch_geometric.nn.conv import GCNConv, GraphConv
from torch_geometric.nn.dense import DenseGraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, to_dense_adj
import math

#https://github.com/FilippoMB/Spectral-Clustering-with-Graph-Neural-Networks-for-Graph-Pooling/blob/master/Clustering.py
#https://github.com/pyg-team/pytorch_geometric/blob/74245f3a680c1f6fd1944623e47d9e677b43e827/torch_geometric/nn/dense/mincut_pool.py
#https://github.com/convei-lab/toptimize/blob/c4ef429c9174d8819279533ed8aead0fd2973791/toptimize/examples/proteins_mincut_pool.py

class MinCutTAD_GraphClass(nn.Module):
    def __init__(self, parameters_user):
        super(MinCutTAD, self).__init__()

        self.max_num_nodes = parameters_user["max_num_nodes"]
        self.encoding_edge = parameters_user["encoding_edge"]
        self.hidden_size = parameters_user["hidden_conv_size"]
        self.number_genomic_annotations = len(parameters_user["genomic_annotations"])
        self.num_layers = parameters_user["num_layers"]
        self.num_classes = len(parameters_user["classes"])
        self.pooling_type = parameters_user["pooling_type"]

        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.rms = []

        if self.encoding_edge:
            self.convs.append(GCNConv(self.number_genomic_annotations, self.hidden_size, add_self_loops=False, aggr='add'))
        else:
            self.convs.append(GraphConv(self.number_genomic_annotations, self.hidden_size, aggr='add'))

        # TODO
        #We need sth. like: GCSConv - https://graphneural.network/layers/convolution/
        #As far as I am able to say we don't have sth. likte this: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GraphConv

        if self.num_layers > 1:
            for i in range(self.num_layers):
                self.convs.append(DenseGraphConv(self.hidden_size, self.hidden_size))

        num_nodes = self.max_num_nodes
        for i in range(self.num_layers - 1):
            num_nodes = math.ceil(0.5 * num_nodes)
            if self.pooling_type == 'linear':
                self.pools.append(nn.Linear(self.hidden_size, num_nodes))
            elif self.pooling_type == 'random':
                self.rms.append(torch.rand(math.ceil(2 * num_nodes), num_nodes))

        self.lin1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.lin2 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, X, edge_index, edge_attr):

        X = F.relu(self.convs[0](X, edge_index))
        #X - torch.Size([2493, 8])

        X, mask = to_dense_batch(x=X)
        #X - torch.Size([1, 2493, 8])
        #mask - torch.Size([1, 2493])
        adj = to_dense_adj(edge_index=edge_index, edge_attr=edge_attr)
        #adj - torch.Size([1, 2493, 2493])

        if self.pooling_type == "linear":
            s = self.pools[0](X)
        elif self.pooling_type == 'random':
            s = self.rms[0][:X.size(1), :].unsqueeze(dim=0).expand(X.size(0), -1, -1).to(X.device)
        #s - torch.Size([1, 2493, 1247])

        X, adj, mc, o = dense_mincut_pool(X, adj, s, mask)
        #X - torch.Size([1, 1247, 8])
        #edge_index - torch.Size([1, 1247, 1247])

        for i in range(1, self.num_layers - 1):
            X = F.relu(self.convs[i](X, adj))
            #x - torch.Size([1, 1247, 8]), torch.Size([1, 624, 8])
            if self.pooling_type == "linear":
                s = self.pools[i](X)
            elif self.pooling_type == 'random':
                s = self.rms[i][:X.size(1), :].unsqueeze(dim=0).expand(X.size(0), -1, -1).to(X.device)
            #s - torch.Size([1, 1247, 624]),  torch.Size([1, 624, 312])
            X, adj, mc_aux, o_aux = dense_mincut_pool(X, adj, s)
            #x - torch.Size([1, 624, 8]), torch.Size([1, 312, 8])
            #adj - torch.Size([1, 624, 624]), torch.Size([1, 312, 312])
            #mc_aux -
            # o_aux
            mc += mc_aux
            o += o_aux

        X = self.convs[self.num_layers-1](X, adj)
        #x - torch.Size([1, 312, 8])

        X = X.mean(dim=1)
        #x - torch.Size([1, 8])
        X = F.relu(self.lin1(X))
        # x - torch.Size([1, 8])
        X = self.lin2(X)
        # x - torch.Size([1, 2])

        return X, mc, o


class MinCutTAD(nn.Module):
    def __init__(self, parameters_user, n_clust):
        super(MinCutTAD, self).__init__()

        self.max_num_nodes = parameters_user["max_num_nodes"]
        self.encoding_edge = parameters_user["encoding_edge"]
        self.hidden_size = parameters_user["hidden_conv_size"]
        self.number_genomic_annotations = len(parameters_user["genomic_annotations"])
        self.num_layers = parameters_user["num_layers"]
        self.num_classes = len(parameters_user["classes"])
        self.pooling_type = parameters_user["pooling_type"]
        self.n_clust = n_clust

        self.hidden_size = 32

        self.conv1 = GraphConv(self.number_genomic_annotations, self.hidden_size, aggr='add')

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, math.ceil(self.hidden_size * 2)),
            nn.ReLU(),
            nn.Linear(math.ceil(self.hidden_size * 2), math.ceil(self.hidden_size * 4)),
            nn.ReLU(),
            nn.Linear(math.ceil(self.hidden_size * 4), math.ceil(self.hidden_size * 2)),
            nn.ReLU(),
            nn.Linear(math.ceil(self.hidden_size * 2), n_clust),
        )


    def forward(self, X, edge_index, edge_attr):

        X = self.conv1(X, edge_index, edge_attr.float()) #hidden_size = 16
        # X - torch.Size([2493, 16])

        X, _ = to_dense_batch(x=X)
        #X - torch.Size([1, 2493, 16])

        adj = to_dense_adj(edge_index=edge_index, edge_attr=edge_attr.float())
        #adj - torch.Size([1, 2493, 2493])

        # s = X.detach()
        s = self.mlp(X)

        #s - torch.Size([1, 2493, 4]) #n_clust = 4

        X, adj, mc, o = dense_mincut_pool(X, adj, s)
        #x - torch.Size([1, 4, 16])
        #adj - torch.Size([1, 4, 4])

        s = torch.softmax(s, dim=-1)
        #s = np.where(s == 1)[2]
        #np.argmax(s.detach().numpy(), axis=-1)

        return s, mc, o