import sys
sys.path.insert(1, './tad_detection/')
sys.path.insert(1, './tad_detection/preprocessing/')
sys.path.insert(1, './tad_detection/model/')
sys.path.insert(1, './tad_detection/evaluation/')

import logging
logger = logging.getLogger('training')

from torch_geometric.nn.dense import dense_mincut_pool, dense_diff_pool
from torch_geometric.nn.conv import GraphConv, GCNConv
from model.gatconv import GATConv
from torch_geometric.nn.dense import DenseGraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, to_dense_adj
import math
from collections import Counter

class MinCutTAD_GraphClass(nn.Module):
    '''
    Model for unsupervised graph classification based on a MLP on a MinCutPool layer (Implementation described in: https://arxiv.org/abs/1907.00481) for pooling.

    :param parameters_user: dictionary with parameters set in parameters.json file
    :param n_clust: number of clusters (Supervised: TAD/ No-TAD; Unsupervised: TAD region count)
    '''

    def __init__(self, parameters_user, n_clust):
        super(MinCutTAD_GraphClass, self).__init__()

        self.max_num_nodes = parameters_user["max_num_nodes"]
        self.encoding_edge = parameters_user["encoding_edge"]
        self.hidden_size = parameters_user["hidden_conv_size"]
        self.number_genomic_annotations = len(parameters_user["genomic_annotations"])
        self.num_layers = parameters_user["num_layers"]
        self.num_classes = n_clust
        self.pooling_type = parameters_user["pooling_type"]

        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.rms = []

        if self.encoding_edge:
            self.convs.append(GCNConv(self.number_genomic_annotations, self.hidden_size, add_self_loops=False, aggr='add'))
        else:
            self.convs.append(GraphConv(self.number_genomic_annotations, self.hidden_size, aggr='add'))

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

        X, mask = to_dense_batch(x=X)
        adj = to_dense_adj(edge_index=edge_index, edge_attr=edge_attr)

        if self.pooling_type == "linear":
            s = self.pools[0](X)
        elif self.pooling_type == 'random':
            s = self.rms[0][:X.size(1), :].unsqueeze(dim=0).expand(X.size(0), -1, -1).to(X.device)

        X, adj, mc, o = dense_mincut_pool(X, adj, s, mask)

        for i in range(1, self.num_layers - 1):
            X = F.relu(self.convs[i](X, adj))
            if self.pooling_type == "linear":
                s = self.pools[i](X)
            elif self.pooling_type == 'random':
                s = self.rms[i][:X.size(1), :].unsqueeze(dim=0).expand(X.size(0), -1, -1).to(X.device)
            X, adj, mc_aux, o_aux = dense_mincut_pool(X, adj, s)
            mc += mc_aux
            o += o_aux

        X = self.convs[self.num_layers-1](X, adj)

        X = X.mean(dim=1)
        X = F.relu(self.lin1(X))
        X = self.lin2(X)

        return X, mc, o

class MinCutTAD(nn.Module):
    '''
    Model for node classification based on GraphConv (Implementation described in: https://arxiv.org/pdf/1810.02244.pdf) or GATConv (Implementation described in: https://arxiv.org/pdf/1710.10903.pdf resp. https://arxiv.org/pdf/2105.14491.pdf; In comparison to papers edge attributes are implemented.) message passing operation, an if trained in unsupervised a MinCutPool layer (Implementation described in: https://arxiv.org/abs/1907.00481) for pooling.

    :param parameters_user: dictionary with parameters set in parameters.json file
    :param n_clust: number of clusters (Supervised: TAD/ No-TAD; Unsupervised: TAD region count)
    '''

    def __init__(self, parameters_user, n_clust=None):
        super(MinCutTAD, self).__init__()

        self.max_num_nodes = parameters_user["max_num_nodes"]
        self.encoding_edge = parameters_user["encoding_edge"]
        self.number_genomic_annotations = len(parameters_user["genomic_annotations"])
        self.num_classes = len(parameters_user["classes"])
        self.pooling_type = parameters_user["pooling_type"]
        self.message_passing_layer = parameters_user["message_passing_layer"]
        self.n_clust = n_clust
        self.heads_num = parameters_user["attention_heads_num"]
        self.task_type = parameters_user["task_type"]

        self.message_passing_layer = parameters_user["message_passing_layer"]
        self.task_type = parameters_user["task_type"]
        if self.message_passing_layer == "GraphConv":
            if self.task_type == "supervised":
                self.conv1 = GraphConv(self.number_genomic_annotations, self.number_genomic_annotations*4)
                self.conv2 = GraphConv(self.number_genomic_annotations*4, self.num_classes)
            elif self.task_type == "unsupervised":
                self.conv1 = GraphConv(self.number_genomic_annotations, self.number_genomic_annotations*4)
                self.conv2 = GraphConv(self.number_genomic_annotations*4, self.n_clust)
        elif self.message_passing_layer == "GATConv":
            if self.task_type == "supervised":
                self.conv1 = GATConv(in_channels=self.number_genomic_annotations, out_channels=self.number_genomic_annotations*4, heads=self.heads_num, add_self_loops=False, edge_dim=1)
                self.conv2 = GATConv(in_channels=self.number_genomic_annotations*4*self.heads_num, out_channels=self.num_classes, heads=1, add_self_loops=False, edge_dim=1)
            elif self.task_type == "unsupervised":
                self.conv1 = GATConv(in_channels=self.number_genomic_annotations, out_channels=16, heads=self.heads_num, add_self_loops=False, edge_dim=1)
                self.conv2 = GATConv(in_channels=self.number_genomic_annotations*4*self.heads_num, out_channels=self.n_clust, heads=1, add_self_loops=False, edge_dim=1)

    def forward(self, x, edge_index, edge_attr):
        s = self.conv1(x, edge_index, edge_attr)
        s = F.relu(s)
        s = F.dropout(s, training=self.training)
        s = self.conv2(s, edge_index, edge_attr)

        if self.task_type == "unsupervised":
            x, _ = to_dense_batch(x=x)
            #X - torch.Size([1, 2493, 16])

            adj = to_dense_adj(edge_index=edge_index, edge_attr=edge_attr.float())
            #adj - torch.Size([1, 2493, 2493])

            x, adj, mc, o = dense_mincut_pool(x, adj, x)
            #x - torch.Size([1, 4, 16])
            #adj - torch.Size([1, 4, 4])

            s = torch.softmax(s, dim=-1)

            return s, mc, o

        elif self.task_type == "supervised":

            return F.log_softmax(s, dim=-1), 0, 0