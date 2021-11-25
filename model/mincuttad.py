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


'''
Example stacking several layers:

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)
'''

'''
model = MincutPool(num_features=stats['num_features'], num_classes=stats['num_classes'],
                   max_num_nodes=stats['max_num_nodes'], hidden=args.hidden_dim,
                   pooling_type=args.pooling_type, num_layers=args.num_layers, encode_edge=encode_edge).to(device)

optimizer = Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, min_lr=1e-5,
                              patience=args.lr_decay_patience, verbose=True)

if args.dataset == 'ZINC':
    train = train_regression
    evaluate = evaluate_regression

train_sup_losses, train_lp_losses, train_entropy_losses = [], [], []
val_sup_losses, val_lp_losses, val_entropy_losses = [], [], []
test_sup_losses, test_lp_losses, test_entropy_losses = [], [], []
val_accuracies, test_accuracies = [], []

epochs_no_improve = 0  # used for early stopping
for epoch in range(1, args.max_epochs + 1):

    # train
    train_sup_loss, train_lp_loss, train_entropy_loss = \
        train(model, optimizer, train_loader, device)

    # validation
    val_acc, val_sup_loss, val_lp_loss, val_entropy_loss \
        = evaluate(model, val_loader, device, evaluator=evaluator)

    # test
    test_acc, test_sup_loss, test_lp_loss, test_entropy_loss = \
        evaluate(model, test_loader, device, evaluator=evaluator)

    val_accuracies.append(val_acc)
    test_accuracies.append(test_acc)

    train_sup_losses.append(train_sup_loss)
    train_lp_losses.append(train_lp_loss)
    train_entropy_losses.append(train_entropy_loss)

    val_sup_losses.append(val_sup_loss)
    val_lp_losses.append(val_lp_loss)
    val_entropy_losses.append(val_entropy_loss)

    test_sup_losses.append(test_sup_loss)
    test_lp_losses.append(test_lp_loss)
    test_entropy_losses.append(test_entropy_loss)

    if (epoch-1) % args.interval == 0:
        print(f'{epoch:03d}: Train Sup Loss: {train_sup_loss:.3f}, '
          f'Val Sup Loss: {val_sup_loss:.3f}, Val Acc: {val_accuracies[-1]:.3f}, '
          f'Test Sup Loss: {test_sup_loss:.3f}, Test Acc: {test_accuracies[-1]:.3f}')

    scheduler.step(val_acc)
        
   
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
