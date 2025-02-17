from sysconfig import get_config_var
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.nn.pytorch import GATConv
from dgl.data import register_data_args, load_data
from dgl.nn.pytorch.conv import SAGEConv

import numpy as np

from pdb import set_trace


'''Stochastic R-GCN
'''
class StochasticRGCN(nn.Module):
    def __init__(self, in_dim, h1_dim, h2_dim, emb_dim):
        super(StochasticRGCN, self).__init__()
        self.conv1 = dgl.nn.RelGraphConv(
            in_dim, h1_dim, num_rels=2, layer_norm=True)
        self.conv2 = dgl.nn.RelGraphConv(
            h1_dim, h2_dim, num_rels=2, layer_norm=True)

        self.dropout = nn.Dropout(p=0.2)

        self.proj = nn.Linear(h2_dim, emb_dim)

        nn.init.xavier_normal_(self.proj.weight, gain=1.414)
        nn.init.xavier_normal_(self.clf.weight, gain=1.414)

    def forward(self, blocks, h):
        h = self.conv1(blocks[0], h, blocks[0].edata['t'])
        h = self.conv2(blocks[1], h, blocks[1].edata['t'])
        emb = self.proj(h)
        return emb


'''Stochastic GCN
'''
class StochasticGCN(nn.Module):
    def __init__(self, in_dim, h1_dim, h2_dim, emb_dim):
        super(StochasticGCN, self).__init__()
        self.conv1 = dgl.nn.GraphConv(
            in_dim, h1_dim, allow_zero_in_degree=True)
        self.conv2 = dgl.nn.GraphConv(
            h1_dim, h2_dim, allow_zero_in_degree=True)

        # self.dropout = nn.Dropout(p=0.2)

        self.proj = nn.Linear(h2_dim, emb_dim)

        # nn.init.xavier_normal_(self.proj.weight, gain=1.414)

    def forward(self, blocks, h):
        h = self.conv1(blocks[0], h)
        h = self.conv2(blocks[1], h)
        h = self.proj(h)
        return h

'''Stochastic GAT
'''
class StochasticGAT(nn.Module):
    def __init__(self, in_dim, h1_dim, h2_dim, num_heads, emb_dim):
        super(StochasticGAT, self).__init__()
        self.conv1 = dgl.nn.GATConv(
            in_dim, h1_dim, num_heads, allow_zero_in_degree=True)
        self.conv2 = dgl.nn.GATConv(
            h1_dim*num_heads, h2_dim, num_heads, allow_zero_in_degree=True)

        self.proj = nn.Linear(h2_dim*num_heads, emb_dim)
        # self.cls = nn.Linear(emb_dim, out_dim)

        nn.init.xavier_normal_(self.proj.weight, gain=1.414)
        # nn.init.xavier_normal_(self.cls.weight, gain=1.414)

    def forward(self, blocks, h):
        h = self.conv1(blocks[0], h)
        h = torch.reshape(h, (h.shape[0], -1))
        h = self.conv2(blocks[1], h)
        h = torch.reshape(h, (h.shape[0], -1))

        out= self.proj(h)
        # logits = self.cls(emb)
        return h, out


'''Stochastic GraphSAGE
'''
class StochasticGraphSAGE(nn.Module):
    def __init__(self, in_dim, h1_dim, h2_dim, out_dim):
        super(StochasticGraphSAGE, self).__init__()
        self.conv1 = dgl.nn.SAGEConv(in_dim, h1_dim, 'pool')
        self.conv2 = dgl.nn.SAGEConv(h1_dim, h2_dim, 'pool')
        self.cls = nn.Linear(h2_dim, out_dim)
        nn.init.xavier_normal_(self.cls.weight, gain=1.414)

    def forward(self, blocks, h):
        h = self.conv1(blocks[0], h)
        h = self.conv2(blocks[1], h)
        logits = self.cls(h)
        return h, logits
