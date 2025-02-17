from sysconfig import get_config_var
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

import numpy as np

from pdb import set_trace


class TweetEmbedder(nn.Module):
    def __init__(
        self, in_dim, h1_dim, h2_dim, 
        com_proj_dim, proj_t1_dim, proj_t2_dim, 
        num_heads=None, conv_type='gat',
        ):
        super(TweetEmbedder, self).__init__()
        if conv_type == 'gat':
            self.conv1 = dgl.nn.GATv2Conv(
                in_dim, h1_dim, num_heads, allow_zero_in_degree=True,
                )
            self.conv2 = dgl.nn.GATv2Conv(
                h1_dim*num_heads, h2_dim, num_heads, allow_zero_in_degree=True,
                )
            self.com_proj = nn.Linear(h1_dim*num_heads, com_proj_dim)
        elif conv_type == 'gcn':
            self.conv1 = dgl.nn.GraphConv(
                in_dim, h1_dim, norm='both', weight=True, bias=True, 
                allow_zero_in_degree=True,
            )
            self.com_proj = nn.Linear(h1_dim, com_proj_dim)
        elif conv_type == 'sage':
            self.conv1 = dgl.nn.SAGEConv(
                in_dim, h1_dim, 'pool',
            )
            self.com_proj = nn.Linear(h1_dim, com_proj_dim)
            
        self.conv_type = conv_type

        self.proj_t1 = nn.Linear(com_proj_dim, proj_t1_dim)
        self.proj_t2 = nn.Linear(com_proj_dim, proj_t2_dim)
        self.cls = nn.Linear(com_proj_dim, 2)

        nn.init.xavier_normal_(self.proj_t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.proj_t2.weight, gain=1.414)
        nn.init.xavier_normal_(self.cls.weight, gain=1.414)

    def forward(self, blocks, h, hop=1):
        if self.conv_type == 'gat':
            h, attn = self.conv1(blocks[0], h, get_attention=True)
        else:
            h = self.conv1(blocks[0], h)
            attn = 0
        
        h = torch.reshape(h, (h.shape[0], -1))
        if hop == 2:
            h = F.relu(h)
            h = self.conv2(blocks[1], h)
            h = torch.reshape(h, (h.shape[0], -1))

        return h, attn

