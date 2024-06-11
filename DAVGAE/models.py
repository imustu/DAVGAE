import torch.nn as nn
from torch.nn import Module
import torch
from torch_geometric.nn import GAE, GCNConv, VGAE, GATConv
import numpy as np
from torch.nn import Linear



class Mish(Module):

    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x * (torch.tanh(nn.functional.softplus(x)))
        return x

class GCNEncoder(Module):#加注意力
    def __init__(self, in_channels, hidden_size, out_channels, dropout):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_size)
        self.attention = GATConv(hidden_size, int(hidden_size/2), heads=2, dropout=dropout)#main函数的参数里的hidden-size=这里的第二个参数*heads
        self.conv2 = GCNConv(hidden_size, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.attention(x, edge_index).relu()#21101 128
        #x = self.dropout(x)#21101 512
        x=self.conv2(x, edge_index)
        return x




class VariationalGCNEncoder(Module):#加注意力
    def __init__(self, in_channels, hidden_size, out_channels, dropout,heads):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_size, cached=False)
        self.attention = GATConv(hidden_size, int(hidden_size/heads), heads=heads, dropout=dropout)
        self.conv_mu = GCNConv(hidden_size, out_channels, cached=False)
        self.conv_logstd = GCNConv(hidden_size, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x_temp1 = self.conv1(x, edge_index)
        #hardswish = nn.Hardswish()
        #x_temp1 = hardswish(x_temp1)
        x_temp1 = nn.functional.gelu(x_temp1)
        #x_temp1 = hardswish(x_temp1)# 添加Hardswish激活函数
        #x_temp1 =Mish()(x_temp1)# 添加Mish激活函数
        x_temp1 = self.dropout(x_temp1)
        x_temp1 = self.attention(x_temp1, edge_index).relu()
        #x_temp1 = nn.functional.gelu(x_temp1)
        #x_temp1 = nn.functional.gelu(x_temp1)
        return self.conv_mu(x_temp1, edge_index), self.conv_logstd(x_temp1, edge_index)

class VariationalLEncoder(Module):
    def __init__(self, in_channels, hidden_size, out_channels, dropout):
        super().__init__()
        self.lin1 = Linear(20, 256)
        self.conv_logstd = GCNConv(256, 128)
        self.lin_mu = Linear(64, 128)
        self.lin_logstd = Linear(64, 128)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x_temp1 = self.lin1(x).relu()
        #x_temp1 = nn.functional.gelu(x_temp1)
        #x_temp1 = self.dropout(x_temp1)
        x_temp1 = self.conv_logstd(x_temp1, edge_index).relu()
        #x_temp1 = nn.functional.gelu(x_temp1)
        #x_temp1 = self.dropout(x_temp1)
        mu, logstd = torch.chunk(x_temp1, 2, dim=-1)  # 将输出张量分成两部分，分别对应 mu 和 logstd
        return self.lin_mu(mu), self.lin_logstd(logstd)
