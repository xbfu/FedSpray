import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv, SAGEConv, SGConv


class MLPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.weight = Parameter(torch.FloatTensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        output = torch.mm(x, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class MLP(nn.Module):
    def __init__(self, in_channel, out_channel, hidden):
        super().__init__()
        self.l1 = nn.Linear(in_channel, hidden)
        self.l2 = nn.Linear(hidden, out_channel)

    def forward(self, x, proxy=None):
        x = self.l1(x)
        x = F.relu(x)
        if proxy is not None:
            x = x + proxy
        x = self.l2(x)
        return x, x


class GCN(nn.Module):
    def __init__(self, in_channel, out_channel, hidden):
        super().__init__()
        self.l1 = GCNConv(in_channel, hidden)
        self.l2 = GCNConv(hidden, out_channel)

        self.reset_parameters()

    def reset_parameters(self):
        for name, para in self.named_parameters():
            para.data.uniform_()

    def forward(self, x, edge_index):
        x = self.l1(x, edge_index)
        x = F.relu(x)
        x = self.l2(x, edge_index)
        return x, x


class SAGE(nn.Module):
    def __init__(self, in_channel, out_channel, hidden):
        super().__init__()
        self.l1 = SAGEConv(in_channel, hidden)
        self.l2 = SAGEConv(hidden, out_channel)

        self.reset_parameters()

    def reset_parameters(self):
        for name, para in self.named_parameters():
            para.data.uniform_()

    def forward(self, x, edge_index):
        x = self.l1(x, edge_index)
        x = F.relu(x)
        x = self.l2(x, edge_index)
        return x, x


class SGC(nn.Module):
    def __init__(self, in_channel, out_channel, hidden):
        super().__init__()
        self.l1 = SGConv(in_channel, out_channel, K=2)

        self.reset_parameters()

    def reset_parameters(self):
        for name, para in self.named_parameters():
            para.data.uniform_()

    def forward(self, x, edge_index):
        x = self.l1(x, edge_index)
        return x, x


class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.l1 = MLPLayer(in_channel, out_channel)

    def forward(self, x, proxy=None):
        x = self.l1(x)
        x = F.relu(x)
        return x


class Classifier(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.l2 = MLPLayer(in_channel, out_channel)

    def forward(self, x, proxy=None):
        if proxy is not None:
            x = x + proxy
        #     x1 = torch.concat([x1, proxy], dim=1)
        x = self.l2(x)
        return x