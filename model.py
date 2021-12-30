import torch
from torch import nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, linear):
        super(SimpleNN, self).__init__()
        self.linears = nn.ModuleList([])
        for ind, fc_num in enumerate(linear[0:-1]):  # Excluding the last one as we need intervals
            self.linears.append(nn.Linear(fc_num, linear[ind + 1]))

    def forward(self, x):
        out = x
        for ind, fc in enumerate(self.linears):
            if ind != len(self.linears) - 1:
                out = F.leaky_relu((fc(out)))  # ReLU + Linear
            else:
                out = fc(out)

        return out

class MLP(nn.Module):
    def __init__(self, linear):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList([])
        self.bn_linears = nn.ModuleList([])
        for ind, fc_num in enumerate(linear[0:-1]):  # Excluding the last one as we need intervals
            self.linears.append(nn.Linear(fc_num, linear[ind + 1]))
            self.bn_linears.append(nn.BatchNorm1d(linear[ind + 1]))

    def forward(self, x):
        out = x
        for ind, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):
            if ind != len(self.linears) - 1:
                out = F.leaky_relu(bn(fc(out)))  # ReLU + BN + Linear
            else:
                out = fc(out)

        return out