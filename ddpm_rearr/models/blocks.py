import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyBlock1(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(MyBlock1, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out


class WideResBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True, dropout=0.0):
        super(WideResBlock, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.normalize = normalize

        self.bn1 = nn.BatchNorm2d(in_c)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)

        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding, bias=False)

        self.dropout = dropout
        self.activation = nn.SiLU() if activation is None else activation
        self.equalInOut = (in_c == out_c)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        x = self.ln(x) if self.normalize else x

        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)
