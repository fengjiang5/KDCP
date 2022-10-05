import torch
import numpy as np
import torch.nn as nn


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, 3)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Linear(out_filters, out_filters)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.conv2(x)

        return x


x = ResContextBlock(3, 6)
print(x.conv1.)
