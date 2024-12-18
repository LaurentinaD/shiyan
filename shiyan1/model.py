import torch
import torch.nn as nn
import numpy

class predict_model(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, out_channels)
        self.activate = nn.Sigmoid()

    def forward(self, x, hidden_show=False):
        x1 = self.linear1(x)
        x2 = self.activate(x1)
        x3 = self.linear2(x2)
        x4 = self.activate(x3)
        if hidden_show == True:
            return x4, x2
        else:
            return x4