import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class CNN_Net(nn.Module):
    def __init__(self, in_channel=3, hidden_channel=(32, 64, 512, 128), out_channel=10):
        super(CNN_Net, self).__init__()
        self.in_channel = in_channel
        self.hidden_channel = hidden_channel
        self.out_channel = out_channel
        self.unit1 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.hidden_channel[0], kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(self.hidden_channel[0]),
            nn.Conv2d(self.hidden_channel[0], self.hidden_channel[0], kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(self.hidden_channel[0]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        self.unit2 = nn.Sequential(
            nn.Conv2d(self.hidden_channel[0], self.hidden_channel[1], kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(self.hidden_channel[1]),
            nn.Conv2d(self.hidden_channel[1], self.hidden_channel[1], kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(self.hidden_channel[1]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        self.unit3 = nn.Sequential(
            nn.Linear(4096, self.hidden_channel[2]),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_channel[2], self.hidden_channel[3]),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_channel[3], self.out_channel),
        )
        
    def forward(self, x):
        x1 = self.unit1(x)
        x2 = self.unit2(x1)
        B, C, H, W = x2.shape
        x2 = x2.permute(0, 2, 3, 1).reshape(B, H*W*C)
        x3 = self.unit3(x2)
        x4 = F.softmax(x3, dim=-1)
        return x4
    
if __name__ == '__main__':
    x = torch.randn(1, 3, 32, 32)
    net = CNN_Net(3,(32, 64, 512, 128), 10)
    summary(net, (1, 3, 32, 32))