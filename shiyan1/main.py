import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import data
import train

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
cudnn.deterministic = True

lr = 1e-3
epochs = 500
batch_size = 64
path = './ex1data.mat'

if __name__ == '__main__':
    train_loader = data.train_dataloader(path, batch_size)
    train.train(train_loader, lr, epochs)
