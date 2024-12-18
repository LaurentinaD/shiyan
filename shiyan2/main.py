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

lr = 1e-4
epochs = 50
batch_size = 32
path = './cifar-10-batches-py/'

if __name__ == '__main__':
    train_loader, valid_loader = data.data_loader(path, batch_size)
    train.train(train_loader, valid_loader, lr, epochs)