from scipy.io import loadmat
import numpy
import torch
from torch.utils.data import Dataset, DataLoader

def data_load(path):
    data = loadmat(path)
    keys = list(data.keys())
    key_X, key_Y = keys[-2],keys[-1]
    data_X, data_Y = data[key_X], data[key_Y]
    return data_X, data_Y

class train_dataset(Dataset):
    def __init__(self, path):
        super(Dataset, self).__init__()
        self.data_X, self.data_Y = data_load(path)

    def __len__(self):
        return self.data_X.shape[0]

    def __getitem__(self, index):
        self.train_data = torch.from_numpy(self.data_X[index])
        self.train_label = torch.zeros(10)
        self.train_label[int(self.data_Y[index])-1] = 1
        return self.train_data.double(), self.train_label.double()

def train_dataloader(path, batch_size=64):
    print('Loading data...')
    train_loader = DataLoader(dataset=train_dataset(path), num_workers=0, batch_size=batch_size, shuffle=True, pin_memory=True)
    return train_loader

