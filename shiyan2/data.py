import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class total_dataset(Dataset):
    def __init__(self, path):
        super(total_dataset, self).__init__()
        self.path = path
        self.data, self.label = self.traindata_get(path)       
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        self.train_data = (self.data[index]/255).reshape(3, 32, 32)
        self.train_data = torch.from_numpy(self.train_data)
        self.train_label = torch.zeros(10)
        self.train_label[int(self.label[index])] = 1
        return self.train_data.float(), self.train_label.float()
    
    def traindata_get(self, path):
        data_batch = []
        train_data = []
        train_label = []
        for i in range(5):
            data_batch.append(unpickle(os.path.join(path, f'data_batch_{i+1}')))
            train_data.append(data_batch[i][b'data'])
            train_label.extend(data_batch[i][b'labels'])
        train_data = np.concatenate(train_data)
        return train_data, train_label

class test_dataset(Dataset):
    def __init__(self, path):
        super(test_dataset, self).__init__()
        self.path = path
        self.data, self.label = self.testdata_get(path)
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        self.test_data = (self.data[index]/255).reshape(3, 32, 32)
        self.test_data = torch.from_numpy(self.test_data)
        self.test_label = torch.zeros(10)
        self.test_label[int(self.label[index])] = 1
        return self.test_data.float(), self.test_label.float()
        
    def testdata_get(self, path):
        data_batch = unpickle(path)
        test_data = data_batch[b'data']
        test_label = data_batch[b'labels']
        return test_data, test_label
    
def data_loader(path, batch_size=32):
    print('Loading data...')
    train_dataset, valid_dataset = random_split(total_dataset(path), [40000, 10000])
    train_loader = DataLoader(dataset=train_dataset, num_workers=0, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_dataset, num_workers=0, batch_size=batch_size, shuffle=True, pin_memory=True)
    return train_loader, valid_loader
def test_loader(path, batch_size=32):
    print('Loading data...')
    test_loader = DataLoader(dataset=test_dataset(path), num_workers=0, batch_size=batch_size, shuffle=False, pin_memory=True)
    return test_loader