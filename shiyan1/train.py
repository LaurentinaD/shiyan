import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import data
import model
from checkgradient import check_gradient 

def train(train_loader, lr, epochs, ck_grad_pt=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Start training...')
    train_model = model.predict_model(400,25,10).double()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(train_model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300,800], gamma=0.1)
    loss_list = []
    prameter_initialize(train_model)
    for epoch in range(epochs):
        epoch_loss = []
        for iter, batch in enumerate(train_loader, 1):
            inputs, labels = batch[0], batch[1]
            optimizer.zero_grad()
            outputs = train_model(inputs)
            L1_penalty = torch.mean(torch.abs(train_model.linear1.weight))
            loss = loss_function(outputs, labels) + L1_penalty*1e-4
            epoch_loss.append(loss.item())
            loss.backward()
            if epoch == ck_grad_pt and iter == 1:
                with torch.no_grad():
                    check_gradient(train_model, inputs, labels)
            optimizer.step()
        scheduler.step()    
        loss_list.append(np.mean(epoch_loss))
        print(f'Epoch:{epoch}/{epochs}, Loss:{loss_list[-1]}')
        if epoch % 50 == 0 and epoch != 0:
            save_checkpoint(train_model, epoch, loss_list)
            
def save_checkpoint(model, epoch, loss_list, path='./checkpoint/'):
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(model.state_dict(), os.path.join(path, f'epoch={epoch}.pth'))
    plt.plot(np.arange(epoch+1), loss_list, 'b*-', alpha=0.5, linewidth=0.5, label='loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(f'./figure/loss_epoch={epoch}.pdf')
    plt.close()
    
def prameter_initialize(model):
    model_dict={}
    model_dict['linear1.bias'] = (torch.rand(25) * 0.24) - 0.12
    model_dict['linear1.weight'] = (torch.rand(25, 400) * 0.24) - 0.12
    model_dict['linear2.bias'] = (torch.rand(10) * 0.24) - 0.12
    model_dict['linear2.weight'] = (torch.rand(10, 25) * 0.24) - 0.12
    model.load_state_dict(model_dict, strict=True)
    return model