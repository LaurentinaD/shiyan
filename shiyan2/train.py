import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import data
import model
from test import predict_acc

def train(train_loader, valid_loader, lr, epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Start training...')
    train_model = model.CNN_Net(3,(32, 64, 512, 128), 10).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(train_model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=5e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300,800], gamma=0.1)
    train_loss_list, valid_loss_list = [] , []
    train_acc_list, valid_acc_list = [], []
    for epoch in range(epochs):
        epoch_train_loss, epoch_valid_loss = [], []
        epoch_train_acc, epoch_valid_acc = [], []
        train_model.train()
        for iter, batch in enumerate(train_loader, 1):
            inputs, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            outputs = train_model(inputs)
            loss = loss_function(outputs, labels)
            epoch_train_loss.append(loss.item())
            accuracy = predict_acc(train_model, inputs, labels)
            epoch_train_acc.append(accuracy)
            loss.backward()
            optimizer.step()
        scheduler.step()    
        train_loss_list.append(np.mean(epoch_train_loss))
        train_acc_list.append(np.mean(epoch_train_acc))
        print(f'Epoch:{epoch}/{epochs}, Loss:{train_loss_list[-1]}, Accuracy:{train_acc_list[-1]}')
     
        train_model.eval()
        with torch.no_grad():
            for iter, batch in enumerate(valid_loader, 1):
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = train_model(inputs)
                loss = loss_function(outputs, labels)
                epoch_valid_loss.append(loss.item())
                accuracy = predict_acc(train_model, inputs, labels)
                epoch_valid_acc.append(accuracy)
            valid_loss_list.append(np.mean(epoch_valid_loss))
            valid_acc_list.append(np.mean(epoch_valid_acc))
        if epoch % 10 == 0 and epoch != 0:
            save_checkpoint(train_model, epoch, train_loss_list, valid_loss_list, train_acc_list, valid_acc_list)
         
def save_checkpoint(model, epoch, train_loss_list, valid_loss_list, train_acc_list, valid_acc_list, path='./checkpoint/'):
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(model.state_dict(), os.path.join(path, f'epoch={epoch}.pth'))
    plt.plot(np.arange(epoch+1), train_loss_list, 'b*-', alpha=0.5, linewidth=0.5, label='train_loss')
    plt.plot(np.arange(epoch+1), valid_loss_list, 'rx--', alpha=0.5, linewidth=0.5, label='valid_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(f'./figure/loss_epoch={epoch}.pdf')
    plt.close()
    plt.plot(np.arange(epoch+1), train_acc_list, 'b*-', alpha=0.5, linewidth=0.5, label='train_acc')
    plt.plot(np.arange(epoch+1), valid_acc_list, 'rx--', alpha=0.5, linewidth=0.5, label='valid_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig(f'./figure/acc_epoch={epoch}.pdf')
    plt.close()  

