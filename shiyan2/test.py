import torch
import numpy as np
import data
import model

def predict_acc(model, datas, labels):
    predict_model = model.eval()
    outputs = predict_model(datas).cpu().detach().numpy()
    labels = labels.cpu().numpy()
    predict = (np.argmax(outputs, axis=1)).reshape(len(labels), 1)
    real = (np.argmax(labels, axis=1)).reshape(len(labels), 1)
    accuracy = np.sum(predict == real)/len(real)
    return accuracy

data_path = './cifar-10-batches-py/test_batch'
ckpt_name = './checkpoint/epoch=40.pth'


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loader = data.test_loader(data_path)
    test_model = model.CNN_Net(3,(32, 64, 512, 128), 10).to(device)
    test_model.load_state_dict(torch.load(ckpt_name), strict=True)
    acc_list = []
    for iter, batch in enumerate(test_loader,1):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        accuracy = predict_acc(test_model, inputs, labels)
        acc_list.append(accuracy)
    accuracy_final = np.mean(acc_list)
    print('预测准确率：', accuracy_final)
    
