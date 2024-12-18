import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat, savemat
import data
from model import predict_model

def parameter_load(path):
    parameters = loadmat(path)
    para_keys = list(parameters.keys())
    key_T1, key_T2 = para_keys[-2],para_keys[-1]
    Theta1, Theta2 = torch.from_numpy(parameters[key_T1]), torch.from_numpy(parameters[key_T2])
    return Theta1, Theta2

def load_from_Theta():
    Theta1, Theta2 = parameter_load('./ex1weights.mat')
    linear1_bias, linear1_weights = Theta1[:,0],Theta1[:,1:401]
    linear2_bias, linear2_weights = Theta2[:,0],Theta2[:,1:26]

    model = predict_model(400,25,10)
    model_dict={}
    model_dict['linear1.bias'] = linear1_bias
    model_dict['linear1.weight'] = linear1_weights
    model_dict['linear2.bias'] = linear2_bias
    model_dict['linear2.weight'] = linear2_weights
    model.load_state_dict(model_dict, strict=True)
    return model
    
def load_from_checkpoint(ckpt_name):
    model =predict_model(400,25,10)
    model.load_state_dict(torch.load(f'./checkpoint/{ckpt_name}.pth'), strict=True)
    return model

predict_type = 'from_Theta'
ckpt_name = 'epoch=450'
hidden_show = False

if __name__ == '__main__':
    if predict_type == 'from_Theta':
        print('Loading model from Theta...')
        model=load_from_Theta()
    elif predict_type == 'from_checkpoint':
        print(f'Loading model from {ckpt_name}...')
        model=load_from_checkpoint(ckpt_name)
    else:
        print('Predict Type Error')
    model.eval()
    data_X, data_Y = data.data_load('./ex1data.mat')
    input_X = torch.from_numpy(data_X).float()
    if hidden_show:
        output_Y, hidden = model(input_X, hidden_show)
        output_Y = output_Y.detach().numpy()
        hidden = hidden.detach().numpy()
        savemat('./hidden.mat', {'hidden':hidden})
    else:
        output_Y = model(input_X).detach().numpy()
    predict_Y = (np.argmax(output_Y, axis=1)+1).reshape(len(data_Y),1)
    accuracy = np.sum(predict_Y == data_Y)/len(data_Y)
    print('预测正确率：', accuracy)
