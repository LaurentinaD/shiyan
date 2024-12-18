import os
import numpy as np
import torch
import torch.nn as nn
import model as md
import predict

def check_gradient(model, inputs, labels):
    e = 1e-7
    loss_function = nn.CrossEntropyLoss()
    auto_gradients = []
    numerical_gradients = []
    for param in model.parameters():
        auto_gradients.append(param.grad.clone())
    
    for param in model.parameters():
        flat_param = param.view(-1)
        num_grad = torch.zeros_like(flat_param)
        for i in range(len(flat_param)):
            original_value = flat_param[i].item()
            flat_param[i] = original_value + e
            outputs_pos = model(inputs)
            loss_pos = loss_function(outputs_pos, labels).item()
            flat_param[i] = original_value - e
            outputs_neg = model(inputs)
            loss_neg = loss_function(outputs_neg, labels).item()
            flat_param[i] = original_value
            num_grad[i] = (loss_pos - loss_neg) / (2 * e)
        numerical_gradients.append(num_grad.view_as(param))
    
    check_flag = True
    for auto_grads, num_grads in zip(auto_gradients, numerical_gradients):
        if not torch.allclose(auto_grads, num_grads, atol=1e-3):
            check_flag = False
            break
    if check_flag:    
        print('Gradients check passed!')
    else:
        print('Gradients check failed!')