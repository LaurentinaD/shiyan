import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import data

def DisplayData(samples, fig_name, nrows=10, ncols=10):
    fig, axes = plt.subplots(nrows, ncols, figsize=(20,20), dpi=400)
    axes = axes.flatten()
    i=0

    for axe in axes:
        axe.set_axis_off()
        axe.imshow(samples[i],cmap=matplotlib.cm.gray)
        i+=1
    plt.tight_layout(pad=0.4, h_pad=0.4, w_pad=0.1, rect=[0,0,0.95,1])
    if not os.path.exists('./figure/'):
        os.mkdir('./figure/')
    fig.savefig(f'./figure/{fig_name}.pdf', dpi=400, bbox_inches='tight')
    plt.close

display_type = 'from_sample'

if __name__ == '__main__':
    if display_type == 'from_sample':
        data_X, data_Y = data.data_load('./ex1data.mat')
        samples_index = np.random.choice(np.arange(4000), size=100)
        samples = []
        for index in samples_index:
            samples.append(data_X[index].reshape(20,20).transpose(1,0))
        DisplayData(samples, 'display_samples')
    elif display_type == 'from_hidden':
        hidden = loadmat('./hidden.mat')['hidden']
        hiddens = []
        for index in range(100):
            hiddens.append(hidden[index].reshape(5,5).transpose(1,0))
        DisplayData(hiddens, 'display_hiddens')
    else:
        print('Display Type Error')