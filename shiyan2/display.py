import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import data

def DisplayData(samples, fig_name, nrows=10, ncols=10):
    fig, axes = plt.subplots(nrows, ncols, figsize=(32,32), dpi=600)
    axes = axes.flatten()
    i=0

    for axe in axes:
        axe.set_axis_off()
        axe.imshow(samples[i],cmap=matplotlib.cm.gray)
        i+=1
    plt.tight_layout(pad=0.4, h_pad=0.4, w_pad=0.1, rect=[0,0,0.95,1])
    if not os.path.exists('./figure/'):
        os.mkdir('./figure/')
    fig.savefig(f'./figure/{fig_name}.pdf', dpi=600, bbox_inches='tight')
    plt.close

path = './cifar-10-batches-py/'
file_name = 'data_batch_1'

if __name__ == '__main__':
    datas = data.unpickle(os.path.join(path, file_name))[b'data']
    samples_index = np.random.choice(np.arange(10000), size=100)
    samples = []
    for index in samples_index:
        samples.append(datas[index].reshape(3, 32, 32).transpose(1, 2, 0))
    DisplayData(samples, 'display_samples')