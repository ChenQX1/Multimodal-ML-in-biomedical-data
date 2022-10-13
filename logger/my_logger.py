from typing import List
import numpy as np
import matplotlib.pyplot as plt


def plot_loss(train_loss: List[float], val_loss: List[int], save_dir: str):
    fig = plt.figure()
    l = len(train_loss)
    plt.plot(np.arange(l), train_loss, label='train loss')
    plt.plot(np.arange(l), val_loss, label='val loss')
    plt.legend(loc='best')
    plt.savefig('/'.join([save_dir, 'loss_log.png']))
