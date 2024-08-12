import matplotlib.pyplot as plt
import numpy as np

def plot(history):

    plt.plot(np.arange(len(history)), history)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss vs. training step')
    plt.show()

    return

def plot_avg(history):
    a = np.array(history)
    b = np.ones(300)/300
    avg_hist = np.convolve(a,b, mode='valid')

    plt.plot(np.arange(len(avg_hist)), avg_hist)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss vs. training step')
    plt.show()

    return