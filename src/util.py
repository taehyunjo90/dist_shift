import os
import pickle
import numpy as np

from sklearn import datasets


def get_preprocessed_noisy_data(total_x, noise_multiple):
    copied = total_x.copy()
    copied /= 255.0
    m, s = np.mean(copied), np.std(copied)
    copied += np.random.normal(m, s, size=copied.shape) * noise_multiple
    return copied


def load_mnist():
    MNIST_PATH = 'mnist_784.pk'

    if os.path.isfile(MNIST_PATH):
        with open(MNIST_PATH, 'rb') as fp:
            mnist = pickle.load(fp)
    else:
        print("Need to download mnist_784 data from sklearn... It will take a few minutes...")
        mnist = datasets.fetch_openml('mnist_784')
        with open(MNIST_PATH, 'wb') as fp:
            pickle.dump(mnist, fp)
    return mnist