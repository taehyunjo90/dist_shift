import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.util import get_preprocessed_noisy_data, load_mnist


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout_prob = 0.25 # 50%의 노드에 대한 가중치 계산을 하지 않기 위한 설정
        self.batch_norm1 = nn.BatchNorm1d(512) # 1dimension이기 때문에 BatchNorm1d를 사용함.
        self.batch_norm2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = F.relu(x) # sigmoid(x)
        # x = F.dropout(x, training=self.training, p=self.dropout_prob)

        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = F.relu(x) # sigmoid(x)
        # x = F.dropout(x, training=self.training, p=self.dropout_prob)

        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)

        return x


def get_correct(pred, real):
    a = pred == real
    corret = sum(a).item()
    total = len(a)
    return corret, total

mnist = load_mnist()
total_x = np.array(mnist.data)
how_many_images = 5


for noise_multiple in [0, 1, 2, 3, 4, 5]:
    print(f"Images with std noise multiple: {noise_multiple}")
    noisy_x = get_preprocessed_noisy_data(total_x, noise_multiple)
    total_y = np.array(mnist.target, dtype=np.int64)
    train_x, train_y, val_x, val_y = noisy_x[:50000], total_y[:50000], noisy_x[50000:], total_y[50000:]

    for i in range(how_many_images):
        plt.imshow(train_x[i].reshape(28, 28))
        plt.show()

    print("-----------------------------------")
