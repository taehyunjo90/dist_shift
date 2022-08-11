import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.util import get_preprocessed_noisy_data, load_mnist


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Linear(16, 10),
        )

    def forward(self, x):
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x


def get_correct(pred, real):
    a = pred == real
    corret = sum(a).item()
    total = len(a)
    return corret, total


# hyperparameters
training_epochs = 20
batch_size = 128
std_noise_multiple = 5

# load data set
mnist = load_mnist()

total_x = np.array(mnist.data)
noisy_x = get_preprocessed_noisy_data(total_x, std_noise_multiple)
total_y = np.array(mnist.target, dtype=np.int64)
train_x, train_y, val_x, val_y = noisy_x[:50000], total_y[:50000], noisy_x[50000:], total_y[50000:]

# set dataset loader
train_tensor_dataset = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y))
val_tensor_dataset = TensorDataset(torch.Tensor(val_x), torch.Tensor(val_y))

data_loader = DataLoader(dataset=train_tensor_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_data_loader = DataLoader(dataset=val_tensor_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# train supversied model
model = MLP().cuda()

optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

train_acc = []
val_acc = []

for epoch in range(training_epochs):

    # train
    model.train()
    train_total = 0
    train_correct = 0

    for batch_idx, (image, label) in enumerate(data_loader):
        image = image.cuda()
        label = label.cuda().long()

        optimizer.zero_grad()
        output = model(image)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        correct, total = get_correct(output.argmax(axis=1), label)
        train_total += total
        train_correct += correct

    # validation
    model.eval()
    val_correct = 0
    val_total = 0

    for x, y in val_data_loader:
        x = x.cuda()
        y = y.cuda().long()

        with torch.no_grad():
            y_pred = model(x).argmax(axis=1)

        correct, total = get_correct(y_pred, y)
        val_total += total
        val_correct += correct

    print(f"{epoch}, train acc: {train_correct / train_total}")
    print(f"{epoch}, validation acc: {val_correct / val_total}")
    print("--------------------------------------------")

    train_acc.append(train_correct / train_total)
    val_acc.append(val_correct / val_total)


# test for distribution shift
print("## test for distribution shift ##")
for test_noise_multiple in range(10):
    noisy_x = get_preprocessed_noisy_data(total_x, test_noise_multiple)
    total_y = np.array(mnist.target, dtype=np.int64)
    val_x, val_y = noisy_x[50000:], total_y[50000:]

    # set dataset loader
    val_tensor_dataset = TensorDataset(torch.Tensor(val_x), torch.Tensor(val_y))
    val_data_loader = DataLoader(dataset=val_tensor_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # validation
    model.eval()
    val_correct = 0
    val_total = 0

    for x, y in val_data_loader:
        x = x.cuda()
        y = y.cuda().long()

        with torch.no_grad():
            y_pred = model(x).argmax(axis=1)

        correct, total = get_correct(y_pred, y)
        val_total += total
        val_correct += correct

    print(f"test_noise_multiple: {test_noise_multiple} validation acc: {val_correct / val_total}")
