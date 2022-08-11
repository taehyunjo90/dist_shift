import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.util import get_preprocessed_noisy_data, load_mnist


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),

            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.flatten = nn.Flatten()

        self.fc0 = nn.Sequential(
            nn.Linear(98, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Linear(64, 64),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Linear(16, 10),
            nn.Softmax(dim=1),
        )
        

    def forward(self, x):
        x = self.cnn1(x)
        x = self.flatten(x)
        x = self.fc0(x)
        x = self.fc1(x)
        return x


def get_correct(pred, real):
    a = pred == real
    corret = sum(a).item()
    total = len(a)
    return corret, total


# hyperparameters
training_epochs = 50
batch_size = 128
noise_multiple = 3

# load data set
mnist = load_mnist()

total_x = np.array(mnist.data)
noisy_x = get_preprocessed_noisy_data(total_x, noise_multiple)
total_y = np.array(mnist.target, dtype=np.int64)
train_x, train_y, val_x, val_y = noisy_x[:50000], total_y[:50000], noisy_x[50000:], total_y[50000:]

# set dataset loader
train_tensor_dataset = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y))
val_tensor_dataset = TensorDataset(torch.Tensor(val_x), torch.Tensor(val_y))

data_loader = DataLoader(dataset=train_tensor_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_data_loader = DataLoader(dataset=val_tensor_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# train supversied model
model = CNN().cuda()

optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

train_acc = []
val_acc = []

for epoch in range(training_epochs):

    # train
    model.train()
    train_total = 0
    train_correct = 0

    for batch_idx, (x, y) in enumerate(data_loader):
        x = x.cuda()
        y = y.cuda().long()

        optimizer.zero_grad()
        output = model(x.reshape(x.shape[0], 1, 28, 28))
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        correct, total = get_correct(output.argmax(axis=1), y)
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
            y_pred = model(x.reshape(x.shape[0], 1, 28, 28)).argmax(axis=1)

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
acc_dist_shift = []
for test_noise_multiple in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
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
            y_pred = model(x.reshape(x.shape[0], 1, 28, 28)).argmax(axis=1)

        correct, total = get_correct(y_pred, y)
        val_total += total
        val_correct += correct

    acc_dist_shift.append(val_correct / val_total)
    print(f"test_noise_multiple: {test_noise_multiple} validation acc: {val_correct / val_total}")
