import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.util import get_preprocessed_noisy_data, load_mnist


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
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
        x = self.fc(x)
        return x


class Enc(nn.Module):
    def __init__(self):
        super(Enc, self).__init__()

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

        self.fc = nn.Sequential(
            nn.Linear(98, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 64),
        )

    def forward(self, x):
        x = self.cnn1(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Dec(nn.Module):
    def __init__(self):
        super(Dec, self).__init__()

        self.defc = nn.Sequential(
            nn.Linear(64, 98),
            nn.BatchNorm1d(98),
            nn.ReLU(),

            nn.Linear(98, 98),
            nn.BatchNorm1d(98),
            nn.ReLU(),

            nn.Linear(98, 98),
        )

        self.decnn1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2, out_channels=4, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.ConvTranspose2d(in_channels=4, out_channels=8, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(3, 3), padding=1),
        )

    def forward(self, x):
        x = self.defc(x)
        x = x.reshape(x.shape[0], 2, 7, 7) # flatten -> conv shape
        x = self.decnn1(x)
        return x

def get_correct_and_total_count(pred, real):
    a = pred == real
    corret = sum(a).item()
    total = len(a)
    return corret, total


# hyperparameters
encoder_training_epochs = 50
mlp_training_epochs = 50
train_noise_std_multiple = 3
batch_size = 128

# load data set
mnist = load_mnist()
total_x = np.array(mnist.data)

noisy_x = get_preprocessed_noisy_data(total_x, train_noise_std_multiple)
total_y = np.array(mnist.target, dtype=np.int64)
train_x, train_y, val_x, val_y = noisy_x[:50000], total_y[:50000], noisy_x[50000:], total_y[50000:]

# set dataset loader
train_tensor_dataset = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y))
val_tensor_dataset = TensorDataset(torch.Tensor(val_x), torch.Tensor(val_y))
train_data_loader = DataLoader(dataset=train_tensor_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_data_loader = DataLoader(dataset=val_tensor_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

## Train Encoder-Decoder(AutoEncoder) ##
print("Train Encoder-Decoder(AutoEncoder)")
enc = Enc().cuda()
dec = Dec().cuda()

optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()))
loss_fn = nn.MSELoss()

for epoch in range(encoder_training_epochs):

    enc.train()
    dec.train()

    for x, _ in train_data_loader:
        x = x.cuda()
        x = x.reshape(x.shape[0], 1, 28, 28)

        latent = enc(x)
        recon_x = dec(latent)

        optimizer.zero_grad()
        loss = loss_fn(recon_x, x)
        loss.backward()
        optimizer.step()

    print(f"{epoch} / mse loss: {loss.item()}")

    ## if you want to see reconstruction image uncomment here. ## 
    # plt.imshow(x[0].reshape(28, 28).cpu().detach())
    # plt.show()
    # plt.imshow(recon_x[0].reshape(28, 28).cpu().detach())
    # plt.show()
    print("----------------------------")


# Train MLP By Encoder latents
print("## Train MLP By Encoder latents ##")
mlp = MLP().cuda()

optimizer = torch.optim.Adam(mlp.parameters())
loss_fn = nn.CrossEntropyLoss()

train_acc = []
val_acc = []

for epoch in range(mlp_training_epochs):
    enc.eval()
    mlp.train()

    train_total = 0
    train_correct = 0

    for x, y in train_data_loader:
        x = x.cuda()
        y = y.cuda().long()

        x = x.reshape(x.shape[0], 1, 28, 28)

        with torch.no_grad():
            latent = enc(x)

        y_pred = mlp(latent)

        optimizer.zero_grad()
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        correct, total = get_correct_and_total_count(y_pred.argmax(axis=1), y)
        train_total += total
        train_correct += correct

    enc.eval()
    mlp.eval()

    val_total = 0
    val_correct = 0

    for x, y in val_data_loader:
        x = x.cuda()
        y = y.cuda().long()

        x = x.reshape(x.shape[0], 1, 28, 28)

        with torch.no_grad():
            latent = enc(x)
            y_pred = mlp(latent)

        correct, total = get_correct_and_total_count(y_pred.argmax(axis=1), y)
        val_total += total
        val_correct += correct

    print(f"{epoch}, train acc: {train_correct / train_total}")
    print(f"{epoch}, validation acc: {val_correct / val_total}")
    print("--------------------------------------------")
    train_acc.append(train_correct / train_total)
    val_acc.append(val_correct / val_total)


# test for distribution shift
print("## test for distribution shift ##")
dist_shift_acc = []
for test_noise_multiple in range(10):
    noisy_x = get_preprocessed_noisy_data(total_x, test_noise_multiple)
    total_y = np.array(mnist.target, dtype=np.int64)
    val_x, val_y = noisy_x[50000:], total_y[50000:]

    # set dataset loader
    val_tensor_dataset = TensorDataset(torch.Tensor(val_x), torch.Tensor(val_y))
    val_data_loader = DataLoader(dataset=val_tensor_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # validation
    enc.eval()
    mlp.eval()

    val_correct = 0
    val_total = 0

    for x, y in val_data_loader:
        x = x.cuda()
        y = y.cuda().long()

        x = x.reshape(x.shape[0], 1, 28, 28)

        with torch.no_grad():
            latent = enc(x)
            y_pred = mlp(latent)

        correct, total = get_correct_and_total_count(y_pred.argmax(axis=1), y)
        val_total += total
        val_correct += correct

    dist_shift_acc.append(val_correct / val_total)
    print(f"test_noise_multiple: {test_noise_multiple} validation acc: {val_correct / val_total}")
