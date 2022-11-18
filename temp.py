# import tensorflow as tf
# import keras
import torch.nn.functional as F
import numpy as np
import torch
import warnings as w
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

w.filterwarnings('ignore')

# print(tf.__version__)
# print(keras.__version__)
# data = torch.tensor([0.12, 2.3, 4.5])
# out = F.softmax(data)
# print(out)
#
# b = torch.nn.Softmax()
# print(b(data))

# data2 = np.array([
#     [
#         [2.1, 2.3],
#         [3.2, 5.0]
#     ],
#     [
#         [2.2, 2.3],
#         [3.2, 5.0]
#     ]
# ])
# data2 = np.float32(data2)
# data2 = torch.tensor(data2)
# print(data2)
# lstm = torch.nn.LSTM(2, 10, 2, bidirectional=True, batch_first=True, dropout=0.3)
# k, _ = lstm(data2)
# print(k)

# data3 = np.array([
#     [1, 2, 5, 6],
#     [3, 4, 4, 2],
#     [1, 2, 5, 6],
#     [3, 4, 4, 2]
# ])
# data3 = np.float32(data3)
# data3 = torch.tensor(data3)
# data3 = data3.unsqueeze(dim=0)
# print(data3)

# data4 = np.array([[[0.1197, 0.1313, 0.1208, 0.1505, 0.1271, 0.1277, 0.1145, 0.1155,
#                     0.1258, 0.1268]],
#
#                   [[0.1333, 0.1250, 0.1356, 0.1333, 0.1189, 0.1277, 0.1283, 0.1471,
#                     0.1258, 0.1268]]])
# data4 = np.float32(data4)
# data4 = torch.tensor(data4)
# data4 = data4.squeeze(dim=1)
# print(data4)


# conv1 = torch.nn.Conv1d(in_channels=4, out_channels=4, kernel_size=1)
# f = torch.nn.Flatten()
# a = f(conv1(data3))
# print(conv1(data3))
# print(a)
# X_train = np.array([
#     [1, 2, 1],
#     [2, 3, 5]
# ])
# X_train = np.reshape(X_train, (1, X_train.shape[0], X_train.shape[1]))
# print(X_train)

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./dataset/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./dataset/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
# print(example_targets)
# print(example_data)
#
import matplotlib.pyplot as plt
fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    print(1)
    print(example_data[i][0])
    print(1)
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        print(x.shape)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(-1, 320)
        print(x.shape)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x1)
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')


train(1)
