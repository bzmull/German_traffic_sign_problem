# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# nclasses = 43 # GTSRB as 43 classes
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(500, 50)
#         self.fc2 = nn.Linear(50, nclasses)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 500)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x)


import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=32)
        # self.conv11 = nn.Conv2d(32, 32, kernel_size=32)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=16)
        # self.conv21 = nn.Conv2d(64, 64, kernel_size=16)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=8)
        # self.conv31 = nn.Conv2d(128, 128, kernel_size=8)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, nclasses)
        # self.linear1 = nn.Linear(128*8*8, 128)
        # self.linear2 = nn.Linear(128, nclasses)
        # self.tanh = nn.Tanh()
        # self.pool1 = nn.MaxPool2d(16, 16)
        # self.pool2 = nn.MaxPool2d(8, 4)
        # self.pool3 = nn.MaxPool2d(4, 4)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = self.tanh(self.pool1(self.conv1(x)))
        # x = self.tanh(self.pool1(self.conv11(x)))
        # x = self.tanh(self.conv2(x))
        # x = self.tanh(self.pool2(self.conv21(x)))
        # x = self.tanh(self.conv3(x))
        # x = self.tanh(self.pool3(self.conv31(x)))
        # x = x.view(-1, 128*8*8)
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # x = self.tanh(self.linear1(x))
        # x = self.tanh(self.linear1(x))
        # return self.linear2(x)
        return F.log_softmax(x)
