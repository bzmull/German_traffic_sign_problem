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
#         print(x.shape)
#         x = x.view(-1, 500)
#         print(x.shape)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         print(x.shape)
#         return F.log_softmax(x)





import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(500, 50)
        # self.fc2 = nn.Linear(50, nclasses)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)  # change kernel to 5 maybe
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2)
        self.linear1 = nn.Linear(32, 128)
        self.linear2 = nn.Linear(128, nclasses)
        self.tanh = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 500)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        # return F.log_softmax(x)

        x = self.tanh(self.conv1(x))
        x = self.tanh(self.pool1(self.conv2(x)))
        x = self.tanh(self.conv3(x))
        x = self.tanh(self.pool2(self.conv4(x)))
        x = self.tanh(self.conv5(x))
        x = self.tanh(self.pool3(self.conv6(x)))
        # print(x.shape)
        x = x.view(-1, 32)
        # print(x.shape)
        x = self.tanh(self.linear1(x))
        x = self.linear2(x)
        # print(x.shape)
        return F.log_softmax(x)
        # return x
