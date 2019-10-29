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
# ---------- ORIGINAL MODEL ------------------------------------------------------------------------------------------ 
#         self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(500, 50)
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, nclasses)
# ----------------------------------------------------------------------------------------------------
# ---------- STN ------------------------------------------------------------------------------------------
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # STN Function
        def stn(self, x):
            xs = self.localization(x)
            xs = xs.view(-1, 10 * 3 * 3)
            theta = self.fc_loc(xs)
            theta = theta.view(-1, 2, 3)

            grid = F.affine_grid(theta, x.size())
            x = F.grid_sample(x, grid)

            return x

# ---------- FIRST MODEL ------------------------------------------------------------------------------------------
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2)  # SWITCH TO GRAYSCALE
#         # self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=2)  # SWITCH TO NON_GRAYSCALE
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2)  # change kernel to 5 maybe
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
#         self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2)
#         self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2)
#         self.conv6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2)
#         self.linear1 = nn.Linear(32*8, 128)
#         self.linear2 = nn.Linear(128, nclasses)
#         self.activation = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.pool3 = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout2d()
# ----------------------------------------------------------------------------------------------------
# ------VGG MODEL----------------------------------------------------------------------------------------------
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)
#         self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
#         self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
#         self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1)
#         self.linear1 = nn.Linear(128*4*4, 128)
#         self.linear2 = nn.Linear(128, 128)
#         self.linear3 = nn.Linear(128, nclasses)
#         self.activation = nn.ReLU()
#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout2d()
# ----------------------------------------------------------------------------------------------------
    def forward(self, x):
# ---------- ORIGINAL MODEL ------------------------------------------------------------------------------------------
        x = self.stn(x)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        # x = x.view(-1, 500)
        x = x.view(-1, 20)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # print(x.shape)
        return F.log_softmax(x)
# ----------------------------------------------------------------------------------------------------
# ---------- FIRST MODEL ------------------------------------------------------------------------------------------        
#         x = self.activation(self.conv1(x))
#         x = self.activation(self.pool1(self.conv2(x)))
#         # print(x.shape)
#         x = self.activation(self.conv3(x))
#         x = self.activation(self.pool2(self.conv4(x)))
#         # print(x.shape)
#         x = self.activation(self.conv5(x))
#         x = self.activation(self.conv6(x))
#         # print(x.shape)
#         x = self.activation(self.conv5(x))
#         x = self.activation(self.conv6(x))
#         # x = F.dropout(x, training=self.training)
#         # print(x.shape)
#         x = x.view(-1, 32*8)
#         # print(x.shape)
#         x = self.activation(self.linear1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.linear2(x)
#         # print(x.shape)
#         return F.log_softmax(x)
#         # return x
# ----------------------------------------------------------------------------------------------------
# -----VGG MODEL-----------------------------------------------------------------------------------------------
#         # print("Start: ", x.shape)
#         x = self.activation(self.conv1(x))
#         # print("Layer1: ", x.shape)
#         x = self.dropout(self.pool(self.activation(self.conv2(x))))
#         # print("Layer3: ", x.shape)
#         x = self.activation(self.conv3(x))
#         # print("Layer4: ", x.shape)
#         x = self.dropout(self.pool(self.activation(self.conv4(x))))
#         # print("Layer6: ", x.shape)
#         x = self.activation(self.conv5(x))
#         # print("Layer7: ", x.shape)
#         x = self.dropout(self.pool(self.activation(self.conv6(x))))
#         # print("Layer9: ", x.shape)
#         x = x.view(-1, 128*4*4)
#         # print(x.shape)
#         x = self.activation(self.linear1(x))
#         # print(x.shape)
#         x = self.activation(self.linear2(x))
#         # print(x.shape)
#         x = self.linear3(x)
#         # print(x.shape)
#         return F.log_softmax(x)
# ----------------------------------------------------------------------------------------------------