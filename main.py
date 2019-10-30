from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
import torchvision.transforms.functional as TF
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

torch.manual_seed(args.seed)

if torch.cuda.is_available():
    gpu_active = True
    print("GPU is active")
else:
    gpu_active = False
    print("GPU is not active")

FloatTensor = torch.cuda.FloatTensor if gpu_active else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if gpu_active else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if gpu_active else torch.ByteTensor
Tensor = FloatTensor

### Data Initialization and Loading
from data import initialize_data, data_transforms, train_data_transform, data_center_crop, data_jitter_brightness, \
    data_jitter_contrast, data_jitter_saturation, data_jitter_hue, data_grayscale, data_horizontal_flip, data_vertical_flip, \
    data_forward_rotation, data_backward_rotation, data_shear, data_translate

initialize_data(args.data)  # extracts the zip files, makes a validation set

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset([
        datasets.ImageFolder(args.data + '/train_images', transform=data_transforms),
        datasets.ImageFolder(args.data + '/train_images', transform=data_center_crop),
        datasets.ImageFolder(args.data + '/train_images', transform=data_jitter_brightness),
        datasets.ImageFolder(args.data + '/train_images', transform=data_jitter_contrast),
        datasets.ImageFolder(args.data + '/train_images', transform=data_jitter_saturation),
        datasets.ImageFolder(args.data + '/train_images', transform=data_grayscale),
        datasets.ImageFolder(args.data + '/train_images', transform=data_horizontal_flip),
        datasets.ImageFolder(args.data + '/train_images', transform=data_vertical_flip),
        datasets.ImageFolder(args.data + '/train_images', transform=data_forward_rotation),
        datasets.ImageFolder(args.data + '/train_images', transform=data_backward_rotation),
        datasets.ImageFolder(args.data + '/train_images', transform=data_shear),
        datasets.ImageFolder(args.data + '/train_images', transform=data_translate)]),
    batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=gpu_active)

# train_loader = torch.utils.data.DataLoader(
#     datasets.ImageFolder(args.data + '/train_images',
#                          transform=train_data_transform),
#     batch_size=args.batch_size, shuffle=True, num_workers=1)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=1)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import Net

model = Net()
if gpu_active:
    model.cuda()

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1, verbose=True)


def plot_history(training_history, validation_history, history_str, epoch_number):
    # Create count of the number of epochs
    epoch_count = range(1, len(training_history) + 1)
    # Visualize loss history
    plt.plot(epoch_count, training_history, 'r-')
    plt.plot(epoch_count, validation_history, 'b-')
    plt.legend(["Training " + history_str, "Validation " + history_str])
    plt.xlabel('Epoch')
    plt.ylabel(history_str)
    plt.savefig("plots/" + history_str + "_epoch_" + str(epoch_number) + ".png")
    plt.clf()


def plot_loss(training_loss_history, validation_loss_history, epoch_number):
    plot_history(training_loss_history, validation_loss_history, "Loss", epoch_number)


def plot_accuracy(training_accuracy_history, validation_accuracy_history, epoch_number):
    plot_history(training_accuracy_history, validation_accuracy_history, "Accuracy", epoch_number)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if gpu_active:
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def train_loss_and_accuracy():
    model.eval()
    training_loss = 0
    correct = 0
    for data, target in train_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        if gpu_active:
            data = data.cuda()
            target = target.cuda()
        output = model(data)
        training_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    training_loss /= len(train_loader.dataset)
    print('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        training_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))

    return training_loss, (100. * int(correct) / len(train_loader.dataset))


def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        if gpu_active:
            data = data.cuda()
            target = target.cuda()
        output = model(data)
        validation_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

    return validation_loss, (100. * int(correct) / len(val_loader.dataset))


# Holders of history and accuracy
training_loss_history = []
training_accuracy_history = []
validation_loss_history = []
validation_accuracy_history = []

if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
    # for epoch in range(1, 3 + 1):
        train(epoch)
        # Calculate training loss and accuracy
        training_loss_accuracy = train_loss_and_accuracy()
        training_loss_history.append(training_loss_accuracy[0])
        training_accuracy_history.append(training_loss_accuracy[1])
        # Calculate validation loss and accuracy
        validation_loss_accuracy = validation()
        validation_loss_history.append(validation_loss_accuracy[0])
        validation_accuracy_history.append(validation_loss_accuracy[1])
        # Check whteher to adjust learning rate based on validation loss
        scheduler.step(validation_loss_accuracy[0])
        # Plot training and validation loss and accuracy every ten epochs
        if (epoch % 10 == 0):
            plot_loss(training_loss_history, validation_loss_history, epoch)
            plot_accuracy(training_accuracy_history, validation_accuracy_history, epoch)

        model_file = 'model/model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file)
        print(
            '\nSaved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file')
    # Plot training and validation loss and accuracy
    plot_loss(training_loss_history, validation_loss_history, args.epochs + 1)
    plot_accuracy(training_accuracy_history, validation_accuracy_history, args.epochs + 1)




# # from __future__ import print_function
# # import argparse
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import torch.optim as optim
# # from torchvision import datasets, transforms, utils
# # from torch.autograd import Variable
# #
# # # import matplotlib.pyplot as plt
# # # import numpy as np
# #
# # # Training settings
# # parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
# # parser.add_argument('--data', type=str, default='data', metavar='D',
# #                     help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
# # parser.add_argument('--batch-size', type=int, default=64, metavar='N',
# #                     help='input batch size for training (default: 64)')
# # parser.add_argument('--epochs', type=int, default=5, metavar='N',
# #                     help='number of epochs to train (default: 10)')
# # parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
# #                     help='learning rate (default: 0.01)')
# # parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
# #                     help='SGD momentum (default: 0.5)')
# # parser.add_argument('--seed', type=int, default=1, metavar='S',
# #                     help='random seed (default: 1)')
# # parser.add_argument('--log-interval', type=int, default=10, metavar='N',
# #                     help='how many batches to wait before logging training status')
# # args = parser.parse_args()
# #
# # torch.manual_seed(args.seed)
# #
# # ### Data Initialization and Loading
# # from data import initialize_data, data_transforms, train_data_transform, vals_data_transforms  # data.py in the same folder
# # initialize_data(args.data) # extracts the zip files, makes a validation set
# #
# # train_loader = torch.utils.data.DataLoader(
# #     datasets.ImageFolder(args.data + '/train_images',
# #                          transform=train_data_transform),
# #                          # transform=train_data_transform),
# #     batch_size=args.batch_size, shuffle=True, num_workers=1)
# # val_loader = torch.utils.data.DataLoader(
# #     datasets.ImageFolder(args.data + '/val_images',
# #                          transform=vals_data_transforms),
# #     batch_size=args.batch_size, shuffle=False, num_workers=1)
# #
# # ### Neural Network and Optimizer
# # # We define neural net in model.py so that it can be reused by the evaluate.py script
# # from model import Net
# # model = Net()
# #
# # # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
# # optimizer = optim.Adam(model.parameters(), lr=args.lr)
# #
# # def train(epoch):
# #     model.train()
# #     for batch_idx, (data, target) in enumerate(train_loader):
# #         data, target = Variable(data), Variable(target)
# #
# #         # images = utils.make_grid(data).numpy()
# #         # plt.imshow(np.transpose(images, (1, 2, 0)))
# #         # plt.show()
# #
# #
# #         optimizer.zero_grad()
# #         output = model(data)
# #         # loss = F.cross_entropy(output, target)
# #         loss = F.nll_loss(output, target)
# #         loss.backward()
# #         optimizer.step()
# #         if batch_idx % args.log_interval == 0:
# #             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
# #                 epoch, batch_idx * len(data), len(train_loader.dataset),
# #                 100. * batch_idx / len(train_loader), loss.item()))
# #
# # def validation():
# #     model.eval()
# #     validation_loss = 0
# #     correct = 0
# #     for data, target in val_loader:
# #         data, target = Variable(data, volatile=True), Variable(target)
# #         output = model(data)
# #         validation_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
# #         pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
# #         correct += pred.eq(target.data.view_as(pred)).cpu().sum()
# #
# #     validation_loss /= len(val_loader.dataset)
# #     print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
# #         validation_loss, correct, len(val_loader.dataset),
# #         100. * correct / len(val_loader.dataset)))
# #
# # if __name__ == "__main__":
# #     for epoch in range(1, args.epochs + 1):
# #         train(epoch)
# #         validation()
# #         model_file = 'model_' + str(epoch) + '.pth'
# #         torch.save(model.state_dict(), model_file)
# #         print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model' + model_file + '` to generate the Kaggle formatted csv file')
#
#
#
#
#
#
#
# from __future__ import print_function
# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# # import numpy as np
# # import matplotlib.pyplot as plt
#
#
# from torchvision import datasets, transforms, utils
# from torch.autograd import Variable
#
# # Training settings
# parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
# parser.add_argument('--data', type=str, default='data', metavar='D',
#                     help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
# parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                     help='input batch size for training (default: 64)')
# parser.add_argument('--epochs', type=int, default=30, metavar='N',
#                     help='number of epochs to train (default: 10)')
# parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#                     help='learning rate (default: 0.01)')
# parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#                     help='SGD momentum (default: 0.5)')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before logging training status')
# args = parser.parse_args()
#
# torch.manual_seed(args.seed)
#
# ### Data Initialization and Loading
# from data import initialize_data, data_transforms, train_data_transform  # data.py in the same folder
# initialize_data(args.data) # extracts the zip files, makes a validation set
#
# train_loader = torch.utils.data.DataLoader(
#     datasets.ImageFolder(args.data + '/train_images',
#                          # transform=data_transforms),  # original transform for training data
#                          transform=train_data_transform),
#     batch_size=args.batch_size, shuffle=True, num_workers=1)
# val_loader = torch.utils.data.DataLoader(
#     datasets.ImageFolder(args.data + '/val_images',
#                          transform=data_transforms),
#     batch_size=args.batch_size, shuffle=False, num_workers=1)
#
# ### Neural Network and Optimizer
# # We define neural net in model.py so that it can be reused by the evaluate.py script
# from model import Net
# model = Net()
#
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
#
# def train(epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = Variable(data), Variable(target)
#
#         # images = utils.make_grid(data).numpy()
#         # plt.imshow(np.transpose(images, (1, 2, 0)))
#         # plt.show()
#
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
#
# def validation():
#     model.eval()
#     validation_loss = 0
#     correct = 0
#     for data, target in val_loader:
#         data, target = Variable(data, volatile=True), Variable(target)
#         output = model(data)
#         validation_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
#         pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()
#
#     validation_loss /= len(val_loader.dataset)
#     print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         validation_loss, correct, len(val_loader.dataset),
#         100. * correct / len(val_loader.dataset)))
#
#
# if __name__ == "__main__":
#     for epoch in range(1, args.epochs + 1):
#         train(epoch)
#         validation()
#         model_file = 'model_' + str(epoch) + '.pth'
#         torch.save(model.state_dict(), model_file)
#         print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model' + model_file + '` to generate the Kaggle formatted csv file')
#
#
#
