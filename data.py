from __future__ import print_function
import zipfile
import os

import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 32 x 32 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set
data_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])
# -----Different data augmentation trnasformations-----

train_data_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.CenterCrop(32),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(1),
    transforms.RandomVerticalFlip(1),
    transforms.RandomRotation((0, 15)),
    transforms.RandomRotation((-15, 0)),
    transforms.RandomAffine(degrees=10, shear=2),
    transforms.RandomAffine(degrees=10, translate=(0.15, 0.15)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])
# Center crop transformation
center_crop_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

jitter_brightness_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ColorJitter(brightness=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

jitter_contrast_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ColorJitter(contrast=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

jitter_saturation_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ColorJitter(saturation=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

jitter_hue_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ColorJitter(hue=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# # Grayscale transformation
# data_grayscale = transforms.Compose([
# 	transforms.Resize((32, 32)),
#     transforms.Grayscale(num_output_channels=3),
#     transforms.ToTensor(),
#     transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
# ])

horizontal_flip_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

vertical_flip_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomVerticalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

rotation_transform_1 = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation((0, 20)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

rotation_transform_2 = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation((-20, 0)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

shear_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomAffine(degrees=10, shear=2),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

translate_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomAffine(degrees=10, translate=(0.15,0.15)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

pad_transform = transforms.Compose([
    transforms.Pad(5, padding_mode="reflect"),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])


def initialize_data(folder):
    train_zip = folder + '/train_images.zip'
    test_zip = folder + '/test_images.zip'
    if not os.path.exists(train_zip) or not os.path.exists(test_zip):
        raise(RuntimeError("Could not find " + train_zip + " and " + test_zip
              + ', please download them from https://www.kaggle.com/c/nyu-cv-fall-2018/data '))
    # extract train_data.zip to train_data
    train_folder = folder + '/train_images'
    if not os.path.isdir(train_folder):
        print(train_folder + ' not found, extracting ' + train_zip)
        zip_ref = zipfile.ZipFile(train_zip, 'r')
        zip_ref.extractall(folder)
        zip_ref.close()
    # extract test_data.zip to test_data
    test_folder = folder + '/test_images'
    if not os.path.isdir(test_folder):
        print(test_folder + ' not found, extracting ' + test_zip)
        zip_ref = zipfile.ZipFile(test_zip, 'r')
        zip_ref.extractall(folder)
        zip_ref.close()

    # make validation_data by using images 00000*, 00001* and 00002* in each class
    val_folder = folder + '/val_images'
    if not os.path.isdir(val_folder):
        print(val_folder + ' not found, making a validation set')
        os.mkdir(val_folder)
        for dirs in os.listdir(train_folder):
            if dirs.startswith('000'):
                os.mkdir(val_folder + '/' + dirs)
                for f in os.listdir(train_folder + '/' + dirs):
                    if f.startswith('00000') or f.startswith('00001') or f.startswith('00002'):
                        # move file to validation folder
                        os.rename(train_folder + '/' + dirs + '/' + f, val_folder + '/' + dirs + '/' + f)




# # from __future__ import print_function
# # import zipfile
# # import os
# #
# # import torchvision.transforms as transforms
# #
# # # once the images are loaded, how do we pre-process them before being passed into the network
# # # by default, we resize the images to 32 x 32 in size
# # # and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# # # the training set
# # data_transforms = transforms.Compose([
# #     transforms.Resize((32, 32)),
# #     transforms.ToTensor(),
# #     transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
# # ])
# # vals_data_transforms = transforms.Compose([
# #     transforms.Grayscale(num_output_channels=1),  # applied grayscale to images # SWITCH TO GRAYSCALE
# #     transforms.Resize((32, 32)),
# #     transforms.ToTensor(),
# #     # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
# #     transforms.Normalize((0.3337,), (0.2672,))  # images are grayscaled which means one channel ==> normalize only one dimension  # SWITCH TO GRAYSCALE
# # ])
# # train_data_transform = transforms.Compose([
# #     transforms.Grayscale(num_output_channels=1),  # applied grayscale to images # SWITCH TO GRAYSCALE
# #     transforms.ColorJitter(brightness=.4, contrast=.4, saturation=.4, hue=.1),  # randomly adjust brightness, contrast, etc. of images
# #     transforms.RandomRotation(15),
# #     transforms.Resize((32, 32)),
# #     transforms.ToTensor(),
# #     # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))  # SWITCH TO NON_GRAYSCALE
# #     transforms.Normalize((0.3337,), (0.2672,))  # images are grayscaled which means one channel ==> normalize only one dimension  # SWITCH TO GRAYSCALE
# # ])
# #
# #
# # def initialize_data(folder):
# #     train_zip = folder + '/train_images.zip'
# #     test_zip = folder + '/test_images.zip'
# #     if not os.path.exists(train_zip) or not os.path.exists(test_zip):
# #         raise(RuntimeError("Could not find " + train_zip + " and " + test_zip
# #               + ', please download them from https://www.kaggle.com/c/nyu-cv-fall-2018/data '))
# #     # extract train_data.zip to train_data
# #     train_folder = folder + '/train_images'
# #     if not os.path.isdir(train_folder):
# #         print(train_folder + ' not found, extracting ' + train_zip)
# #         zip_ref = zipfile.ZipFile(train_zip, 'r')
# #         zip_ref.extractall(folder)
# #         zip_ref.close()
# #     # extract test_data.zip to test_data
# #     test_folder = folder + '/test_images'
# #     if not os.path.isdir(test_folder):
# #         print(test_folder + ' not found, extracting ' + test_zip)
# #         zip_ref = zipfile.ZipFile(test_zip, 'r')
# #         zip_ref.extractall(folder)
# #         zip_ref.close()
# #
# #     # make validation_data by using images 00000*, 00001* and 00002* in each class
# #     val_folder = folder + '/val_images'
# #     if not os.path.isdir(val_folder):
# #         print(val_folder + ' not found, making a validation set')
# #         os.mkdir(val_folder)
# #         for dirs in os.listdir(train_folder):
# #             if dirs.startswith('000'):
# #                 os.mkdir(val_folder + '/' + dirs)
# #                 for f in os.listdir(train_folder + '/' + dirs):
# #                     if f.startswith('00000') or f.startswith('00001') or f.startswith('00002'):
# #                         # move file to validation folder
# #                         os.rename(train_folder + '/' + dirs + '/' + f, val_folder + '/' + dirs + '/' + f)
#
#
#
#
#
# from __future__ import print_function
# import zipfile
# import os
#
#
# import torchvision.transforms as transforms
#
# # once the images are loaded, how do we pre-process them before being passed into the network
# # by default, we resize the images to 32 x 32 in size
# # and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# # the training set
# data_transforms = transforms.Compose([
# transforms.Grayscale(num_output_channels=1),  # use if doing grayscale
#     transforms.Resize((32, 32)),
#     transforms.ToTensor(),
#     # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
#     transforms.Normalize((0.3337,), (0.2672,))  # images are grayscaled which means one channel ==> normalize only one dimension
# ])
#
# train_data_transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),  # applied grayscale to images
#     transforms.ColorJitter(brightness=.4, contrast=.4, saturation=.4, hue=.1),  # randomly adjust brightness, contrast, etc. of images
#     transforms.RandomRotation(15),
#     transforms.Resize((32, 32)),
#     transforms.ToTensor(),  # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
#     transforms.Normalize((0.3337,), (0.2672,))  # images are grayscaled which means one channel ==> normalize only one dimension
# ])
#
#
# def initialize_data(folder):
#     train_zip = folder + '/train_images.zip'
#     test_zip = folder + '/test_images.zip'
#     if not os.path.exists(train_zip) or not os.path.exists(test_zip):
#         raise(RuntimeError("Could not find " + train_zip + " and " + test_zip
#               + ', please download them from https://www.kaggle.com/c/nyu-cv-fall-2018/data '))
#     # extract train_data.zip to train_data
#     train_folder = folder + '/train_images'
#     if not os.path.isdir(train_folder):
#         print(train_folder + ' not found, extracting ' + train_zip)
#         zip_ref = zipfile.ZipFile(train_zip, 'r')
#         zip_ref.extractall(folder)
#         zip_ref.close()
#     # extract test_data.zip to test_data
#     test_folder = folder + '/test_images'
#     if not os.path.isdir(test_folder):
#         print(test_folder + ' not found, extracting ' + test_zip)
#         zip_ref = zipfile.ZipFile(test_zip, 'r')
#         zip_ref.extractall(folder)
#         zip_ref.close()
#
#     # make validation_data by using images 00000*, 00001* and 00002* in each class
#     val_folder = folder + '/val_images'
#     if not os.path.isdir(val_folder):
#         print(val_folder + ' not found, making a validation set')
#         os.mkdir(val_folder)
#         for dirs in os.listdir(train_folder):
#             if dirs.startswith('000'):
#                 os.mkdir(val_folder + '/' + dirs)
#                 for f in os.listdir(train_folder + '/' + dirs):
#                     if f.startswith('00000') or f.startswith('00001') or f.startswith('00002'):
#                         # move file to validation folder
#                         os.rename(train_folder + '/' + dirs + '/' + f, val_folder + '/' + dirs + '/' + f)
