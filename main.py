import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import string
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import multiprocessing as mp

# here in the begnining the i am deffing
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-3
CLIP_NORM = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu_count = mp.cpu_count()

print(f'Device: {DEVICE}')
print(f'cpu_count: {cpu_count}')

#  I am using my local directory as path so  when you copy my code take care of it.
path = 'D:/chair problem/recapa/samples/samples/'

# checking the number of images and unique label
images = os.listdir(path)
unique_len = len(np.unique([img.split(".")[0] for img in images]))
print(f'Total images: {len(images)}')
print(f'Unique length of capthca: {unique_len}')


# check for non-captcha if len of capthca not equal to 5
for idx, filename in enumerate(images):
    if len(filename.split(".")[0]) != 5:
        print(
            f'Found file "{filename}" at index {idx} that is non-captcha image')
        images.remove(filename)

print(f'After cleaning, there are: {len(images)} images')

train_images, test_images = train_test_split(images, random_state=0)
print(f'Train images: {len(train_images)}')
print(f'Test images: {len(test_images)}')

# Character and numbers Mapping

all_letter = string.ascii_lowercase + string.digits
mapping = {}
mapping_inv = {}
i = 0
for x in all_letter:
    mapping[x] = i
    mapping_inv[i] = x
    i += 1


num_chart = len(mapping)
# print(num_chart)
# print(mapping)
# print(mapping_inv)

# data loader now I am going to load images with


def remove_file_extension(filename):
    """this function for removing extension like file.png it will convert it file"""
    return filename.split('.')[0]


class CaptchaDataset(Dataset):
    # this init function is a constructer which takes folder directory and a file (image) name
    # using the self keyword making goble the varible name base_dir and image_filename
    def __init__(self, base_dir, image_filename):
        self.base_dir = base_dir
        self.image_filename = image_filename


# The function len retrun the size of arry image or number of images in the dataset


    def __len__(self):
        return len(self.image_filename)


# function getitem take index as a input and return the the name of image and the image


    def __getitem__(self, index):
        image_filename = self.image_filename[index]
        image_path = os.path.join(self.base_dir, image_filename)
        image = Image.open(image_path).convert("RGB")
        image = self.transfrom(image)
        label = remove_file_extension(image_filename)
        return (image, label)
# this function take image as an imput and transfrom the image into tensor then normalize
# the image and retrun image as output

# Here is the explation of of function of transfroms
# transformation normalizes the tensor image by subtracting the mean
# values specified in mean and dividing by the standard deviation
# values specified in std. The values in mean and std are expected to
# be sequences of length 3, corresponding to the mean and standard
# deviation values for the red, green, and blue channels of the image.
    def transfrom(self, image):
        transform_op = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                  std=(0.229, 0.224, 0.225))
             ])
        return transform_op(image)


# loading data in  as train and test dataset
train_dataset = CaptchaDataset(path, train_images)
test_dataset = CaptchaDataset(path, test_images)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, num_workers=cpu_count, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         num_workers=cpu_count, shuffle=False)

print(f'{len(train_loader)} batches in the train_loader')
print(f'{len(test_loader)} batches in the test_loader')

#this solution works on Kaggle's notebook but it not working on vscode

dataiter = iter(train_loader)
batch_images, batch_labels = next(dataiter)
print(batch_images.shape)
print(batch_labels)
