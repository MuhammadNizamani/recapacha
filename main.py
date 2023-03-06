import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import string
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
#  I am using my local directory as path so  when you copy my code take care of it.
path = 'D:/chair problem/recapa/samples/samples/'

# checking the number of images and unique label
images = os.listdir(path)
unique_len = len(np.unique([img.split(".")[0] for img in images]))
print(f'Total images: {len(images)}')
print(f'Unique length of capthca: {unique_len}')


#check for non-captcha if len of capthca not equal to 5
for idx, filename in enumerate(images):
    if len(filename.split(".")[0]) != 5:
        print(f'Found file "{filename}" at index {idx} that is non-captcha image')
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
print(num_chart)
# print(mapping)
print(mapping_inv)

#data loader now I am going to load images with 


def remove_file_extension(filename):
    """this function for removing extension like file.png it will convert it file"""
    return filename.split('.')[0]