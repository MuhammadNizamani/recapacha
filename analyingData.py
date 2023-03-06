# In this file image I am checking the size of images to make it sure
# images are same size

import os
import numpy as np
import pandas as pd
from PIL import Image


# adding function which remove extention for the name of the image becasue the name of the
# is the labal which I am going use as label and extention is a noise
def remove_extension(filename):
    return filename.split('.')[0]


#  I am using my local directory as path so  when you copy my code take care of it.
path = "D:/chair problem/recapa/samples/samples/"


listofName = os.listdir(path)
count_of_label = len(np.unique([img.split(".")[0] for img in listofName]))
print(
    f"number of images in this dataset {len(listofName)} and number of unique labals {count_of_label} ")
# print(listofName)

for i in range(2):
    image = Image.open(path+listofName[i])
    width, height = image.size
    print(remove_extension(listofName[i]))
    # print(f"the size of image name {listofName[i]} width {width} and height {height}")
