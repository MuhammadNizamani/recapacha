# In this file image I am checking the size of images to make it sure
# images are same size

import os
import numpy as np
import pandas as pd
from PIL import Image

listofName = os.listdir("D:/chair problem/recapa/samples/samples")
for i in range(500):
    image = Image.open(
        f"D:/chair problem/recapa/samples/samples/{listofName[i]}")
    width, height = image.size
    print(
        f"the size of image name {listofName[i]} width {width} and height {height}")
