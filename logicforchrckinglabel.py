# in this file I am creating logic for checking the length of the labal weither it 5 or not
# if not then remove the that label and the image
import os
path = "D:/chair problem/recapa/samples/samples/"
images = os.listdir(path)
for idx, filename in enumerate(images):
    if len(filename.split(".")[0]) != 5:
        print(f"name of the file is {filename} at the index {idx}")
        images.remove(filename) 

print(f"After cleaning, there are: {len(images)} images")
