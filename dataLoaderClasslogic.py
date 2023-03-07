import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


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
#this function take image as an imput and transfrom the image into tensor then normalize 
# the image and retrun image as output 

# Here is the explation of of function of transfroms
# transformation normalizes the tensor image by subtracting the mean 
# values specified in mean and dividing by the standard deviation 
# values specified in std. The values in mean and std are expected to
# be sequences of length 3, corresponding to the mean and standard 
# deviation values for the red, green, and blue channels of the image.
    def transfrom(image):
        transform_op = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                  std=(0.229, 0.224, 0.225))
             ])
        return transform_op(image)
