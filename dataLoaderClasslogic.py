import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def remove_file_extension(filename):
    """this function for removing extension like file.png it will convert it file"""
    return filename.split('.')[0]


class CaptchaDataset(Dataset):
    def __init__(self, base_dir, image_filename):
        self.base_dir = base_dir
        self.image_filename = image_filename

    def __len__(self):
        return len(self.image_filename)

    def __getitem__(self, index):
        image_filename = self.image_filename[index]
        image_path = os.path.join(self.base_dir, image_filename)
        image = Image.open(image_path).convert("RGB")
        image = self.transfrom(image)
        label = remove_file_extension(image_filename)
        return (image, label)

    def transfrom(image):
        transform_op = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                  std=(0.229, 0.224, 0.225))
             ])
        return transform_op(image)