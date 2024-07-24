import numpy as np
import torch 
import torchvision
import sklearn as sk
from PIL import Image
import shutil
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder

# This is for the progress bar.
from tqdm.auto import tqdm

import pandas as pd

data_dir='/oscar/home/dfurtad1/projects/COVID19-DATASET/'
normal_dir = data_dir + 'train/normal'
covid_dir = data_dir + 'train/covid19' 

import os
from sklearn.model_selection import train_test_split

print('Loading data ...')

#DEFINE PATH
data_dir='/oscar/home/dfurtad1/projects/COVID19-DATASET/'

#LOADING IMAGES
normal_images = [img for img in os.listdir(normal_dir) ]
covid_images = [img for img in os.listdir(covid_dir) ] 

print(len(normal_images))
print(len(covid_images))

images = normal_images + covid_images 
labels = [0] * len(normal_images) + [1] * len(covid_images)

#SPLIT DATA
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.25, stratify=labels)

print(f"Total images: {len(images)}")
print(f"Training images: {len(train_images)}")
print(f"Testing images: {len(test_images)}")

class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(np.int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data) 

def find_file_directory(root_dir, file_name):
    """
    Search for the directory containing the specified file starting from the root directory.
    
    :param root_dir: The root directory to start the search from.
    :param file_name: The name of the file to search for.
    :return: The path of the directory containing the file, or None if not found.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if file_name in filenames:
            return dirpath
    return None

def clear_directory(directory_path):
    """
    Clear all files in the specified directory.
    
    :param directory_path: The path of the directory to clear.
    """
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return
    
    # List all files in the directory
    files = os.listdir(directory_path)
    
    # Loop through the files and delete each one
    for file_name in files:
        file_path = os.path.join(directory_path, file_name)
        
        # Check if it is a file (not a directory)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_name}")
    
    print("All files have been deleted.")

directory_to_clear = '/oscar/home/dfurtad1/projects/COVID19-DATASET/train/total/data'
clear_directory(directory_to_clear)

for img in train_images:
    path = find_file_directory('/oscar/home/dfurtad1/projects/COVID19-DATASET', img) + '/' + img
    destination = '/oscar/home/dfurtad1/projects/COVID19-DATASET/train/total/data'
    shutil.copy(path, destination)

directory_to_clear = '/oscar/home/dfurtad1/projects/COVID19-DATASET/test/data'
clear_directory(directory_to_clear)

for img in test_images:
    path = find_file_directory('/oscar/home/dfurtad1/projects/COVID19-DATASET', img) + '/' + img
    destination = '/oscar/home/dfurtad1/projects/COVID19-DATASET/test/data'
    shutil.copy(path, destination)

train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
])

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

batch_size = 64

from torch.utils.data import DataLoader

# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = DatasetFolder("COVID19-DATASET/train/total", loader=lambda x: Image.open(x), extensions=("jpg", "jpeg", "png"), transform=train_tfm)
# valid_set = DatasetFolder("food-11/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
test_set = DatasetFolder("COVID19-DATASET/test", loader=lambda x: Image.open(x), extensions=("jpg", "jpeg", "png"), transform=test_tfm)

# Construct data loaders.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
# valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
