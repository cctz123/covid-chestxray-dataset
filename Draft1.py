import numpy as np
import torch 
import torchvision
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder

# This is for the progress bar.
from tqdm.auto import tqdm

import pandas as pd

data_dir='./projects/COVID19-DATASET/'
normal_dir = data_dir + 'train/normal'
covid_dir = data_dir + 'train/covid19' 

import os
from sklearn.model_selection import train_test_split

print('Loading data ...')

#DEFINE PATH
data_dir='./projects/COVID19-DATASET/'


#LOADING IMAGES
normal_images = [img for img in os.listdir(normal_dir) ]
covid_images = [img for img in os.listdir(covid_dir) ] 

print(len(normal_images))
print(len(covid_images))

images  = normal_images + covid_images 
labels = [0] * len(normal_images) + [1] * len(covid_images)


#SPLIT DATA
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.25, stratify=labels)

print(f"Total images: {len(images)}")
print(f"Training images: {len(train_images)}")
print(f"Testing images: {len(test_images)}")


#pasted code 

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


BATCH_SIZE = 64

from torch.utils.data import DataLoader

train_set = TIMITDataset(train_x, train_y)
val_set = TIMITDataset(val_x, val_y)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) #only shuffle the training data
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False) 


