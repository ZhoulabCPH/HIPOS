
import os
import torchvision
import cv2
import numpy as np



import torch
import cv2
from torch.utils.data import Dataset
import h5py
from torch import nn, optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps, ImageFilter
is_amp = True
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from augmentation import *

############################################################
####### Folds
############################################################
def make_fold_WSI(arg):
    train_df=pd.read_csv(arg.Train_Patient)
    Index = np.arange(len(train_df))
    np.random.shuffle(Index)
    train_df = train_df.iloc[Index].reset_index(drop=True)
    train_df['Index']=Index


    test_df=pd.read_csv(arg.Test_Patient)
    Index = np.arange(len(test_df))
    np.random.shuffle(Index)
    test_df = test_df.iloc[Index].reset_index(drop=True)
    test_df['Index']=Index
    return train_df,test_df

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            # transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomRotation(
                degrees=90,
                resample=False,
                expand=False,
                center=None,
                fill=255,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.4),
            Solarization(p=0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            # transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomRotation(
                degrees=90,
                resample=False,
                expand=False,
                center=None,
                fill=255,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2

class Transform_:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),

            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        return y1,y1



class CreatDataset(Dataset):
    def __init__(self, df,X, transform1=None):
        self.df = df
        self.Data=X
        self.transform1 = transform1()

        self.length = len(self.df)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        img_name = self.df.loc[index, 'Patch_Name']
        img_index=self.df.loc[index, 'Unnamed: 0']
        image = Image.fromarray(self.Data[img_index])
        image1,image2 = self.transform1(image)
        r = {}
        r['index'] = img_index
        r['image_Argument1'] = image1
        r['image_Argument2'] = image2
        r['name']=img_name.split("_")[0]+"_"+img_name.split("_")[-1].split(".")[0]
        return r
tensor_list = [
     'image_Argument1','image_Argument2'
]

def image_to_tensor(image, mode='bgr'):  # image mode
    if mode == 'bgr':
        image = image[:, :, ::-1]
    x = image
    x = x.transpose(2, 0, 1)
    x = np.ascontiguousarray(x)
    x = torch.tensor(x, dtype=torch.float)
    return x


def tensor_to_image(x, mode='bgr'):
    image = x.data.cpu().numpy()
    image = image.transpose(1, 2, 0)
    if mode == 'bgr':
        image = image[:, :, ::-1]
    image = np.ascontiguousarray(image)
    image = image.astype(np.float32)
    return image



tensor_list = ['image_Argument1','image_Argument2']


def null_collate(batch):
    d = {}
    key = batch[0].keys()
    for k in key:
        v = [b[k] for b in batch]
        if k in tensor_list:
            v = torch.stack(v)
        d[k] = v
    # d['organ'] = d['organ'].reshape(-1)
    return d

