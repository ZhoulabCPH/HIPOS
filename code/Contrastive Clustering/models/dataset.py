import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
import h5py

# Suppress warnings
warnings.filterwarnings('ignore')

############################################################
# Fold Creation Function
############################################################
def make_fold_WSI(args):
    """
    Generate training and testing dataframes with shuffled indices.
    """
    def shuffle_and_add_index(df):
        indices = np.arange(len(df))
        np.random.shuffle(indices)
        df = df.iloc[indices].reset_index(drop=True)
        df['Index'] = indices
        return df

    train_df = shuffle_and_add_index(pd.read_csv(args.Train_Patient))
    test_df = shuffle_and_add_index(pd.read_csv(args.Test_Patient))
    return train_df, test_df

############################################################
# Custom Transformations
############################################################
class GaussianBlur:
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        return img

class Solarization:
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img

class Transform:
    """
    Transformation pipeline for generating two augmented versions of an image.
    """
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=90, fill=255),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.4),
            Solarization(p=0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomRotation(degrees=90, fill=255),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2

class Transform_:
    """
    Basic transformation pipeline for single image processing.
    """
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y = self.transform(x)
        return y, y

############################################################
# Custom Dataset
############################################################
class CustomDataset(Dataset):
    """
    Dataset class for handling image patches and transformations.
    """
    def __init__(self, df, data, transform1):
        self.df = df
        self.data = data
        self.transform1 = transform1()
        self.length = len(df)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_name = self.df.loc[index, 'Patch_Name']
        img_index = self.df.loc[index, 'Unnamed: 0']
        image = Image.fromarray(self.data[img_index])
        image1, image2 = self.transform1(image)
        return {
            'index': img_index,
            'image_Argument1': image1,
            'image_Argument2': image2,
            'name': f"{img_name.split('_')[0]}_{img_name.split('_')[-1].split('.')[0]}"
        }

############################################################
# Utility Functions
############################################################
tensor_list = ['image_Argument1', 'image_Argument2']

def image_to_tensor(image, mode='bgr'):
    """
    Convert an image to a PyTorch tensor.
    """
    if mode == 'bgr':
        image = image[:, :, ::-1]
    x = image.transpose(2, 0, 1)
    return torch.tensor(np.ascontiguousarray(x), dtype=torch.float)

def tensor_to_image(tensor, mode='bgr'):
    """
    Convert a PyTorch tensor back to an image.
    """
    image = tensor.cpu().numpy().transpose(1, 2, 0)
    if mode == 'bgr':
        image = image[:, :, ::-1]
    return image.astype(np.float32)

def null_collate(batch):
    """
    Custom collate function for DataLoader.
    """
    collated_batch = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        collated_batch[key] = torch.stack(values) if key in tensor_list else values
    return collated_batch

############################################################
# Main Execution (Example Usage)
############################################################
if __name__ == "__main__":
    # Example arguments
    class Args:
        Train_Patient = "train_patients.csv"
        Test_Patient = "test_patients.csv"

    args = Args()
    train_df, test_df = make_fold_WSI(args)

    # Example HDF5 data (mocked for demonstration)
    h5_data = np.random.randint(0, 255, (100, 224, 224, 3), dtype=np.uint8)
    transform = Transform_

    # Create dataset and dataloader
    dataset = CustomDataset(train_df, h5_data, transform)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=null_collate)

    for batch in dataloader:
        print(batch['index'], batch['image_Argument1'].shape, batch['name'])
