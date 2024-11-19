import torch
import torch.nn as nn
from torch.utils.data import Dataset,RandomSampler,DataLoader
is_amp = True
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
############################################################
####### Folds
############################################################
def make_fold_WSI(arg):
    Datasets = pd.read_csv(arg.Train_csv).iloc[:, 1:]
    return Datasets


class HubmapDataset(Dataset):
    def __init__(self, df,arg):
        self.arg=arg
        self.df = df
        self.length = len(self.df)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        img_name = self.df.loc[index, 'Sample_Name']
        Image_Datasets=torch.from_numpy(np.array(self.df.values[index, 1:16], dtype=np.float32))
        Image_label=torch.from_numpy(np.array(self.df.values[index, -1], dtype=np.int64))

        r = {}
        r['index'] = index
        r['image'] = Image_Datasets
        r['label']=Image_label
        r['name']=img_name

        return r


tensor_list = [
     'image','label'
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

tensor_list = ['image','label']


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

