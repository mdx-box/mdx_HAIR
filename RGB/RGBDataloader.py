from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from tqdm.auto import tqdm
class MyRGBDataset(Dataset):
    def __init__(self, data_root_path:str):

        super(MyRGBDataset).__init__()
        self.data_root_path = data_root_path
        self.RGB_data, self.labels = self.data_load(data_path=self.data_root_path)
        self.len = len(self.labels)
        print('数据已经准备好了....')
    def __getitem__(self, index) :

        return self.RGB_data[index],  self.labels[index]
    def __len__(self):
        return self.len

    def data_load(self, data_path):
        path_list = os.listdir(data_path)[0:10]
        X_RGB = []
        y = []
        for path in tqdm(path_list):
            RGB_data = torch.from_numpy(np.load(f'{data_path}/{path}/{path}_RGB.npy'))
            X_RGB.append(RGB_data)
            label_data = torch.from_numpy(np.load(f'{data_path}/{path}/{path}_label.npy'))
            y.append(label_data)
        return torch.cat(X_RGB,dim=0), torch.cat(y,dim=0)