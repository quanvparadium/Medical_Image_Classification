import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import os

class MedFMDataset_Competition(data.Dataset):
    def __init__(self, df, img_dir, transforms):
        self.img_dir = img_dir
        self.transforms = transforms
        self.df = pd.read_csv(df)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        path = self.img_dir + self.df.iloc[idx, 0]