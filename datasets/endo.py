import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import os
import torch

class EndoDataset(data.Dataset):
    def __init__(self, df, img_dir, transforms):
        self.img_dir = img_dir
        self.transforms = transforms
        self.df = pd.read_csv(df)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        unnamed = self.df.iloc[idx, 0]
        path = os.path.join(self.img_dir, self.df.iloc[idx, 1])
        image = Image.open(path)
        image = np.array(image)
        image = Image.fromarray(image)
        image = self.transforms(image)

        study_id = self.df.iloc[idx, 2]
        ulcer = self.df.iloc[idx, 3]
        erosion = self.df.iloc[idx, 4]
        polyp = self.df.iloc[idx, 5]
        tumor = self.df.iloc[idx, 6]

        endo_tensor = torch.tensor([ulcer, erosion, polyp, tumor])

        return image, endo_tensor
