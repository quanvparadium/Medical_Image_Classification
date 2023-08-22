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
        self.start_idx = 0
        if df.endswith('.csv'):
            self.df = pd.read_csv(df)
            self.start_idx = 1 
        elif df.endswith('.txt'):
            unsave_df = pd.read_csv(df, sep=" ", header=None)
            # label_df = unsave_df[1].str.split(',', expand=True)
            # label_df.columns = [f'column{i+1}' for i in range(label_df.shape[0])]
            # label_df = label_df.astype(dtype=np.float64)
            # self.df = pd.concat([unsave_df[0], label_df], axis=1)
            self.df = unsave_df
        else:
            raise Exception("Format errors")

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # unnamed = self.df.iloc[idx, 0]
        path = os.path.join(self.img_dir, self.df.iloc[idx, 0 + self.start_idx])
        image = Image.open(path)
        image = np.array(image)
        image = Image.fromarray(image)
        image = self.transforms(image)

        # study_id = self.df.iloc[idx, 2]
        ulcer = self.df.iloc[idx, 1 + self.start_idx]
        erosion = self.df.iloc[idx, 2 + self.start_idx]
        polyp = self.df.iloc[idx, 3 + self.start_idx]
        tumor = self.df.iloc[idx, 4 + self.start_idx]

        endo_tensor = torch.tensor([ulcer, erosion, polyp, tumor])

        return image, endo_tensor
