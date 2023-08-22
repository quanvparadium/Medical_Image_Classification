import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import os
import torch

class ChestDataset(data.Dataset):
    def __init__(self, df, img_dir, transforms):
        self.img_dir = img_dir
        self.transforms = transforms
        self.start_idx = 0
        if df.endswith('.csv'):
            self.df = pd.read_csv(df)
            self.start_idx = 1
        elif df.endswith('.txt'):
            unsave_df = pd.read_csv(df, sep=" ", header=None)
            label_df = unsave_df[1].str.split(',', expand=True)
            label_df.columns = [f'column{i+1}' for i in range(label_df.shape[0])]
            label_df = label_df.astype(dtype=np.float64)
            self.df = pd.concat([unsave_df[0], label_df], axis=1)
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

        pleural_effusion = self.df.iloc[idx, 1 + self.start_idx]
        nodule = self.df.iloc[idx, 2 + self.start_idx]
        pneumonia = self.df.iloc[idx, 3 + self.start_idx]	
        cardiomegaly = self.df.iloc[idx, 4 + self.start_idx]
        hilar_enlargement = self.df.iloc[idx, 5 + self.start_idx]
        fracture_old = self.df.iloc[idx, 6 + self.start_idx]
        fibrosis = self.df.iloc[idx, 7 + self.start_idx]
        aortic_calcification = self.df.iloc[idx, 8 + self.start_idx]
        tortuous_aorta = self.df.iloc[idx, 9 + self.start_idx]
        thickened_pleura = self.df.iloc[idx, 10 + self.start_idx]
        TB = self.df.iloc[idx, 11 + self.start_idx]
        pneumothorax = self.df.iloc[idx, 12 + self.start_idx]
        emphysema = self.df.iloc[idx, 13 + self.start_idx]
        atelectasis = self.df.iloc[idx, 14 + self.start_idx]
        calcification = self.df.iloc[idx, 15 + self.start_idx]
        pulmonary_edema = self.df.iloc[idx, 16 + self.start_idx]
        increased_lung_markings = self.df.iloc[idx, 17 + self.start_idx]
        elevated_diaphragm = self.df.iloc[idx, 18 + self.start_idx]
        consolidation = self.df.iloc[idx, 19 + self.start_idx]
        
        chest_tensor = torch.tensor([pleural_effusion, nodule, pneumonia, cardiomegaly, hilar_enlargement, fracture_old, fibrosis, aortic_calcification, tortuous_aorta, thickened_pleura, TB, pneumothorax, emphysema, atelectasis, calcification, pulmonary_edema, increased_lung_markings, elevated_diaphragm, consolidation])

        return image, chest_tensor
