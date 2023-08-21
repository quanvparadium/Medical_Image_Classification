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
        self.df = pd.read_csv(df)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        unnamed = self.df.iloc[idx, 0]
        path = self.img_dir + self.df.iloc[idx, 1]
        image = Image.open(path)
        image = np.array(image)
        image = Image.fromarray(image)
        image = self.transforms(image)

        pleural_effusion = self.df.iloc[idx, 2]
        nodule = self.df.iloc[idx, 3]
        pneumonia = self.df.iloc[idx, 4]	
        cardiomegaly = self.df.iloc[idx, 5]
        hilar_enlargement = self.df.iloc[idx, 6]
        fracture_old = self.df.iloc[idx, 7]
        fibrosis = self.df.iloc[idx, 8]
        aortic_calcification = self.df.iloc[idx, 9]
        tortuous_aorta = self.df.iloc[idx, 10]
        thickened_pleura = self.df.iloc[idx, 11]
        TB = self.df.iloc[idx, 12]
        pneumothorax = self.df.iloc[idx, 13]
        emphysema = self.df.iloc[idx, 14]
        atelectasis = self.df.iloc[idx, 15]
        calcification = self.df.iloc[idx, 16]
        pulmonary_edema = self.df.iloc[idx, 17]
        increased_lung_markings = self.df.iloc[idx, 18]
        elevated_diaphragm = self.df.iloc[idx, 19]
        consolidation = self.df.iloc[idx, 20]
        
        chest_tensor = torch.tensor([pleural_effusion, nodule, pneumonia, cardiomegaly, hilar_enlargement, fracture_old, fibrosis, aortic_calcification, tortuous_aorta, thickened_pleura, TB, pneumothorax, emphysema, atelectasis, calcification, pulmonary_edema, increased_lung_markings, elevated_diaphragm, consolidation])

        return image, chest_tensor
