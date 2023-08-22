import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import os
import torch

class ColonDataset(data.Dataset):
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
        # slide_id = self.df.iloc[idx, 0]
        path = os.path.join(self.img_dir, self.df.iloc[idx, 0 + self.start_idx])
        image = Image.open(path)
        image = np.array(image)
        image = Image.fromarray(image)
        image = self.transforms(image)

        tumor = self.df.iloc[idx, 1 + self.start_idx]
        colon_tensor = torch.tensor([tumor])
        # chest_tensor = torch.tensor([pleural_effusion, nodule, pneumonia, cardiomegaly, hilar_enlargement, fracture_old, fibrosis, aortic_calcification, tortuous_aorta, thickened_pleura, TB, pneumothorax, emphysema, atelectasis, calcification, pulmonary_edema, increased_lung_markings, elevated_diaphragm, consolidation])

        return image, colon_tensor
