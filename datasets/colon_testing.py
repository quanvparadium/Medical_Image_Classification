import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import os

class ColonDataset_Testing(data.Dataset):
    def __init__(self, df, img_dir, transforms):
        print("Dataset: Colon_MedFM_Competition_Testing")
        self.img_dir = img_dir
        self.transforms = transforms
        self.df = pd.read_csv(df)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        slide_id = self.df.iloc[idx, 0]
        img_name = self.df.iloc[idx, 1]
        path = os.path.join(self.img_dir, img_name)
        image = Image.open(path)
        image = np.array(image)
        image = Image.fromarray(image)
        image = self.transforms(image)
        
        return image, img_name