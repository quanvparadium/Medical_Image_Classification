import pandas as pd
import numpy as np

def isSubsample(subfile, originfile):
    sub_df = pd.read_csv(subfile, sep=" ", header=None)
    pass




if __name__ == "__main__":
    subfile = './data/Datasets/chest_1-shot_train_exp1.txt'
    originfile = 'chest_train.csv'
    if isSubsample(subfile, originfile):
        print("True")