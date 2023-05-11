import os
import pandas as pd
from pathlib import Path
from globals import TRAIN_PATH

import torch
from torch.utils.data import DataLoader, Dataset

class DigitDataset(Dataset):
    def __init__(self, data_path:Path, transforms=None) -> None:
        super().__init__()
        self.dataset = pd.read_csv(data_path)
        self.transforms = transforms

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        data = self.dataset.iloc[index]
        label, data = data[0], data[1:]
        # data = data[1:]

        if self.transforms:
            pass
        return data, label
    
if __name__ == '__main__':
    # print(f"[INFO] Current working directory: {os.getcwdb()}")
    # set_cwdb = 'c:\\Users\\Studen\\Documents\\vscode_projects\\kaggleCompetitions\\'
    # print(f"[INFO] Changing working directory to: {set_cwdb}")
    # os.chdir(set_cwdb)
    dataset = DigitDataset(TRAIN_PATH)

    print(dataset.__getitem__(0))
