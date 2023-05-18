import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from globals import TRAIN_PATH, TRAIN_TRANSFORMS

import torch
from torch.utils.data import DataLoader, Dataset


class DigitDataset(Dataset):
    def __init__(self, data_path: Path, transforms=None) -> None:
        super().__init__()
        self.dataset = pd.read_csv(data_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset.iloc[index]
        label, data = data[0], data[1:]
        data = np.reshape(np.array(data, dtype=np.float32), (28, 28))
        # data = data[1:]

        if self.transforms:
            data = self.transforms(data)

        return data, label


def data_viz(data, label, tile: bool = False):
    if tile:
        num_samples = data.shape[0]
        plt_row = num_samples // 2
        plt_col = num_samples - plt_row
        for i in range(num_samples):
            img = data[i].squeeze()
            plt.subplot(plt_row, plt_col, i + 1)
            plt.title("Label: " + str(int(label[i])))
            plt.imshow(img)
        plt.tight_layout()
    else:
        img = data.squeeze()
        plt.title("Label: " + str(int(label)))
        plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    # print(f"[INFO] Current working directory: {os.getcwdb()}")
    # set_cwdb = 'c:\\Users\\Studen\\Documents\\vscode_projects\\kaggleCompetitions\\'
    # print(f"[INFO] Changing working directory to: {set_cwdb}")
    # os.chdir(set_cwdb)
    tarin_set = DigitDataset(TRAIN_PATH, TRAIN_TRANSFORMS)
    train_loader = DataLoader(tarin_set, batch_size=8, shuffle=False, num_workers=10)

    images, labels = next(iter(train_loader))
    print(images.shape)
    data_viz(data=images, label=labels, tile=True)
    data_viz(data=images[0], label=labels[0])

    # print(tarin_set.__getitem__(0))
