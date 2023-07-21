import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from globals import *

import torch
from torch.utils.data import DataLoader, Dataset

# TODO:
# Ideas
# 1. Shift one or more of the chanels before input to model


class DigitDataset(Dataset):
    def __init__(
        self, data_path: Path, transforms: bool = False, type: str = "train"
    ) -> None:
        super().__init__()
        self.dataset = pd.read_csv(data_path)
        self.transforms = transforms
        self.type = type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset.iloc[index]
        if self.type == "train":
            label, data = data[0], data[1:]
        data = np.reshape(np.array(data, dtype=np.uint8), IMAGE_SHAPE)
        data = REQUIRED_TRANSFORMS(data)

        if self.transforms and self.type == "train":
            data, label = apply_transforms(data, label)

        if self.type == "test":
            return data
        else:
            return data, label


def data_viz(data, label, tile: bool = False):
    if tile:
        num_samples = data.shape[0]
        plt_row = int(np.sqrt(num_samples))
        plt_col = int(np.ceil(num_samples / plt_row))
        for i in range(num_samples):
            img = data[i].squeeze()
            plt.subplot(plt_row, plt_col, i + 1)
            plt.title("Label: " + str(int(label[i])))
            plt.axis("off")
            plt.imshow(img)
        plt.tight_layout()
    else:
        img = data.squeeze()
        plt.title("Label: " + str(int(label)))
        plt.axis("off")
        plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    train_set = DigitDataset(TRAIN_PATH, type="train")
    train_loader = DataLoader(train_set, batch_size=16, shuffle=False, num_workers=10)

    images, labels = next(iter(train_loader))
    print(
        f"[INFO] Minimum data value: {torch.min(images[0])}, Maximum data value: {torch.max(images[0])}"
    )
    print(images.shape)
    data_viz(data=images, label=labels, tile=True)
