import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import os
import time
from pathlib import Path

from dataset import TitanicDataset


if __name__ == '__main__':
    print(f"[INFO] Current working directory: {os.getcwd()}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Current running device: {device}")

    current_time = time.strftime('%H_%M_%S-%d_%m_%Y', time.localtime(time.time()))
    log_dir = Path(f"Titanic/logs/{current_time}")

    data_path = Path('Titanic/titanic_dataset/train_prepared.csv')
    full_dataset = TitanicDataset(csv_file=data_path)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

