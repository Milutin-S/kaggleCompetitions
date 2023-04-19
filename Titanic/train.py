import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import os
import time
from tqdm import tqdm
from pathlib import Path

from dataset import TitanicDataset
from nn import TitanicNet

def train(epoch:int):
    model.train()
    running_loss = 0.
    train_loss = 0.
    total = 0
    correct = 0

    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        threshold_out = torch.where(outputs >= threshold, torch.ones_like(outputs), torch.zeros_like(outputs))
        correct += (threshold_out == labels).sum().item()
        # correct += ((outputs >= 0.6).astype(int) == labels).sum().item()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[INFO] [{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            writer.add_scalar('Train Loss [per 100]', running_loss / 100, epoch * len(train_loader) + i)
            running_loss = 0.
    
    train_loss /= len(train_loader)
    return train_loss

def main(epoch_num:int=10):
    best_performance = None

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"[INFO] Created models dir at: {out_dir}")

    print(f"[INFO] TRAINING STARTED!")
    for epoch in tqdm(range(epoch_num)):
        train_loss = train(epoch)
        print(f'[INFO] Epoch {epoch+1}/{epoch_num}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}')
        writer.add_scalar('Train Loss [epoch]', train_loss, epoch)

    print(f"[INFO] FINISHED TRAINING!")


if __name__ == '__main__':
    print(f"[INFO] Current working directory: {os.getcwd()}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Current running device: {device}")

    current_time = time.strftime('%H_%M_%S-%d_%m_%Y', time.localtime(time.time()))
    log_dir = Path(f"Titanic/logs/{current_time}")
    out_dir = Path(f"Titanic/models/{current_time}")

    data_path = Path('Titanic/titanic_dataset/train_prepared.csv')
    full_dataset = TitanicDataset(csv_file=data_path)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = TitanicNet(input_size=8).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    writer = SummaryWriter(log_dir)
    epoch_num = 30
    threshold = 0.8

    main(epoch_num)


