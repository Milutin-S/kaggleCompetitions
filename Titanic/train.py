import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import os
import json
import time
from tqdm import tqdm
from pathlib import Path

from dataset import TitanicDataset
from nn import TitanicNet

def train(epoch:int):
    model.train()
    running_loss = 0.
    train_loss = 0.
    total = 0.
    correct = 0.

    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        threshold_out = torch.where(outputs >= threshold, torch.ones_like(outputs), torch.zeros_like(outputs))
        correct += (threshold_out == labels).sum().item()
        total += labels.size(0)

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
    train_acc = correct/total
    return train_loss, train_acc

def test(epoch:int):
    model.eval()
    running_loss = 0.
    test_loss = 0.
    total = 0.
    correct = 0.

    with torch.no_grad():
        for i, (inputs, labels), in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            threshold_out = torch.where(outputs >= threshold, torch.ones_like(outputs), torch.zeros_like(outputs))
            correct += (threshold_out == labels).sum().item()
            total += labels.size(0)

            loss = criterion(outputs, labels)
            test_loss += loss.item()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[INFO] [{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                writer.add_scalar('Test Loss [per 100]', running_loss / 100, epoch * len(test_loader) + i)
                running_loss = 0.
        
    test_loss /= len(test_loader)
    test_acc = correct / total
    return test_loss, test_acc

def CreateMetadata():
    """Creates meta data json with all hyperparameters for current model"""

    meta_data = {
        "time_stamp" : current_time,
        "epoch_num" : epoch_num,
        "batch_size" : batch_size,
        "input_size" : input_size,
        "optimizer" : type (optimizer).__name__,
        "learning_rate" : learning_rate,
        "treshold" : threshold,
    }

    with open(os.path.join(out_dir, "model_meta_data.json"), "w") as outfile:
        json.dump(meta_data, outfile, indent=4)

    print(f"[INFO] Meta data saved at: {out_dir}")

def main(epoch_num:int=10):
    best_performance = None

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"[INFO] Created models dir at: {out_dir}")
        CreateMetadata()

    print(f"[INFO] TRAINING STARTED!")
    for epoch in tqdm(range(epoch_num)):
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)

        print(f'[INFO] Epoch {epoch+1}/{epoch_num}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}')
        print(f'[INFO] Epoch {epoch+1}/{epoch_num}, Train Accuracy: {train_acc:.2%}, Test Accuracy: {test_acc:.2%}')
        writer.add_scalar('Train Loss [epoch]', train_loss, epoch)
        writer.add_scalar('Test Loss [epoch]', test_loss, epoch)
        writer.add_scalar('Train Accuracy epoch [%]', train_acc * 100, epoch)
        writer.add_scalar('Test Accuracy  epoch [%]', test_acc * 100, epoch)

        if best_performance is None or test_acc > best_performance:
            best_performance = test_acc
            torch.save(model.state_dict(), os.path.join(out_dir, f'epoch_{epoch}_weights.pth'))

    print(f"[INFO] FINISHED TRAINING!")


def main2(epoch_num:int=10):
    best_performance = None

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"[INFO] Created models dir at: {out_dir}")
        CreateMetadata()

    print(f"[INFO] TRAINING STARTED!")
    for epoch in tqdm(range(epoch_num)):
        train_loss, train_acc = train(epoch)
        # test_loss, test_acc = test(epoch)

        # print(f'[INFO] Epoch {epoch+1}/{epoch_num}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}')
        # print(f'[INFO] Epoch {epoch+1}/{epoch_num}, Train Accuracy: {train_acc:.2%}, Test Accuracy: {test_acc:.2%}')
        writer.add_scalar('Train Loss [epoch]', train_loss, epoch)
        # writer.add_scalar('Test Loss [epoch]', test_loss, epoch)
        writer.add_scalar('Train Accuracy epoch [%]', train_acc * 100, epoch)
        # writer.add_scalar('Test Accuracy  epoch [%]', test_acc * 100, epoch)

        if best_performance is None or train_acc > best_performance:
            best_performance = train_acc
            torch.save(model.state_dict(), os.path.join(out_dir, f'epoch_{epoch}_weights.pth'))

    print(f"[INFO] FINISHED TRAINING!")


if __name__ == '__main__':
    thresholds = [0.5, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        print(f"[INFO] Current working directory: {os.getcwd()}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Current running device: {device}")

        current_time = time.strftime('%H_%M_%S-%d_%m_%Y', time.localtime(time.time()))
        log_dir = Path(f"Titanic/logs/{current_time}")
        out_dir = Path(f"Titanic/models/{current_time}")

        data_path = Path('Titanic/titanic_dataset/train_prepared.csv')
        full_dataset = TitanicDataset(csv_file=data_path)
        # train_size = int(0.8 * len(full_dataset))
        # test_size = len(full_dataset) - train_size
        # train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

        batch_size = 4
        train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        input_size = 8
        model = TitanicNet(input_size=input_size).to(device)
        criterion = nn.BCELoss()
        learning_rate = 0.0003 # 0.001
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)

        writer = SummaryWriter(log_dir)
        epoch_num = 300
        # threshold = 0.6

        main2(epoch_num)


