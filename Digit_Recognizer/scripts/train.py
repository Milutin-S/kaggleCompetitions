import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import time
import json
from tqdm import tqdm

from nn import Digit_Net
from dataset_loader import DigitDataset
from globals import (
    OUTPUT_DIR,
    MODELS_DIR,
    PARAMETERS_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCH_NUM,
    TRAIN_PATH,
    TRAIN_TRANSFORMS,
)


def train():
    model.train()
    train_loss = 0
    train_acc = 0
    total_predictions = 0
    correct_predictions = 0

    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, predictions = torch.max(outputs, 1)
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)

    train_loss /= len(train_loader)
    train_acc = correct_predictions / total_predictions
    return train_loss, train_acc


def test():
    model.eval()
    test_loss = 0
    test_acc = 0
    total_predictions = 0
    correct_predictions = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predictions = torch.max(outputs, 1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    test_loss /= len(test_loader)
    test_acc = correct_predictions / total_predictions
    return test_loss, test_acc


def main(validation_only: bool = False):
    best_accuracy = None

    print(f"[INFO] TRAINING STARTED!🚀")
    for epoch in tqdm(range(EPOCH_NUM)):
        print(f"[INFO] Epoch {epoch+1}/{EPOCH_NUM}")
        if not validation_only:
            train_loss, train_acc = train()
            print(f"    Train loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2%}")

        test_loss, test_acc = test()
        print(f"    Test loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2%}")

        writer.add_scalar("Train Loss [epoch]", train_loss, epoch)
        writer.add_scalar("Test Loss [epoch]", test_loss, epoch)
        writer.add_scalar("Train Accuracy epoch [%]", train_acc * 100, epoch)
        writer.add_scalar("Test Accuracy  epoch [%]", test_acc * 100, epoch)

        if best_accuracy is None or test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(
                model.state_dict(), model_dir.joinpath(f"epoch_{epoch}_weights.pth")
            )
            print(f"    Best accuracy in {epoch} epoch saved.")

    print(f"[INFO] TRAINING FINISHED! 🏁")


def Create_Parameters_Metadata():
    """Creates meta data json with all hyperparameters for current model"""

    meta_data = {
        "time_stamp": current_time,
        "epoch_num": EPOCH_NUM,
        "batch_size": BATCH_SIZE,
        "optimizer": type(optimizer).__name__,
        "learning_rate": LEARNING_RATE,
    }

    with open(params_dir.joinpath("model_meta_data.json"), "w") as outfile:
        json.dump(meta_data, outfile, indent=4)

    print(f"[INFO] Meta data saved at: {params_dir.joinpath('model_meta_data.json')} 💾")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running device is: {device}")

    current_time = time.strftime("%H_%M_%S-%d_%m_%Y", time.localtime(time.time()))
    writer = SummaryWriter(log_dir=f"{OUTPUT_DIR}/logs/{current_time}")
    model_dir = MODELS_DIR.joinpath(current_time)
    model_dir.mkdir(parents=True, exist_ok=True)
    params_dir = PARAMETERS_DIR.joinpath(current_time)
    params_dir.mkdir(parents=True, exist_ok=True)

    model = Digit_Net()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    Create_Parameters_Metadata()

    dataset = DigitDataset(data_path=TRAIN_PATH, transforms=TRAIN_TRANSFORMS)
    dataset_size = dataset.__len__()
    train_size = int(0.9 * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(13),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    main()
