import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pandas as pd
from natsort import natsorted
from pathlib import Path

from nn import Digit_Net
from dataset_loader import DigitDataset
from globals import TEST_PATH, MODELS_DIR, PREDICTIONS_PATH, TRAIN_TRANSFORMS


def get_best_weights(path: Path):
    model_list = natsorted(path.glob("*.pth"))
    print(f"[INFO] Best weights path: {model_list[-1]}")
    return model_list[-1]


def run_inference(test_loader):
    for inputs in test_loader:
        # print(inputs.shape)
        with torch.no_grad():
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

    return predictions


def save_results(predictions, test_df):
    test_output = pd.DataFrame(
        {"ImageId": test_df.index, "Label": predictions.cpu().astype(int)}
    )
    test_output.to_csv(
        PREDICTIONS_PATH
        / f"model-date-{model_date}-epoch-{best_weights_path.name.split('_')[1]}.csv",
        index=False,
    )


if __name__ == "__main__":
    model_date = "21_04_29-18_05_2023"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running device is: {device}")

    best_weights_path = get_best_weights(MODELS_DIR.joinpath(model_date))
    model = Digit_Net()
    model.to(device)
    model.load_state_dict(torch.load(best_weights_path))

    test_set = DigitDataset(data_path=TEST_PATH, transforms=TRAIN_TRANSFORMS, test=True)

    test_loader = DataLoader(test_set, batch_size=test_set.__len__(), shuffle=False)
    # print(test_set.__len__())
    # print(test_loader.batch_size)
    test_loader.batch_sampler

    predictions = run_inference(test_loader)
    print(predictions.shape)
    print(predictions.cpu().numpy())
