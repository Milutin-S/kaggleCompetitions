import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import time
import pandas as pd
from natsort import natsorted
from pathlib import Path

from nn import Digit_Net, Digit_Net2
from dataset_loader import DigitDataset
from globals import TEST_PATH, MODELS_DIR, PREDICTIONS_PATH

# TODO: add time count for inference ✅


def get_best_weights(path: Path):
    model_list = natsorted(path.glob("*.pth"))
    print(f"[INFO] Best weights path: {model_list[-1]}")
    return model_list[-1]


def run_inference(test_loader):
    predictions = torch.empty(0).to(device)
    for inputs in test_loader:
        # print(inputs.shape)
        with torch.no_grad():
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, temp_pred = torch.max(outputs, 1)
            predictions = torch.cat((predictions, temp_pred), dim=0)

    return predictions


def save_results(predictions, test_df):
    test_output = pd.DataFrame(
        {"ImageId": test_df.index + 1, "Label": predictions.cpu().numpy().astype(int)}
    )
    output_path = (
        PREDICTIONS_PATH
        / f"model-date-{model_date}-epoch-{best_weights_path.name.split('_')[1]}.csv"
    )
    test_output.to_csv(
        output_path,
        index=False,
    )

    print(f"[INFO] Successfully saved the results to: {output_path}")


if __name__ == "__main__":
    model_date = "12_43_16-26_05_2023"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running device is: {device}")

    best_weights_path = get_best_weights(MODELS_DIR.joinpath(model_date))
    # model = Digit_Net()
    model = Digit_Net2()
    model.to(device)
    model.load_state_dict(torch.load(best_weights_path))

    test_set = DigitDataset(data_path=TEST_PATH, type="test")

    test_loader = DataLoader(
        test_set, batch_size=int(test_set.__len__() / 4), shuffle=False
    )
    # print(test_set.__len__())
    # print(test_loader.batch_size)
    # test_loader.batch_sampler

    start_time = time.time()
    predictions = run_inference(test_loader)
    duration = time.time() - start_time
    print(f"[INFO] Inference time: {duration:.2f} s.")
    print(predictions.shape)
    print(type(predictions))
    print(predictions.cpu().numpy())

    save_results(predictions=predictions, test_df=test_set.dataset)
