import torch
import torch.nn as nn
import torch.optim as optim
# import torchmetrics
from torchvision import transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

import os
import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from nn import TitanicNet
# from dataset import datasets, imshow

# weights = 'Titanic/models/10_21_46-22_04_2023/epoch_6_weights.pth'
weights = 'Titanic/models/20_40_17-05_05_2023/epoch_284_weights.pth'

csv_file = 'Titanic/titanic_dataset/test_prepared.csv'
features = ['HasCabin', 'Pclass', 'SexCode', 'EmbarkedCode', 
                    'TitleCode', 'Fare', 'Age', 'FamilySize']
out_path = Path('Titanic/titanic_dataset/')

def loadTest():
    data_df = pd.read_csv(csv_file)
    print(f"[INFO] Initial data shape: {data_df.shape}")
    # row = data_df.iloc[index]
    inputs = torch.Tensor(data_df[features].values)
    # label = torch.Tensor([row['Survived']])
    return data_df, inputs



def Test(inputs):

    with torch.no_grad():
        inputs = inputs.to(device)
        outputs = model(inputs)
        threshold_out = torch.where(outputs >= threshold, torch.ones_like(outputs), torch.zeros_like(outputs))
        print(f"[INFO] Predicted shape: {threshold_out.shape}")
    
    return threshold_out, outputs

def saveResults(data_df, threshold_out, raw_output):
        # output_raw = pd.DataFrame({'PassengerId' : data_df.index + 892,'Survived' : raw_output.cpu().numpy().squeeze().astype(int)})
        # output_raw.to_csv(out_path / 'best_submission_torch_raw#3.csv', index=False)

        output = pd.DataFrame({'PassengerId' : data_df.index + 892,'Survived' : threshold_out.cpu().numpy().squeeze().astype(int)})
        output.to_csv(out_path / 'submission_treshold_05.csv', index=False)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Current running device: {device}")

    input_size = 8
    model = TitanicNet(input_size=input_size).to(device)
    model.load_state_dict(torch.load(weights))
    # criterion = nn.BCELoss()
    # learning_rate = 0.0003 # 0.001
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)
    threshold = 0.5

    data_df, inputs = loadTest()
    threshold_out, raw_output = Test(inputs)
    saveResults(data_df, threshold_out, raw_output)

