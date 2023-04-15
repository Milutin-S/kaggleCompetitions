import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class TitanicDataset(Dataset):
    def __init__(self, csv_file):
        self.data_df = pd.read_csv(csv_file)
        self.features = ['HasCabin', 'Pclass', 'SexCode', 'EmbarkedCode', 
                    'TitleCode', 'Fare', 'Age', 'FamilySize']
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, index):
        row = self.data_df.iloc[index]
        features = torch.Tensor(row[self.features])
        label = torch.Tensor([row['Survived']])
        return features, label

if __name__ == '__main__':
    data_path = 'Titanic/titanic_dataset/train_prepared.csv'

    dataset = TitanicDataset(csv_file=data_path)

    batch_size = 32
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    data_iter = iter(train_dataloader)
    batch = next(data_iter)
    features, labels = batch

    print(f"[INFO] Features shape: {features.shape}")
    print(f"[INFO] Labels shape: {labels.shape}")
    print(f"[INFO] Features: {features}")
    print(f"[INFO] Labels: {labels}")

