import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

class TitanicNet(nn.Module):
    def __init__(self, input_size):
        super(TitanicNet, self).__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x
    
if __name__ == '__main__':
    model = TitanicNet(input_size=8)

    print(model)
    print('-'*50)
    summary(model.cuda(), (8,))
