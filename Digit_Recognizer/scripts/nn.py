import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

# TODO:
# 1. Revers droput to inital state, only one


class Digit_Net(nn.Module):
    def __init__(self):
        super(Digit_Net, self).__init__()

        # self.conv1 = nn.Conv2d(1, 64, kernel_size = 5, padding = 'same', bias = False)

        resnet18 = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        # print(resnet18)
        self.resnet18_core = nn.Sequential(*(list(resnet18.children())[:-1]))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout1 = nn.Dropout(p=0.20)
        self.dropout2 = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        # self.fc1 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        self.softmax = nn.Softmax(1)

    def forward(self, x):
        # x = self.conv1(x)
        x = torch.cat([x, x, x], dim=1)
        x = self.resnet18_core(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        # x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        # x = self.dropout1(x)
        x = self.relu(self.fc3(x))
        # x = self.dropout2(x)
        x = self.softmax(x)

        return x


class Digit_Net2(nn.Module):
    def __init__(self) -> None:
        super(Digit_Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        resnet18 = models.resnet18(weights=None)
        self.resnet18_core = nn.Sequential(*(list(resnet18.children())[1:-4]))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.20)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.resnet18_core(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = Digit_Net()

    print(model)
    print("-" * 100)
    summary(model.cuda(), (1, 28, 28))
    print("=" * 100)

    model = Digit_Net2()
    print(model)
    summary(model.cuda(), (1, 28, 28))
