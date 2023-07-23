import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier


class Autoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # N, 1, 28, 28
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1
            ),  # N, 16, 14, 14
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1
            ),  # N, 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7),  # N, 64, 1, 1
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=7
            ),  # N, 32, 7, 7
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),  # N, 16, 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),  # N, 1, 28, 28
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # N, 1, 28, 28
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1
            ),  # N, 16, 14, 14
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1
            ),  # N, 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7),  # N, 64, 1, 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class KNN(nn.Module):
    def __init__(self, input_dim, num_classes, k: int = 3) -> None:
        super().__init__()
        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.input_dim = input_dim
        self.num_classes = num_classes

    def forward(self, x):
        # This method is not needed for KNN, as it's a non-parametric model
        pass

    def fit(self, x_train, y_train):
        self.knn.fit(x_train, y_train)

    def predict(self, x_test):
        return self.knn.predict(x_test)


class EncoderWithKNN(nn.Module):
    def __init__(self, encoder, num_classes, k=3):
        super(EncoderWithKNN, self).__init__()
        self.encoder = encoder
        self.knn_head = KNN(input_dim=64, num_classes=num_classes, k=k)

    def forward(self, x):
        x_encoded = self.encoder(x)
        return x_encoded

    def fit_knn(self, x_train, y_train):
        self.knn_head.fit(x_train, y_train)

    def predict_knn(self, x_test):
        return self.knn_head.predict(x_test)
