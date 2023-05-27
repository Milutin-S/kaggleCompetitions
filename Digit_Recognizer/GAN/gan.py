import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import datetime
from tqdm import tqdm
from pathlib import Path

from Digit_Recognizer.scripts.dataset_loader import DigitDataset
from Digit_Recognizer.scripts.globals import *

OUTPUT_PATH = Path("./Digit_Recognizer/GAN/Tran_Output")


class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            # Fake = 0, Real = 1
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    # z_dim = latent noise or noise
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),  # 1x28x28 = 784
            # Tanh = pixel values to be [-1, 1], because we normalized the input images also
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64  # 128, 256 ...
image_dim = 28 * 28 * 1  # 784
batch_size = 32
num_epochs = 50

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    # Actual mean and standard deviation of MNIST dataset
)

dataset = DigitDataset(data_path=TRAIN_PATH, type="train")
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

optimizer_gen = optim.Adam(gen.parameters(), lr=lr)
optimizer_disc = optim.Adam(disc.parameters(), lr=lr)
criterion = nn.BCELoss()

current_time = datetime.datetime.now().isoformat()[:-7]
OUTPUT_DIR.joinpath(current_time)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
writer_fake = SummaryWriter(log_dir=OUTPUT_DIR, filename_suffix="fake")
writer_real = SummaryWriter(log_dir=OUTPUT_DIR, filename_suffix="real")
step = 0

for epoch in tqdm(num_epochs):
    for batch_idx, (real_imgs, _) in enumerate(train_loader):
        real_imgs = real_imgs.view(-1, 784).to(device)
        batch_size = real_imgs.shape[0]

        ### Train Discriminator: max log(D(real)) + log(1-D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)  # G(z)
        disc_real = disc(real_imgs).view(-1)
        lossD_real = criterion(
            disc_real, torch.ones_like(disc_real)
        )  # min -log(D(real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(
            disc_fake, torch.zeros_like(disc_fake)
        )  # min -log(1-D(G(z)))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(
            retain_graph=True
        )  # Clears chash, except if you use .detach() or retain_graph=True
        optimizer_disc.step()

        ### Train Generator: min log(1 - D(G(z))) (Wanishing gradient) <--> max log(D(G(z)))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        optimizer_gen.step()

        if batch_idx == 0:
            tqdm.write(f"[INFO] Epoch {epoch}/{num_epochs}")
            tqdm.write(f"   Loss D: {lossD:.4f}, Loss G: {lossG:.4f}")
            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real_imgs.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )

                step += 1
