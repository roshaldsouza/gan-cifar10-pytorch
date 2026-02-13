import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
latent_dim = 100
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
num_epochs = 10
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU(),

            nn.Unflatten(1, (128, 8, 8)),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),

            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):

        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # ---- Train Discriminator ----
        optimizer_D.zero_grad()

        z = torch.randn(batch_size, latent_dim, device=device)
        fake_images = generator(z)

        real_loss = criterion(discriminator(real_images), valid)
        fake_loss = criterion(discriminator(fake_images.detach()), fake)

        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # ---- Train Generator ----
        optimizer_G.zero_grad()

        g_loss = criterion(discriminator(fake_images), valid)
        g_loss.backward()
        optimizer_G.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Batch [{i}/{len(dataloader)}] "
                  f"D Loss: {d_loss.item():.4f} "
                  f"G Loss: {g_loss.item():.4f}")

    # ---- Show generated images ----
    with torch.no_grad():
        z = torch.randn(16, latent_dim, device=device)
        generated = generator(z).cpu()
        grid = torchvision.utils.make_grid(generated, nrow=4, normalize=True)
        plt.figure(figsize=(4,4))
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.axis("off")
        plt.savefig(f"generated_epoch_{epoch+1}.png")
        plt.close()



