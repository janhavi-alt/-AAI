import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# Define the generator class
class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Define the discriminator class
class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Hyperparameters
latent_size = 100
image_size = 28 * 28  # MNIST images are 28x28
batch_size = 64
epochs = 50
lr = 0.0002

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models
generator = Generator(input_size=latent_size, output_size=image_size)
discriminator = Discriminator(input_size=image_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=lr)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    for batch_idx, (real_images, _) in enumerate(data_loader):
        # Prepare real and fake data
        real_images = real_images.view(-1, image_size).to(device)
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train the discriminator
        outputs = discriminator(real_images)
        d_loss_real = criterion(outputs, real_labels)
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        # Train the generator
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

    print(f"Epoch [{epoch + 1}/{epochs}]  D Loss: {d_loss.item():.4f}  G Loss: {g_loss.item():.4f}")

    # Save and display generated images every 10 epochs
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            z = torch.randn(64, latent_size).to(device)
            fake_images = generator(z).view(-1, 1, 28, 28).cpu()

            # Debug: Ensure the generated images are not all zeros
            print(f"Epoch {epoch + 1}: Generated Images Mean Pixel Value = {fake_images.mean().item():.4f}")

            grid = vutils.make_grid(fake_images, nrow=8, normalize=True)
            plt.figure(figsize=(8, 8))
            plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
            plt.title(f"Generated Images at Epoch {epoch + 1}")
            plt.axis("off")
            filename = f"generated_images_epoch_{epoch + 1}.png"
            plt.savefig(filename)
            plt.close()

            print(f"Images saved to {filename}")
