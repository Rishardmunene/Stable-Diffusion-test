import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.data_loader import DataLoader as CustomDataLoader
from src.model import Generator, Discriminator
from src.utils import log_metrics, save_loss_plot, generate_noise
from torchvision import transforms

# Hyperparameters
num_epochs = 35
batch_size = 16
learning_rate = 0.0002
n_samples = 100  # Number of images to visualize
log_interval = 100  # Log every N steps
save_interval = 5  # Save plots every N epochs

# Initialize DataLoader
data_loader = CustomDataLoader(root_dir='path/to/images', transform=transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

train_loader = DataLoader(data_loader, batch_size=batch_size, shuffle=True)

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Loss function
criterion = torch.nn.BCELoss()

# Lists to store losses
generator_losses = []
discriminator_losses = []

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        current_batch_size = images.size(0)
        images = images.to(device)

        # Generate fake images
        noise = generate_noise(current_batch_size, device)
        fake_images = generator(noise)

        # Create labels
        real_labels = torch.ones(current_batch_size, 1, device=device)
        fake_labels = torch.zeros(current_batch_size, 1, device=device)

        # Train Discriminator
        optimizer_D.zero_grad()
        real_outputs = discriminator(images)
        fake_outputs = discriminator(fake_images.detach())
        real_loss = criterion(real_outputs, real_labels)
        fake_loss = criterion(fake_outputs, fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        fake_outputs = discriminator(fake_images)
        g_loss = criterion(fake_outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

        # Log metrics
        if (i + 1) % log_interval == 0:
            log_metrics(epoch, i, d_loss.item(), g_loss.item())

        # Store losses
        generator_losses.append(g_loss.item())
        discriminator_losses.append(d_loss.item())

    # Visualize generated images every N epochs
    if (epoch + 1) % save_interval == 0:
        with torch.no_grad():
            generated_images = generator(generate_noise(n_samples, device))
            plt.figure(figsize=(10, 10))
            plt.imshow(generated_images[0].permute(1, 2, 0).cpu().numpy())
            plt.axis('off')
            plt.title(f'Epoch {epoch + 1}')
            plt.savefig(f'logs/loss_plots/generated_epoch_{epoch + 1}.png')
            plt.close()

# Save loss plots
save_loss_plot(generator_losses, 'logs/loss_plots/generator_loss.png')
save_loss_plot(discriminator_losses, 'logs/loss_plots/discriminator_loss.png')