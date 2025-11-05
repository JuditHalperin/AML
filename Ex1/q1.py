import os
import torch
import torch.optim as optim
from tqdm import tqdm
from model import ConvVAE, loss_function
from data import load_MNIST_dataset, get_random_images
from visualization import plot_losses, plot_reconstructions


# Create output directory
out_dir = 'weights_amortization'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# Load datasets
train_dataset, train_loader, test_dataset, test_loader = load_MNIST_dataset()

# Initialize the model, loss function and optimizer
model = ConvVAE(latent_dim=200)
criterion = loss_function
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
train_losses, val_losses = [], []
num_epochs = 30
save_epochs = [1, 5, 10, 20, 30]

for epoch in range(1, num_epochs + 1):
    print(f'Epoch {epoch}:')

    # Training
    model.train()
    train_loss = 0.0
    for x, _ in tqdm(train_loader):
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)
        loss = criterion(x, recon_x, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, _ in tqdm(test_loader):
            recon_x, mu, logvar = model(x)
            loss = criterion(x, recon_x, mu, logvar)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(test_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f'Train loss = {train_loss}')
    print(f'Val loss = {val_loss}')

    if epoch in save_epochs:
        torch.save(model.state_dict(), f'{out_dir}/model_q1_epoch_{epoch}.pth')

# Plot losses
plot_losses(num_epochs, train_losses, val_losses)

# Plot reconstructions in validation set
images = get_random_images(test_dataset)[1]
recon_images = []

for epoch in save_epochs:
    model.load_state_dict(torch.load(f'{out_dir}/model_q1_epoch_{epoch}.pth'))
    model.eval()
    reconstructed_images_epoch = []
    with torch.no_grad():
        for x in images:
            x = x.unsqueeze(0)  # add batch dimension
            recon_x = model.reconstruct(x)
            reconstructed_images_epoch.append(recon_x)
    recon_images.append(reconstructed_images_epoch)

plot_reconstructions(images, recon_images, save_epochs)

# Plot reconstructions in training set
images = get_random_images(train_dataset)[1]
recon_images = []

for epoch in save_epochs:
    model.load_state_dict(torch.load(f'{out_dir}/model_q1_epoch_{epoch}.pth'))
    model.eval()
    reconstructed_images_epoch = []
    with torch.no_grad():
        for x in images:
            x = x.unsqueeze(0)  # add batch dimension
            recon_x = model.reconstruct(x)
            reconstructed_images_epoch.append(recon_x)
    recon_images.append(reconstructed_images_epoch)

plot_reconstructions(images, recon_images, save_epochs)

