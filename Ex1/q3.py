import os
import torch
import torch.optim as optim
from tqdm import tqdm
from model import ConvVAE, loss_function, std2logvar
from data import load_MNIST_dataset, get_random_images
from visualization import plot_losses, plot_reconstructions, plot_generations


# Create output directory
out_dir = 'weights_latent_optimization'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# Load train dataset without shuffling as images should be in the same order in each iteration so they match to their corresponding mu and sigma vectors
batch_size = 64
train_dataset, train_loader, _, _ = load_MNIST_dataset(batch_size, shuffle_train=False)
num_images = len(train_dataset)

# Initialize the model using latent optimization and loss function
latent_dim = 200
model = ConvVAE(latent_dim=latent_dim, latent_optimization=True)
criterion = loss_function

# Initialize latent vectors from standard normal distribution
mu = torch.randn(num_images, latent_dim, requires_grad=True)
sigma = torch.randn(num_images, latent_dim, requires_grad=True)

# Initialize optimizer including the latent vectors
optimizer = optim.Adam([
    {'params': [mu, sigma], 'lr': 0.01},
    {'params': model.parameters()}
], lr=0.001)

# Training loop
train_losses = []
num_epochs = 30
save_epochs = [1, 5, 10, 20, 30]

indices, images = get_random_images(train_dataset)
recon_images = []

for epoch in range(1, num_epochs + 1):
    print(f'Epoch {epoch}:')

    # Training
    model.train()
    train_loss = 0.0
    for i, (x, _) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        # Select mu and sigma vectors corresponding to current batch images
        mu_batch = mu[i * batch_size: (i + 1) * batch_size]
        logvar_batch = std2logvar(sigma[i * batch_size: (i + 1) * batch_size])
        recon_x = model(mu_batch, logvar_batch)
        loss = criterion(x, recon_x, mu_batch, logvar_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    print(f'Train loss = {train_loss}')

    if epoch in save_epochs:
        torch.save(model.state_dict(), f'{out_dir}/model_q3_epoch_{epoch}.pth')

        # Reconstruction
        model.eval()
        reconstructed_images_epoch = []
        with torch.no_grad():
            for index in indices:
                # Select mu vector corresponding to current image
                mu_image = mu[index].unsqueeze(0)  # add batch dimension
                recon_x = model.reconstruct(mu_image)
                reconstructed_images_epoch.append(recon_x)    
        recon_images.append(reconstructed_images_epoch)
    
# # Plot training loss
plot_losses(num_epochs, train_losses)

# # Plot reconstructions in training set
plot_reconstructions(images, recon_images, save_epochs)

# Plot generations
generated_images = {}
for epoch in save_epochs:
    model.load_state_dict(torch.load(f'{out_dir}/model_q3_epoch_{epoch}.pth'))
    model.eval()
    with torch.no_grad():
        generated_images[epoch] = model.sample()
plot_generations(generated_images, save_epochs)
