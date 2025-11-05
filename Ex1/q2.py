import torch
from model import ConvVAE
from visualization import plot_generations


out_dir = 'weights_amortization'
model = ConvVAE(latent_dim=200)

save_epochs = [1, 5, 10, 20, 30]

generated_images = {}
for epoch in save_epochs:
    model.load_state_dict(torch.load(f'{out_dir}/model_q1_epoch_{epoch}.pth'))
    model.eval()
    with torch.no_grad():
        generated_images[epoch] = model.sample()

plot_generations(generated_images, save_epochs)
