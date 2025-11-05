import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


# Utility function to convert between standard deviation and log-variance
logvar2std = lambda logvar: torch.exp(0.5 * logvar)
std2logvar = lambda std: torch.log(std.pow(2))


def loss_function(x: torch.Tensor, recon_x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute the ELBO loss combining the reconstruction loss (MSE) and KL divergence
    Each term is summed over dimensions and averaged across batches
    """
    mse_loss = torch.mean(torch.sum(F.mse_loss(x, recon_x, reduction='none'), dim=(1, 2, 3)))
    kl_loss = torch.mean(- 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return mse_loss + kl_loss


class ConvVAE(nn.Module):

    def __init__(self, latent_dim: int = 200, latent_optimization: bool = False) -> None:
        """
        Initialize model layers. In amortized VAE, we define encoder, latent space and decoder, while in latent optimization we only use the decoder
        latent_dim: size of latent vectors
        latent_optimization: whether to use latent optimization or amortization for optimizing the q distributions
        """
        super(ConvVAE, self).__init__()

        self.latent_dim = latent_dim
        
        self.forward = self.amortized_forward if not latent_optimization else self.latent_optimization_forward
        self.reconstruct = self.amortized_reconstruct if not latent_optimization else self.latent_optimization_reconstruct

        if not latent_optimization:
            
            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # (batch_size, 32, 14, 14)
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (batch_size, 64, 7, 7)
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (batch_size, 128, 4, 4)
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=2)  # (batch_size, 512, 1, 1)
            )

            # Latent space
            self.fc_mu = nn.Linear(128, latent_dim)
            self.fc_logvar = nn.Linear(128, latent_dim)
            
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 128)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2),  # (batch_size, 128, 2, 2)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch_size, 128, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),  # (batch_size, 64, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch_size, 32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch_size, 1, 28, 28)
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.fc_decode(z)
        z = z.view(z.size(0), 128, 1, 1)
        z = self.decoder(z)
        return z

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Re-parameterization trick
        """
        # Transform log-variance into standard deviation
        sigma = logvar2std(logvar)
        # Sample random variable from standard normal distribution
        eps = torch.randn_like(sigma)
        # Treat mu and sigma as deterministic by defining the random variable z
        return mu + sigma * eps

    def amortized_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward used in amortized VAE
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def latent_optimization_forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Forward used in latent optimization
        """
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x

    def amortized_reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct an image using amortized VAE
        """
        mu, _ = self.encode(x)
        recon_x = self.decode(mu)
        return recon_x

    def latent_optimization_reconstruct(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct an image using latent optimization
        """
        recon_x = self.decode(mu)
        return recon_x

    def sample(self, n: int = 10) -> torch.Tensor:
        """
        Sample images using latent vectors sampled from a standard normal distribution
        n: number of images
        """
        latent_samples = torch.randn(n, self.latent_dim)
        new_x = self.decode(latent_samples)
        return new_x
    
    def estimate_likelihood(self, x: torch.Tensor, sigma_p: float = 0.4, m: int = 1000) -> torch.Tensor:
        """
        Estimate log-likelihood of an image using Monte Carlo
        x: image
        sigma_p: standard deviation of p(x|z)
        m: Monte Carlo repeats 
        """

        def log_pdf_gaussian(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
            log_normalization = -0.5 * (np.log(2 * np.pi) + torch.sum(torch.log(std ** 2)))
            log_likelihood = -0.5 * torch.sum(((x - mean) / std) ** 2)
            return log_normalization + log_likelihood

        log_likelihoods = []
        for _ in range(m):

            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            recon_x = self.decode(z)
            
            log_p_z = log_pdf_gaussian(z, torch.zeros_like(mu), torch.ones_like(logvar))
            log_p_x_given_z = log_pdf_gaussian(x, recon_x, torch.tensor(sigma_p))
            log_q_z = log_pdf_gaussian(z, mu, logvar2std(logvar))
            
            log_likelihoods.append(log_p_z + log_p_x_given_z - log_q_z)
        
        return torch.logsumexp(torch.stack(log_likelihoods), dim=0) - np.log(m)
