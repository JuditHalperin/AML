import torch
import torch.nn as nn
from torch import Tensor


class AffineCouplingLayer(nn.Module):

    def __init__(self, dim_size: int, num_layers: int = 5, hidden_size: int = 8) -> None:
        """
        dim_size: number of data dimensions
        num_layers: depth of fully-connected
        hidden_size: number of neurons in each layer
        """
        super(AffineCouplingLayer, self).__init__()
        self.half_dim = dim_size // 2
        b, log_s = [], []
        for i in range(num_layers):
            in_features = self.half_dim if i == 0 else hidden_size
            out_features = hidden_size if i != num_layers - 1 else self.half_dim
            b += [nn.Linear(in_features, out_features), nn.LeakyReLU()]
            log_s += [nn.Linear(in_features, out_features), nn.LeakyReLU()]
        self.b, self.log_s = nn.Sequential(*b), nn.Sequential(*log_s)

    def forward(self, z: Tensor) -> Tensor:
        """From noise to real points"""
        z_l, z_r = z[..., :self.half_dim], z[..., self.half_dim:]
        b, log_s = self.b(z_l), self.log_s(z_l)
        y_r = torch.exp(log_s) * z_r + b
        return torch.cat([z_l, y_r], dim=-1)

    def inverse(self, y: Tensor) -> Tensor:
        """From real points to noise"""
        y_l, y_r = y[..., :self.half_dim], y[..., self.half_dim:]
        b, log_s = self.b(y_l), self.log_s(y_l)  # as y_l == z_l
        z_r = (y_r - b) / torch.exp(log_s)
        return torch.cat([y_l, z_r], dim=-1)

    def log_inverse_jacobian_det(self, y: Tensor) -> Tensor:
        y_l = y[..., :self.half_dim]
        log_s = self.log_s(y_l)
        return -torch.sum(log_s, dim=-1)


class PermutationLayer(nn.Module):
    
    def __init__(self, dim_size: int) -> None:
        super(PermutationLayer, self).__init__()
        self.perm = torch.randperm(dim_size)

    def forward(self, z: Tensor) -> Tensor:
        return z[..., self.perm]

    def inverse(self, z: Tensor) -> Tensor:
        return z[..., torch.argsort(self.perm)]


class NormalizingFlow(nn.Module):
    """Normalizing flow model"""

    def __init__(self, dim_size: int, num_affine_coupling: int = 15, num_linear: int = 5, hidden_size: int = 8) -> None:
        """
        Create a flow of affine coupling layers with permutation between them 
        dim_size: number of data dimensions
        num_affine_coupling: number of affine coupling layers
        num_linear: depth of each affine coupling layer
        hidden_size: number of neurons in each layer
        """
        super(NormalizingFlow, self).__init__()
        self.dim_size = dim_size
        layers = []
        for i in range(num_affine_coupling):
            layers.append(AffineCouplingLayer(dim_size, num_linear, hidden_size))
            if i < num_affine_coupling - 1:
                layers.append(PermutationLayer(dim_size))
        self.flow = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        """From noise to real points"""
        return self.flow(z)

    def inverse(self, x: Tensor) -> Tensor:
        """From real points to noise"""
        z = x
        for layer in self.flow[::-1]:
            z = layer.inverse(z)
        return z

    def log_inverse_jacobian_det(self, x: Tensor) -> Tensor:
        """
        Compute the log-inverse determinant summing log-inverse determinant of each layer
        As the permutation layers have an inverse jacobian determinant of 1 we simply ignore them
        """
        log_det = torch.zeros(x.size(0))
        z = x
        for layer in self.flow[::-1]:
            if isinstance(layer, AffineCouplingLayer):
                log_det += layer.log_inverse_jacobian_det(z)
            z = layer.inverse(z)
        return log_det
    
    def log_prob_multi_gaussian(self, z: Tensor) -> Tensor:
        """
        Compute the Gaussian PDF of multivariate data in log-space, assuming p(z) is a standard normal distribution
        """
        mvn = torch.distributions.MultivariateNormal(
            loc=torch.zeros_like(z),
            covariance_matrix=torch.eye(z.size(-1))
        )
        return mvn.log_prob(z)

    def get_objective_terms(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        return: tuple of two loss terms in log-space
        """
        z = self.inverse(x)
        log_prob_z = self.log_prob_multi_gaussian(z)
        log_det = self.log_inverse_jacobian_det(x)
        return log_prob_z, log_det

    def compute_loss(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute loss averaged over batch
        return: tuple of loss composed of two terms, and each term separately
        """
        log_prob_z, log_det = self.get_objective_terms(x)
        return torch.mean(- log_prob_z - log_det), torch.mean(-log_prob_z), torch.mean(-log_det)

    def estimate(self, x: Tensor) -> Tensor:
        """Estimate log-likelihood of point"""
        log_prob_z, log_det = self.get_objective_terms(x)
        return log_prob_z + log_det
        
    def sample(self, n: int = 1000, seed: int = 1234) -> Tensor:
        """Get real points from sampled noise"""
        torch.manual_seed(seed)
        z = torch.randn(n, self.dim_size)
        return self(z)
