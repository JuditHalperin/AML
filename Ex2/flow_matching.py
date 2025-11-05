import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


class FlowMatching(nn.Module):
    """Conditional or unconditional flow matching model"""

    def __init__(
            self,
            dim_size: int,
            num_classes: int = None,
            num_layers: int = 5,
            initial_neurons: int = 64,
            additional_neurons: int = 16,
            embed_dim: int = 3
        ) -> None:
        """
        dim_size: number of data dimensions
        num_classes: number of classes in conditional model. For unconditional model specify num_classes = None
        num_layers: depth of fully-connected
        initial_neurons: number of neurons in the first layer
        additional_neurons: number of neurons to add to each following layer
        embed_dim: number of embedding dimensions
        """
        super(FlowMatching, self).__init__()
        self.dim_size = dim_size
        input_size = dim_size + 1  # time concatenation

        if num_classes:
            self.embedding = nn.Embedding(num_classes, embed_dim)
            input_size += embed_dim  # class concatenation

        num_neurons = initial_neurons
        layers = []
        for i in range(num_layers):
            in_features = input_size if i == 0 else num_neurons
            num_neurons += additional_neurons
            out_features = num_neurons if i != num_layers - 1 else dim_size
            layers += [nn.Linear(in_features, out_features), nn.LeakyReLU()]
        self.net = nn.Sequential(*layers)

    def forward(self, y: Tensor, t: Tensor, c: Tensor) -> Tensor:
        """
        y: input data in shape (batch_size, dim_size)
        t: input time n shape (batch_size, 1)
        c: input class in shape (batch_size, 1) used in conditional flow matching (otherwise None)
        return: output data in shape (batch_size, dim_size)
        """
        if c is not None:  # conditional
            emb = self.embedding(c)  # (batch_size, embed_dim)
            y = torch.cat([y, emb], dim=-1)  # (batch_size, dim_size + embed_dim)
        y = torch.cat([y, t], dim=-1)  # (batch_size, dim_size + (embed_dim)? + 1)
        return self.net(y)  # (batch_size, dim_size)
    
    def compute_loss(self, y1: Tensor, c: Tensor) -> Tensor:
        """
        Compute loss using linear flows
        y: data points
        c: labels used in conditional flow matching
        return: MSE loss averaged over batch
        """
        y0 = torch.randn_like(y1)
        t = torch.rand((y1.shape[0], 1))
        y = t * y1 + (1 - t) * y0
        v_t = self(y, t, c)
        return F.mse_loss(v_t, y1 - y0, reduction='mean')
    
    def sample(self, num_samples: int = 1000, delta_t: float = 1e-3, y_rate: float = 1.0, y_bias: float = 0, condition: list[int] = None, prior_points: Tensor = None) -> list[Tensor]:
        """
        Sample real points from noise
        condition: label list for each sampled point used in conditional flow matching
        y_rate: new parameter used to control the rate of updating y (by default no rate)
        y_bias: new parameter used to control the rate of updating y (by default no bias)
        prior_points: prior points instead of random noise sampled from a standard Gaussian
        return: list of all time steps of all points
        """
        y = torch.randn(num_samples, self.dim_size) if prior_points is None else prior_points
        assert not condition or len(condition) == num_samples
        c = torch.tensor(condition) if condition is not None else None
        trajectory = []
        for t in np.arange(0, 1 + delta_t, delta_t, dtype=np.float32):
            t = torch.tensor(t).repeat(y.shape[0], 1)
            v_t = self(y, t, c)
            y = y + v_t * delta_t * y_rate + y_bias
            trajectory.append(y)
        return trajectory

    def reverse_sample(self, y: Tensor, delta_t: float = 1e-3, condition: list[int] = None) -> list[Tensor]:
        """
        Sample noise from real points
        y: points
        condition: label list for each sampled point used in conditional flow matching
        return: list of all reversed time steps of all points
        """
        assert not condition or len(condition) == y.shape[0]
        c = torch.tensor(condition) if condition is not None else None
        trajectory = []
        for t in np.arange(1, -delta_t, -delta_t, dtype=np.float32):
            t = torch.tensor(t).repeat(y.shape[0], 1)
            v_t = self(y, t, c)
            y = y - v_t * delta_t
            trajectory.append(y)
        return trajectory
