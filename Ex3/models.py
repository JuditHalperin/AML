import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class Encoder(nn.Module):

    def __init__(self, D: int = 128, device: str = 'cuda'):
        super(Encoder, self).__init__()
        self.resnet = resnet18(pretrained=False).to(device)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(512, 512)
        self.fc = nn.Sequential(nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, D))

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

    def encode(self, x):
        return self.forward(x)


class Projector(nn.Module):

    def __init__(self, D, proj_dim=512):
        super(Projector, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(D, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        return self.model(x)


class VICReg(nn.Module):

    def __init__(self, emb_dim: int = 128, proj_dim: int = 512, num_classes: int = None, device: str = 'cuda'):
        """
        num_classes: whether to add a classification layer using linear probing
        """
        super(VICReg, self).__init__()
        self.encoder = Encoder(emb_dim, device)

        # Projection
        if not num_classes:
            self.projector = Projector(emb_dim, proj_dim)
            self.forward = self.forward_projection
            self.compute_loss = self.compute_loss_projection

        # Classification
        else:
            self.classifier = nn.Linear(emb_dim , num_classes)
            self.forward = self.forward_classification
            self.compute_loss = self.compute_loss_classification

    def forward_projection(self, x):
        x = self.encoder(x)
        x = self.projector(x)
        return x
    
    def forward_classification(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
    
    def encode(self, x):
        return self.encoder(x)
    
    def classify(self, x):
        return self.forward_classification(x)
    
    def _invariance_objective(self, z, z_tag):
        return F.mse_loss(z, z_tag, reduction='mean')

    def _variance_objective(self, z, gamma: int = 1, eps: float = 1e-4):
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.mean(F.relu(gamma - std))

    def _covariance_objective(self, z):
        z = z - z.mean(dim=0)
        cov_z = ((z.T @ z) / (z.shape[0] - 1)).square()
        return (cov_z.sum() - cov_z.diagonal().sum()) / z.shape[1]

    def compute_loss_projection(self, z, z_tag, invar_weight: int = 25, var_weight: int = 25, covar_weight: int = 1):
        invariance = invar_weight * self._invariance_objective(z, z_tag)
        variance = var_weight * (self._variance_objective(z) + self._variance_objective(z_tag))
        covariance = covar_weight * (self._covariance_objective(z) + self._covariance_objective(z_tag))
        loss = invariance + variance + covariance
        return loss, invariance, variance, covariance
    
    def compute_loss_classification(self, pred, true):
        return F.cross_entropy(pred, true)
