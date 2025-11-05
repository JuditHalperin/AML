import torch
from tqdm import tqdm
from normalizing_flow import NormalizingFlow
from flow_matching import FlowMatching
from visualization import plot_losses, plot_data
from data import load_datasets
from utils import create_output_dirs


def train_normalizing_flow(
        num_epochs: int = 20,
        learning_rate: float = 1e-3,
        plot_loss: bool = False,
        plot_samples: bool = False,
        plot_input: bool = False,
        exp_title: str = 'normalizing_flow'
    ):
    """
    Train normalizing flow
    """

    create_output_dirs(exp_title)

    train_loader, val_loader, dim_size, _ = load_datasets(conditional=False, verbose=plot_input)
    model = NormalizingFlow(dim_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_losses, train_log_priors, train_log_dets = [], [], []
    val_losses, val_log_priors, val_log_dets = [], [], []

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}:')

        # Training
        model.train()
        train_loss, train_log_prior, train_log_det = 0.0, 0.0, 0.0
        for x in tqdm(train_loader):
            optimizer.zero_grad()
            loss, log_prior, log_det = model.compute_loss(x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_log_prior += log_prior.item()
            train_log_det += log_det.item()
        scheduler.step()

        train_loss /= len(train_loader)
        train_log_prior /= len(train_loader)
        train_log_det /= len(train_loader)

        train_losses.append(train_loss)
        train_log_priors.append(train_log_prior)
        train_log_dets.append(train_log_det)

        print(f'Train loss = {train_loss} | Train log-prior = {train_log_prior} | Train log-det = {train_log_det}')

        # Validation
        model.eval()
        val_loss, val_log_prior, val_log_det = 0.0, 0.0, 0.0
        with torch.no_grad():
            for x in tqdm(val_loader):
                loss, log_prior, log_det = model.compute_loss(x)
                val_loss += loss.item()
                val_log_prior += log_prior.item()
                val_log_det += log_det.item()

        val_loss /= len(val_loader)
        val_log_prior /= len(val_loader)
        val_log_det /= len(val_loader)

        val_losses.append(val_loss)
        val_log_priors.append(val_log_prior)
        val_log_dets.append(val_log_det)

        print(f'Val loss = {val_loss} | Val log-prior = {val_log_prior} | Val log-det = {val_log_det}')

        # Sampling
        if plot_samples:
            with torch.no_grad():
                x = model.sample()
                plot_data(x, title=f'Samples - Epoch {epoch}', exp_name=exp_title)

        # Saving weights
        torch.save(model, f'weights/{exp_title}/weights_epoch_{epoch}.pth')

    if plot_loss:
        plot_losses({'Train loss': train_losses, 'log-prior': train_log_priors, 'log-det': train_log_dets}, num_epochs, title='Train Loss', exp_name=exp_title)
        plot_losses({'Validation loss': val_losses, 'log-prior': val_log_priors, 'log-det': val_log_dets}, num_epochs, title='Val Loss', exp_name=exp_title)


def train_flow_matching(
        conditional: bool,
        num_epochs: int = 20,
        learning_rate: float = 1e-3,
        plot_loss: bool = False,
        plot_samples: bool = False,
        plot_input: bool = False,
        exp_title: str = 'flow_matching'
    ):
    """
    Train conditional / unconditional flow matching
    conditional: whether to train a conditional model
    """

    create_output_dirs(exp_title)

    train_loader, val_loader, dim_size, num_classes = load_datasets(conditional, verbose=plot_input)
    model = FlowMatching(dim_size, num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}:')

        # Training
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader):
            y1, c = tuple(batch) if isinstance(batch, list) else (batch, None)
            optimizer.zero_grad()
            loss = model.compute_loss(y1, c)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        print(f'Train loss = {train_loss}')

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x in tqdm(val_loader):
                # loss = model.compute_loss(x)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f'Val loss = {val_loss}')

        # Sampling (unconditional)
        if plot_samples and not conditional:
            with torch.no_grad():
                x = model.sample()[-1]
                plot_data(x, title=f'Samples - Epoch {epoch}', exp_name=exp_title)

        # Saving weights
        torch.save(model, f'weights/{exp_title}/weights_epoch_{epoch}.pth')

    if plot_loss:
        plot_losses({'Train loss': train_losses}, num_epochs, title='Train Loss', exp_name=exp_title)
        plot_losses({'Validation loss': val_losses}, num_epochs, title='Validation Loss', exp_name=exp_title)

