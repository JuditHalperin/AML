import torch
from tqdm import tqdm
from models import VICReg
from visualization import plot_losses
from utils import create_output_dirs, get_device


def train_representation(
        train_loader,
        val_loader = None,
        num_epochs: int = 50,
        plot_loss: bool = False,
        var_weight: int = 25,
        exp_title: str = 'VICReg'
    ):

    create_output_dirs(exp_title)

    device = get_device()
    model = VICReg(device=device).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4, betas = (0.9, 0.999), weight_decay = 1e-6)

    train_losses, train_invariances, train_variances, train_covariances = [], [], [], []
    val_losses, val_invariances, val_variances, val_covariances = [], [], [], []

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}:')

        # Training
        model.train()
        train_loss, train_invariance, train_variance, train_covariance = 0.0, 0.0, 0.0, 0.0
        for (x1, x2), _ in tqdm(train_loader):           
            optimizer.zero_grad()

            x1, x2 = x1.to(device), x2.to(device)
            z1, z2 = model(x1), model(x2)
            loss, invariance, variance, covariance = model.compute_loss(z1, z2, var_weight=var_weight)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_invariance += invariance.item()
            train_variance += variance.item()
            train_covariance += covariance.item()

        train_loss /= len(train_loader)
        train_invariance /= len(train_loader)
        train_variance /= len(train_loader)
        train_covariance /= len(train_loader)

        train_losses.append(train_loss)
        train_invariances.append(train_invariance)
        train_variances.append(train_variance)
        train_covariances.append(train_covariance)

        print(f'Train loss = {train_loss} | Invariance = {train_invariance} | Variance = {train_variance} | Covariance = {train_covariance}')

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss, val_invariance, val_variance, val_covariance = 0.0, 0.0, 0.0, 0.0
            with torch.no_grad():
                for (x1, x2), _ in tqdm(val_loader):
                    x1, x2 = x1.to(device), x2.to(device)
                    z1, z2 = model(x1), model(x2)
                    loss, invariance, variance, covariance = model.compute_loss(z1, z2, var_weight=var_weight)

                    val_loss += loss.item()
                    val_invariance += invariance.item()
                    val_variance += variance.item()
                    val_covariance += covariance.item()

            val_loss /= len(val_loader)
            val_invariance /= len(val_loader)
            val_variance /= len(val_loader)
            val_covariance /= len(val_loader)

            val_losses.append(val_loss)
            val_invariances.append(val_invariance)
            val_variances.append(val_variance)
            val_covariances.append(val_covariance)

            print(f'Val loss = {val_loss} | Invariance = {val_invariance} | Variance = {val_variance} | Covariance = {val_covariance}')

        # Saving weights
        torch.save(model, f'weights/{exp_title}/weights_epoch_{epoch}.pth')

    if plot_loss:
        plot_losses({'Train': train_losses, 'Validation': val_losses}, num_epochs, title='VICReg Loss', exp_name=exp_title)
        plot_losses({'Train': train_invariances, 'Validation': val_invariances}, num_epochs, title='Invariance Loss', exp_name=exp_title)
        plot_losses({'Train': train_variances, 'Validation': val_variances}, num_epochs, title='Variance Loss', exp_name=exp_title)
        plot_losses({'Train': train_covariances, 'Validation': val_covariances}, num_epochs, title='Covariance Loss', exp_name=exp_title)


def train_classification(
        train_loader,
        val_loader,
        trained_model: str,
        num_epochs: int = 10,
        plot_loss: bool = False,
        exp_title: str = 'classifier'
    ):
    """
    trained_model: path to encoder checkpoints
    """

    create_output_dirs(exp_title)

    device = get_device()
    model = VICReg(num_classes=10, device=device).to(device)

    model.encoder = torch.load(trained_model, map_location=device).encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}:')

        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader):           
            optimizer.zero_grad()

            images, labels = images.to(device), labels.to(device)
            pred_labels = model(images)
            loss = model.compute_loss(pred_labels, labels)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += labels.size(0)
            _, pred_labels = torch.max(pred_labels, dim=1)
            correct += (pred_labels == labels).sum().item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_accuracy = correct / total
        train_accuracies.append(train_accuracy)
        print(f'Train loss = {train_loss} | Train accuracy = {train_accuracy}')

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                
                images, labels = images.to(device), labels.to(device)
                pred_labels = model(images)
                loss = model.compute_loss(pred_labels, labels)

                val_loss += loss.item()
                total += labels.size(0)
                _, pred_labels = torch.max(pred_labels, dim=1)
                correct += (pred_labels == labels).sum().item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        print(f'Val loss = {val_loss} | Val accuracy {val_accuracy}')

        # Saving weights
        torch.save(model, f'weights/{exp_title}/weights_epoch_{epoch}.pth')

    if plot_loss:
        plot_losses({'Train': train_losses, 'Validation': val_losses}, num_epochs, title='Classification Loss', exp_name=exp_title)
        plot_losses({'Train': train_accuracies, 'Validation': val_accuracies}, num_epochs, title='Classification Accuracy', exp_name=exp_title)
