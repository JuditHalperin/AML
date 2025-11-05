import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_losses(num_epochs: int, train_losses: list[float], val_losses: list[float] = None) -> None:
    """
    Plot train and test losses across epochs
    """
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    if val_losses:
        plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.show()


def plot_reconstructions(original_images: list[torch.Tensor], reconstructed_imagess: list[torch.Tensor], epochs: list[int]) -> None:
    """
    Plot reconstructed images in several epochs
    """
    num_epochs = len(reconstructed_imagess)
    num_images = len(original_images)
    
    _, axes = plt.subplots(num_epochs + 1, num_images, figsize=(num_images * 2, (num_epochs + 1) * 2))
    
    # Plot original images in the first row
    for j in range(num_images):
        axes[0, j].imshow(original_images[j].squeeze(), cmap='gray')
        axes[0, j].axis('off')
    axes[0, 1].set_title('Original images')
    
    # Plot reconstructed images for each epoch in subsequent rows
    for i in range(num_epochs):
        for j in range(num_images):
            axes[i + 1, j].imshow(reconstructed_imagess[i][j].squeeze(), cmap='gray')
            axes[i + 1, j].axis('off')
        axes[i + 1, 1].set_title(f'Reconstructed images at epoch {epochs[i]}')
        
    plt.tight_layout()
    plt.show()


def plot_generations(generated_images: list[torch.Tensor], epochs: list[int]) -> None:
    """
    Plot generated images in several epochs
    """
    num_epochs = len(epochs)
    num_images = generated_images[epochs[0]].shape[0]
    
    _, axes = plt.subplots(num_epochs, num_images, figsize=(num_images * 2, num_epochs * 2))
    
    # Plot generated images for each epoch
    for i, epoch in enumerate(epochs):
        for j in range(num_images):
            axes[i, j].imshow(generated_images[epoch][j].squeeze(), cmap='gray')
            axes[i, j].axis('off')
        axes[i, 1].set_title(f'Generated images at epoch {epoch}')
        
    plt.tight_layout()
    plt.show()


def plot_image_probabilities(images: list[torch.Tensor], probs: list[float]) -> None:
    """
    Plot images with their log-probabilities
    """
    num_images = len(images)
    rows = 2
    cols = (num_images + 1) // rows
    
    plt.figure(figsize=(cols * 3, rows * 3))
    
    for i in range(num_images):
        image = images[i].squeeze() if len(images[i].shape) == 3 else images[i]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'log-probability = {round(float(probs[i]), 2)}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def present_prob(probs: list[float]) -> float:
    return round(np.mean(probs), 2)
