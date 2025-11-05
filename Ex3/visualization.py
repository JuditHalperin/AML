import warnings
import matplotlib.pyplot as plt
from torch import Tensor
import sklearn.metrics as metrics
from utils import save_plot


def plot_losses(losses: dict[str, list[float]], num_epochs: int, title: str = 'Loss', exp_name: str = None) -> None:
    """
    Plot any type of losses across epochs
    """
    for loss_name, loss_vals in losses.items():
        if loss_vals:
            plt.plot(range(1, num_epochs + 1), loss_vals, label=loss_name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    save_plot(title, exp_name)


def plot_representations(embedding_list: list, label_list: list, name_list: list[str], mark_points = None, title: str = '2D Representation', exp_name: str = None):
    num_embeddings = len(embedding_list)
    plt.figure(figsize=(6 * num_embeddings, 6))

    for i, (embeddings, labels, name) in enumerate(zip(embedding_list, label_list, name_list)):
        plt.subplot(1, num_embeddings, i + 1)
        plt.scatter(embeddings[:, 0], embeddings[:, 1], s=10, c=labels, cmap='tab10', alpha=0.6)
        plt.title(f'{exp_name} - {name}')
        plt.xlabel(f'Component 1')
        plt.ylabel(f'Component 2')
        plt.colorbar()
        
        if mark_points is not None and mark_points[i] is not None:
            for mark_point in mark_points[i]:
                plt.scatter(mark_point[0], mark_point[1], c='black', s=100, marker='o')

    save_plot(title, exp_name)


def plot_images(images: dict[int, list[Tensor]], title: str = 'Neighboring Images', exp_name: str = None):
    num_classes = len(images)
    num_images = len(next(iter(images.values())))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Clipping input data")

        plt.figure()
        for i, image_list in enumerate(images.values()):  # columns
            for j, image in enumerate(image_list):  # rows
                plt.subplot(num_images, num_classes, j * num_classes + i + 1)
                plt.imshow(image.numpy().transpose(1, 2, 0))
                plt.axis('off')
        plt.suptitle(title)

    save_plot(title, exp_name)


def plot_roc_curve(y_true, y_score, title: str = 'ROC Curve', exp_name: str = None):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve AUC = {auc}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')

    save_plot(title, exp_name)


def _plot_views(x1, x2, num_views: int = 10):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Clipping input data")

        for i in range(num_views):
            plt.figure(figsize=(10, 5))

            # First image
            plt.subplot(1, 2, 1)
            plt.imshow(x1[i].numpy().transpose(1, 2, 0))
            plt.axis('off')

            # Second image
            plt.subplot(1, 2, 2)
            plt.imshow(x2[i].numpy().transpose(1, 2, 0))
            plt.axis('off')

            save_plot()
