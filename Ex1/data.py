import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


NUM_DIGITS = 10


def load_MNIST_dataset(batch_size: int = 64, shuffle_train: bool = True):
    """
    Load the MNIST data and split into train and test sets
    return: dataset and data loader of both train and test sets
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_targets = train_dataset.targets
    train_idx, _ = train_test_split(range(len(train_targets)), train_size=20000, stratify=train_targets)
    train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)

    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, train_loader, test_dataset, test_loader


def get_random_indices(dataset, num_images: int = 1, by_label: bool = False) -> dict[int, list[int]] | list[int]:
    """
    Randomly select image indices from the dataset
    dataset: MNIST dataset - either train or test
    num_images: number of random images per digit
    by_label: whether to return a dict when each key is a digit and each value is a list of corresponding indices, or just a list of all selected indices
    return: dict of lists with random indices, or list of random indices
    """
    # Shuffle the dataset indices to select random images
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)

    # Select indices until each digit has num_images indices
    digit_indices = {i: [] for i in range(NUM_DIGITS)}
    found_digits = 0
    for idx in all_indices:
        if found_digits == NUM_DIGITS:
            break
        _, label = dataset[idx]
        if len(digit_indices[label]) < num_images:
            digit_indices[label].append(idx)
            if len(digit_indices[label]) == num_images:
                found_digits += 1

    # Sort by digits
    digit_indices = dict(sorted(digit_indices.items(), key=lambda x: int(x[0])))

    # Convert to list of all selected indices
    if not by_label:
        digit_indices = [index for sublist in digit_indices.values() for index in sublist]

    return digit_indices


def get_random_images(dataset, num_images: int = 1, by_label: bool = False) -> tuple[list[int] | dict[int, list[int]], list[torch.Tensor] | dict[int, list[torch.Tensor]]]:
    """
    Randomly select images from the dataset
    dataset: MNIST dataset - either train or test
    num_images: number of random images per digit
    by_label: whether to return a dict when each key is a digit and each value is a list of corresponding images, or just a list of all selected images
    return: random indices and images
    """
    indices = get_random_indices(dataset, num_images, by_label)

    # Get the images in the selected indices - when each image is stored in index 0 (and label in index 1)
    if not by_label:
        images = [dataset[idx][0] for idx in indices]
    else:
        images = {digit: [dataset[idx][0] for idx in idx_list] for digit, idx_list in indices.items()}
    
    return indices, images
