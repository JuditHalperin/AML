import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from sklearn.model_selection import train_test_split
from create_data import create_unconditional_olympic_rings, create_olympic_rings


class OlympicRingsDataset(Dataset):

    def __init__(self, data, labels=None, class_names=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.int) if labels is not None else None
        self.class_names = class_names

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor] | Tensor:
        return (self.data[idx], self.labels[idx]) if self.labels is not None else self.data[idx]
    
    def get_dim_size(self) -> int:
        return int(self.data.shape[-1])

    def get_num_classes(self) -> int | None:
        return len(self.class_names) if self.class_names else None


def get_unconditional_dataset(train_size: int, test_size: int, verbose: bool = True) -> tuple[OlympicRingsDataset, OlympicRingsDataset]:
    data = create_unconditional_olympic_rings(train_size + test_size, verbose=verbose)
    X_train, X_test = train_test_split(data, train_size=train_size)
    return OlympicRingsDataset(X_train), OlympicRingsDataset(X_test)


def get_conditional_dataset(train_size: int, test_size: int, verbose: bool = True) -> tuple[OlympicRingsDataset, OlympicRingsDataset]:
    data, labels, class_names = create_olympic_rings(train_size + test_size, verbose=verbose)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=train_size, stratify=labels)
    return OlympicRingsDataset(X_train, y_train, class_names), OlympicRingsDataset(X_test, y_test, class_names)


def load_datasets(
        conditional: bool = False,
        train_size: int = 250000,
        test_size: int = 10000,
        batch_size: int = 128,
        verbose: bool = False
    ):
    """
    Get conditional / unconditional train and test datasets
    return: train set loader, test set loader, dimension size in datasets, number of classes in conditional case
    """
    creation_func = get_conditional_dataset if conditional else get_unconditional_dataset
    train_dataset, test_dataset = creation_func(train_size, test_size, verbose)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset.get_dim_size(), train_dataset.get_num_classes()


def pick_points():
    """
    Pick some points on, inside and outside of the rings.
    The points are chosen based on the normalized value range between (-2, 2)
    """
    points_on = torch.Tensor([
        [0.0, 0.0],
        [-1.5, 1.5],
        [1.0, 0.0],
        [-0.5, 1.0],
    ])
    points_in = torch.Tensor([
        [0.0, 1.0],
        [-1.0, 1.0],
        [0.5, -1.0],
    ])
    points_out = torch.Tensor([
        [1.0, 2.0],
        [-2.0, 1.0],
        [0.0, -2.3],
    ])
    return points_on, points_in, points_out
