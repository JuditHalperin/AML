import random, torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

no_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])


class PairedCIFAR10(datasets.CIFAR10):
    """CIFAR10 dataset for representation models when 2 image views are used during training"""

    def __init__(
            self,
            train: bool,
            augmentation: bool = False,
            keep_original: bool = False,
            neighboring_indices: list[int] = None
        ) -> None:
        """
        train: whether to use train set or test set
        augmentation: whether to apply train_transform, used in normal VICReg
        keep_original: whether to apply no_transform
        If both augmentation and keep_original are false, test_transform is applied
        neighboring_indices: index of neighboring image for each instance, used in VICReg without generated neighbors
        """
        super().__init__(train=train, root='./data', download=True)
        assert not (augmentation and neighboring_indices)

        # Define data transformation
        if augmentation:
            self.transform = train_transform
        elif not keep_original:
            self.transform = test_transform
        else:
            self.transform = no_transform
            
        self.neighboring_indices = neighboring_indices

        # Define get item method
        if augmentation:
            self.get_item = self.get_generated_neighboring_views
        elif neighboring_indices:
            self.get_item = self.get_knn_neighboring_views
        else:
            self.get_item = self.get_single_view
    
    def get_generated_neighboring_views(self, idx: int):
        """Return two views using generated augmentation of an image"""
        img = self.data[idx]
        return self.transform(img), self.transform(img)
    
    def get_knn_neighboring_views(self, idx: int):
        """Return two views using knn neighboring image"""
        neighbor_idx = self.neighboring_indices[idx]
        return self.transform(self.data[idx]), self.transform(self.data[neighbor_idx])
    
    def get_single_view(self, idx: int):
        """Return single view"""
        return self.transform(self.data[idx])

    def __getitem__(self, index: int) -> tuple:
        """Return item and target"""
        return self.get_item(index), self.targets[index]


def load_datasets(
        augmentation: bool = True,
        neighboring_indices_train: list[int] = None,
        shuffle_train: bool = True,
        batch_size: int = 256,
        num_workers: int = 2
    ) -> tuple[DataLoader, DataLoader]:
    """
    augmentation: whether to apply train_transform, used in normal VICReg
    neighboring_indices_train: index of neighboring image for each instance, used in VICReg without generated neighbors
    """
    train_dataset = PairedCIFAR10(train=True, augmentation=augmentation, neighboring_indices=neighboring_indices_train)
    test_dataset = PairedCIFAR10(train=False, augmentation=augmentation, neighboring_indices=None)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def get_random_indices(dataset, num_images: int = 1, num_classes: int = 10):
    """
    Randomly select image indices from the dataset
    dataset: either train or test dataset
    num_images: number of random images per label
    num_classes: number of label classes
    return: dict of lists with random indices per label
    """
    # Shuffle the dataset indices to select random images
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)

    # Select indices until each digit has num_images indices
    digit_indices = {i: [] for i in range(num_classes)}
    found_digits = 0
    for idx in all_indices:
        if found_digits == num_classes:
            break
        _, label = dataset[idx]
        if len(digit_indices[label]) < num_images:
            digit_indices[label].append(idx)
            if len(digit_indices[label]) == num_images:
                found_digits += 1

    # Sort by digits
    digit_indices = dict(sorted(digit_indices.items(), key=lambda x: int(x[0])))
    return digit_indices


def load_anomaly_datasets(
        batch_size: int = 256,
        num_workers: int = 2
    ):

    cifer_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    mnist_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), 
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=cifer_transform)
    cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=cifer_transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)

    combined_test_data = ConcatDataset([cifar10_test, mnist_test])
    test_labels = torch.cat([torch.zeros(len(cifar10_test), dtype=torch.long), torch.ones(len(mnist_test), dtype=torch.long)])

    train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(combined_test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, combined_test_data, test_labels
