from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    input_shape: tuple[int, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]


DATASET_SPECS = {
    "mnist": DatasetSpec(
        name="mnist",
        num_classes=10,
        input_shape=(1, 28, 28),
        mean=(0.1307,),
        std=(0.3081,),
    ),
    "cifar10": DatasetSpec(
        name="cifar10",
        num_classes=10,
        input_shape=(3, 32, 32),
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    ),
}


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    key = dataset_name.lower()
    if key not in DATASET_SPECS:
        raise KeyError(f"Unknown dataset: {dataset_name}. Choices: {sorted(DATASET_SPECS)}")
    return DATASET_SPECS[key]


def build_dataloaders(
    dataset_name: str,
    data_root: str | Path,
    batch_size: int,
    test_batch_size: int | None = None,
    num_workers: int = 4,
    download: bool = False,
    augment: bool = False,
    val_split: float = 0.0,
    seed: int = 0,
):
    dataset_name = dataset_name.lower()
    data_root = Path(data_root).expanduser()
    test_batch_size = test_batch_size or batch_size
    if dataset_name == "mnist":
        train_transform, test_transform = _mnist_transforms()
        dataset_cls = datasets.MNIST
    elif dataset_name == "cifar10":
        train_transform, test_transform = _cifar10_transforms(augment=augment)
        dataset_cls = datasets.CIFAR10
    else:
        raise KeyError(f"Unknown dataset: {dataset_name}. Choices: {sorted(DATASET_SPECS)}")

    train_dataset = dataset_cls(
        root=str(data_root),
        train=True,
        transform=train_transform,
        download=download,
    )
    test_dataset = dataset_cls(
        root=str(data_root),
        train=False,
        transform=test_transform,
        download=download,
    )

    train_loader = None
    val_loader = None
    if val_split > 0:
        if not 0 < val_split < 1:
            raise ValueError("--val-split must be in (0, 1)")
        val_len = int(len(train_dataset) * val_split)
        train_len = len(train_dataset) - val_len
        generator = torch.Generator().manual_seed(seed)
        train_subset, val_subset = random_split(
            train_dataset, [train_len, val_len], generator=generator
        )
        train_loader = _loader(train_subset, batch_size, True, num_workers)
        val_loader = _loader(val_subset, test_batch_size, False, num_workers)
    else:
        train_loader = _loader(train_dataset, batch_size, True, num_workers)

    test_loader = _loader(test_dataset, test_batch_size, False, num_workers)
    return train_loader, val_loader, test_loader


def _mnist_transforms():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(DATASET_SPECS["mnist"].mean, DATASET_SPECS["mnist"].std),
        ]
    )
    return transform, transform


def _cifar10_transforms(augment: bool):
    normalize = transforms.Normalize(
        DATASET_SPECS["cifar10"].mean,
        DATASET_SPECS["cifar10"].std,
    )
    train_ops = []
    if augment:
        train_ops.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    train_ops.extend([transforms.ToTensor(), normalize])
    test_ops = [transforms.ToTensor(), normalize]
    return transforms.Compose(train_ops), transforms.Compose(test_ops)


def _loader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
