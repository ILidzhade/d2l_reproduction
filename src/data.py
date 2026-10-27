"""Load images, create leakage-free partitions, and build the task stream."""

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .config import Config


class ImageDataset(Dataset):
    """Load labelled RGB images from explicit paths using a supplied transform.

    Args:
        samples: Image-path and integer-label pairs.
        classes: Ordered class names corresponding to integer labels.
        transform: Function converting a PIL image into a model tensor.
    """

    def __init__(
        self, samples: list[tuple[Path, int]], classes: list[str],
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        self.samples = samples
        self.classes = classes
        self.targets = [label for _, label in samples]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label


@dataclass(frozen=True)
class TaskData:
    """Bundle training, mapping, validation, and test loaders for one task."""
    train_loader: DataLoader
    test_loader: DataLoader
    validation_loader: DataLoader
    mapping_loader: DataLoader
    class_ids: tuple[int, ...]


def _loader(dataset: Dataset, cfg: Config, shuffle: bool, seed: int) -> DataLoader:
    """Build a reproducibly seeded data loader for one dataset partition."""
    return DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=shuffle,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
    )


def _transforms(cfg: Config) -> tuple[Callable, Callable]:
    """Return stochastic training and deterministic evaluation transforms."""
    return (
        transforms.Compose([
            transforms.RandomResizedCrop(cfg.image_size),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        ]),
        transforms.Compose([
            transforms.Resize(round(cfg.image_size * 256 / 224)),
            transforms.CenterCrop(cfg.image_size), transforms.ToTensor(),
        ]),
    )


def _partition(cfg: Config) -> tuple[ImageDataset, ImageDataset, ImageDataset]:
    """Create non-overlapping training, validation, and test image datasets."""
    root = cfg.data_root
    csv_paths = [root / f"{split}.csv" for split in ("train", "val", "test")]
    by_class: dict[str, list[Path]] = {}
    separate_test: dict[str, list[Path]] | None = None
    if (root / "images").is_dir() and all(path.is_file() for path in csv_paths):
        seen: set[str] = set()
        for path in csv_paths:
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if not {"filename", "label"} <= set(reader.fieldnames or []):
                    raise ValueError(f"{path} needs filename and label columns.")
                for row in reader:
                    filename = row["filename"]
                    if filename in seen:
                        raise ValueError(f"Duplicate image across CSVs: {filename}")
                    seen.add(filename)
                    image_path = root / "images" / filename
                    if not image_path.is_file():
                        raise FileNotFoundError(image_path)
                    by_class.setdefault(row["label"], []).append(image_path)
    else:
        folder = root if (root / "train").is_dir() else root / "seqMINIIMG" / "images"
        if not (folder / "train").is_dir() or not (folder / "val").is_dir():
            raise FileNotFoundError("Expected CSV images or ImageFolder train/val directories.")
        train = datasets.ImageFolder(folder / "train")
        test = datasets.ImageFolder(folder / "val")
        if train.classes != test.classes:
            raise ValueError("Train/evaluation class names differ.")
        separate_test = {}
        for path, label in train.samples:
            by_class.setdefault(train.classes[label], []).append(Path(path))
        for path, label in test.samples:
            separate_test.setdefault(test.classes[label], []).append(Path(path))

    names = sorted(by_class)
    if len(names) < cfg.total_classes:
        raise ValueError(f"Need {cfg.total_classes} classes, found {len(names)}.")
    if len(names) > cfg.total_classes:
        names = sorted(random.Random(cfg.seed).sample(names, cfg.total_classes))
    partitions: list[list[tuple[Path, int]]] = [[], [], []]
    for label, name in enumerate(names):
        paths = sorted(by_class[name])
        random.Random(cfg.seed + label).shuffle(paths)
        if separate_test is None:
            test_paths = paths[:cfg.csv_test_samples_per_class]
            paths = paths[cfg.csv_test_samples_per_class:]
            if len(test_paths) != cfg.csv_test_samples_per_class:
                raise ValueError(f"Insufficient test samples for {name}.")
        else:
            test_paths = sorted(separate_test[name])
        val_paths = paths[:cfg.validation_samples_per_class]
        train_paths = paths[cfg.validation_samples_per_class:]
        if len(val_paths) != cfg.validation_samples_per_class or not train_paths or not test_paths:
            raise ValueError(f"Insufficient training/validation/evaluation samples for {name}.")
        if cfg.train_samples_per_class is not None:
            train_paths = train_paths[:cfg.train_samples_per_class]
        if cfg.eval_samples_per_class is not None:
            test_paths = test_paths[:cfg.eval_samples_per_class]
        for partition, selected in zip(partitions, (train_paths, val_paths, test_paths), strict=True):
            partition.extend((path, label) for path in selected)
    train_transform, eval_transform = _transforms(cfg)
    resolved = [
        {path.resolve() for path, _ in partition} for partition in partitions
    ]
    if any(resolved[i] & resolved[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Training, validation and test paths overlap.")
    return (
        ImageDataset(partitions[0], names, train_transform),
        ImageDataset(partitions[1], names, eval_transform),
        ImageDataset(partitions[2], names, eval_transform),
    )


def build_task_stream(cfg: Config) -> list[TaskData]:
    """Partition configured classes into the deterministic continual task stream.

    Args:
        cfg: Experiment configuration defining data paths, splits, and task sizes.

    Returns:
        Ordered task bundles beginning with the base task.
    """
    train, validation, test = _partition(cfg)
    mapping_data = ImageDataset(train.samples, train.classes, _transforms(cfg)[1])
    order = torch.randperm(
        cfg.total_classes, generator=torch.Generator().manual_seed(cfg.seed)
    ).tolist()
    tasks = []
    offset = 0
    for task_id, size in enumerate(
        [cfg.base_classes] + [cfg.classes_per_task] * (cfg.num_tasks - 1)
    ):
        classes = tuple(order[offset:offset + size])
        offset += size
        subsets = [
            Subset(dataset, [i for i, label in enumerate(dataset.targets) if label in classes])
            for dataset in (train, test, validation)
        ]
        tasks.append(TaskData(
            train_loader=_loader(subsets[0], cfg, True, cfg.seed + task_id),
            test_loader=_loader(subsets[1], cfg, False, cfg.seed + task_id),
            validation_loader=_loader(subsets[2], cfg, False, cfg.seed + task_id),
            mapping_loader=_loader(
                Subset(mapping_data, list(subsets[0].indices)), cfg, False, cfg.seed + task_id
            ),
            class_ids=classes,
        ))
    return tasks


def stream_manifest(tasks: list[TaskData], root: Path) -> dict:
    """Serialize every task's classes and file paths for reproducibility checks.

    Args:
        tasks: Ordered continual-learning tasks.
        root: Dataset root relative to which image paths are recorded.

    Returns:
        A JSON-compatible manifest containing task partitions and class names.
    """
    records = []
    for task in tasks:
        record: dict = {"class_ids": list(task.class_ids)}
        for name, loader in (
            ("train", task.train_loader), ("validation", task.validation_loader),
            ("test", task.test_loader),
        ):
            subset = loader.dataset
            if not isinstance(subset, Subset) or not isinstance(subset.dataset, ImageDataset):
                raise TypeError("Manifest requires ImageDataset task subsets.")
            dataset = subset.dataset
            record[name] = [
                {"path": str(dataset.samples[i][0].relative_to(root)),
                 "label": dataset.samples[i][1]}
                for i in subset.indices
            ]
            record["classes"] = dataset.classes
        records.append(record)
    return {"tasks": records}
