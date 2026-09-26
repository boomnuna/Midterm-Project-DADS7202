"""
src/data/dataset.py

DataModule wraps everything about "getting images into the model" behind
one class with a stable interface, regardless of which dataset topic is
plugged in.

EXPECTED FOLDER LAYOUT (any of these three works):

Option A — fully pre-split (all three folders fixed by hand):
    data_root/
    ├── train/<class_name>/*.jpg
    ├── val/<class_name>/*.jpg
    └── test/<class_name>/*.jpg

Option B — one flat folder per class (DataModule splits ALL THREE for you):
    data_root/
    ├── <class_name_1>/*.jpg
    ├── <class_name_2>/*.jpg
    └── ...

Option C — train/test pre-split by hand, val carved from train automatically
(recommended when you want the test set locked away manually — e.g. to
control for near-duplicate images across splits — but don't want to hand-
split validation every time you add more training images):
    data_root/
    ├── train/<class_name>/*.jpg   <- val is split OUT of this at load time
    └── test/<class_name>/*.jpg    <- used exactly as given, never touched

    NOTE: in this mode, config.val_fraction means "fraction of the TRAIN
    folder" (e.g. 0.15 = carve 15% of train/ into validation), NOT a
    fraction of the whole dataset like in Option B. config.test_fraction
    is ignored entirely in this mode since test/ is already fixed.
"""

from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

from src.data.transforms import TransformFactory


class DataModule:
    def __init__(self, config):
        self.config = config
        self.transform_factory = TransformFactory(image_size=config.image_size)
        self._prepare_datasets()

    # ---------------------------------------------------------------
    def _has_presplit_layout(self) -> bool:
        return all((self.config.data_root / split).is_dir()
                   for split in ("train", "val", "test"))

    def _has_train_test_only_layout(self) -> bool:
        root = self.config.data_root
        return (root / "train").is_dir() and (root / "test").is_dir() and not (root / "val").is_dir()

    def _prepare_datasets(self):
        if self._has_presplit_layout():
            self._load_presplit()
        elif self._has_train_test_only_layout():
            self._load_train_test_only()
        else:
            self._load_and_split_flat()

        # class names/order come from ImageFolder's own sorted() logic —
        # this is what lets the rest of the pipeline stay dataset-agnostic
        self.class_names = self.train_dataset_base.classes
        self.num_classes = len(self.class_names)

    def _load_presplit(self):
        root = self.config.data_root
        self.train_dataset_base = ImageFolder(root / "train", transform=self.transform_factory.train_transform())
        self.val_dataset = ImageFolder(root / "val", transform=self.transform_factory.eval_transform())
        self.test_dataset = ImageFolder(root / "test", transform=self.transform_factory.eval_transform())
        self.train_dataset = self.train_dataset_base
        self.train_targets = [label for _, label in self.train_dataset_base.samples]

    def _load_train_test_only(self):
        """
        Option C: test/ is used exactly as given (never re-split, never
        touched) — this is what protects it from the near-duplicate/
        leakage concerns that come with re-splitting from scratch every
        time images are added. val/ is carved OUT of train/ using a
        stratified split, seeded for reproducibility.
        """
        root = self.config.data_root

        train_full_raw = ImageFolder(root / "train")  # transform applied per-subset below
        targets = [label for _, label in train_full_raw.samples]
        indices = list(range(len(train_full_raw)))

        train_idx, val_idx, _, _ = train_test_split(
            indices, targets,
            test_size=self.config.val_fraction,  # fraction OF TRAIN, not of the whole dataset
            stratify=targets,
            random_state=self.config.split_seed,
        )

        train_full = ImageFolder(root / "train", transform=self.transform_factory.train_transform())
        eval_full = ImageFolder(root / "train", transform=self.transform_factory.eval_transform())

        self.train_dataset_base = train_full_raw  # for .classes access
        self.train_dataset = Subset(train_full, train_idx)
        self.val_dataset = Subset(eval_full, val_idx)
        self.test_dataset = ImageFolder(root / "test", transform=self.transform_factory.eval_transform())
        self.train_targets = [targets[i] for i in train_idx]

    def _load_and_split_flat(self):
        """
        Stratified split (keeps class proportions roughly equal across
        train/val/test) — matches the assignment's suggestion of
        "stratified data splitting" as one valid, defensible approach.
        """
        root = self.config.data_root
        full_raw = ImageFolder(root)  # transform applied later per-subset
        targets = [label for _, label in full_raw.samples]
        indices = list(range(len(full_raw)))

        train_idx, temp_idx, train_y, temp_y = train_test_split(
            indices, targets,
            test_size=(self.config.val_fraction + self.config.test_fraction),
            stratify=targets,
            random_state=self.config.split_seed,
        )
        relative_test_size = self.config.test_fraction / (self.config.val_fraction + self.config.test_fraction)
        val_idx, test_idx, _, _ = train_test_split(
            temp_idx, temp_y,
            test_size=relative_test_size,
            stratify=temp_y,
            random_state=self.config.split_seed,
        )

        train_full = ImageFolder(root, transform=self.transform_factory.train_transform())
        eval_full = ImageFolder(root, transform=self.transform_factory.eval_transform())

        self.train_dataset_base = full_raw  # for .classes access
        self.train_dataset = Subset(train_full, train_idx)
        self.val_dataset = Subset(eval_full, val_idx)
        self.test_dataset = Subset(eval_full, test_idx)
        self.train_targets = [targets[i] for i in train_idx]

    # ---------------------------------------------------------------
    def class_distribution(self, split: str = "train") -> dict:
        """Returns {class_name: count} — used for EDA and imbalance checks."""
        targets = {
            "train": self.train_targets,
        }.get(split)
        if targets is None:
            raise ValueError("class_distribution() only supported for 'train' split here; "
                              "use eda.py for full-dataset EDA across all splits.")
        counts = Counter(targets)
        return {self.class_names[i]: counts.get(i, 0) for i in range(self.num_classes)}

    def class_weights(self) -> torch.Tensor:
        """
        Inverse-frequency class weights for nn.CrossEntropyLoss(weight=...).
        This is the "class_weights" imbalance strategy from config.py —
        rarer classes get proportionally more weight in the loss.
        """
        dist = self.class_distribution("train")
        counts = torch.tensor([dist[name] for name in self.class_names], dtype=torch.float)
        weights = counts.sum() / (len(counts) * counts.clamp(min=1))
        return weights

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    # pin_memory speeds up CPU->GPU transfer, but only means anything
    # when there's a GPU to transfer to — on a CPU-only machine it does
    # nothing useful and just prints a warning every run. Checking once
    # here keeps that warning from showing up on CPU-only setups while
    # still getting the speedup automatically wherever a GPU IS available.
    @property
    def _pin_memory(self) -> bool:
        return torch.cuda.is_available()

    def train_loader(self, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.config.batch_size,
            shuffle=shuffle, num_workers=self.config.num_workers, pin_memory=self._pin_memory,
        )

    def val_loader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=self.config.num_workers, pin_memory=self._pin_memory,
        )

    def test_loader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=self.config.num_workers, pin_memory=self._pin_memory,
        )
