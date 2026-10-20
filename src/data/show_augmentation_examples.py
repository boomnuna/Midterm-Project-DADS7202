"""
scripts/show_augmentation_examples.py

Assignment section 2 (Data preparation) requires explaining which
augmentation operations were chosen and why. This script runs REAL
images through the ACTUAL augmentation pipeline used during training
(TransformFactory.train_transform_display() — same ops as
train_transform(), just without ToTensor/Normalize so they're viewable)
and saves before/after grids for the slide.

NEW: shows N DIFFERENT source images from EACH class (default 4).
Every row = one source image: column 0 is the original (resized only),
the other columns are fresh random augmentations of that same image.

Produces (in config.output_root):
  augmentation_examples_<class>.png   one figure per class
                                      (images_per_class rows x (1 + versions) columns)
  augmentation_examples.png           all classes in one tall figure
                                      (skip it with --no-combined)

Usage:
    python -m scripts.show_augmentation_examples
    python -m scripts.show_augmentation_examples --images-per-class 4 --versions-per-image 4
    python -m scripts.show_augmentation_examples --seed 7        # a different random pick
    python -m scripts.show_augmentation_examples --no-combined
"""

import sys
import random
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from PIL import Image
import matplotlib.pyplot as plt

from config import Config
from src.data.dataset import DataModule
from src.data.transforms import TransformFactory


def get_sample_paths(data_module: DataModule, images_per_class: int, seed: int) -> dict:
    """
    Picks `images_per_class` DIFFERENT images from EACH class, randomly
    (seeded, so you get the same pictures every run), from the TRAIN split
    specifically — that's the split augmentation actually applies to.

    Returns {class_name: [image_path, ...]} in class order.
    """
    train_dataset = data_module.train_dataset  # load training dataset
    underlying = train_dataset.dataset if hasattr(train_dataset, "dataset") else train_dataset  # handles Subset wrapping
    indices = train_dataset.indices if hasattr(train_dataset, "indices") else range(len(train_dataset))

    # group the training images by class
    by_class = {}
    for idx in indices:
        path, label = underlying.samples[idx]
        by_class.setdefault(label, []).append(path)

    rng = random.Random(seed)
    samples = {}
    for label in sorted(by_class):
        paths = by_class[label]
        k = min(images_per_class, len(paths))
        if k < images_per_class:
            print(f"WARNING: class '{data_module.class_names[label]}' has only {len(paths)} "
                  f"training images (wanted {images_per_class}).")
        samples[data_module.class_names[label]] = rng.sample(paths, k)  # distinct images, no repeats
    return samples


def draw_rows(rows, display_transform, resize_only, versions, title, save_path, tile=3.0):
    """rows = list of (class_name, image_path). One row per source image."""
    n_rows, n_cols = len(rows), 1 + versions
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(tile * n_cols, (tile + 0.2) * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for r, (class_name, path) in enumerate(rows):
        img = Image.open(path).convert("RGB")

        # column 0: original (resized only, no augmentation)
        original_np = resize_only(img).permute(1, 2, 0).numpy()
        axes[r, 0].imshow(original_np)
        axes[r, 0].set_title(f"{class_name}\n(original)", fontsize=9)
        axes[r, 0].axis("off")

        # columns 1..N: fresh random augmentation each time (online augmentation)
        for c in range(1, n_cols):
            axes[r, c].imshow(display_transform(img))  # returns a PIL Image
            axes[r, c].set_title(f"augmented #{c}", fontsize=9)
            axes[r, c].axis("off")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-per-class", type=int, default=4,
                        help="How many DIFFERENT source images to show from EACH class (default: 4)")
    parser.add_argument("--versions-per-image", type=int, default=4,
                        help="How many augmented versions to show per source image (default: 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for picking the source images (same seed = same pictures)")
    parser.add_argument("--no-combined", action="store_true",
                        help="Do not save the single tall figure with all classes")
    args = parser.parse_args()

    config = Config()
    data_module = DataModule(config)

    display_transform = TransformFactory(config.image_size).train_transform_display()  # images augmentation
    resize_only = TransformFactory(config.image_size).raw_transform()  # for the "original" column

    samples = get_sample_paths(data_module, args.images_per_class, args.seed)
    if not samples:
        print("No training images found — check config.data_root.")
        return

    config.output_root.mkdir(parents=True, exist_ok=True)

    # one figure per class
    for class_name, paths in samples.items():
        save_path = config.output_root / f"augmentation_examples_{class_name}.png"
        draw_rows([(class_name, p) for p in paths], display_transform, resize_only,
                  args.versions_per_image,
                  f"Data augmentation examples — {class_name} "
                  f"({len(paths)} different images, online augmentation)",
                  save_path)
        print(f"Saved {len(paths)} images x {1 + args.versions_per_image} columns to {save_path}")

    # all classes in one picture
    if not args.no_combined:
        all_rows = [(c, p) for c, paths in samples.items() for p in paths]
        save_path = config.output_root / "augmentation_examples.png"
        draw_rows(all_rows, display_transform, resize_only, args.versions_per_image,
                  "Data augmentation examples (online — fresh random transform per epoch)",
                  save_path, tile=2.4)
        print(f"Saved combined figure ({len(all_rows)} rows) to {save_path}")

    print("Use the per-class pictures on your 'Data preparation / augmentation' slide.")


if __name__ == "__main__":
    main()