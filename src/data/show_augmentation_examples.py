"""
scripts/show_augmentation_examples.py

Assignment section 2 (Data preparation) requires explaining which
augmentation operations were chosen and why. This script runs REAL
images through the ACTUAL augmentation pipeline used during training
(TransformFactory.train_transform_display() — same ops as
train_transform(), just without ToTensor/Normalize so they're viewable)
and saves ONE figure for the slide.

Layout (ONE figure, default 4 images per class):
  - columns = 4 DIFFERENT training images of that class
  - for every class there is a block of rows:
        row 1 : original (resized only, no augmentation)
        row 2 : one random augmentation of the SAME image
        (row 3, 4, ... : more random augmentations if --versions-per-image > 1)
  - blocks are stacked: class 1, class 2, class 3, class 4

Produces: outputs/augmentation_examples.png

Usage:
    python -m scripts.show_augmentation_examples
    python -m scripts.show_augmentation_examples --images-per-class 4 --versions-per-image 2
    python -m scripts.show_augmentation_examples --seed 7      # a different random pick
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
    train_dataset = data_module.train_dataset
    underlying = train_dataset.dataset if hasattr(train_dataset, "dataset") else train_dataset  # handles Subset wrapping
    indices = train_dataset.indices if hasattr(train_dataset, "indices") else range(len(train_dataset))

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-per-class", type=int, default=4,
                        help="How many DIFFERENT source images from EACH class (= columns, default: 4)")
    parser.add_argument("--versions-per-image", type=int, default=1,
                        help="How many random augmentations to show under each original (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for picking the source images (same seed = same pictures)")
    args = parser.parse_args()

    config = Config()
    data_module = DataModule(config)

    display_transform = TransformFactory(config.image_size).train_transform_display()  # augmentation (PIL out)
    resize_only = TransformFactory(config.image_size).raw_transform()  # for the "original" rows

    samples = get_sample_paths(data_module, args.images_per_class, args.seed)
    if not samples:
        print("No training images found — check config.data_root.")
        return

    n_classes = len(samples)
    n_cols = args.images_per_class
    block = 1 + args.versions_per_image          # rows per class: original + augmented
    n_rows = n_classes * block

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.8 * n_rows + 0.8))
    axes = axes.reshape(n_rows, n_cols) if n_rows * n_cols > 1 else [[axes]]

    for ci, (class_name, paths) in enumerate(samples.items()):
        r0 = ci * block
        for col in range(n_cols):
            for r in range(block):
                axes[r0 + r][col].axis("off")
        for col, path in enumerate(paths):
            img = Image.open(path).convert("RGB")

            # first row of the block: original (resized only, no augmentation)
            axes[r0][col].imshow(resize_only(img).permute(1, 2, 0).numpy())
            axes[r0][col].set_title(f"{class_name} (original)", fontsize=9, fontweight="bold")

            # following rows: fresh random augmentation each time (online augmentation)
            for v in range(1, block):
                axes[r0 + v][col].imshow(display_transform(img))
                axes[r0 + v][col].set_title(f"augmented #{v}", fontsize=9)

    fig.suptitle(f"Data augmentation examples — {n_cols} different images per class "
                 f"(online: fresh random transform per epoch)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.985])

    config.output_root.mkdir(parents=True, exist_ok=True)
    save_path = config.output_root / "augmentation_examples.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved ONE figure: {n_classes} classes x {n_cols} images "
          f"(original + {args.versions_per_image} augmented each) to {save_path}")


if __name__ == "__main__":
    main()