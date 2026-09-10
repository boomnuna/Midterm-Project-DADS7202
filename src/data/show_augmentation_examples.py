"""
scripts/show_augmentation_examples.py

Assignment section 2 (Data preparation) requires explaining which
augmentation operations were chosen and why. This script runs REAL
images through the ACTUAL augmentation pipeline used during training
(TransformFactory.train_transform_display() — same ops as
train_transform(), just without ToTensor/Normalize so they're viewable)
and saves a before/after grid — this is the image to put on your slide,
not a text description of the operations.

Produces: outputs/augmentation_examples.png
  - Left column: original image (resized only, no augmentation)
  - Right 4 columns: 4 different random augmented versions of the SAME
    image — shows the augmentation is applied freshly each time (online
    augmentation), not a single fixed transformation

Usage:
    python -m scripts.show_augmentation_examples
    python -m scripts.show_augmentation_examples --num-images 5
"""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from PIL import Image
import matplotlib.pyplot as plt

from config import Config
from src.data.dataset import DataModule
from src.data.transforms import TransformFactory


def get_sample_paths(data_module: DataModule, num_images: int) -> list:
    """
    Picks one image per class where possible (falls back to multiple
    per class if num_images > num_classes), pulling from the TRAIN split
    specifically — that's the split augmentation actually applies to.
    """
    train_dataset = data_module.train_dataset # load traning dataset 
    underlying = train_dataset.dataset if hasattr(train_dataset, "dataset") else train_dataset # handles the possibility that train_dataset is wrapped inside another dataset object.
    indices = train_dataset.indices if hasattr(train_dataset, "indices") else range(len(train_dataset)) # check if the training dataset is a Subset, it has specific indexes.

    # group available indices by class so we can spread the sample across classes
    by_class = {}
    for idx in indices:
        _, label = underlying.samples[idx]
        by_class.setdefault(label, []).append(idx)

    samples = []
    class_ids = list(by_class.keys())
    i = 0
    while len(samples) < num_images and any(by_class.values()):
        label = class_ids[i % len(class_ids)] # cycles through the classes.
        if by_class[label]:
            idx = by_class[label].pop(0) # remove from list 
            path, _ = underlying.samples[idx] # gets the actual image path.
            samples.append((path, data_module.class_names[label])) # store img path and class name 
        i += 1

    return samples[:num_images]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=5, # How many different original images to select.
                         help="How many distinct source images to show (default: 5, "
                              "each gets 1 original + 4 augmented versions = 5 tiles/row)")
    parser.add_argument("--versions-per-image", type=int, default=4, # How many augmented versions to generate for each original image.
                         help="How many different augmented versions to show per image (default: 4)")
    args = parser.parse_args()

    config = Config()
    data_module = DataModule(config)

    display_transform = TransformFactory(config.image_size).train_transform_display() # images augmentation
    resize_only = TransformFactory(config.image_size).raw_transform()  # for the "original" column

    samples = get_sample_paths(data_module, args.num_images)
    if not samples:
        print("No training images found — check config.data_root.")
        return

    # creating the figure
    n_rows = len(samples)
    n_cols = 1 + args.versions_per_image  # 1 original + N augmented versions

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for row, (path, class_name) in enumerate(samples): # open each original image
        img = Image.open(path).convert("RGB")

        # column 0: original (resized only, no augmentation)
        original_tensor = resize_only(img)
        original_np = original_tensor.permute(1, 2, 0).numpy()
        axes[row, 0].imshow(original_np)
        axes[row, 0].set_title(f"{class_name}\n(original)", fontsize=9)
        axes[row, 0].axis("off")

        # columns 1..N: fresh random augmentation each time (online augmentation)
        for col in range(1, n_cols):
            augmented_img = display_transform(img)  # returns a PIL Image
            axes[row, col].imshow(augmented_img)
            axes[row, col].set_title(f"augmented #{col}", fontsize=9)
            axes[row, col].axis("off")

    fig.suptitle("Data augmentation examples (online — fresh random transform per epoch)",
                 fontsize=13)
    plt.tight_layout()

    save_path = config.output_root / "augmentation_examples.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {n_rows} images x {n_cols} columns ({n_rows * n_cols} tiles total) to {save_path}")
    print("Use this image directly on your 'Data preparation / augmentation' slide.")


if __name__ == "__main__":
    main()
