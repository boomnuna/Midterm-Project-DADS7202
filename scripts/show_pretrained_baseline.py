"""
scripts/show_pretrained_baseline.py

Assignment section 3 (Model architecture) requires showing that a
PRE-TRAINED, NOT-YET-FINETUNED CNN either misclassifies our images
entirely or has no matching class at all — this is the evidence for
"why we need to finetune it." No accuracy/precision/recall needed here,
just example predictions to screenshot for your slides.

Produces:
  - a printed text table (quick terminal check)
  - a saved image grid: outputs/pretrained_baseline_<backbone>.png
    (real images from EVERY class, each labeled with true class vs. ImageNet's top-1
    guess — THIS is what actually goes on your slide, not the text table)

Usage:
    python -m scripts.show_pretrained_baseline                          # resnet50 (default)
    python -m scripts.show_pretrained_baseline --backbone vgg16
    python -m scripts.show_pretrained_baseline --backbone efficientnet_b0 --per-class 5

  --backbone: resnet50 | vgg16 | efficientnet_b0 | mobilenet_v3_small
  Output file includes the model name: outputs/pretrained_baseline_<backbone>.png
"""

import sys
import json
import argparse
import random
from collections import defaultdict
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models

from config import Config
from src.data.dataset import DataModule
from src.data.transforms import TransformFactory

# get all 1000 imagenet class name 
BACKBONES = ["resnet50", "vgg16", "efficientnet_b0", "mobilenet_v3_small"]


def load_pretrained_model(backbone: str):
    """
    Loads the torchvision model with its default ImageNet weights and the
    matching 1000 ImageNet class names (taken from the same weights object,
    so the names always line up with the model).
    """
    print("-" * 70)
    print(f"Loading pretrained {backbone} (ImageNet weights)...")
    weights = models.get_model_weights(backbone).DEFAULT
    model = models.get_model(backbone, weights=weights)
    return model, weights.meta["categories"]


def get_sample_image_paths(data_module: DataModule, per_class: int, seed=None) -> list:
    """
    Pulls real file paths (not just tensors) from the test set so we can
    both DISPLAY the original image and run it through the model —
    mirrors the same underlying-dataset access pattern used for GradCAM
    sampling in run_all_experiments.py.

    FIXED: picks `per_class` images from EACH class. The old version took
    the first N test images in dataset order, but ImageFolder sorts samples
    by class, so the first 10 images were all from the first class
    (gomu_gomu) and the other 3 classes never showed up.
    """
    test_dataset = data_module.test_dataset
    underlying = test_dataset.dataset if hasattr(test_dataset, "dataset") else test_dataset
    if hasattr(test_dataset, "indices"):
        all_samples = [underlying.samples[i] for i in test_dataset.indices]
    else:
        all_samples = list(underlying.samples)

    by_class = defaultdict(list)
    for path, label in all_samples:
        by_class[label].append((path, label))

    rng = random.Random(seed)  # seed=None -> different random images every run
    paths_and_labels = []
    for label in sorted(by_class):
        items = by_class[label]
        paths_and_labels.extend(rng.sample(items, min(per_class, len(items))))
    return paths_and_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet50", choices=BACKBONES,
                         help="Which pretrained model to show (default: resnet50)")
    parser.add_argument("--per-class", type=int, default=4,
                         help="How many real images to show PER CLASS (default: 4 -> 4x4 = 16 total for 4 classes)")
    parser.add_argument("--seed", type=int, default=None,
                         help="Optional: fix the random pick so the SAME images are chosen again "
                              "(default: none, i.e. different random images every run)")
    args = parser.parse_args()

    config = Config()
    data_module = DataModule(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, imagenet_classes = load_pretrained_model(args.backbone)
    model = model.to(device)
    model.eval()

    raw_transform = TransformFactory(config.image_size).raw_transform()
    eval_transform = TransformFactory(config.image_size).eval_transform()

    samples = get_sample_image_paths(data_module, args.per_class, args.seed)

    print(f"Pretrained {args.backbone} (ImageNet, untouched) predictions on OUR dataset:\n")
    print("-" * 70)
    print(f"{'true class (ours)':25s} | top-1 ImageNet prediction")
    print("-" * 70)

    results = []  # (raw_image_np, true_name, pred_name) for plotting
    with torch.no_grad():
        for path, label in samples:
            img = Image.open(path).convert("RGB")
            raw_np = raw_transform(img).permute(1, 2, 0).numpy()          # for display
            input_tensor = eval_transform(img).unsqueeze(0).to(device)     # for the model

            output = model(input_tensor)
            pred_idx = output.argmax(dim=1).item()

            true_name = data_module.class_names[label]
            pred_name = imagenet_classes[pred_idx]

            print(f"{true_name:25s} | {pred_name}")
            results.append((raw_np, true_name, pred_name))

    # ---- build and save the image grid — THIS is the actual slide asset ----
    n = len(results)
    n_cols = data_module.num_classes  # 4 classes x 4 per class -> each ROW is one class
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.5 * n_rows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, (raw_np, true_name, pred_name) in zip(axes, results):
        ax.imshow(raw_np)
        ax.axis("off")
        # red title = wrong/no-match, as expected for an un-finetuned model —
        # if it ever happens to guess right, title still shows it plainly
        is_match = true_name.lower() in pred_name.lower() or pred_name.lower() in true_name.lower()
        color = "green" if is_match else "crimson"
        ax.set_title(f"true: {true_name}\nImageNet says: {pred_name}", fontsize=9, color=color)

    # hide any unused subplot slots if num_examples isn't a multiple of n_cols
    for ax in axes[len(results):]:
        ax.axis("off")

    fig.suptitle(f"Pretrained {args.backbone} (ImageNet, NOT finetuned) on our dataset", fontsize=13)
    plt.tight_layout()

    save_path = config.output_root / f"pretrained_baseline_{args.backbone}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("-" * 70)
    print(f"\nSaved image grid to {save_path}")
    print("Use this image directly on your slide as the 'pretrained model doesn't "
          "know our classes' evidence — no need to re-screenshot the terminal.")


if __name__ == "__main__":
    main()