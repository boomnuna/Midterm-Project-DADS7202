"""
scripts/show_pretrained_baseline.py

Assignment section 3 (Model architecture) requires showing that a
PRE-TRAINED, NOT-YET-FINETUNED CNN either misclassifies our images
entirely or has no matching class at all — this is the evidence for
"why we need to finetune it." No accuracy/precision/recall needed here,
just example predictions to screenshot for your slides.

Produces:
  - a printed text table (quick terminal check)
  - a saved image grid: outputs/pretrained_baseline_examples.png
    (10 real images, each labeled with true class vs. ImageNet's top-1
    guess — THIS is what actually goes on your slide, not the text table)

Usage:
    python -m scripts.show_pretrained_baseline
    python -m scripts.show_pretrained_baseline --num-examples 15
"""

import sys
import json
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision.models import ResNet50_Weights

from config import Config
from src.data.dataset import DataModule
from src.data.transforms import TransformFactory

# get all 1000 imagenet class name 
def load_imagenet_class_names():
    print("-" * 70)
    print("Loading pretrained ResNet50 (ImageNet weights)...")
    weights = ResNet50_Weights.IMAGENET1K_V2
    return weights.meta["categories"]


def get_sample_image_paths(data_module: DataModule, num_examples: int) -> list:
    """
    Pulls real file paths (not just tensors) from the test set so we can
    both DISPLAY the original image and run it through the model —
    mirrors the same underlying-dataset access pattern used for GradCAM
    sampling in run_all_experiments.py.
    """
    test_dataset = data_module.test_dataset
    underlying = test_dataset.dataset if hasattr(test_dataset, "dataset") else test_dataset

    paths_and_labels = []
    n = min(num_examples, len(test_dataset))
    for i in range(n):
        idx = test_dataset.indices[i] if hasattr(test_dataset, "indices") else i
        path, label = underlying.samples[idx]
        paths_and_labels.append((path, label))
    return paths_and_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, default=10,
                         help="How many real images to show/save (default: 10)")
    args = parser.parse_args()

    config = Config()
    data_module = DataModule(config)
    imagenet_classes = load_imagenet_class_names()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
    model.eval()

    raw_transform = TransformFactory(config.image_size).raw_transform()
    eval_transform = TransformFactory(config.image_size).eval_transform()

    samples = get_sample_image_paths(data_module, args.num_examples)

    print("Pretrained ResNet50 (ImageNet, untouched) predictions on OUR dataset:\n")
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
    n_cols = 5
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

    fig.suptitle("Pretrained ResNet50 (ImageNet, NOT finetuned) on our dataset", fontsize=13)
    plt.tight_layout()

    save_path = config.output_root / "pretrained_baseline_examples.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("-" * 70)
    print(f"\nSaved image grid to {save_path}")
    print("Use this image directly on your slide as the 'pretrained model doesn't "
          "know our classes' evidence — no need to re-screenshot the terminal.")


if __name__ == "__main__":
    main()
