"""
scripts/show_predictions.py

Assignment section 7 (Discussion — "Eyeball analysis") requires capturing
specific example results as images: true label vs. what the model
actually predicted, especially for cases that look wrong or surprising.

Loads a TRAINED model from its checkpoint (saved by Trainer during
train_single.py or run_all_experiments.py) and runs it on the test set,
producing TWO saved images:

  1. outputs/<backbone>/predictions_sample.png
     A random sample of test images with true vs. predicted label
     (green title = correct, red title = wrong) — the general
     "here's what our model does" slide.

  2. outputs/<backbone>/predictions_errors.png
     ONLY the misclassified examples (up to --max-errors) — this is the
     one to actually study for error analysis: what does the model
     confuse, and does the mistake look reasonable to a human or not.

Usage:
    python -m scripts.show_predictions --backbone resnet50 --seed 100
    python -m scripts.show_predictions --backbone resnet50 --run-id single_resnet50_seed100
    python -m scripts.show_predictions --backbone resnet50 --num-sample 15 --max-errors 15
"""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
import matplotlib.pyplot as plt

from config import Config
from src.data.dataset import DataModule
from src.data.transforms import TransformFactory
from src.models.classifier import CNNClassifier
from src.utils.checkpoint import CheckpointManager

# load test sataset 
def get_test_samples_in_order(data_module: DataModule) -> list:
    """
    Returns [(path, true_label_idx), ...] in the SAME order test_loader()
    iterates in (shuffle=False), so predictions line up with paths by
    index. Handles both plain ImageFolder (Option A/C layouts) and
    Subset (Option B auto-split layout) the same way other scripts in
    this project do.
    """
    test_dataset = data_module.test_dataset
    if hasattr(test_dataset, "dataset") and hasattr(test_dataset, "indices"):
        underlying = test_dataset.dataset
        return [underlying.samples[i] for i in test_dataset.indices]
    return list(test_dataset.samples)  # plain ImageFolder — samples already in order

# load trained model
def load_trained_model(config, data_module, backbone_name: str, run_id: str, device: str) -> CNNClassifier:
    checkpoint_manager = CheckpointManager(config.output_root / "checkpoints", run_id)

    # ---- failure mode 1: no checkpoint at all for this run_id ----
    if not checkpoint_manager.has_checkpoint():
        raise FileNotFoundError(
            f"No checkpoint found for run_id='{run_id}' under {config.output_root / 'checkpoints'}. "
            f"Check --backbone/--seed/--run-id match a run you've actually trained "
            f"(train_single.py uses run_id='single_<backbone>_seed<seed>', "
            f"run_all_experiments.py uses run_id='<backbone>_seed<seed>')."
        )

    # ---- failure mode 2: latest.pt exists but best.pt doesn't (rare —
    # e.g. a Colab disconnect right after the very first epoch's
    # save_latest() call but before save_best() ran). Fall back to
    # latest.pt with a warning rather than crashing — it's usable, just
    # maybe not the best epoch. ----
    if checkpoint_manager.has_best_checkpoint():
        try:
            ckpt = checkpoint_manager.load_best(map_location=device)
        except Exception as e:
            raise RuntimeError(
                f"Found best.pt for run_id='{run_id}' but failed to read it (file may be "
                f"corrupted — this can happen if a Colab session died mid-write despite the "
                f"atomic-rename safeguard, e.g. Drive sync issues). Original error: {e}\n"
                f"Try re-running training for this run_id, or check outputs/checkpoints/{run_id}/ "
                f"manually."
            ) from e
    else:
        print(f"WARNING: run_id='{run_id}' has no best.pt (only latest.pt) — "
              f"using the LATEST checkpoint instead, which may not be the best epoch. "
              f"This usually means training was interrupted very early.")
        try:
            ckpt = checkpoint_manager.load_latest(map_location=device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read latest.pt for run_id='{run_id}' either. Original error: {e}\n"
                f"This checkpoint directory may be corrupted — re-run training for this run_id."
            ) from e

    model = CNNClassifier(
        backbone_name=backbone_name,
        num_classes=data_module.num_classes,
        head_hidden_dim=config.head_hidden_dim,
        head_dropout=config.head_dropout,
        pretrained=False,  # weights get overwritten by the checkpoint right below anyway
    )

    # ---- failure mode 3: architecture mismatch — e.g. --backbone doesn't
    # match what this run_id was actually trained with, or config.py's
    # head_hidden_dim/head_dropout/num_classes changed since training ----
    try:
        model.load_state_dict(ckpt["model_state"])
    except RuntimeError as e:
        raise RuntimeError(
            f"Checkpoint for run_id='{run_id}' doesn't match the model architecture being "
            f"built (backbone='{backbone_name}', num_classes={data_module.num_classes}, "
            f"head_hidden_dim={config.head_hidden_dim}, head_dropout={config.head_dropout}). "
            f"This usually means --backbone is wrong for this run_id, or config.py's "
            f"head_hidden_dim/head_dropout/dataset changed since this run was trained.\n"
            f"Original error: {e}"
        ) from e

    model.to(device)
    model.eval()
    return model

# image evaluation 
@torch.no_grad()
def run_inference(model, samples: list, eval_transform, device: str) -> list:
    """Returns [(path, true_idx, pred_idx), ...] for every sample."""
    results = []
    for path, true_idx in samples:
        img = Image.open(path).convert("RGB")
        input_tensor = eval_transform(img).unsqueeze(0).to(device)
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        results.append((path, true_idx, pred_idx))
    return results

# save evaluation result 
def save_grid(results: list, class_names: list, raw_transform, title: str, save_path: Path, n_cols: int = 5):
    if not results:
        print(f"  (nothing to plot for '{title}' — skipping)")
        return

    n_rows = (len(results) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.3 * n_rows))
    axes = axes.flatten() if len(results) > 1 else [axes]

    for ax, (path, true_idx, pred_idx) in zip(axes, results):
        img = Image.open(path).convert("RGB")
        raw_np = raw_transform(img).permute(1, 2, 0).numpy()
        ax.imshow(raw_np)
        ax.axis("off")

        true_name = class_names[true_idx]
        pred_name = class_names[pred_idx]
        is_correct = true_idx == pred_idx
        color = "green" if is_correct else "crimson"
        ax.set_title(f"true: {true_name}\npred: {pred_name}", fontsize=9, color=color)

    for ax in axes[len(results):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {len(results)} images to {save_path}")

# main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--run-id", default=None,
                         help="Override the checkpoint run_id directly instead of deriving it "
                              "from --backbone/--seed (use this if the checkpoint came from "
                              "train_single.py, which prefixes run_id with 'single_')")
    parser.add_argument("--num-sample", type=int, default=10,
                         help="How many random test images to show in the general sample grid (default: 10)")
    parser.add_argument("--max-errors", type=int, default=10,
                         help="Max number of misclassified examples to show in the errors grid (default: 10)")
    parser.add_argument("--seed-for-sampling", type=int, default=0,
                         help="Random seed for WHICH test images get picked for the sample grid "
                              "(separate from --seed, which identifies the trained model's checkpoint)")
    args = parser.parse_args()

    import random
    random.seed(args.seed_for_sampling)

    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_module = DataModule(config)
    run_id = args.run_id or f"{args.backbone}_seed{args.seed}"

    print(f"Loading trained model from checkpoint run_id='{run_id}'...")
    model = load_trained_model(config, data_module, args.backbone, run_id, device)

    eval_transform = TransformFactory(config.image_size).eval_transform()
    raw_transform = TransformFactory(config.image_size).raw_transform()

    samples = get_test_samples_in_order(data_module)
    print(f"Running inference on {len(samples)} test images...")
    all_results = run_inference(model, samples, eval_transform, device)

    correct = [r for r in all_results if r[1] == r[2]]
    incorrect = [r for r in all_results if r[1] != r[2]]
    accuracy = len(correct) / len(all_results) if all_results else 0.0
    print(f"Test accuracy: {accuracy:.4f}  ({len(correct)}/{len(all_results)} correct, "
          f"{len(incorrect)} misclassified)")

    out_dir = config.output_root / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)

    # general sample grid: random mix of correct + incorrect, however they naturally fall
    sample = random.sample(all_results, min(args.num_sample, len(all_results)))
    save_grid(sample, data_module.class_names, raw_transform,
              title=f"{args.backbone} — sample predictions (green=correct, red=wrong)",
              save_path=out_dir / "predictions_sample.png")

    # errors-only grid — the one to actually study for error analysis
    errors_to_show = incorrect[:args.max_errors]
    save_grid(errors_to_show, data_module.class_names, raw_transform,
              title=f"{args.backbone} — misclassified examples ({len(incorrect)} total wrong)",
              save_path=out_dir / "predictions_errors.png")

    if not incorrect:
        print("\nNo misclassified examples found on the test set — nothing to save for error analysis.")


if __name__ == "__main__":
    main()
