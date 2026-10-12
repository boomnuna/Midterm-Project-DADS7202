"""
scripts/reevaluate_runs.py

Re-computes the TEST metrics of runs that are already trained, using their saved
checkpoints and the CURRENT test folder -- no retraining.

Use it when the test set was wrong/damaged while the runs were evaluated
(e.g. a missing class folder in test/ made every label shift by one, giving
accuracy = 0). It then fixes, for every (backbone, seed):
  - logs/<run_id>_metrics.json   (this file is what run_all_experiments reads
                                  for runs it skips as "already completed")
  - experiment_log.csv           (accuracy / precision_macro / recall_macro / f1_macro
                                  columns only; every other column is kept)
Backups of both are written first.

Usage (from the project root, same place you run the other scripts):
    python -m scripts.reevaluate_runs --dry-run     # only prints the new numbers
    python -m scripts.reevaluate_runs               # writes the fixes
    python -m scripts.reevaluate_runs --backbones resnet50 vgg16

Afterwards run  python -m scripts.run_all_experiments  once more. It will skip
training, rebuild the mean±SD table, significance tests, confusion matrices,
training curves and GradCAM from the corrected metrics.
"""

import sys
import json
import shutil
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from config import Config
from src.data.dataset import DataModule
from src.models.classifier import CNNClassifier
from src.evaluation.metrics import Evaluator
from src.utils.checkpoint import CheckpointManager

CSV_METRIC_COLUMNS = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]


def to_python(value):
    """numpy -> plain python so it can be written to JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def test_class_counts(data_module) -> Counter:
    """Number of test images per class name (same logic as run_all_experiments)."""
    test_dataset = data_module.test_dataset
    underlying = test_dataset.dataset if hasattr(test_dataset, "dataset") else test_dataset
    if hasattr(test_dataset, "indices"):
        samples = [underlying.samples[i] for i in test_dataset.indices]
    else:
        samples = list(underlying.samples)
    return Counter(data_module.class_names[label] for _, label in samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", nargs="+", default=None,
                        help="Only these backbones (default: all in config.py)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evaluate and print only; do not touch any file")
    args = parser.parse_args()

    config = Config()
    if args.backbones:
        config.backbone_names = args.backbones

    data_module = DataModule(config)
    class_names = data_module.class_names

    # ---- safety check: refuse to continue on a broken test set ----
    counts = test_class_counts(data_module)
    print("Test images per class:")
    for name in class_names:
        print(f"  {name:20s}: {counts.get(name, 0)}")
    print(f"  total: {sum(counts.values())}")
    missing = [n for n in class_names if counts.get(n, 0) == 0]
    if missing:
        sys.exit(f"\nSTOP: no test images for {missing}. Fix the dataset folder first, "
                 f"otherwise the metrics would be wrong again.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = Evaluator(class_names)
    test_loader = data_module.test_loader()

    logs_dir = config.output_root / "logs"
    ckpt_root = config.output_root / "checkpoints"
    csv_path = config.output_root / "experiment_log.csv"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    new_metrics = {}   # run_id -> {metric: value}
    for backbone in config.backbone_names:
        for i in range(config.num_repeats):
            seed = config.base_seed + i
            run_id = f"{backbone}_seed{seed}"

            ckpt_mgr = CheckpointManager(ckpt_root, run_id)
            if not ckpt_mgr.has_checkpoint():
                print(f"  WARNING {run_id}: no checkpoint found -> skipped")
                continue

            # head size used for THIS run (tuned per backbone) -- read from its own config
            try:
                with open(logs_dir / f"{run_id}_config.json") as f:
                    run_cfg = json.load(f)
                hidden = run_cfg.get("head_hidden_dim", config.head_hidden_dim)
                dropout = run_cfg.get("head_dropout", config.head_dropout)
            except (OSError, json.JSONDecodeError):
                hidden, dropout = config.head_hidden_dim, config.head_dropout
                print(f"  WARNING {run_id}: run config json unreadable, using config.py head size")

            try:
                ckpt = (ckpt_mgr.load_best(map_location="cpu") if ckpt_mgr.has_best_checkpoint()
                        else ckpt_mgr.load_latest(map_location="cpu"))
                model = CNNClassifier(backbone_name=backbone, num_classes=data_module.num_classes,
                                      head_hidden_dim=hidden, head_dropout=dropout, pretrained=False)
                model.load_state_dict(ckpt["model_state"])
            except (RuntimeError, KeyError, OSError) as e:
                print(f"  WARNING {run_id}: could not load checkpoint ({e}) -> skipped")
                continue

            model.to(device)
            model.eval()
            result = evaluator.evaluate(model, test_loader, device)
            new_metrics[run_id] = {k: to_python(v) for k, v in result.items() if k != "confusion_matrix"}
            print(f"  {run_id:32s} acc={result['accuracy']:.4f}  f1_macro={result['f1_macro']:.4f}")

    if not new_metrics:
        sys.exit("Nothing was evaluated.")

    # ---- summary ----
    print("\nNew f1_macro (mean ± SD over repeats):")
    worst_mean = 1.0
    for backbone in config.backbone_names:
        vals = [m["f1_macro"] for rid, m in new_metrics.items() if rid.startswith(backbone + "_seed")]
        if vals:
            worst_mean = min(worst_mean, float(np.mean(vals)))
            print(f"  {backbone:20s}: {np.mean(vals):.4f} ± {np.std(vals, ddof=1) if len(vals) > 1 else 0:.4f}  (n={len(vals)})")
    if worst_mean < 0.5:
        print("\nWARNING: some means are still very low -> the test folder or the checkpoints "
              "may not match. Files will NOT be written.")
        sys.exit(1)

    if args.dry_run:
        print("\n--dry-run: nothing was written.")
        return

    # ---- backups ----
    backup_dir = logs_dir / f"metrics_backup_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) logs/<run_id>_metrics.json ----
    for run_id, metrics in new_metrics.items():
        path = logs_dir / f"{run_id}_metrics.json"
        old = {}
        if path.exists():
            shutil.copy2(path, backup_dir / path.name)
            with open(path) as f:
                old = json.load(f)
        else:
            old = {"seed": int(run_id.rsplit("seed", 1)[1])}
        old.update(metrics)
        with open(path, "w") as f:
            json.dump(old, f, indent=2)
    print(f"\nUpdated {len(new_metrics)} metrics json files (old ones: {backup_dir})")

    # ---- 2) experiment_log.csv ----
    if csv_path.exists():
        shutil.copy2(csv_path, config.output_root / f"experiment_log_before_reeval_{stamp}.csv")
        df = pd.read_csv(csv_path)
        updated = 0
        for run_id, metrics in new_metrics.items():
            mask = df["run_id"] == run_id
            if not mask.any():
                print(f"  WARNING {run_id}: not in experiment_log.csv (row not added)")
                continue
            for col in CSV_METRIC_COLUMNS:
                if col in metrics:
                    df.loc[mask, col] = metrics[col]
            updated += int(mask.sum())
        df.to_csv(csv_path, index=False)
        print(f"Updated {updated} rows in {csv_path}")
    else:
        print(f"WARNING: {csv_path} not found -> csv not updated")

    print("\nDone. Now run:  python -m scripts.run_all_experiments   (training is skipped)")


if __name__ == "__main__":
    main()
