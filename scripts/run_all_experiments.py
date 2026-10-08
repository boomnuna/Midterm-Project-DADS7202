"""
scripts/run_all_experiments.py

THE main script for the assignment deliverables — runs every backbone in
config.backbone_names, each repeated config.num_repeats times, then
produces:
  - mean±SD table (assignment section 6)
  - pairwise statistical significance (assignment section 6)
  - training curves + confusion matrix per architecture (section 6)
  - GradCAM examples using the BEST run of each architecture (section 7)

PRIMARY METRIC: f1_macro, not accuracy — the dataset is imbalanced across
the 4 classes, so macro-F1 (unweighted average of each class's F1) is a
fairer "which architecture actually wins" signal than raw accuracy, which
can look good just by nailing the majority class(es). Every selection
below (best run per backbone, best architecture overall, pairwise
significance) is keyed on f1_macro for this reason. accuracy is still
reported in the mean±SD table for completeness, just not used to pick
winners anymore.

RUNNING THIS ACROSS MULTIPLE COLAB SESSIONS:
Use --backbones to run just ONE (or a few) architecture(s) per session —
e.g. one Colab session/person per backbone, run in parallel:
    python -m scripts.run_all_experiments --backbones resnet50
    python -m scripts.run_all_experiments --backbones vgg16
Then once ALL backbones have completed runs saved (check
outputs/experiment_log.csv), run this script once more with no
--backbones argument (or with all of them) to generate the combined
mean±SD table, significance tests, and GradCAM outputs across everything.

Already-completed (backbone, seed) runs are automatically skipped (see
ExperimentRunner) and in-progress runs resume from their last saved
epoch — so this is safe to just re-run after any Colab disconnect.

Usage:
    python -m scripts.run_all_experiments
    python -m scripts.run_all_experiments --backbones resnet50 vgg16
"""

import sys
import json
import pickle
import argparse
import random
from collections import defaultdict
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from config import Config
from src.data.dataset import DataModule
from src.data.transforms import TransformFactory
from src.models.classifier import CNNClassifier
from src.training.repeated_runs import ExperimentRunner
from src.evaluation.metrics import Evaluator
from src.evaluation.statistics import StatisticalComparator
from src.interpretability.gradcam import GradCAM
from src.utils.checkpoint import CheckpointManager

# the metric used to pick "best run" / "best architecture" everywhere in
# this script — see module docstring for why f1_macro, not accuracy.
PRIMARY_METRIC = "f1_macro"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", nargs="+", default=None,
                         help="Run only these backbones this session (default: all in config.py). "
                              "Useful for splitting work across Colab sessions/group members.")
    parser.add_argument("--gradcam-per-class", type=int, default=5,
                         help="How many random test images PER CLASS to explain with GradCAM "
                              "(default: 5). The same images are used for every architecture.")
    parser.add_argument("--gradcam-seed", type=int, default=None,
                         help="Optional: fix the random pick so the SAME images are chosen again "
                              "(default: none = different random images every run)")
    args = parser.parse_args()

    config = Config()
    if args.backbones:
        config.backbone_names = args.backbones

    data_module = DataModule(config)
    print(f"Classes: {data_module.class_names}")
    print(f"Architectures to compare THIS RUN: {config.backbone_names}")
    print(f"Repeats per architecture: {config.num_repeats}")
    print(f"Primary metric for best-run/best-architecture selection: {PRIMARY_METRIC}")
    print(f"Output root: {config.output_root}  "
          f"({'Google Drive path — good, survives disconnects' if 'drive' in str(config.output_root).lower() else 'LOCAL PATH — will be LOST on Colab disconnect unless this is a mounted Drive path!'})\n")

    runner = ExperimentRunner(config, data_module)
    results = runner.run_all(keep_last_model=True)

    # ---- persist raw results so you don't have to retrain to re-plot ----
    results_path = config.output_root / "experiment_results.pkl"
    with open(results_path, "wb") as f:
        # confusion matrices are numpy arrays, fine for pickle; skip
        # history/model refs that aren't needed for the summary tables
        pickle.dump(results, f)
    print(f"\nSaved raw results to {results_path}")

    # ---- mean±SD + significance (assignment section 6) ----
    comparator = StatisticalComparator(results)
    comparator.print_mean_std_table()  # prints accuracy/precision/recall/f1_macro all at once, for the report
    print()
    comparator.print_pairwise_significance(metric_name=PRIMARY_METRIC)

    best_name = comparator.best_architecture(metric_name=PRIMARY_METRIC)
    print(f"\nBest architecture by mean {PRIMARY_METRIC}: {best_name}")

    # ---- plots + best-run bookkeeping for each architecture ----
    # best_run_ids[backbone_name] = (seed, run_id) of the run with the
    # highest f1_macro for that backbone — reused below for GradCAM so
    # GradCAM reflects the SAME "best" run as everything else in the
    # report, not whichever seed happened to finish training last.
    evaluator = Evaluator(data_module.class_names)
    best_run_ids = {}
    for backbone_name, runs in results.items():
        best_run = max(runs, key=lambda r: r[PRIMARY_METRIC])
        best_run_ids[backbone_name] = (best_run["seed"], f"{backbone_name}_seed{best_run['seed']}")

        out_dir = config.output_root / backbone_name
        out_dir.mkdir(parents=True, exist_ok=True)

        evaluator.plot_training_curves(
            best_run["history"], title=f"{backbone_name} (best run, {PRIMARY_METRIC}={best_run[PRIMARY_METRIC]:.4f})",
            save_path=out_dir / "training_curves_best_run.png",
        )
        evaluator.plot_confusion_matrix(
            best_run["confusion_matrix"], title=f"{backbone_name} (best run, {PRIMARY_METRIC}={best_run[PRIMARY_METRIC]:.4f})",
            save_path=out_dir / "confusion_matrix_best_run.png",
        )

    # ---- GradCAM on a few test images, using each architecture's BEST
    # run (by f1_macro) — reloaded from its saved checkpoint rather than
    # from runner.last_models, which only ever holds the LAST-trained
    # seed (not necessarily the best one). This keeps GradCAM consistent
    # with which run every other plot/table in this script treats as
    # "the" result for that architecture. ----
    print(f"\nGenerating GradCAM examples (using each architecture's best run by {PRIMARY_METRIC})...")
    raw_transform = TransformFactory(config.image_size).raw_transform()
    eval_transform = TransformFactory(config.image_size).eval_transform()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # pick N RANDOM test images from EACH class (picked once, so every architecture is
    # explained on the same images). The old version took the first 5 test images, which
    # are all from one class whenever the test folder is sorted by class.
    test_dataset = data_module.test_dataset
    underlying = test_dataset.dataset if hasattr(test_dataset, "dataset") else test_dataset
    if hasattr(test_dataset, "indices"):
        all_test_samples = [underlying.samples[i] for i in test_dataset.indices]
    else:
        all_test_samples = list(underlying.samples)

    paths_by_class = defaultdict(list)
    for path, label in all_test_samples:
        paths_by_class[label].append(path)

    rng = random.Random(args.gradcam_seed)
    sample_items = []  # (true_class_name, image_path)
    for label in sorted(paths_by_class):
        chosen = rng.sample(paths_by_class[label], min(args.gradcam_per_class, len(paths_by_class[label])))
        sample_items.extend((data_module.class_names[label], p) for p in chosen)
    print(f"GradCAM sample: {len(sample_items)} test images "
          f"({args.gradcam_per_class} per class, {'random' if args.gradcam_seed is None else f'seed={args.gradcam_seed}'})")

    for backbone_name, (best_seed, best_run_id) in best_run_ids.items():
        checkpoint_manager = CheckpointManager(config.output_root / "checkpoints", best_run_id)

        # ---- error handling: best run's checkpoint should always exist
        # right after run_all() finishes, but be defensive anyway (e.g.
        # someone re-runs just this GradCAM section later after manually
        # clearing outputs/checkpoints/ but keeping experiment_log.csv) ----
        if not checkpoint_manager.has_checkpoint():
            print(f"  WARNING: no checkpoint found for {backbone_name}'s best run "
                  f"(run_id='{best_run_id}') under {config.output_root / 'checkpoints'} — "
                  f"skipping GradCAM for {backbone_name}.")
            continue

        # FIXED: read THIS run's own saved head_hidden_dim/head_dropout
        # instead of trusting config.head_hidden_dim/config.head_dropout.
        # Each backbone gets its own Optuna-tuned hyperparameters (see
        # "Loaded tuned hyperparameters for <backbone>: {...}" at the start
        # of each architecture's block above) which can differ from the
        # global config defaults — e.g. vgg16 was actually trained with
        # head_hidden_dim=128, not whatever config.head_hidden_dim happens
        # to be. Building the model with the wrong dims here causes a
        # state_dict shape mismatch when loading the checkpoint below.
        # logs/<run_id>_config.json already records the EXACT values used
        # to train this specific run, so read from there instead.
        run_config_path = config.output_root / "logs" / f"{best_run_id}_config.json"
        try:
            with open(run_config_path) as f:
                run_config = json.load(f)
            head_hidden_dim = run_config.get("head_hidden_dim", config.head_hidden_dim)
            head_dropout = run_config.get("head_dropout", config.head_dropout)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  WARNING: could not read {run_config_path} ({e}) — falling back to "
                  f"config.head_hidden_dim/head_dropout, which may not match this run's "
                  f"actual architecture and could cause a checkpoint load failure below.")
            head_hidden_dim = config.head_hidden_dim
            head_dropout = config.head_dropout

        model = CNNClassifier(
            backbone_name=backbone_name,
            num_classes=data_module.num_classes,
            head_hidden_dim=head_hidden_dim,
            head_dropout=head_dropout,
            pretrained=False,  # weights get overwritten by the checkpoint right below
        )

        try:
            if checkpoint_manager.has_best_checkpoint():
                ckpt = checkpoint_manager.load_best(map_location="cpu")
            else:
                print(f"  WARNING: run_id='{best_run_id}' has no best.pt (only latest.pt) — "
                      f"using the latest checkpoint instead for GradCAM.")
                ckpt = checkpoint_manager.load_latest(map_location="cpu")
            model.load_state_dict(ckpt["model_state"])
        except (RuntimeError, KeyError, OSError) as e:
            print(f"  WARNING: failed to load checkpoint for {backbone_name}'s best run "
                  f"(run_id='{best_run_id}'): {e} — skipping GradCAM for {backbone_name}.")
            continue

        model.to(device)
        model.eval()

        gradcam = GradCAM(model)
        out_dir = config.output_root / backbone_name / "gradcam"
        out_dir.mkdir(parents=True, exist_ok=True)
        # the sample is random, so clear GradCAM images from earlier runs of this
        # script (otherwise old and new pictures would pile up in the same folders)
        for old_png in out_dir.rglob("gradcam_*.png"):
            old_png.unlink()

        for true_class, img_path in sample_items:
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            raw_np = raw_transform(img).permute(1, 2, 0).numpy()
            input_tensor = eval_transform(img).unsqueeze(0)

            # one sub-folder per TRUE class, e.g. gradcam/mera_mera/gradcam_<image>.png
            # (the overlay title shows what the model PREDICTED)
            class_dir = out_dir / true_class
            class_dir.mkdir(parents=True, exist_ok=True)
            fname = Path(img_path).stem
            gradcam.visualize(
                input_tensor, raw_np, data_module.class_names,
                save_path=class_dir / f"gradcam_{fname}.png",
            )

        best_run_metrics = next(r for r in results[backbone_name] if r["seed"] == best_seed)
        print(f"  GradCAM done for {backbone_name} using best run "
              f"(run_id='{best_run_id}', {PRIMARY_METRIC}={best_run_metrics[PRIMARY_METRIC]:.4f})")

    print(f"\nAll outputs saved under {config.output_root}/")


if __name__ == "__main__":
    main()