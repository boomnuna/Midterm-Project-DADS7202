"""
scripts/run_all_experiments.py

THE main script for the assignment deliverables — runs every backbone in
config.backbone_names, each repeated config.num_repeats times, then
produces:
  - mean±SD table (assignment section 6)
  - pairwise statistical significance (assignment section 6)
  - training curves + confusion matrix per architecture (section 6)
  - GradCAM examples using the best run of each architecture (section 7)

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
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

from config import Config
from src.data.dataset import DataModule
from src.training.repeated_runs import ExperimentRunner
from src.evaluation.metrics import Evaluator
from src.evaluation.statistics import StatisticalComparator
from src.interpretability.gradcam import GradCAM
from src.data.transforms import TransformFactory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", nargs="+", default=None,
                         help="Run only these backbones this session (default: all in config.py). "
                              "Useful for splitting work across Colab sessions/group members.")
    args = parser.parse_args()

    config = Config()
    if args.backbones:
        config.backbone_names = args.backbones

    data_module = DataModule(config)
    print(f"Classes: {data_module.class_names}")
    print(f"Architectures to compare THIS RUN: {config.backbone_names}")
    print(f"Repeats per architecture: {config.num_repeats}")
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
    comparator.print_mean_std_table()
    print()
    comparator.print_pairwise_significance(metric_name="accuracy")

    best_name = comparator.best_architecture(metric_name="accuracy")
    print(f"\nBest architecture by mean accuracy: {best_name}")

    # ---- plots for the best run of each architecture ----
    evaluator = Evaluator(data_module.class_names)
    for backbone_name, runs in results.items():
        best_run = max(runs, key=lambda r: r["accuracy"])
        out_dir = config.output_root / backbone_name
        out_dir.mkdir(parents=True, exist_ok=True)

        evaluator.plot_training_curves(
            best_run["history"], title=f"{backbone_name} (best run)",
            save_path=out_dir / "training_curves_best_run.png",
        )
        evaluator.plot_confusion_matrix(
            best_run["confusion_matrix"], title=f"{backbone_name} (best run)",
            save_path=out_dir / "confusion_matrix_best_run.png",
        )

    # ---- GradCAM on a few test images, using each architecture's last-trained model ----
    print("\nGenerating GradCAM examples...")
    raw_transform = TransformFactory(config.image_size).raw_transform()
    eval_transform = TransformFactory(config.image_size).eval_transform()

    # grab a small fixed sample of test images to keep GradCAM comparable across architectures
    sample_paths = []
    test_dataset = data_module.test_dataset
    underlying = test_dataset.dataset if hasattr(test_dataset, "dataset") else test_dataset
    for i in range(min(5, len(test_dataset))):
        idx = test_dataset.indices[i] if hasattr(test_dataset, "indices") else i
        sample_paths.append(underlying.samples[idx][0])

    for backbone_name, model in runner.last_models.items():
        gradcam = GradCAM(model)
        out_dir = config.output_root / backbone_name / "gradcam"
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sample_paths:
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            raw_np = raw_transform(img).permute(1, 2, 0).numpy()
            input_tensor = eval_transform(img).unsqueeze(0)

            fname = Path(img_path).stem
            gradcam.visualize(
                input_tensor, raw_np, data_module.class_names,
                save_path=out_dir / f"gradcam_{fname}.png",
            )

    print(f"\nAll outputs saved under {config.output_root}/")


if __name__ == "__main__":
    main()
