"""
scripts/tune_hyperparameters.py

Runs Optuna hyperparameter search for ONE backbone at a time (run it
once per architecture you're comparing). Prints and saves the best
hyperparameters found — copy those into config.py before your final
run_all_experiments.py run for that architecture.

Usage:
    python -m scripts.tune_hyperparameters --backbone resnet50 --n-trials 20

    # to share a study across group members (see README for setup):
    python -m scripts.tune_hyperparameters --backbone resnet50 \\
        --storage "sqlite:///outputs/optuna_study.db" --study-name resnet50_group_search
"""

import sys
import json
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import Config
from src.data.dataset import DataModule
from src.training.hyperparameter_tuning import OptunaTuner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--storage", default=None)
    parser.add_argument("--study-name", default=None)
    args = parser.parse_args()

    config = Config()
    data_module = DataModule(config)

    tuner = OptunaTuner(config, data_module, backbone_name=args.backbone)
    best_params = tuner.run(n_trials=args.n_trials, storage=args.storage, study_name=args.study_name)

    out_path = config.output_root / f"best_params_{args.backbone}.json"
    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nSaved best params to {out_path}")
    print("Copy these values into config.py (or a per-backbone override) "
          "before running the full multi-repeat experiment.")


if __name__ == "__main__":
    main()
