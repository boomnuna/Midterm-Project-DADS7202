"""
scripts/find_best_run.py

Closes the loop the logging system exists for: "which run had the best
result, and what EXACT settings do I need to reproduce it?"

Reads outputs/experiment_log.csv (the summary of every run ever), finds
the best row by a chosen metric, then loads that run's full config
snapshot (outputs/logs/<run_id>_config.json) and prints it as ready-to-
paste Python — copy that into config.py (or a notebook cell) to retrain
with EXACTLY those settings.

Usage:
    python -m scripts.find_best_run
    python -m scripts.find_best_run --metric f1_macro --backbone resnet50
"""

import sys
import json
import argparse
import csv
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config


def main():
    parser = argparse.ArgumentParser() # This allows you to run the script like: python find_best.py --metric accuracy
    parser.add_argument("--metric", default="accuracy", # which metrics you use to  define best  
                         help="Which column in experiment_log.csv to maximize (default: accuracy)")
    parser.add_argument("--backbone", default=None, # restrict the search that make you can run one CNN backbone for each time.
                         help="Restrict search to one backbone (default: search across all)")
    args = parser.parse_args() # parse the arguments

    config = Config()
    csv_path = config.output_root / "experiment_log.csv" # log file path

    # if no excel file
    if not csv_path.exists():
        print(f"No {csv_path} found yet — run some experiments first "
              f"(train_single.py or run_all_experiments.py).")
        return

    # read file 
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    # in-case you run 'python find_best.py --backbone resnet50' it will filter only 'resnet50'
    if args.backbone:
        rows = [r for r in rows if r["backbone"] == args.backbone]

    # if it empty file 
    if not rows:
        print("No matching rows found in experiment_log.csv.")
        return

    # find best row 
    best_row = max(rows, key=lambda r: float(r[args.metric]))
    run_id = best_row["run_id"]

    # print best result
    print("=" * 60)
    print(f"BEST RUN by {args.metric}: {run_id}")
    print("=" * 60)
    print(f"  {args.metric} = {best_row[args.metric]}")
    print(f"  backbone      = {best_row['backbone']}")
    print(f"  seed          = {best_row['seed']}")

    config_path = config.output_root / "logs" / f"{run_id}_config.json"

    # if config path not exist 
    if not config_path.exists():
        print(f"\nNOTE: {config_path} not found — this run predates the "
              f"full-config logging update, or its logs were deleted. "
              f"Only the summary row above is available; full hyperparameters "
              f"for this specific run can't be recovered.")
        return

    # load config file with match best result  
    with open(config_path) as f:
        run_config = json.load(f)

    print(f"\nFull config for this run (from {config_path}):\n")
    print("-" * 60)
    print("# Paste into config.py's Config dataclass defaults, or set on")
    print("# a Config() instance before running scripts.train_single:")
    for key, value in run_config.items():
        if key in ("data_root", "output_root"):
            continue  # paths — usually don't want to blindly copy these across machines
        print(f"config.{key} = {value!r}")
    print("-" * 60)
    print(f"\nTo retrain this exact configuration:")
    print(f"  python -m scripts.train_single --backbone {best_row['backbone']} --seed {best_row['seed']}")
    print(f"  (after applying the config values printed above, if they differ from your current config.py)")


if __name__ == "__main__":
    main()



"""
experiment_log.csv
       ↓
Filter by backbone (optional)
       ↓
Find highest accuracy
       ↓
Get run_id
       ↓
Load {run_id}_config.json
       ↓
Print the configuration
       ↓
Tell you how to retrain it
"""