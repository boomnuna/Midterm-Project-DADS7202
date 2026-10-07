"""
scripts/run_eda.py

Assignment section 1 (Dataset description / EDA).

Always analyzes the dataset fresh (nothing is loaded from a previous run),
then SAVES the results so you don't have to re-run just to see them again:

  - outputs/eda_summary.txt  the printed summary table, as plain text
  - outputs/eda_summary.png  the plot

Each run overwrites these two files with the results for the current dataset.

Usage:
    python -m scripts.run_eda
"""

import sys
import io
import contextlib
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import Config
from src.data.dataset import DataModule
from src.data.eda import EDAAnalyzer


# run full pipeline for EDA data
def main():
    print("-"*70)
    config = Config()
    data_module = DataModule(config)
    config.output_root.mkdir(parents=True, exist_ok=True)

    summary_path = config.output_root / "eda_summary.txt"
    plot_path = config.output_root / "eda_summary.png"

    print(f"Classes found: {data_module.class_names}")
    print(f"Number of classes: {data_module.num_classes}")
    print("-"*70)

    print("Starting dataset analysis...")
    analyzer = EDAAnalyzer(config.data_root, data_module.class_names)
    report = analyzer.analyze()

    # capture the printed summary so it is both shown here and saved as text
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        analyzer.print_summary(report)
    summary_text = buffer.getvalue()
    print(summary_text)
    try:
        summary_path.write_text(summary_text, encoding="utf-8")
        print(f"Saved summary text to {summary_path}")
    except OSError as e:
        print(f"WARNING: could not save summary text to {summary_path} ({e})")

    print("plotting graph...")
    analyzer.plot_summary(report, save_path=plot_path)
    print("-"*70)


if __name__ == "__main__":
    main()