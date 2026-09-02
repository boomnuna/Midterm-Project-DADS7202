"""
scripts/run_eda.py

Assignment section 1 (Dataset description / EDA).

Usage:
    python -m scripts.run_eda
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import Config
from src.data.dataset import DataModule
from src.data.eda import EDAAnalyzer


# run full pipeline for EDA data 
def main():
    config = Config()
    data_module = DataModule(config)

    print(f"Classes found: {data_module.class_names}")
    print(f"Number of classes: {data_module.num_classes}\n")

    analyzer = EDAAnalyzer(config.data_root, data_module.class_names)
    report = analyzer.analyze()
    analyzer.print_summary(report)

    plot_path = config.output_root / "eda_summary.png"
    analyzer.plot_summary(report, save_path=plot_path)


if __name__ == "__main__":
    main()
