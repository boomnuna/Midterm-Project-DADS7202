"""
src/utils/logger.py

Two layers of logging, both plain local files (no external service
required — this is what still works even if Optuna/W&B aren't set up
yet, or the wifi in the room dies during a live demo):

  1. Per-run text log  -> outputs/logs/<run_id>.log
     Human-readable, timestamped. Use this when ONE specific run
     crashed or behaved weirdly and you need the full detail.

  2. Shared summary CSV -> outputs/experiment_log.csv
     ONE ROW appended per completed run, across every experiment you've
     EVER run (single runs, full sweeps, everything). This is the file
     you actually open in Excel/pandas to compare all your experiments
     at a glance — e.g. `pd.read_csv("outputs/experiment_log.csv")` and
     sort by accuracy — without re-running or re-parsing anything.
"""

import csv
import json
import logging
from pathlib import Path
from datetime import datetime

# records what happened during each experiment
class ExperimentLogger:
    def __init__(self, output_root: Path, run_id: str):
        self.output_root = Path(output_root)
        self.run_id = run_id

        self.log_dir = self.output_root / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.summary_csv_path = self.output_root / "experiment_log.csv"

        self._logger = logging.getLogger(f"exp.{run_id}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._logger.handlers.clear()

        file_handler = logging.FileHandler(self.log_dir / f"{run_id}.log")
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self._logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(console_handler)

    # ---- plain logging passthroughs ----
    def info(self, msg: str):
        self._logger.info(msg)

    def warning(self, msg: str):
        self._logger.warning(msg)

    def error(self, msg: str):
        self._logger.error(msg)

    # ---- structured helpers used by Trainer/ExperimentRunner ----
    def log_epoch(self, epoch: int, train_loss: float, train_acc: float,
                  val_loss: float, val_acc: float):
        self.info(f"epoch {epoch:3d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    def log_run_summary(self, record: dict):
        """
        Appends one row to the shared CSV. `record` should be a FLAT dict
        of scalar values only (no nested dicts/arrays like confusion
        matrices or full histories — those stay in the per-run .log file
        and separate plot files, since they don't belong in a comparison
        spreadsheet).
        """
        record = {"timestamp": datetime.now().isoformat(timespec="seconds"), **record}
        file_exists = self.summary_csv_path.exists()

        with open(self.summary_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

        self.info(f"Logged run summary -> {self.summary_csv_path}")

    def save_json(self, data: dict, filename: str):
        """For saving anything structured that doesn't fit the flat CSV row
        (e.g. full hyperparameter dict, per-class metrics)."""
        path = self.log_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        self.info(f"Saved {path}")



"""
  ExperimentRunner
                          │
                          ↓
                       Trainer
                          │
                ┌─────────┴─────────┐
                ↓                   ↓
           Train model         Evaluate model
                │                   │
                └─────────┬─────────┘
                          ↓
                   ExperimentLogger
                          │
             ┌────────────┼────────────┐
             ↓            ↓            ↓
          .log file   CSV summary    JSON
"""