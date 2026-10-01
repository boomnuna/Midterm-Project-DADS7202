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

SCHEMA STABILITY (NEW): log_run_summary() used to build a fresh CSV
header from whatever keys happened to be in each record, every single
call. If two runs ever logged records with different sets of fields
(e.g. a hyperparameter like finetune_unfreeze_last_n_blocks got added
to summary_row after some runs were already logged, or train_single.py
logs a differently-shaped record into the same shared CSV), later rows
would silently land under the WRONG column headers — csv.DictWriter
doesn't check that a row's keys still match the header the file was
first created with. Now log_run_summary() reads the file's ACTUAL
existing header first and always writes against THAT (extra fields in
a record are dropped, missing ones written blank), with a warning
either way — so the CSV can never get silently misaligned again, even
if it's a little lossy for the odd mismatched run in the meantime.
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

    def log_config(self, config_dict: dict):
        """
        Logs the FULL hyperparameter/setting snapshot for this run — both
        as readable lines in the .log file (so you can just open it and
        see exactly what was used) AND as a standalone JSON file
        (outputs/logs/<run_id>_config.json), which is the file to diff
        against if you're trying to figure out exactly what differed
        between two runs, or to reproduce a specific run exactly.

        Call this once, right after starting a run, before training.
        """
        self.info("=" * 50)
        self.info("FULL CONFIG FOR THIS RUN:")
        for key, value in config_dict.items():
            self.info(f"  {key}: {value}")
        self.info("=" * 50)
        self.save_json(config_dict, filename=f"{self.run_id}_config.json")

    # ---- structured helpers used by Trainer/ExperimentRunner ----
    def log_epoch(self, epoch: int, train_loss: float, train_acc: float,
                  val_loss: float, val_acc: float, epoch_time: float = None):
        time_str = f" | time={epoch_time:.1f}s" if epoch_time is not None else ""
        self.info(f"epoch {epoch:3d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}{time_str}")

    # ------------------------------------------------------------
    # NEW: figure out what fieldnames to write THIS row with. If the CSV
    # already exists, reads its real header off disk rather than trusting
    # this call's own record.keys() — that's what keeps every row aligned
    # to the SAME columns no matter how summary_row's shape has drifted
    # over the life of the project.
    def _resolve_fieldnames(self, record: dict) -> list:
        if not self.summary_csv_path.exists():
            return list(record.keys())

        try:
            with open(self.summary_csv_path, "r", newline="") as f:
                existing_header = next(csv.reader(f), None)
        except OSError as e:
            self.warning(f"log_run_summary(): could not read existing header from "
                         f"{self.summary_csv_path} ({e}) — falling back to this record's "
                         f"own keys as the header. If the file already has a different "
                         f"column order, this row may not line up with earlier ones.")
            return list(record.keys())

        if not existing_header:
            # file exists but is empty (e.g. created then never written to) — safe to
            # treat this record's keys as the header, same as a brand-new file.
            return list(record.keys())

        missing_from_record = [k for k in existing_header if k not in record]
        extra_in_record = [k for k in record if k not in existing_header]
        if extra_in_record:
            self.warning(f"log_run_summary(): this run's record has field(s) not in "
                         f"{self.summary_csv_path}'s existing header: {extra_in_record}. "
                         f"These values will be DROPPED from this row so the CSV stays "
                         f"column-aligned with earlier rows (e.g. a hyperparameter added "
                         f"after earlier runs were already logged). If you want them kept, "
                         f"back up/delete {self.summary_csv_path} to start a fresh file "
                         f"with the new schema.")
        if missing_from_record:
            self.warning(f"log_run_summary(): this run's record is missing field(s) that "
                         f"{self.summary_csv_path}'s existing header expects: "
                         f"{missing_from_record}. They'll be written blank for this row.")
        return existing_header

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
        fieldnames = self._resolve_fieldnames(record)

        with open(self.summary_csv_path, "a", newline="") as f:
            # extrasaction="ignore": drop any record key not in fieldnames
            # (already warned about above) instead of raising ValueError.
            # restval="": any fieldname missing from record (already
            # warned about above) is written as a blank cell instead of
            # raising KeyError.
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
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
