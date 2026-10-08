"""
src/training/repeated_runs.py

Directly implements the assignment's most distinctive requirement:
  "train model A repeatedly from scratch, 3-10 times, same architecture/
   hyperparameters but different initial weights, report mean±SD"

ExperimentRunner does this for EVERY backbone in config.backbone_names,
so the end result is a nested structure ready for both the mean±SD
table and the statistical significance test (see evaluation/statistics.py).

RESUME SUPPORT (for Colab): before starting each (backbone, seed) run,
checks whether that run's metrics were already saved to disk from a
PREVIOUS session — if so, loads the saved result instead of retraining.
Combined with Trainer's per-epoch checkpointing, this means a Colab
disconnect at ANY point costs at most the current epoch's progress,
never a fully-completed run.

Every run also gets:
  - its own text log file (outputs/logs/<backbone>_seed<N>.log)
  - one row appended to the shared outputs/experiment_log.csv
  - a per-run checkpoint folder (outputs/checkpoints/<run_id>/)
  - an optional W&B run, if config.use_wandb is True (see README for setup)

PER-BACKBONE TUNED HYPERPARAMETERS (NEW):
Before training each backbone, run_all() looks for
outputs/best_params_<backbone>.json — the file scripts/tune_hyperparameters.py
writes after an Optuna search for that backbone. If found, those values
override config.py's defaults for THAT backbone only (every other backbone
keeps its own independent copy of the config, unaffected). If no tuned
file exists yet for a backbone, it just trains with config.py's defaults
and prints a note saying so — nothing crashes or blocks on a missing file.
"""

import json
import copy
import dataclasses
from pathlib import Path

import torch

from src.utils.seed import set_seed
from src.utils.logger import ExperimentLogger
from src.utils.checkpoint import CheckpointManager
from src.models.classifier import CNNClassifier
from src.training.trainer import Trainer
from src.evaluation.metrics import Evaluator

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class ExperimentRunner:
    def __init__(self, config, data_module):
        self.config = config
        self.data_module = data_module
        # results[backbone_name] = list of per-run metric dicts (one dict per repeat)
        self.results = {}
        self.last_models = {}

        if config.use_wandb and not _WANDB_AVAILABLE:
            print("WARNING: config.use_wandb=True but the 'wandb' package isn't installed. "
                  "Run `pip install wandb` or set use_wandb=False. Continuing without W&B.")

    # metrics file path
    def _completed_metrics_path(self, run_id: str) -> Path:
        return self.config.output_root / "logs" / f"{run_id}_metrics.json"

    # load metrics file path
    def _load_completed_run(self, run_id: str, backbone_name: str, cfg) -> dict:
        """
        Loads a previously-saved run's result. <run_id>_metrics.json alone
        doesn't include confusion_matrix/history (those are deliberately
        left out of that file — see the save_json call in run_all()), so
        this reconstructs both from that run's saved CHECKPOINT instead of
        requiring a full retrain:

          - history: stored directly inside the checkpoint already (Trainer
            saves self.history every epoch) — just read it back, no
            recomputation needed.
          - confusion_matrix: NOT stored in the checkpoint, so it's
            recomputed by reloading the checkpointed model weights and
            re-running evaluation on the test set. This is cheap (one
            forward pass over the test set, no training) compared to
            retraining the whole run from scratch.

        This is what lets run_all_experiments.py's plotting/GradCAM code
        treat every run uniformly — freshly trained or resumed from a
        previous session — without ever needing to retrain just to get a
        training-curve plot or confusion matrix back.
        """
        with open(self._completed_metrics_path(run_id)) as f:
            test_metrics = json.load(f)

        checkpoint_manager = CheckpointManager(self.config.output_root / "checkpoints", run_id)
        if not checkpoint_manager.has_checkpoint():
            print(f"  WARNING: '{run_id}' is marked completed but has no saved checkpoint — "
                  f"'history'/'confusion_matrix' will be missing from its result, which will "
                  f"break plotting/GradCAM for this run. (Did outputs/checkpoints/{run_id} "
                  f"get deleted after this run finished?)")
            return test_metrics

        try:
            ckpt = (checkpoint_manager.load_best(map_location="cpu")
                    if checkpoint_manager.has_best_checkpoint()
                    else checkpoint_manager.load_latest(map_location="cpu"))
            test_metrics["history"] = ckpt["history"]

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = CNNClassifier(
                backbone_name=backbone_name,
                num_classes=self.data_module.num_classes,
                head_hidden_dim=cfg.head_hidden_dim,
                head_dropout=cfg.head_dropout,
                pretrained=False,  # weights get overwritten by the checkpoint right below
            )
            model.load_state_dict(ckpt["model_state"])
            model.to(device)

            evaluator = Evaluator(self.data_module.class_names)
            recomputed = evaluator.evaluate(model, self.data_module.test_loader(), device)
            test_metrics["confusion_matrix"] = recomputed["confusion_matrix"]
        except (RuntimeError, KeyError, OSError) as e:
            print(f"  WARNING: could not reconstruct history/confusion_matrix for completed "
                  f"run '{run_id}' from its checkpoint ({e}) — plotting/GradCAM may fail or "
                  f"use incomplete data for this run.")

        return test_metrics

    # ------------------------------------------------------------
    # NEW: load this backbone's tuned hyperparameters (if any), with
    # defensive error handling so a missing/corrupt/stale JSON file can
    # NEVER crash or silently corrupt a multi-hour unattended training run
    # — worst case, it just falls back to config.py's defaults and prints
    # a clear warning explaining why.
    # ------------------------------------------------------------
    def _load_best_params(self, backbone_name: str) -> dict:
        path = self.config.output_root / f"best_params_{backbone_name}.json"

        if not path.exists():
            print(f"  (no tuned hyperparameters found for {backbone_name} at {path} "
                  f"— using config.py defaults instead)")
            return {}

        try:
            with open(path) as f:
                best_params = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  WARNING: {path} exists but is not valid JSON ({e}) — "
                  f"ignoring it and using config.py defaults for {backbone_name} instead. "
                  f"Check the file, or re-run tune_hyperparameters.py for this backbone.")
            return {}
        except OSError as e:
            print(f"  WARNING: could not read {path} ({e}) — "
                  f"using config.py defaults for {backbone_name} instead.")
            return {}

        if not isinstance(best_params, dict):
            print(f"  WARNING: {path} does not contain a JSON object (got "
                  f"{type(best_params).__name__} instead) — ignoring it and using "
                  f"config.py defaults for {backbone_name} instead.")
            return {}

        # Only accept keys that are actually real Config fields. A typo'd
        # or stale key (e.g. left over from an older search space in
        # hyperparameter_tuning.py) should never silently get set as a
        # bogus attribute on cfg — dataclasses allow setattr() for ANY
        # name, so without this check a mistake here would fail silently.
        valid_fields = {f.name for f in dataclasses.fields(self.config)}
        clean_params = {}
        for key, value in best_params.items():
            if key not in valid_fields:
                print(f"  WARNING: '{key}' in {path} is not a known Config field — "
                      f"skipping it (did the search space in "
                      f"hyperparameter_tuning.py change since this file was saved?)")
                continue
            clean_params[key] = value

        if clean_params:
            print(f"  Loaded tuned hyperparameters for {backbone_name}: {clean_params}")

        return clean_params

    def run_all(self, keep_last_model: bool = True):
        # loop through each transfer model
        for backbone_name in self.config.backbone_names:
            print(f"\n{'=' * 60}\nArchitecture: {backbone_name}\n{'=' * 60}")
            self.results[backbone_name] = []  # prepare result storage

            # ---- NEW: build a per-backbone config — starts as a copy of
            # the shared config, then gets this backbone's tuned
            # hyperparameters (if any) layered on top. Each backbone gets
            # its OWN independent copy, so tuning one never affects
            # another, and nothing here mutates self.config itself. ----
            best_params = self._load_best_params(backbone_name)
            cfg = copy.deepcopy(self.config)
            for key, value in best_params.items():
                setattr(cfg, key, value)

            # calculate class-weights (imbalance_strategy isn't part of the
            # Optuna search space, so it's identical across backbones —
            # fine to read from cfg here too, for consistency)
            class_weights = None
            if cfg.imbalance_strategy == "class_weights":
                class_weights = self.data_module.class_weights()

            for repeat_i in range(cfg.num_repeats):  # repeat each experiment
                seed = cfg.base_seed + repeat_i  # create different seeds
                run_id = f"{backbone_name}_seed{seed}"

                # ---- RESUME CHECK: was this run already completed in a
                # previous (now-disconnected) session? ----
                if self._completed_metrics_path(run_id).exists():  # check whether this experiment already finished
                    print(f"\n--- {run_id} — already completed, loading saved result, skipping retrain ---")
                    self.results[backbone_name].append(self._load_completed_run(run_id, backbone_name, cfg))
                    continue

                print(f"\n--- {run_id} ---")
                set_seed(seed)  # makes experiment reproducible.

                logger = ExperimentLogger(self.config.output_root, run_id)  # create logger
                logger.info(f"Starting run: backbone={backbone_name} seed={seed} "
                            f"training_mode={cfg.training_mode}")
                logger.log_config(cfg.to_dict())  # full snapshot -> reproducible later (now reflects tuned values)

                # create checkpoint manager
                checkpoint_manager = CheckpointManager(
                    self.config.output_root / "checkpoints", run_id
                )
                if checkpoint_manager.has_checkpoint():
                    logger.info("Found existing checkpoint for this run — will resume from it.")

                # (Optional) Weights & Biases
                wandb_run = None
                if cfg.use_wandb and _WANDB_AVAILABLE:
                    wandb_run = wandb.init(
                        project=cfg.wandb_project,
                        entity=cfg.wandb_entity,
                        group=backbone_name,
                        name=run_id,
                        config={
                            "backbone": backbone_name, "seed": seed,
                            "learning_rate": cfg.learning_rate,
                            "weight_decay": cfg.weight_decay,
                            "optimizer": cfg.optimizer_name,
                            "training_mode": cfg.training_mode,
                            "num_epochs": cfg.num_epochs,
                        },
                        reinit=True,
                    )

                # create the CNN model
                model = CNNClassifier(
                    backbone_name=backbone_name,
                    num_classes=self.data_module.num_classes,
                    head_hidden_dim=cfg.head_hidden_dim,
                    head_dropout=cfg.head_dropout,
                )

                # create the Trainer — note: cfg, not self.config, so this
                # backbone actually trains with ITS OWN tuned hyperparameters.
                # (Trainer.__init__ is what freezes the backbone, so it must be
                # created BEFORE model.summary() below — otherwise the summary's
                # "trainable" column wrongly shows every layer as trainable.)
                trainer = Trainer(model, cfg, class_weights=class_weights,
                                   logger=logger, wandb_run=wandb_run,
                                   checkpoint_manager=checkpoint_manager)

                # print the architecture summary once per backbone (not once
                # per repeat — same architecture every repeat, only seed
                # differs, so repeating this 5-10x would just be noise)
                if repeat_i == 0:
                    print(f"\nModel summary for {backbone_name} "
                          f"(training_mode={cfg.training_mode}):")
                    print(f"Trainable parameters: {model.trainable_parameter_count():,}")
                    summary_text = model.summary(image_size=cfg.image_size)
                    logger.save_json({"model_summary": summary_text},
                                     filename=f"{backbone_name}_model_summary.json")

                # train the model
                history = trainer.fit(
                    self.data_module.train_loader(),
                    self.data_module.val_loader(),
                )

                # evaluate on test set
                evaluator = Evaluator(self.data_module.class_names)
                test_metrics = evaluator.evaluate(model, self.data_module.test_loader(), trainer.device)

                # save training history
                test_metrics["history"] = history
                test_metrics["seed"] = seed

                # create summary
                summary_row = {
                    "run_id": run_id,
                    "backbone": backbone_name,
                    "seed": seed,
                    "training_mode": cfg.training_mode,
                    # ---- hyperparameters — this is what makes the CSV
                    # useful for "which settings gave the best result?"
                    # without having to open each run's individual JSON.
                    # These now reflect the TUNED values actually used for
                    # this backbone, not just config.py's shared defaults ----
                    "learning_rate": cfg.learning_rate,
                    "weight_decay": cfg.weight_decay,
                    "optimizer": cfg.optimizer_name,
                    "lr_scheduler": cfg.lr_scheduler,
                    "batch_size": cfg.batch_size,
                    "image_size": cfg.image_size,
                    "head_hidden_dim": cfg.head_hidden_dim,
                    "head_dropout": cfg.head_dropout,
                    "label_smoothing": cfg.label_smoothing,
                    "early_stopping_patience": cfg.early_stopping_patience,
                    "early_stopping_min_delta": cfg.early_stopping_min_delta,
                    "finetune_unfreeze_last_n_blocks": cfg.finetune_unfreeze_last_n_blocks,
                    "imbalance_strategy": cfg.imbalance_strategy,
                    # ---- results ----
                    "accuracy": test_metrics["accuracy"],
                    "precision_macro": test_metrics["precision_macro"],
                    "recall_macro": test_metrics["recall_macro"],
                    "f1_macro": test_metrics["f1_macro"],
                    "num_epochs_run": len(history["train_loss"]),
                    "final_train_loss": history["train_loss"][-1],
                    "final_val_loss": history["val_loss"][-1],
                    "total_training_time_sec": trainer.elapsed_seconds,
                    "avg_epoch_time_sec": sum(history["epoch_time"]) / len(history["epoch_time"]),
                }
                logger.log_run_summary(summary_row)
                # NOTE: this JSON file's existence is what run_all() checks
                # above to decide a run is "completed" — it's written LAST,
                # after training fully finishes, so a run interrupted
                # mid-training will correctly be seen as NOT completed
                # (and will resume via its checkpoint instead).

                # save metrics to JSON
                logger.save_json(
                    {k: v for k, v in test_metrics.items() if k not in ("confusion_matrix", "history")},
                    filename=f"{run_id}_metrics.json",
                )
                # sends final test metrics to W&B and closes the experiment.
                if wandb_run is not None:
                    wandb_run.log({
                        "final_test_accuracy": test_metrics["accuracy"],
                        "final_test_f1_macro": test_metrics["f1_macro"],
                    })
                    wandb_run.finish()

                # store results in memory
                self.results[backbone_name].append(test_metrics)

                # keep the last model
                if keep_last_model:
                    self.last_models[backbone_name] = model

        return self.results