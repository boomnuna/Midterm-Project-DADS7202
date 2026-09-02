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
"""

import json
from pathlib import Path

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
    def _load_completed_run(self, run_id: str) -> dict:
        """Loads a previously-saved run's metrics (without confusion_matrix/
        history, which weren't saved to JSON — see log_run_summary usage
        below). Good enough for the mean±SD/significance tables; if you
        need the confusion matrix or training curves for an ALREADY
        completed run, retrain it (or don't delete outputs/ between
        sessions if you'll want those plots)."""
        with open(self._completed_metrics_path(run_id)) as f:
            return json.load(f)


    def run_all(self, keep_last_model: bool = True):
        # calculate class-weights
        class_weights = None
        if self.config.imbalance_strategy == "class_weights":
            class_weights = self.data_module.class_weights()

        # loop through each transfer model
        for backbone_name in self.config.backbone_names:
            print(f"\n{'=' * 60}\nArchitecture: {backbone_name}\n{'=' * 60}")
            self.results[backbone_name] = [] # prepare result storage

            for repeat_i in range(self.config.num_repeats): # repeat each experiment
                seed = self.config.base_seed + repeat_i # create different seeds
                run_id = f"{backbone_name}_seed{seed}"

                # ---- RESUME CHECK: was this run already completed in a
                # previous (now-disconnected) session? ----
                if self._completed_metrics_path(run_id).exists(): # check whether this experiment already finished
                    print(f"\n--- {run_id} — already completed, loading saved result, skipping retrain ---")
                    self.results[backbone_name].append(self._load_completed_run(run_id))
                    continue

                print(f"\n--- {run_id} ---")
                set_seed(seed) # makes experiment reproducible.

                logger = ExperimentLogger(self.config.output_root, run_id) # create logger
                logger.info(f"Starting run: backbone={backbone_name} seed={seed} "
                            f"training_mode={self.config.training_mode}")

                # create checkpoint manager
                checkpoint_manager = CheckpointManager( 
                    self.config.output_root / "checkpoints", run_id
                )
                if checkpoint_manager.has_checkpoint():
                    logger.info("Found existing checkpoint for this run — will resume from it.")

                # (Optional) Weights & Biases
                wandb_run = None
                if self.config.use_wandb and _WANDB_AVAILABLE:
                    wandb_run = wandb.init(
                        project=self.config.wandb_project,
                        entity=self.config.wandb_entity,
                        group=backbone_name,
                        name=run_id,
                        config={
                            "backbone": backbone_name, "seed": seed,
                            "learning_rate": self.config.learning_rate,
                            "weight_decay": self.config.weight_decay,
                            "optimizer": self.config.optimizer_name,
                            "training_mode": self.config.training_mode,
                            "num_epochs": self.config.num_epochs,
                        },
                        reinit=True,
                    )

                # create the CNN model
                model = CNNClassifier(
                    backbone_name=backbone_name,
                    num_classes=self.data_module.num_classes,
                    head_hidden_dim=self.config.head_hidden_dim,
                    head_dropout=self.config.head_dropout,
                )

                # create the Trainer
                trainer = Trainer(model, self.config, class_weights=class_weights,
                                   logger=logger, wandb_run=wandb_run,
                                   checkpoint_manager=checkpoint_manager)

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
                    "training_mode": self.config.training_mode,
                    "accuracy": test_metrics["accuracy"],
                    "precision_macro": test_metrics["precision_macro"],
                    "recall_macro": test_metrics["recall_macro"],
                    "f1_macro": test_metrics["f1_macro"],
                    "num_epochs_run": len(history["train_loss"]),
                    "final_train_loss": history["train_loss"][-1],
                    "final_val_loss": history["val_loss"][-1],
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


"""
## Pipeline Conclusion

              run_all()
                       │
          ┌────────────┴────────────┐
          │                         │
    Backbone 1                Backbone 2
    ResNet18                 EfficientNet
          │                         │
    ┌─────┼─────┐             ┌─────┼─────┐
    ↓     ↓     ↓             ↓     ↓     ↓
 Seed42 Seed43 Seed44       Seed42 Seed43 Seed44
    │     │     │             │     │     │
    ↓     ↓     ↓             ↓     ↓     ↓
  Trainer Trainer Trainer   Trainer Trainer Trainer
    │
    ↓
  fit()
    │
    ├── _try_resume()
    │       │
    │       └── load_latest() ← checkpoint
    │
    ├── train epoch
    ├── validation
    ├── save checkpoint
    └── repeat
    │
    ↓
  Test evaluation
    │
    ↓
  Save metrics JSON
    │
    ↓
  self.results
"""

"""
This code automatically
1. Tries multiple CNN architectures
2. Repeats each architecture with different random seeds
3. Resumes interrupted experiments from checkpoints
4. Trains the model
5. Evaluates on the test set
6. Saves metrics
6. Logs to Weights & Biases if enabled
7. Keeps the trained model if requested
"""