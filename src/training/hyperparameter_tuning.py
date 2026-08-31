"""
src/training/hyperparameter_tuning.py

Optuna integration. OptunaTuner searches over a fixed hyperparameter
space using SHORT training runs (config.optuna_quick_epochs, not the
full config.num_epochs) to keep the search itself affordable, then you
train your FINAL reported models (via ExperimentRunner) using the best
params found here, for the full epoch budget.

GROUP USE: pass a shared `storage` (e.g. a database URL, or even a
SQLite file on a shared drive) and the same `study_name` so multiple
group members' searches accumulate into ONE study — Optuna will not
repeat trials another member already ran once they share the same
storage. See README for why W&B Sweep is usually simpler for this in
practice (SQLite over a shared drive can have write-lock issues; W&B's
cloud storage doesn't).
"""

import copy
import optuna

from src.utils.seed import set_seed
from src.models.classifier import CNNClassifier
from src.training.trainer import Trainer
from src.evaluation.metrics import Evaluator


class OptunaTuner:
    def __init__(self, base_config, data_module, backbone_name: str):
        self.base_config = base_config
        self.data_module = data_module
        self.backbone_name = backbone_name

    def _objective(self, trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(self.base_config)

        # ---- search space — adjust ranges based on what you observe ----
        cfg.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        cfg.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        cfg.head_dropout = trial.suggest_float("head_dropout", 0.1, 0.5)
        cfg.head_hidden_dim = trial.suggest_categorical("head_hidden_dim", [128, 256, 512])
        cfg.optimizer_name = trial.suggest_categorical("optimizer_name", ["adamw", "sgd"])
        cfg.num_epochs = self.base_config.optuna_quick_epochs  # short runs during search

        set_seed(cfg.base_seed)
        class_weights = None
        if cfg.imbalance_strategy == "class_weights":
            class_weights = self.data_module.class_weights()

        model = CNNClassifier(
            self.backbone_name, self.data_module.num_classes,
            head_hidden_dim=cfg.head_hidden_dim, head_dropout=cfg.head_dropout,
        )
        trainer = Trainer(model, cfg, class_weights=class_weights)
        trainer.fit(self.data_module.train_loader(), self.data_module.val_loader(), verbose=False)

        evaluator = Evaluator(self.data_module.class_names)
        val_metrics = evaluator.evaluate(model, self.data_module.val_loader(), trainer.device)

        # report intermediate value so trial can be pruned early if it's
        # clearly underperforming (saves time across many trials)
        trial.report(val_metrics["accuracy"], step=cfg.num_epochs)

        return val_metrics["accuracy"]

    def run(self, n_trials: int = None, storage: str = None, study_name: str = None) -> dict:
        n_trials = n_trials or self.base_config.optuna_n_trials
        storage = storage or self.base_config.optuna_storage
        study_name = study_name or f"{self.backbone_name}_tuning"

        study = optuna.create_study(
            direction="maximize",
            storage=storage,
            study_name=study_name,
            load_if_exists=True,  # lets multiple people/runs contribute to the same study
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(self._objective, n_trials=n_trials)

        print(f"\nBest validation accuracy for {self.backbone_name}: {study.best_value:.4f}")
        print(f"Best hyperparameters: {study.best_params}")
        print(f"Total trials in this study so far: {len(study.trials)}")

        return study.best_params
