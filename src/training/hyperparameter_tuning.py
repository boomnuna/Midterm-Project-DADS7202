"""
What hyperparameters should I use for this model?
"""

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

PRUNING (NEW): trial.report()/should_prune() now happen INSIDE
Trainer.fit(), once per epoch, instead of once here after training
already finished. That's what actually lets MedianPruner cut off a
clearly-bad trial partway through instead of always paying for the
full optuna_quick_epochs on every trial. See trainer.py's
_report_to_trial_and_maybe_prune(). Note the per-epoch pruning signal
is val_acc (tracked every epoch already inside fit()), while the final
objective returned below is still val f1_macro — that's fine, they
don't need to be the same metric.
"""


import copy
import optuna

from src.utils.seed import set_seed
from src.models.classifier import CNNClassifier
from src.training.trainer import Trainer
from src.evaluation.metrics import Evaluator

# hyperparameter tuning with optuna
class OptunaTuner:
    def __init__(self, base_config, data_module, backbone_name: str):
        self.base_config = base_config
        self.data_module = data_module
        self.backbone_name = backbone_name

    def _objective(self, trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(self.base_config)

        # ---- search space — adjust ranges based on what you observe ----
        # define value that allow to try
        cfg.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True) # learning rate
        cfg.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True) # weight decay
        cfg.head_dropout = trial.suggest_float("head_dropout", 0.1, 0.5) # num of dropout
        cfg.head_hidden_dim = trial.suggest_categorical("head_hidden_dim", [128, 256, 512]) # hidden layer size
        cfg.optimizer_name = trial.suggest_categorical("optimizer_name", ["adamw", "sgd"]) # optimizer type
        # NEW: how many of the backbone's last blocks get unfrozen during
        # fine-tuning. Only meaningful when cfg.training_mode == "finetune"
        # (it is, for every backbone in this project) — different
        # backbones/depths may want a different amount unfrozen.
        cfg.finetune_unfreeze_last_n_blocks = trial.suggest_int("finetune_unfreeze_last_n_blocks", 1, 4)
        cfg.num_epochs = self.base_config.optuna_quick_epochs  # short runs during search (don't use all epoch for trail)

        set_seed(cfg.base_seed)
        # handle class imbalance
        class_weights = None
        if cfg.imbalance_strategy == "class_weights":
            class_weights = self.data_module.class_weights()

        # create the model
        model = CNNClassifier(
            self.backbone_name, self.data_module.num_classes,
            head_hidden_dim=cfg.head_hidden_dim, head_dropout=cfg.head_dropout,
        )

        # create Trainer
        trainer = Trainer(model, cfg, class_weights=class_weights)

        # train model — pass `trial` through so fit() can report val_acc
        # to Optuna every epoch and raise optuna.TrialPruned() early if
        # this trial is clearly underperforming (see trainer.py). That
        # exception propagates straight out of this function, which is
        # exactly what optuna.Study.optimize() expects: it catches
        # TrialPruned itself and records the trial as pruned, then moves
        # on to the next one — nothing extra needed here for that part.
        trainer.fit(self.data_module.train_loader(), self.data_module.val_loader(),
                    verbose=False, trial=trial)

        # evaluate validation performance
        evaluator = Evaluator(self.data_module.class_names)
        val_metrics = evaluator.evaluate(model, self.data_module.val_loader(), trainer.device)

        return val_metrics["f1_macro"]

    # starts the whole search
    def run(self, n_trials: int = None, storage: str = None, study_name: str = None) -> dict:
        n_trials = n_trials or self.base_config.optuna_n_trials # If don't provide n_trials, it uses your config.
        storage = storage or self.base_config.optuna_storage
        study_name = study_name or f"{self.backbone_name}_tuning"

        # creates Optuna experiment.
        study = optuna.create_study(
            direction="maximize",
            storage=storage,
            study_name=study_name,
            load_if_exists=True,  # lets multiple people/runs contribute to the same study
            pruner=optuna.pruners.MedianPruner(),
        )

        # run the trials
        study.optimize(self._objective, n_trials=n_trials)

        print(f"\nBest validation F1-score for {self.backbone_name}: {study.best_value:.4f}")
        print(f"Best hyperparameters: {study.best_params}")
        print(f"Total trials in this study so far: {len(study.trials)}")

        return study.best_params
