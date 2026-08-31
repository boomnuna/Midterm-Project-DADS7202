"""
scripts/wandb_sweep_train.py

This is the script `wandb agent` actually runs, over and over, once per
sweep trial (see configs/wandb_sweep.yaml). Each invocation:
  1. wandb.init() — the sweep agent injects the sampled hyperparameters
     into wandb.config automatically, no argparse needed for those
  2. builds a Config with those values overridden
  3. trains ONE model, ONE seed (fast — this is a search run, not a
     final reported result)
  4. logs val_accuracy per epoch so the sweep's Bayesian search can learn
     from it

Usage (do NOT run this directly for a solo test — see the note below):
    wandb agent <SWEEP_ID>

For a one-off local test without a real sweep controller:
    python -m scripts.wandb_sweep_train
(this uses wandb's default/random config values since no sweep is driving it)
"""

import sys
import copy
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import wandb

from config import Config
from src.utils.seed import set_seed
from src.data.dataset import DataModule
from src.models.classifier import CNNClassifier
from src.training.trainer import Trainer
from src.evaluation.metrics import Evaluator


def main():
    run = wandb.init()
    sweep_config = run.config  # hyperparameters sampled by the sweep controller

    config = Config()
    config.learning_rate = sweep_config.get("learning_rate", config.learning_rate)
    config.weight_decay = sweep_config.get("weight_decay", config.weight_decay)
    config.head_dropout = sweep_config.get("head_dropout", config.head_dropout)
    config.head_hidden_dim = sweep_config.get("head_hidden_dim", config.head_hidden_dim)
    config.optimizer_name = sweep_config.get("optimizer_name", config.optimizer_name)
    config.num_epochs = config.optuna_quick_epochs  # keep sweep trials short, same idea as Optuna path
    backbone_name = sweep_config.get("backbone_name", config.backbone_names[0])

    set_seed(config.base_seed)
    data_module = DataModule(config)

    class_weights = None
    if config.imbalance_strategy == "class_weights":
        class_weights = data_module.class_weights()

    model = CNNClassifier(
        backbone_name, data_module.num_classes,
        head_hidden_dim=config.head_hidden_dim, head_dropout=config.head_dropout,
    )

    trainer = Trainer(model, config, class_weights=class_weights, wandb_run=run)
    trainer.fit(data_module.train_loader(), data_module.val_loader())

    evaluator = Evaluator(data_module.class_names)
    val_metrics = evaluator.evaluate(model, data_module.val_loader(), trainer.device)

    # this is the metric name declared in wandb_sweep.yaml's `metric.name`
    run.log({"val_accuracy": val_metrics["accuracy"]})
    run.finish()


if __name__ == "__main__":
    main()
