"""
scripts/train_single.py

Quick single training run — one architecture, one seed. Use this while
developing/debugging the pipeline (fast feedback loop); use
run_all_experiments.py for the real assignment results (multi-backbone,
multi-repeat).

Usage:
    python -m scripts.train_single --backbone resnet50
"""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import Config
from src.utils.seed import set_seed
from src.utils.logger import ExperimentLogger
from src.utils.checkpoint import CheckpointManager
from src.data.dataset import DataModule
from src.models.classifier import CNNClassifier
from src.training.trainer import Trainer
from src.evaluation.metrics import Evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    config = Config()
    set_seed(args.seed)

    run_id = f"single_{args.backbone}_seed{args.seed}"
    logger = ExperimentLogger(config.output_root, run_id=run_id)
    logger.log_config(config.to_dict())
    checkpoint_manager = CheckpointManager(config.output_root / "checkpoints", run_id)
    if checkpoint_manager.has_checkpoint():
        logger.info("Found existing checkpoint — will resume from it (e.g. after a Colab disconnect).")

    data_module = DataModule(config)
    print(f"Classes: {data_module.class_names}")
    print(f"Class distribution (train): {data_module.class_distribution()}")

    class_weights = None
    if config.imbalance_strategy == "class_weights":
        class_weights = data_module.class_weights()
        print(f"Class weights: {class_weights}")

    model = CNNClassifier(
        backbone_name=args.backbone,
        num_classes=data_module.num_classes,
        head_hidden_dim=config.head_hidden_dim,
        head_dropout=config.head_dropout,
    )
    print(f"Trainable parameters: {model.trainable_parameter_count():,}")

    trainer = Trainer(model, config, class_weights=class_weights, logger=logger,
                       checkpoint_manager=checkpoint_manager)
    history = trainer.fit(data_module.train_loader(), data_module.val_loader())

    evaluator = Evaluator(data_module.class_names)
    test_metrics = evaluator.evaluate(model, data_module.test_loader(), trainer.device)

    print(f"\nTest accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    evaluator.print_classification_report(model, data_module.test_loader(), trainer.device)

    out_dir = config.output_root / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluator.plot_training_curves(history, title=args.backbone, save_path=out_dir / "training_curves.png")
    evaluator.plot_confusion_matrix(test_metrics["confusion_matrix"], title=args.backbone,
                                     save_path=out_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()
