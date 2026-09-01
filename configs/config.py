"""
config.py

Single source of truth for paths and hyperparameters. Every script in
this project imports Config from here instead of hardcoding values —
this is what makes swapping the demo dataset (chihuahua vs cookie) for
the real assignment dataset a one-file change.

TO SWITCH TO YOUR REAL DATASET LATER:
    1. Point DATA_ROOT at your real dataset folder (same train/val/test/
       <class_name>/ structure as the demo dataset — see data/dataset.py)
    2. That's it. num_classes, class names, everything downstream reads
       this dynamically from the folder structure, not from a hardcoded
       number here.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ---- paths ----
    # TEMP: demo dataset for pipeline development. Replace with your real
    # dataset path once class topic + collected images are finalized.
    data_root: Path = Path("data")
    output_root: Path = Path("outputs")

    # ---- image / dataloader ----
    image_size: int = 224          # standard input size for most torchvision backbones
    batch_size: int = 64
    num_workers: int = 4

    # ---- train/val/test split (used only if your data_root has no
    # pre-made train/val/test subfolders yet — see data/dataset.py) ----
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    split_seed: int = 42           # fixed so the split itself is reproducible
                                    # (separate from the N repeated-run seeds)

    # ---- backbones to compare ----
    # Must be "significantly different architectures" per the assignment
    # (e.g. don't count VGG16+VGG19 as 2 architectures). Names must match
    # keys registered in models/backbone_factory.py.
    backbone_names: list = field(default_factory=lambda: [
        "resnet50",
        "vgg16",
        "efficientnet_b0",
        "mobilenet_v3_large",
    ])

    # ---- training method ----
    # "feature_extract" = freeze backbone entirely, train new head only
    #                      (matches assignment's "transfer learning" option)
    # "finetune"        = unfreeze last N backbone blocks too
    #                      (matches assignment's "finetuning" option)
    training_mode: str = "finetune"
    finetune_unfreeze_last_n_blocks: int = 2

    # ---- optimizer / schedule ----
    optimizer_name: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"   # "cosine" | "step" | "none"
    num_epochs: int = 20
    early_stopping_patience: int = 5

    # ---- classifier head added on top of the backbone ----
    head_hidden_dim: int = 256
    head_dropout: float = 0.3

    # ---- repeated runs (for mean±SD reporting, assignment section 6) ----
    num_repeats: int = 5           # assignment asks for 3-10 repeats per architecture
    base_seed: int = 100           # repeat i uses seed = base_seed + i

    # ---- class imbalance handling ----
    # "none" | "class_weights" | "oversample"
    imbalance_strategy: str = "class_weights"

    # ---- logging ----
    # Local file logging always happens regardless of the flags below —
    # see src/utils/logger.py. These two are OPTIONAL cloud/tracking add-ons.
    use_wandb: bool = False
    wandb_project: str = "Midterm-Project-DADS7202" 
    wandb_entity: str = None   # your W&B team/org name, if using one

    use_optuna: bool = False
    optuna_n_trials: int = 20
    optuna_quick_epochs: int = 8   # shorter than num_epochs, just for the search phase
    optuna_storage: str = "sqlite:///outputs/optuna_study.db"
    # ^ file-based storage works for one person tuning locally. For a
    # group to share trials across machines, point this at a shared
    # database instead (e.g. "mysql://..." or "postgresql://...") —
    # see README for why W&B Sweep is usually the easier group option.

    def __post_init__(self):
        self.data_root = Path(self.data_root)
        self.output_root = Path(self.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
