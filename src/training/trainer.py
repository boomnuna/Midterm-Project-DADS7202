"""
src/training/trainer.py

Trainer runs ONE full train/val loop for ONE model instance. It's used
both for a single ad-hoc run (scripts/train_single.py) and, repeatedly
with different seeds, by RepeatedExperimentRunner (repeated_runs.py) to
build the mean±SD numbers the assignment requires.

CHECKPOINT/RESUME SUPPORT (for Google Colab): if a checkpoint_manager is
passed in, fit() saves full training state EVERY epoch and will resume
from the last saved epoch automatically if one already exists — so a
Colab disconnect mid-training loses at most 1 epoch of progress, not the
whole run. See src/utils/checkpoint.py and src/utils/colab_utils.py.
"""

import copy
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


class Trainer:
    def __init__(self, model, config, class_weights=None, device=None,
                 logger=None, wandb_run=None, checkpoint_manager=None):
        self.model = model
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.logger = logger                          # ExperimentLogger, or None -> print()
        self.wandb_run = wandb_run                     # active wandb run, or None
        self.checkpoint_manager = checkpoint_manager   # CheckpointManager, or None -> no resume support

        self._apply_training_mode() # freeze bbackbone or not 

        weight_tensor = class_weights.to(self.device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # show log message 
    def _log(self, msg: str):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    # ------------------------------------------------------------
    # freeze backbone or not 
    def _apply_training_mode(self):
        if self.config.training_mode == "feature_extract":
            self.model.freeze_backbone(fully=True)
        elif self.config.training_mode == "finetune":
            self.model.freeze_backbone(fully=True) # reset everything back to freeze 
            self.model.unfreeze_last_n_blocks(self.config.finetune_unfreeze_last_n_blocks) # then unfreeze later 
        else:
            raise ValueError(f"Unknown training_mode: {self.config.training_mode}")

    # choose optimizer type
    def _build_optimizer(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if self.config.optimizer_name == "adamw":
            return AdamW(trainable_params, lr=self.config.learning_rate,
                         weight_decay=self.config.weight_decay)
        if self.config.optimizer_name == "sgd":
            return SGD(trainable_params, lr=self.config.learning_rate,
                       momentum=0.9, weight_decay=self.config.weight_decay)
        raise ValueError(f"Unknown optimizer_name: {self.config.optimizer_name}")

    # choose learning rate schedule 
    def _build_scheduler(self):
        if self.config.lr_scheduler == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.config.num_epochs)
        if self.config.lr_scheduler == "step":
            return StepLR(self.optimizer, step_size=max(1, self.config.num_epochs // 3), gamma=0.1)
        return None

    # ------------------------------------------------------------
    # run epoch for train and test 
    def _run_epoch(self, loader, train: bool):
        # set model to train or validation mode 
        self.model.train() if train else self.model.eval()
        # init metrics
        total_loss, correct, total = 0.0, 0, 0


        with torch.set_grad_enabled(train): # enable/disable gradient calculation
            for images, labels in loader: # loader = num batch size 
                images, labels = images.to(self.device), labels.to(self.device) # move data to GPU/CPU

                if train:
                    self.optimizer.zero_grad() # clear old gradients
                outputs = self.model(images) # forward pass 
                loss = self.criterion(outputs, labels) # calculate loss 

                if train:
                    loss.backward() # get gradient 
                    self.optimizer.step() # update weight 

                total_loss += loss.item() * images.size(0) 
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += images.size(0)

        return total_loss / total, correct / total

    # ------------------------------------------------------------
    # resuming training from a checkpoint if training was interrupted.
    def _try_resume(self):
        """
        Returns (start_epoch, best_val_loss, epochs_without_improvement).
        If no checkpoint exists (or none was configured), returns the
        fresh-start defaults — this makes fit() work identically whether
        or not checkpointing is enabled.
        """
        # check if a checkpoint exists
        if self.checkpoint_manager is None or not self.checkpoint_manager.has_checkpoint():
            return 0, float("inf"), 0
        
        # load the latest checkpoint
        ckpt = self.checkpoint_manager.load_latest(map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"]) # restore the model
        self.optimizer.load_state_dict(ckpt["optimizer_state"]) # restore the optimizer 
        if self.scheduler is not None and ckpt["scheduler_state"] is not None: # restore the scheduler
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.history = ckpt["history"]

        start_epoch = ckpt["epoch"] + 1 # set the next epoch
        self._log(f"RESUMED from checkpoint at epoch {ckpt['epoch']} "
                  f"-> continuing from epoch {start_epoch + 1}")
        return start_epoch, ckpt["best_val_loss"], ckpt["epochs_without_improvement"]

    # model fitting 
    def fit(self, train_loader, val_loader, verbose: bool = True):
        # try to resume 
        start_epoch, best_val_loss, epochs_without_improvement = self._try_resume()

        # epoch reached 
        if start_epoch >= self.config.num_epochs:
            self._log(f"Checkpoint already reached target num_epochs="
                      f"{self.config.num_epochs} — nothing to train, returning existing history.")
            return self.history

        best_state = None

        for epoch in range(start_epoch, self.config.num_epochs):
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            val_loss, val_acc = self._run_epoch(val_loader, train=False)

            if self.scheduler is not None:
                self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # print result or not
            if verbose:
                if self.logger is not None:
                    self.logger.log_epoch(epoch + 1, train_loss, train_acc, val_loss, val_acc)
                else:
                    print(f"  epoch {epoch + 1:2d}/{self.config.num_epochs} "
                          f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            if self.wandb_run is not None:
                self.wandb_run.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss, "train_acc": train_acc,
                    "val_loss": val_loss, "val_acc": val_acc,
                })

            # for early stopping record 
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # ---- checkpoint EVERY epoch, not just at the end — this is
            # what limits a Colab disconnect's damage to <=1 epoch ----
            if self.checkpoint_manager is not None:
                self.checkpoint_manager.save_latest(
                    epoch, self.model, self.optimizer, self.scheduler,
                    self.history, best_val_loss, epochs_without_improvement,
                )
                if improved:
                    self.checkpoint_manager.save_best(
                        epoch, self.model, self.optimizer, self.scheduler,
                        self.history, best_val_loss, epochs_without_improvement,
                    )

            if epochs_without_improvement >= self.config.early_stopping_patience:
                if verbose:
                    self._log(f"  early stopping at epoch {epoch + 1} "
                              f"(no val improvement for {self.config.early_stopping_patience} epochs)")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.history
