"""
src/utils/checkpoint.py

Saves/loads everything needed to resume training mid-run: model weights,
optimizer state, scheduler state, training history, and the early-
stopping bookkeeping. Without the optimizer/scheduler state, "resuming"
would restart momentum/learning-rate schedule from scratch even if the
model weights carry over — this saves ALL of it so a resumed run behaves
as if it never stopped.

Designed around Colab's failure mode specifically: the runtime can die
at ANY point, so checkpoints must be saved to disk FREQUENTLY (every
epoch here, not just at the end) and to a location that survives a
runtime reset — see colab_utils.py for pointing this at Google Drive.
"""

from pathlib import Path
import torch

# Checkpoint   
class CheckpointManager:
    def __init__(self, checkpoint_root: Path, run_id: str):
        self.dir = Path(checkpoint_root) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.dir / "latest.pt"
        self.best_path = self.dir / "best.pt"

    # save training progress 
    def _save(self, path: Path, epoch: int, model, optimizer, scheduler,
              history: dict, best_val_loss: float, epochs_without_improvement: int):
        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "history": history,
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
        }
        # save to a temp file then rename — avoids a corrupted checkpoint
        # if the runtime dies mid-write (rename is effectively atomic)
        tmp_path = path.with_suffix(".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(path)

    # save lastest epoch
    def save_latest(self, epoch, model, optimizer, scheduler, history,
                     best_val_loss, epochs_without_improvement):
        self._save(self.latest_path, epoch, model, optimizer, scheduler,
                   history, best_val_loss, epochs_without_improvement)

    # save best epoch 
    def save_best(self, epoch, model, optimizer, scheduler, history,
                  best_val_loss, epochs_without_improvement):
        self._save(self.best_path, epoch, model, optimizer, scheduler,
                   history, best_val_loss, epochs_without_improvement)

    # check checkpoint exist 
    def has_checkpoint(self) -> bool:
        return self.latest_path.exists()

    # load lastest checkpoint 
    def load_latest(self, map_location="cpu") -> dict:
        return torch.load(self.latest_path, map_location=map_location, weights_only=False)

    # load best checkpoint
    def load_best(self, map_location="cpu") -> dict:
        return torch.load(self.best_path, map_location=map_location, weights_only=False)
