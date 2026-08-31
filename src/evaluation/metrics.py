"""
src/evaluation/metrics.py

Evaluator computes the full metric set the assignment asks for:
"the more metrics the better — overall, per-class, confusion matrix."
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt


class Evaluator:
    def __init__(self, class_names: list):
        self.class_names = class_names

    @torch.no_grad()
    def _collect_predictions(self, model, loader, device):
        model.eval()
        all_preds, all_labels = [], []
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
        return np.array(all_labels), np.array(all_preds)

    def evaluate(self, model, loader, device) -> dict:
        y_true, y_pred = self._collect_predictions(model, loader, device)

        accuracy = accuracy_score(y_true, y_pred)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        precision_per_class, recall_per_class, f1_per_class, support_per_class = (
            precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        )
        cm = confusion_matrix(y_true, y_pred)

        return {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_per_class": dict(zip(self.class_names, precision_per_class)),
            "recall_per_class": dict(zip(self.class_names, recall_per_class)),
            "f1_per_class": dict(zip(self.class_names, f1_per_class)),
            "support_per_class": dict(zip(self.class_names, support_per_class)),
            "confusion_matrix": cm,
        }

    def print_classification_report(self, model, loader, device):
        y_true, y_pred = self._collect_predictions(model, loader, device)
        print(classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0))

    def plot_confusion_matrix(self, confusion_mat, title: str, save_path=None):
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(confusion_mat, cmap="Blues")
        ax.set_xticks(range(len(self.class_names)))
        ax.set_yticks(range(len(self.class_names)))
        ax.set_xticklabels(self.class_names, rotation=45, ha="right")
        ax.set_yticklabels(self.class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

        for i in range(confusion_mat.shape[0]):
            for j in range(confusion_mat.shape[1]):
                ax.text(j, i, str(confusion_mat[i, j]), ha="center", va="center",
                        color="white" if confusion_mat[i, j] > confusion_mat.max() / 2 else "black")

        fig.colorbar(im)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved confusion matrix to {save_path}")
        plt.close(fig)

    def plot_training_curves(self, history: dict, title: str, save_path=None):
        """Loss/accuracy curves — for spotting overfit/underfit per the assignment."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].plot(history["train_loss"], label="train")
        axes[0].plot(history["val_loss"], label="val")
        axes[0].set_title(f"{title} — Loss")
        axes[0].set_xlabel("epoch")
        axes[0].legend()

        axes[1].plot(history["train_acc"], label="train")
        axes[1].plot(history["val_acc"], label="val")
        axes[1].set_title(f"{title} — Accuracy")
        axes[1].set_xlabel("epoch")
        axes[1].legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved training curves to {save_path}")
        plt.close(fig)
