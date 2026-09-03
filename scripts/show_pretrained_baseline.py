"""
scripts/show_pretrained_baseline.py

Assignment section 3 (Model architecture) requires showing that a
PRE-TRAINED, NOT-YET-FINETUNED CNN either misclassifies our images
entirely or has no matching class at all — this is the evidence for
"why we need to finetune it." No accuracy/precision/recall needed here,
just example predictions to screenshot for your slides.

Usage:
    python -m scripts.show_pretrained_baseline
"""

import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torchvision import models
from torchvision.models import ResNet50_Weights

from config import Config
from src.data.dataset import DataModule

# get all 1000 imagenet class name 
def load_imagenet_class_names():
    weights = ResNet50_Weights.IMAGENET1K_V2
    return weights.meta["categories"]


def main():
    config = Config()
    data_module = DataModule(config)
    imagenet_classes = load_imagenet_class_names()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device) # call pre-trained model
    model.eval()

    loader = data_module.test_loader() # load test data
    print("Pretrained ResNet50 (ImageNet, untouched) predictions on OUR dataset:\n")
    print(f"{'true class (ours)':25s} | top-1 ImageNet prediction")
    print("-" * 70)

    shown = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            for i in range(len(labels)):
                if shown >= 15:  # just enough examples for a slide, not a full pass
                    break
                true_name = data_module.class_names[labels[i].item()]
                pred_name = imagenet_classes[preds[i]]
                print(f"{true_name:25s} | {pred_name}")
                shown += 1
            if shown >= 15:
                break

    print("\nUse this output (or better, matching example images) as your "
          "'pretrained model doesn't know our classes' evidence slide.")


if __name__ == "__main__":
    main()
