"""
src/data/transforms.py

Separates "what preprocessing/augmentation to do" from "how to load
data" (see dataset.py) — so you can swap augmentation strategy without
touching data loading code, and vice versa.
"""

from torchvision import transforms

# Standard ImageNet normalization stats — correct to use even for a
# non-ImageNet target dataset, because the backbones were PRETRAINED on
# ImageNet with these stats. Keeping them matches what the backbone
# already expects at its input.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TransformFactory:
    """
    Builds train/val/test transform pipelines.

    Train pipeline includes augmentation (required by the assignment —
    online augmentation here means "applied fresh every epoch", as
    opposed to offline augmentation where you'd pre-generate and save
    augmented copies to disk once).

    Val/test pipelines do NOT augment — you want to evaluate on
    representative, undistorted images.
    """

    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def train_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            # --- augmentation operations (justify choice per-dataset in
            # your presentation — e.g. horizontal flip makes sense for
            # most objects but NOT for text/asymmetric items where left-
            # right matters) ---
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def eval_transform(self) -> transforms.Compose:
        """Used for both validation AND test sets — no augmentation."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def raw_transform(self) -> transforms.Compose:
        """
        No normalization — useful for EDA (looking at raw pixel
        brightness/color stats) and for GradCAM overlay images (you want
        to overlay the heatmap on a normal-looking image, not a
        normalized one).
        """
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
