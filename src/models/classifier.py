"""
src/models/classifier.py

CNNClassifier = pretrained backbone (feature extractor) + a freshly
initialized classifier head. This is the "cut the original head off,
attach a new one" step the assignment explicitly asks you to describe.
"""

import torch.nn as nn
from src.models.backbone_factory import BackboneFactory


class ClassifierHead(nn.Module):
    """
    The "part we add" on top of the backbone. Kept as its own small
    class so its architecture is easy to describe precisely in your
    presentation (assignment asks: how many layers, how do they connect,
    what are the parameters of each).
    """

    def __init__(self, in_features: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class CNNClassifier(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int,
                 head_hidden_dim: int = 256, head_dropout: float = 0.3,
                 pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone, feature_dim, self.block_groups = BackboneFactory.build(
            backbone_name, pretrained=pretrained
        )
        self.head = ClassifierHead(feature_dim, head_hidden_dim, num_classes, head_dropout)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    # ------------------------------------------------------------
    # Freeze/unfreeze control — used by Trainer to implement both
    # training modes the assignment allows:
    #   "feature_extract" -> freeze_backbone(fully=True)
    #   "finetune"         -> freeze_backbone(fully=True) then
    #                          unfreeze_last_n_blocks(n)
    # ------------------------------------------------------------
    def freeze_backbone(self, fully: bool = True):
        for param in self.backbone.parameters():
            param.requires_grad = not fully

    def unfreeze_last_n_blocks(self, n: int):
        """Unfreezes the last n entries of self.block_groups (deepest
        layers first — these adapt most to a new dataset during
        finetuning, while early layers stay frozen as generic
        edge/texture detectors)."""
        blocks_to_unfreeze = self.block_groups[-n:] if n > 0 else []
        for block in blocks_to_unfreeze:
            for param in block.parameters():
                param.requires_grad = True

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def gradcam_target_layer(self):
        return BackboneFactory.gradcam_target_layer(self.backbone_name, self.backbone)
