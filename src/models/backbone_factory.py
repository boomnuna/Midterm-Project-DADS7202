"""
src/models/backbone_factory.py

Registry pattern: each backbone is registered under a short name, and
BackboneFactory.build() returns a consistent interface (feature extractor
+ output feature dimension) regardless of which underlying torchvision
architecture it wraps. This is the piece that makes "try 3-5 significantly
different architectures" (assignment requirement) a config change instead
of a rewrite.

Adding a new backbone later = add one entry to _BUILDERS, nothing else
in the codebase needs to change.
"""

import torch.nn as nn
from torchvision import models

def _resnet_block_groups(net):
    # ResNet's natural "blocks" for progressive unfreezing during finetuning
    return [net.layer1, net.layer2, net.layer3, net.layer4]

# resnet50
def _build_resnet50(pretrained: bool = True):
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    net = models.resnet50(weights=weights)
    feature_dim = net.fc.in_features
    net.fc = nn.Identity()  # remove original 1000-class ImageNet head
    return net, feature_dim, _resnet_block_groups(net)

# vgg16
def _build_vgg16(pretrained: bool = True):
    weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.vgg16(weights=weights)
    feature_dim = net.classifier[0].in_features  # 25088 (after flatten)
    net.classifier = nn.Identity()  # remove original FC classifier stack
    return net, feature_dim, [net.features]  # VGG has one big conv block

# EfficientNet
def _build_efficientnet_b0(pretrained: bool = True):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.efficientnet_b0(weights=weights)
    feature_dim = net.classifier[1].in_features
    net.classifier = nn.Identity()
    return net, feature_dim, list(net.features)  # list of stages for block-wise unfreeze

# MobileNet
def _build_mobilenet_v3_large(pretrained: bool = True):
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    net = models.mobilenet_v3_large(weights=weights)
    feature_dim = net.classifier[0].in_features
    net.classifier = nn.Identity()
    return net, feature_dim, list(net.features)


_BUILDERS = {
    "resnet50": _build_resnet50,
    "vgg16": _build_vgg16,
    "efficientnet_b0": _build_efficientnet_b0,
    "mobilenet_v3_large": _build_mobilenet_v3_large,
}


class BackboneFactory:
    @staticmethod
    # check how many model we use 
    def available() -> list:
        return list(_BUILDERS.keys())

    # build model before using
    @staticmethod
    def build(name: str, pretrained: bool = True):
        """
        Returns (backbone_module, feature_dim, block_groups)
          - backbone_module: nn.Module with its original classifier head
            removed (nn.Identity), so forward(x) returns raw features
          - feature_dim: size of that feature vector, needed to build the
            matching classifier head
          - block_groups: list of sub-modules, ordered shallow->deep, used
            by Trainer for "unfreeze last N blocks" finetuning
        """
        if name not in _BUILDERS:
            raise ValueError(f"Unknown backbone '{name}'. Available: {BackboneFactory.available()}")
        return _BUILDERS[name](pretrained=pretrained)

    # show GradCAM 
    @staticmethod
    def gradcam_target_layer(name: str, backbone_module: nn.Module):
        """
        Returns the layer GradCAM should hook into (the last convolutional
        layer before global pooling) — this differs per architecture, so
        it's centralized here rather than guessed in interpretability code.
        """
        if name == "resnet50":
            return backbone_module.layer4[-1]
        if name == "vgg16":
            return backbone_module.features[-1]
        if name == "efficientnet_b0":
            return backbone_module.features[-1]
        if name == "mobilenet_v3_large":
            return backbone_module.features[-1]
        raise ValueError(f"No GradCAM target layer defined for '{name}'")
    
