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

# ---------------------------------------------------------------------------
# block_groups = the units that "unfreeze last n blocks" counts in.
# To make n mean the same thing for every backbone, each backbone is split
# into STAGES that end at the same feature-map resolution (shallow -> deep):
#
#   n=1 -> last stage  (7x7  feature map, the most task-specific layers)
#   n=2 -> + stage     (14x14)
#   n=3 -> + stage     (28x28)
#   n=4 -> + stage     (56x56 / 112x112)
#
#   resnet50           layer1 | layer2 | layer3 | layer4          (4 stages; conv1/bn1 stem never unfrozen)
#   vgg16              conv block1 | 2 | 3 | 4 | 5                (5 blocks; n=4 -> blocks 2-5)
#   efficientnet_b0    features[0:3] | [3] | [4:6] | [6:9]        (4 stages)
#   mobilenet_v3_small features[0:2] | [2:4] | [4:9] | [9:13]     (4 stages)
#
# Slicing an nn.Sequential returns a NEW Sequential that shares the same
# layers, so no weights/state_dict keys change — only the grouping does.
# ---------------------------------------------------------------------------
def _resnet_block_groups(net):
    # ResNet's natural "blocks" for progressive unfreezing during finetuning
    return [net.layer1, net.layer2, net.layer3, net.layer4]

def _vgg16_block_groups(net):
    # the 5 conv blocks, each ending with a MaxPool (indices of the MaxPool: 4, 9, 16, 23, 30)
    f = net.features
    return [f[0:5], f[5:10], f[10:17], f[17:24], f[24:31]]

def _efficientnet_b0_block_groups(net):
    # grouped by output resolution: 112/56 | 28 | 14 | 7
    f = net.features
    return [f[0:3], f[3:4], f[4:6], f[6:9]]

def _mobilenet_v3_small_block_groups(net):
    # grouped by output resolution: 112/56 | 28 | 14 | 7
    f = net.features
    return [f[0:2], f[2:4], f[4:9], f[9:13]]

# resnet50
def _build_resnet50(pretrained: bool = True):
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
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
    return net, feature_dim, _vgg16_block_groups(net)  # 5 conv blocks

# EfficientNet
def _build_efficientnet_b0(pretrained: bool = True):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.efficientnet_b0(weights=weights)
    feature_dim = net.classifier[1].in_features
    net.classifier = nn.Identity()
    return net, feature_dim, _efficientnet_b0_block_groups(net)  # 4 resolution stages

# MobileNet
def _build_mobilenet_v3_small(pretrained: bool = True):
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.mobilenet_v3_small(weights=weights)
    feature_dim = net.classifier[0].in_features
    net.classifier = nn.Identity()
    return net, feature_dim, _mobilenet_v3_small_block_groups(net)  # 4 resolution stages


_BUILDERS = {
    "resnet50": _build_resnet50,
    "vgg16": _build_vgg16,
    "efficientnet_b0": _build_efficientnet_b0,
    "mobilenet_v3_small": _build_mobilenet_v3_small,
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
        if name == "mobilenet_v3_small":
            return backbone_module.features[-1]
        raise ValueError(f"No GradCAM target layer defined for '{name}'")
    
