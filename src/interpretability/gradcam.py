"""
src/interpretability/gradcam.py

Grad-CAM (Selvaraju et al. 2017) implementation via forward/backward
hooks on the backbone's last convolutional layer. Works for any backbone
registered in backbone_factory.py because CNNClassifier.gradcam_target_layer()
already knows which layer that is per architecture.

This satisfies the assignment's mandatory GradCAM analysis requirement,
AND doubles as the "pre-trained vs our model" eyeball comparison tool
(assignment section 7) since it works on any CNNClassifier instance.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.target_layer = model.gradcam_target_layer()

        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, target_class: int = None):
        """
        input_tensor: single image, shape (1, C, H, W), already normalized
                      the same way the model was trained on.
        target_class: class index to explain; if None, uses the model's
                       own top prediction (standard GradCAM usage).

        Returns: (heatmap as HxW numpy array in [0,1], predicted_class_idx)
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)

        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # weight each activation channel by the average gradient flowing
        # into it — this is the core Grad-CAM weighting step
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        weighted_activations = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        heatmap = F.relu(weighted_activations).squeeze().cpu().numpy()

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap, target_class

    @staticmethod
    def overlay_heatmap(raw_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4):
        """
        raw_image: HxWx3 numpy array, values in [0,1] (use TransformFactory
                   .raw_transform() output, NOT the normalized training
                   transform, so colors look correct)
        heatmap: HxW numpy array from generate(), values in [0,1]

        Returns: HxWx3 numpy array ready for plt.imshow()
        """
        heatmap_resized = np.array(
            plt.cm.jet(heatmap)[:, :, :3]  # apply colormap, drop alpha channel
        )
        # resize heatmap to match raw_image if sizes differ
        if heatmap_resized.shape[:2] != raw_image.shape[:2]:
            from PIL import Image
            heatmap_img = Image.fromarray((heatmap_resized * 255).astype(np.uint8))
            heatmap_img = heatmap_img.resize((raw_image.shape[1], raw_image.shape[0]))
            heatmap_resized = np.array(heatmap_img) / 255.0

        overlay = (1 - alpha) * raw_image + alpha * heatmap_resized
        return np.clip(overlay, 0, 1)

    def visualize(self, input_tensor: torch.Tensor, raw_image: np.ndarray,
                  class_names: list, target_class: int = None, save_path=None):
        """Convenience: generate + overlay + side-by-side plot in one call."""
        heatmap, predicted_class = self.generate(input_tensor, target_class)
        overlay = self.overlay_heatmap(raw_image, heatmap)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(raw_image)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM heatmap")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay — predicted: {class_names[predicted_class]}")
        axes[2].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved GradCAM visualization to {save_path}")
        plt.close(fig)

        return predicted_class
