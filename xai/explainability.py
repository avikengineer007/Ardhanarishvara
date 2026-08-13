"""
Explainability Module for Ardhanarishvara Phase 5.
Full implementations of:
  1. BranchGradCAM — Grad-CAM heatmaps on conv3 of each CNN encoder branch
  2. SHAPExplainer — SHAP values on connectivity matrix features per modality

Replaces the Phase 5 placeholder with production-grade XAI utilities.
"""

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.nn.functional as F

import config
from security.sanitized_logging import sanitize_errors, log_info


class BranchGradCAM:
    """
    Compute Grad-CAM heatmaps for fMRI or EEG CNN encoder branches.
    Hooks into the target convolutional layer (default: conv3) to capture
    activations and gradients, then computes class-discriminative spatial maps.

    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", IJCV 2020.
    """

    def __init__(self, model: nn.Module, target_layer_name: str = "conv3"):
        """
        Args:
            model: Trained encoder (FMRI2DCNNEncoder or EEG2DCNNEncoder)
            target_layer_name: Name of the conv layer to hook (default: 'conv3')
        """
        self.model = model
        self.model.eval()

        # Resolve target layer
        if not hasattr(model, target_layer_name):
            raise ValueError(f"Model has no layer named '{target_layer_name}'. "
                             f"Available: {[n for n, _ in model.named_modules()]}")
        self.target_layer = getattr(model, target_layer_name)

        self._activations = None
        self._gradients = None

        # Register hooks
        self._fwd_hook = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Forward hook: capture feature map activations."""
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook: capture gradients flowing through target layer."""
        self._gradients = grad_output[0].detach()

    @sanitize_errors("Failed to compute Grad-CAM heatmap.")
    def compute(self, input_tensor: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a single input.

        Args:
            input_tensor: (1, 1, H, W) single connectivity matrix
            target_class: Class index to explain (None = predicted class)

        Returns:
            cam: (H, W) normalized heatmap in [0, 1]
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        # Clone input and enable gradient tracking
        input_tensor = input_tensor.clone().detach().to(device).requires_grad_(True)

        # Forward pass
        logits, _ = self.model(input_tensor)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)

        # Grad-CAM computation
        if self._gradients is None or self._activations is None:
            raise RuntimeError("Gradients or activations were not captured. Ensure forward and backward hooks fired correctly.")
        gradients = self._gradients[0]    # (C, H', W')
        activations = self._activations[0]  # (C, H', W')

        # Global Average Pool of gradients → channel weights
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted combination of feature maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=device)
        for c in range(weights.shape[0]):
            cam += weights[c] * activations[c]

        # ReLU — only positive contributions
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max

        # Resize to input spatial dimensions via bilinear interpolation
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, H', W')
        cam = F.interpolate(cam, size=(input_h, input_w), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        return cam

    @sanitize_errors("Failed to compute batch Grad-CAM heatmaps.")
    def compute_batch(self, inputs: np.ndarray, labels: np.ndarray | None = None) -> list:
        """
        Compute Grad-CAM heatmaps for a batch of inputs.

        Args:
            inputs: (N, H, W) or (N, 1, H, W) connectivity matrices
            labels: (N,) target class for each sample (None = predicted)

        Returns:
            List of (H, W) numpy heatmaps
        """
        if inputs.ndim == 3:
            inputs = np.expand_dims(inputs, axis=1)

        heatmaps = []
        for i in range(len(inputs)):
            single_input = torch.tensor(inputs[i:i+1], dtype=torch.float32)
            target = int(labels[i]) if labels is not None else None
            cam = self.compute(single_input, target_class=target)
            heatmaps.append(cam)

        log_info(f"Computed {len(heatmaps)} Grad-CAM heatmaps (shape: {heatmaps[0].shape})")
        return heatmaps

    def cleanup(self):
        """Remove registered hooks to prevent memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


class SHAPExplainer:
    """
    SHAP analysis on connectivity matrix features for either modality.
    Uses gradient-based SHAP (fast) with fallback to input×gradient saliency.

    Identifies which ROI-ROI connections (fMRI) or channel-channel PLV pairs (EEG)
    most strongly drive the ASD/TD classification decision.
    """

    def __init__(self, model: nn.Module, background_data: np.ndarray,
                 feature_names: list[str] | None = None):
        """
        Args:
            model: Trained encoder (FMRI2DCNNEncoder or EEG2DCNNEncoder)
            background_data: (N_bg, H, W) or (N_bg, 1, H, W) background samples
            feature_names: Optional list of feature labels for the matrix entries
        """
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        self.feature_names = feature_names

        # Prepare background data
        if background_data.ndim == 3:
            background_data = np.expand_dims(background_data, axis=1)
        self.background = torch.tensor(background_data, dtype=torch.float32).to(self.device)

        self._shap_available = False
        try:
            # pyrefly: ignore [missing-import]
            import shap
            self._shap_lib = shap
            self._shap_available = True
            log_info("SHAP library available — using GradientExplainer")
        except ImportError:
            log_info("WARNING: SHAP library not available. Falling back to gradient×input saliency.")

    @sanitize_errors("Failed to compute SHAP values.")
    def explain(self, samples: np.ndarray, target_class: int = 1) -> np.ndarray:
        """
        Compute per-feature importance values for the given samples.

        Args:
            samples: (N, H, W) or (N, 1, H, W) connectivity matrices to explain
            target_class: Class to explain (default: 1 = ASD)

        Returns:
            importance_values: (N, H, W) per-feature importance scores
        """
        if samples.ndim == 3:
            samples = np.expand_dims(samples, axis=1)

        sample_tensor = torch.tensor(samples, dtype=torch.float32).to(self.device)

        if self._shap_available:
            return self._shap_gradient_explain(sample_tensor, target_class)
        else:
            return self._gradient_input_saliency(sample_tensor, target_class)

    def _shap_gradient_explain(self, sample_tensor: torch.Tensor,
                                target_class: int) -> np.ndarray:
        """Compute SHAP values using GradientExplainer."""
        # Wrapper that returns only logits (required by SHAP)
        class _LogitWrapper(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder
            def forward(self, x):
                logits, _ = self.encoder(x)
                return logits

        wrapper = _LogitWrapper(self.model).to(self.device)
        wrapper.eval()

        # Limit background samples for speed
        n_bg = min(len(self.background), config.SHAP_N_BACKGROUND)
        bg_subset = self.background[:n_bg]

        try:
            explainer = self._shap_lib.GradientExplainer(wrapper, bg_subset)
            shap_values = explainer.shap_values(sample_tensor)

            # shap_values is a list per class or array with class as last dimension
            if isinstance(shap_values, (list, tuple)):
                values = shap_values[target_class]
            else:
                values = shap_values

            if isinstance(values, torch.Tensor):
                values = values.cpu().numpy()
            else:
                values = np.asarray(values)

            # If class dimension is at the end: e.g. (N, 1, H, W, num_classes) or (N, H, W, num_classes)
            if values.ndim >= 4 and values.shape[-1] == 2:
                values = values[..., target_class]

            # Remove singleton channel dimension if present: (N, 1, H, W) -> (N, H, W)
            while values.ndim > 3:
                values = values.squeeze(1)

            # If 2D (single sample), expand to 3D (1, H, W)
            if values.ndim == 2:
                values = np.expand_dims(values, axis=0)

            return values

        except Exception as e:
            log_info(f"SHAP GradientExplainer failed ({e}). Falling back to gradient×input.")
            return self._gradient_input_saliency(sample_tensor, target_class)

    def _gradient_input_saliency(self, sample_tensor: torch.Tensor,
                                  target_class: int) -> np.ndarray:
        """Fallback: gradient × input saliency maps."""
        sample_tensor = sample_tensor.clone().detach().requires_grad_(True)

        logits, _ = self.model(sample_tensor)

        self.model.zero_grad()
        target_scores = logits[:, target_class].sum()
        target_scores.backward()

        # Saliency = gradient × input (element-wise)
        gradients = sample_tensor.grad.detach()
        saliency = (gradients * sample_tensor.detach()).abs()

        # (N, 1, H, W) → (N, H, W)
        result = saliency.squeeze(1).cpu().numpy()
        log_info(f"Computed gradient×input saliency maps: shape={result.shape}")
        return result

    def get_top_features(self, importance_values: np.ndarray, n: int = 20) -> list:
        """
        Identify top-N most important matrix entries from aggregated SHAP/saliency values.

        Args:
            importance_values: (N, H, W) per-sample importance
            n: Number of top features to return

        Returns:
            List of (row_idx, col_idx, importance_score) tuples
        """
        # Average absolute importance across samples
        mean_importance = np.abs(importance_values).mean(axis=0)  # (H, W)

        # Get upper triangle indices (connectivity matrices are symmetric)
        H, W = mean_importance.shape
        triu_indices = np.triu_indices(min(H, W), k=1)
        triu_values = mean_importance[triu_indices]

        # Sort by importance (descending)
        sorted_idx = np.argsort(triu_values)[::-1][:n]

        top_features = []
        for idx in sorted_idx:
            row = triu_indices[0][idx]
            col = triu_indices[1][idx]
            score = float(triu_values[idx])

            if self.feature_names:
                label = f"{self.feature_names[row]} ↔ {self.feature_names[col]}"
            else:
                label = f"({row}, {col})"

            top_features.append((row, col, score, label))

        return top_features
