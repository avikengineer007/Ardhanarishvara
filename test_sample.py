"""
Ardhanarishvara — Synthetic Test Sample

Purpose: let any collaborator verify the encoder architecture, validation
layer, and fusion wiring run correctly end-to-end WITHOUT needing real
ABIDE-I or KAU EEG data access (which is gated/pending).

This uses synthetic connectivity matrices shaped exactly like the real
data:
  - fMRI: 200x200 (CC200 parcellation, ABIDE-I)
  - EEG:  16x16   (KAU dataset, 16-channel 10-20 layout)

IMPORTANT: results from this script are NOT real performance numbers.
They exist only to confirm the pipeline runs — never report accuracy
from this script as a project result (see project's no-fabrication rule).

Run:
    pip install torch numpy scikit-learn
    python test_sample.py
"""
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.optim as optim

# ---------------------------------------------------------------------
# 1. Shared encoder architecture (same as models/encoder.py)
# ---------------------------------------------------------------------

class ConnectivityCNNEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
        )
        # AdaptiveAvgPool handles ANY input size (200x200 fMRI or 16x16 EEG)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x).flatten(1)
        return self.fc(x)


class ConnectivityClassifier(nn.Module):
    def __init__(self, embedding_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.encoder = ConnectivityCNNEncoder(embedding_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))


class FusionClassifier(nn.Module):
    """Concat-based fusion of frozen fMRI + EEG embeddings (Phase 4 v1)."""
    def __init__(self, embedding_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, emb_fmri, emb_eeg):
        return self.net(torch.cat([emb_fmri, emb_eeg], dim=1))


# ---------------------------------------------------------------------
# 2. Validation helpers (same rules as utils/validation.py)
# ---------------------------------------------------------------------

def validate_connectivity_matrix(matrix: np.ndarray, expected_dim: int) -> np.ndarray:
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1], "must be square"
    assert matrix.shape[0] == expected_dim, f"expected {expected_dim}, got {matrix.shape[0]}"
    assert not np.isnan(matrix).any() and not np.isinf(matrix).any(), "contains NaN/Inf"
    return matrix


# ---------------------------------------------------------------------
# 3. Synthetic data generators (shaped like real data, NOT real signal)
# ---------------------------------------------------------------------

def make_synthetic_fmri_batch(n_subjects: int = 8, seed: int = 42):
    rng = np.random.default_rng(seed)
    matrices, labels = [], []
    for i in range(n_subjects):
        m = rng.uniform(-1, 1, size=(200, 200))
        m = (m + m.T) / 2  # symmetric, like a real correlation matrix
        np.fill_diagonal(m, 1.0)
        validate_connectivity_matrix(m, expected_dim=200)
        matrices.append(m)
        labels.append(i % 2)  # alternate ASD/TD
    return np.array(matrices, dtype=np.float32), np.array(labels)


def make_synthetic_eeg_batch(n_subjects: int = 8, seed: int = 43):
    rng = np.random.default_rng(seed)
    matrices, labels = [], []
    for i in range(n_subjects):
        m = rng.uniform(0, 1, size=(16, 16))  # PLV values are in [0,1]
        m = (m + m.T) / 2
        np.fill_diagonal(m, 1.0)
        validate_connectivity_matrix(m, expected_dim=16)
        matrices.append(m)
        labels.append(i % 2)
    return np.array(matrices, dtype=np.float32), np.array(labels)


# ---------------------------------------------------------------------
# 4. End-to-end smoke test
# ---------------------------------------------------------------------

def run_smoke_test():
    print("=" * 65)
    print("ARDHANARISHVARA — SYNTHETIC PIPELINE SMOKE TEST")
    print("(shapes/wiring only — NOT real performance numbers)")
    print("=" * 65)

    # --- fMRI branch ---
    fmri_data, fmri_labels = make_synthetic_fmri_batch(n_subjects=8)
    fmri_tensor = torch.tensor(fmri_data).unsqueeze(1)  # (N, 1, 200, 200)
    fmri_model = ConnectivityClassifier()
    fmri_out = fmri_model(fmri_tensor)
    print(f"[fMRI]  input {tuple(fmri_tensor.shape)} -> output {tuple(fmri_out.shape)}  OK")

    # --- EEG branch ---
    eeg_data, eeg_labels = make_synthetic_eeg_batch(n_subjects=8)
    eeg_tensor = torch.tensor(eeg_data).unsqueeze(1)  # (N, 1, 16, 16)
    eeg_model = ConnectivityClassifier()
    eeg_out = eeg_model(eeg_tensor)
    print(f"[EEG]   input {tuple(eeg_tensor.shape)} -> output {tuple(eeg_out.shape)}  OK")

    # --- Fusion branch (unpaired: just checking wiring, not real fusion) ---
    emb_fmri = fmri_model.encoder(fmri_tensor)  # (8, 128)
    emb_eeg = eeg_model.encoder(eeg_tensor)      # (8, 128)
    fusion_model = FusionClassifier()
    fusion_out = fusion_model(emb_fmri, emb_eeg)
    print(f"[Fusion] embeddings {tuple(emb_fmri.shape)}+{tuple(emb_eeg.shape)} -> output {tuple(fusion_out.shape)}  OK")

    # --- One training step, just to confirm backward pass works ---
    optimizer = optim.Adam(fmri_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    labels_tensor = torch.tensor(fmri_labels, dtype=torch.long)

    optimizer.zero_grad()
    loss = criterion(fmri_out, labels_tensor)
    loss.backward()
    optimizer.step()
    print(f"[Backward pass] loss={loss.item():.4f}  gradients flowed correctly  OK")

    print("=" * 65)
    print("ALL WIRING CHECKS PASSED.")
    print("Next step: replace synthetic data with real ABIDE-I / KAU data")
    print("before reporting any accuracy numbers.")
    print("=" * 65)


if __name__ == "__main__":
    run_smoke_test()
