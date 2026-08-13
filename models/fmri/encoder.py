"""
fMRI 2D-CNN Encoder & Training Pipeline for Ardhanarishvara.
Phase 2 Deliverable:
- 2D-CNN Encoder (Conv2D(1->32, k=5, p=2) -> Conv2D(32->64, k=3, p=1) -> Conv2D(64->128, k=3, p=1) -> GlobalAvgPool -> Linear(128, 128))
- Generates 128-dim embedding vector.
- Saves model checkpoint to models/fmri/fmri_encoder.pt.
"""

import os
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.optim as optim
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader, TensorDataset
# pyrefly: ignore [missing-import]
import numpy as np

import config
from security.sanitized_logging import sanitize_errors, log_info
from security.validation import validate_connectivity_matrix


class FMRI2DCNNEncoder(nn.Module):
    """
    2D-CNN Encoder for fMRI CC200 Functional Connectivity Matrices (200x200).
    Mirrored architecture: Conv2D(32, k=5) -> Conv2D(64, k=3) -> Conv2D(128, k=3) -> GlobalAvgPool -> Dense(128).
    """
    def __init__(self, embedding_dim: int = 128, num_classes: int = 2):
        super().__init__()
        
        # Layer 1: Conv2D(1 -> 32, k=5, p=2)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 200x200 -> 100x100

        # Layer 2: Conv2D(32 -> 64, k=3, p=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 100x100 -> 50x50

        # Layer 3: Conv2D(64 -> 128, k=3, p=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # 50x50 -> 1x1 Global Avg Pool

        # Dense Embedding Layer: 128 -> 128
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.ReLU(inplace=True)
        )

        # Classification Head
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 128-dim feature embedding."""
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.gap(self.relu3(self.bn3(self.conv3(x))))
        embed = self.embedding(x)
        return embed

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, 128-dim embedding)."""
        embed = self.extract_embedding(x)
        logits = self.classifier(embed)
        return logits, embed


@sanitize_errors("Failed to train fMRI 2D-CNN encoder.")
def train_fmri_encoder(matrices: np.ndarray, labels: np.ndarray, epochs: int = 15, batch_size: int = 16, lr: float = 1e-3) -> tuple[FMRI2DCNNEncoder, dict]:
    """Train fMRI 2D-CNN Encoder on CC200 matrices and save weights to models/fmri/fmri_encoder.pt."""
    log_info("=== Phase 2: Training fMRI 2D-CNN Encoder ===")

    # Ensure shape (N, 1, 200, 200)
    if matrices.ndim == 3:
        matrices = np.expand_dims(matrices, axis=1)

    X_tensor = torch.tensor(matrices, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    # Train / Val Split (80% train, 20% val)
    n_samples = len(X_tensor)
    val_size = max(4, int(n_samples * 0.2))
    train_size = n_samples - val_size

    torch.manual_seed(config.RANDOM_SEED)
    indices = torch.randperm(n_samples)
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
    val_dataset = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = FMRI2DCNNEncoder(embedding_dim=config.EMBEDDING_DIM, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_acc": [], "val_auc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            logits, _ = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_b.size(0)

        epoch_loss = running_loss / train_size
        history["train_loss"].append(epoch_loss)

        # Validation
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_b, y_b in val_loader:
                logits, _ = model(X_b)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                correct += (preds == y_b).sum().item()
                total += y_b.size(0)
                all_preds.extend(probs[:, 1].cpu().numpy())
                all_targets.extend(y_b.cpu().numpy())

        val_acc = correct / total if total > 0 else 0.0
        history["val_acc"].append(val_acc)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(config.FMRI_MODEL_DIR, "fmri_encoder.pt")
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            log_info(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc*100:.1f}%")

    save_path = os.path.join(config.FMRI_MODEL_DIR, "fmri_encoder.pt")
    log_info(f"Saved fMRI Encoder model weights to {save_path}")

    # Verify 128-dim embedding extraction
    model.eval()
    with torch.no_grad():
        test_in = X_tensor[:2]
        test_logits, test_embed = model(test_in)
        log_info(f"Verified fMRI Encoder Output Embedding Shape: {test_embed.shape} (Expected: (2, 128))")

    return model, history


if __name__ == "__main__":
    # Test script with synthetic CC200 data representing ABIDE-I cohort
    rng = np.random.RandomState(config.RANDOM_SEED)
    N = 60
    matrices = np.zeros((N, 200, 200))
    labels = rng.choice([0, 1], size=N, p=[0.5, 0.5])
    for i in range(N):
        mat = rng.randn(200, 200)
        sym = (mat + mat.T) / 2.0
        np.fill_diagonal(sym, 1.0)
        matrices[i] = sym

    train_fmri_encoder(matrices, labels, epochs=10)
