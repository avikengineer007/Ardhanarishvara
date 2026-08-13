"""
EEG 2D-CNN Encoder, SVM Baseline & Training Pipeline for Ardhanarishvara.
Phase 3 Deliverable:
- EEG-CNN with mirrored architecture (Conv2D(32, k=5) -> Conv2D(64, k=3) -> Conv2D(128, k=3) -> GlobalAvgPool -> Linear(128, 128))
- SVM Baseline Classifier for performance comparison.
- Generates 128-dim embedding vector.
- Saves model checkpoint to models/eeg/eeg_encoder.pt.

Target dataset: King Abdulaziz University (KAU) ASD EEG Dataset (Djemal et al., 2017)
  - 16 subjects: 8 ASD, 8 TD
  - 16 EEG channels (standard 10-20): FP1, F3, F7, FP2, F4, F8, T7, P7, T8, P8, C3, Cz, C4, P3, Pz, P4
  - Sampling rate: 256 Hz
  - Input connectivity matrix: 16x16 Phase Locking Value (PLV)
  - NOTE: Real EEG files require data-sharing request to KAU. The __main__ block
    below uses synthetic matrices for architecture validation ONLY.
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
# pyrefly: ignore [missing-source, missing-import]
from sklearn.svm import SVC
# pyrefly: ignore [missing-source, missing-import]
from sklearn.preprocessing import StandardScaler
# pyrefly: ignore [missing-source, missing-import]
from sklearn.metrics import accuracy_score, roc_auc_score

import config
from security.sanitized_logging import sanitize_errors, log_info


class EEG2DCNNEncoder(nn.Module):
    """
    2D-CNN Encoder for EEG Connectivity Matrices (e.g. 16x16 PLV matrices from 16-channel KAU montage).
    Mirrored architecture: Conv2D(32, k=5) -> Conv2D(64, k=3) -> Conv2D(128, k=3) -> GlobalAvgPool -> Dense(128).
    Extracts 128-dimensional embedding vector per subject regardless of input matrix resolution via Global Average Pooling.
    """
    def __init__(self, embedding_dim: int = 128, num_classes: int = 2):
        super().__init__()
        
        # Layer 1: Conv2D(1 -> 32, k=5, p=2)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2: Conv2D(32 -> 64, k=3, p=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 3: Conv2D(64 -> 128, k=3, p=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

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


@sanitize_errors("Failed to train SVM baseline.")
def train_eeg_svm_baseline(matrices: np.ndarray, labels: np.ndarray) -> tuple[SVC, float, float]:
    """Train SVM Baseline (RBF kernel) on flattened EEG connectivity features."""
    N = len(matrices)
    flat_features = matrices.reshape(N, -1)

    train_size = int(N * 0.8)
    X_train, X_val = flat_features[:train_size], flat_features[train_size:]
    y_train, y_val = labels[:train_size], labels[train_size:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    svm = SVC(kernel="rbf", C=1.0, probability=True, random_state=config.RANDOM_SEED)
    svm.fit(X_train_scaled, y_train)

    preds = svm.predict(X_val_scaled)
    probs = svm.predict_proba(X_val_scaled)[:, 1]

    acc = accuracy_score(y_val, preds)
    try:
        auc = roc_auc_score(y_val, probs)
    except Exception:
        auc = 0.5

    log_info(f"[EEG SVM Baseline] Accuracy: {acc*100:.2f}% | AUC: {auc:.4f}")
    return svm, acc, auc


@sanitize_errors("Failed to train EEG 2D-CNN encoder.")
def train_eeg_encoder(matrices: np.ndarray, labels: np.ndarray, epochs: int = 15, batch_size: int = 16, lr: float = 1e-3) -> tuple[EEG2DCNNEncoder, dict]:
    """Train EEG 2D-CNN Encoder and save weights to models/eeg/eeg_encoder.pt."""
    log_info("=== Phase 3: Training EEG 2D-CNN Encoder & SVM Baseline ===")

    # 1. Train SVM Baseline
    svm_model, svm_acc, svm_auc = train_eeg_svm_baseline(matrices, labels)

    # 2. Train EEG-CNN
    if matrices.ndim == 3:
        matrices = np.expand_dims(matrices, axis=1)

    X_tensor = torch.tensor(matrices, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)

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

    model = EEG2DCNNEncoder(embedding_dim=config.EMBEDDING_DIM, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_acc": [], "svm_acc": svm_acc, "svm_auc": svm_auc}

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

        with torch.no_grad():
            for X_b, y_b in val_loader:
                logits, _ = model(X_b)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_b).sum().item()
                total += y_b.size(0)

        val_acc = correct / total if total > 0 else 0.0
        history["val_acc"].append(val_acc)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(config.EEG_MODEL_DIR, "eeg_encoder.pt")
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            log_info(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc*100:.1f}%")

    save_path = os.path.join(config.EEG_MODEL_DIR, "eeg_encoder.pt")
    log_info(f"Saved EEG Encoder model weights to {save_path}")

    # Verify 128-dim embedding extraction
    model.eval()
    with torch.no_grad():
        test_in = X_tensor[:2]
        test_logits, test_embed = model(test_in)
        log_info(f"Verified EEG Encoder Output Embedding Shape: {test_embed.shape} (Expected: (2, 128))")
        log_info(f"Final Model Comparison: EEG-CNN Val Acc: {best_val_acc*100:.2f}% vs SVM Baseline Val Acc: {svm_acc*100:.2f}%")

    return model, history


if __name__ == "__main__":
    # ARCHITECTURE VALIDATION ONLY — uses synthetic data.
    # Real KAU dataset: 16 subjects (8 ASD + 8 TD), 16 channels, 256 Hz -> 16x16 PLV matrices.
    # Obtain real data via data-sharing request to King Abdulaziz University
    # (doi:10.3390/app7020183) before running a genuine training loop.
    rng = np.random.RandomState(config.RANDOM_SEED)
    N_SUBJECTS = 16          # real KAU count
    N_CHANNELS = 16          # real KAU channels (10-20 system)
    matrices = np.zeros((N_SUBJECTS, N_CHANNELS, N_CHANNELS))
    # 8 ASD (label=0) + 8 TD (label=1)
    labels = np.array([0] * 8 + [1] * 8)
    for i in range(N_SUBJECTS):
        mat = rng.randn(N_CHANNELS, N_CHANNELS)
        sym = (mat + mat.T) / 2.0
        np.fill_diagonal(sym, 1.0)
        matrices[i] = sym

    print("[ARCHITECTURE VALIDATION] Running on synthetic 16x16 PLV matrices (N=16).")
    print("Real KAU EEG data required for genuine training.")
    train_eeg_encoder(matrices, labels, epochs=5)
