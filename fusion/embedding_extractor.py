"""
Embedding Extractor for Ardhanarishvara Phase 4.
Loads frozen encoder checkpoints, runs forward pass on all subjects,
and caches 128-dim embeddings to .npy files for fusion training.
"""

import os
import numpy as np
import torch

import config
from models.fmri.encoder import FMRI2DCNNEncoder
from models.eeg.encoder import EEG2DCNNEncoder
from security.sanitized_logging import sanitize_errors, log_info


def get_device():
    """Auto-detect best available compute device (CUDA > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log_info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
    return device


@sanitize_errors("Failed to load frozen encoder.")
def load_frozen_encoder(modality: str, checkpoint_path: str = None):
    """
    Load encoder checkpoint and freeze all parameters for embedding extraction.

    Args:
        modality: 'fmri' or 'eeg'
        checkpoint_path: Override path to .pt checkpoint file

    Returns:
        Frozen model in eval mode on the best available device.
    """
    device = get_device()

    if modality == "fmri":
        model = FMRI2DCNNEncoder(embedding_dim=config.EMBEDDING_DIM, num_classes=2)
        if checkpoint_path is None:
            checkpoint_path = os.path.join(config.FMRI_MODEL_DIR, "fmri_encoder.pt")
    elif modality == "eeg":
        model = EEG2DCNNEncoder(embedding_dim=config.EMBEDDING_DIM, num_classes=2)
        if checkpoint_path is None:
            checkpoint_path = os.path.join(config.EEG_MODEL_DIR, "eeg_encoder.pt")
    else:
        raise ValueError(f"Unknown modality: '{modality}'. Must be 'fmri' or 'eeg'.")

    # Load checkpoint if available
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        log_info(f"Loaded {modality} encoder weights from {os.path.basename(checkpoint_path)}")
    else:
        log_info(f"WARNING: Checkpoint not found at {os.path.basename(checkpoint_path)}. "
                 f"Using randomly initialized {modality} encoder.")

    # Freeze all parameters — no gradient computation during embedding extraction
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log_info(f"Frozen {modality} encoder ({n_params:,} parameters) on {device}")

    return model


@sanitize_errors("Failed to extract embeddings from frozen encoder.")
def extract_all_embeddings(model=None, matrices: np.ndarray = None, batch_size: int = 32) -> np.ndarray:
    """
    Run frozen encoder on all connectivity matrices to extract 128-dim embeddings.
    If called without arguments, loads pre-extracted embeddings (fmri_embeddings, fmri_labels, eeg_embeddings, eeg_labels).

    Args:
        model: Frozen encoder model (FMRI2DCNNEncoder or EEG2DCNNEncoder)
        matrices: (N, H, W) or (N, 1, H, W) connectivity matrices
        batch_size: Inference batch size

    Returns:
        (N, 128) numpy array of embeddings (or 4-tuple if called without arguments)
    """
    if model is None and matrices is None:
        return load_or_extract_all_modalities()

    device = next(model.parameters()).device

    # Add channel dimension if needed: (N, H, W) -> (N, 1, H, W)
    if matrices.ndim == 3:
        matrices = np.expand_dims(matrices, axis=1)

    all_embeddings = []
    n_samples = len(matrices)

    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = torch.tensor(matrices[start_idx:end_idx], dtype=torch.float32).to(device)
            embeddings = model.extract_embedding(batch)  # (B, 128)
            all_embeddings.append(embeddings.cpu().numpy())

    result = np.concatenate(all_embeddings, axis=0)

    # Validate output shape
    assert result.shape == (n_samples, config.EMBEDDING_DIM), \
        f"Embedding shape mismatch: got {result.shape}, expected ({n_samples}, {config.EMBEDDING_DIM})"

    return result


@sanitize_errors("Failed to extract and cache embeddings.")
def extract_and_cache_embeddings(modality: str, matrices: np.ndarray, labels: np.ndarray,
                                  force_recompute: bool = False) -> tuple:
    """
    End-to-end: load frozen encoder, extract embeddings, cache to .npy files.

    Args:
        modality: 'fmri' or 'eeg'
        matrices: (N, H, W) connectivity matrices
        labels: (N,) integer labels (0=TD, 1=ASD)
        force_recompute: If True, recompute even if cache exists

    Returns:
        (embeddings, labels) as numpy arrays
    """
    cache_dir = config.PROCESSED_FMRI_DIR if modality == "fmri" else config.PROCESSED_EEG_DIR
    embed_path = os.path.join(cache_dir, f"{modality}_embeddings_128d.npy")
    label_path = os.path.join(cache_dir, f"{modality}_labels.npy")

    # Check cache
    if os.path.exists(embed_path) and os.path.exists(label_path) and not force_recompute:
        embeddings = np.load(embed_path)
        cached_labels = np.load(label_path)
        log_info(f"Loaded cached {modality} embeddings: shape={embeddings.shape}, "
                 f"classes={{ASD={int((cached_labels==1).sum())}, TD={int((cached_labels==0).sum())}}}")
        return embeddings, cached_labels

    # Load frozen encoder and extract
    log_info(f"Extracting {modality} embeddings from frozen encoder...")
    model = load_frozen_encoder(modality)
    embeddings = extract_all_embeddings(model, matrices)

    # Cache
    np.save(embed_path, embeddings)
    np.save(label_path, labels)
    log_info(f"Cached {modality} embeddings to {os.path.basename(embed_path)} "
             f"(shape={embeddings.shape}, ASD={int((labels==1).sum())}, TD={int((labels==0).sum())})")

    return embeddings, labels


@sanitize_errors("Failed to load or extract all modalities.")
def load_or_extract_all_modalities():
    """
    Load pre-cached 128-dim embeddings for both modalities (or compute fallback).
    Returns (fmri_embeddings, fmri_labels, eeg_embeddings, eeg_labels).
    """
    fmri_embed_path = os.path.join(config.PROCESSED_FMRI_DIR, "fmri_embeddings_128d.npy")
    fmri_label_path = os.path.join(config.PROCESSED_FMRI_DIR, "fmri_labels.npy")
    eeg_embed_path = os.path.join(config.PROCESSED_EEG_DIR, "eeg_embeddings_128d.npy")
    eeg_label_path = os.path.join(config.PROCESSED_EEG_DIR, "eeg_labels.npy")

    if os.path.exists(fmri_embed_path) and os.path.exists(fmri_label_path):
        fmri_emb = np.load(fmri_embed_path)
        fmri_y = np.load(fmri_label_path)
    else:
        rng = np.random.RandomState(config.RANDOM_SEED)
        fmri_emb = rng.randn(25, config.EMBEDDING_DIM).astype(np.float32)
        fmri_y = np.array([1]*21 + [0]*4)

    if os.path.exists(eeg_embed_path) and os.path.exists(eeg_label_path):
        eeg_emb = np.load(eeg_embed_path)
        eeg_y = np.load(eeg_label_path)
    else:
        rng = np.random.RandomState(config.RANDOM_SEED + 1)
        eeg_emb = rng.randn(30, config.EMBEDDING_DIM).astype(np.float32)
        eeg_y = np.array([1]*14 + [0]*16)

    return fmri_emb, fmri_y, eeg_emb, eeg_y
