"""
fMRI CPAC Preprocessing & CC200 Connectivity Matrix Pipeline for Ardhanarishvara.
Fetches authentic ABIDE-I preprocessed datasets (CPAC pipeline, CC200 atlas).
Extracts CC200 ROI timeseries and computes Fisher z-transformed Pearson correlation matrices (200x200).
Caches matrices as .npy and .h5 to prevent redundant downloads.
Enforces strict security validation, rate-limiting, and error sanitization.
"""

import os
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import h5py
# pyrefly: ignore [missing-import]
from nilearn import datasets
import config
from security.sanitized_logging import sanitize_errors, log_info
from security.validation import validate_connectivity_matrix, validate_file_path
from security.rate_limiter import rate_limit_downloads


@rate_limit_downloads()
def fetch_real_abide_cc200(n_subjects: int = 30):
    """Fetch authentic ABIDE-I preprocessed CC200 ROI timeseries from Nilearn/CPAC with rate limiting."""
    log_info(f"Fetching {n_subjects} authentic ABIDE-I subjects (CPAC pipeline, CC200 atlas)...")
    abide_data = datasets.fetch_abide_pcp(
        data_dir=config.FMRI_DIR,
        pipeline="cpac",
        derivatives=["rois_cc200"],
        n_subjects=n_subjects
    )
    return abide_data


@sanitize_errors("Failed to compute fMRI CC200 connectivity matrix.")
def process_fmri_subject(timeseries_file_or_data, subject_id: str = "sub_001") -> np.ndarray:
    """
    Process CC200 ROI timeseries into a 200x200 Fisher z-transformed correlation connectivity matrix.
    Uses caching (.npy / .h5) to prevent duplicate preprocessing.
    """
    cache_npy_path = os.path.join(config.PROCESSED_FMRI_DIR, f"{subject_id}_cc200.npy")
    cache_h5_path = os.path.join(config.PROCESSED_FMRI_DIR, f"{subject_id}_cc200.h5")

    # Check cache first
    if os.path.exists(cache_npy_path):
        validate_file_path(cache_npy_path)
        matrix = np.load(cache_npy_path)
        return validate_connectivity_matrix(matrix, expected_dim=(200, 200), name=f"fMRI CC200 Matrix ({subject_id})")

    # Load timeseries data
    if isinstance(timeseries_file_or_data, str):
        validate_file_path(timeseries_file_or_data)
        ts_data = np.loadtxt(timeseries_file_or_data)
    elif isinstance(timeseries_file_or_data, np.ndarray):
        ts_data = timeseries_file_or_data
    else:
        ts_data = np.asarray(timeseries_file_or_data)

    # Ensure shape is (T, 200)
    if ts_data.ndim == 2 and ts_data.shape[0] == 200 and ts_data.shape[1] != 200:
        ts_data = ts_data.T

    # Compute Pearson Correlation connectivity matrix (Vectorized)
    # Filter any constant ROI columns that have zero variance
    std = np.std(ts_data, axis=0)
    ts_data[:, std == 0] = np.random.randn(ts_data.shape[0], int((std == 0).sum())) * 1e-6

    corr_matrix = np.corrcoef(ts_data.T)  # (200, 200)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Apply Fisher z-transform: z = arctanh(r), clipping r to (-0.9999, 0.9999)
    clipped_corr = np.clip(corr_matrix, -0.9999, 0.9999)
    z_matrix = np.arctanh(clipped_corr)
    # Set self-correlation diagonal
    np.fill_diagonal(z_matrix, 1.0)

    # Security validation
    validated_matrix = validate_connectivity_matrix(z_matrix, expected_dim=(200, 200), name=f"fMRI CC200 Matrix ({subject_id})")

    # Cache to .npy and .h5
    np.save(cache_npy_path, validated_matrix)
    with h5py.File(cache_h5_path, "w") as f:
        f.create_dataset("connectivity", data=validated_matrix)

    log_info(f"Generated & cached fMRI CC200 matrix for {subject_id} (Shape: {validated_matrix.shape})")
    return validated_matrix
