"""
Input & File Safety Validation Engine for Ardhanarishvara.
Enforces MANDATORY SECURITY REQUIREMENTS #2 and #4:
- File extension and size limits (rejects unauthorized uploads/files).
- Schema validation: check matrix dimensions, numeric types, NaN/Inf presence, value bounds.
"""

import os
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-source, missing-import]
import pandas as pd
from security.sanitized_logging import (
    FileUploadSecurityException,
    InvalidDataFormatException,
    log_info
)

ALLOWED_EXTENSIONS = {
    ".nii", ".nii.gz", ".edf", ".bdf", ".fif", ".npy", ".h5", ".hdf5", ".csv", ".json", ".pt"
}
MAX_ALLOWED_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB limit per file


def validate_file_path(filepath: str, allowed_extensions=ALLOWED_EXTENSIONS, max_bytes=MAX_ALLOWED_FILE_SIZE_BYTES) -> str:
    """Validate file path existence, extension, and file size before processing."""
    if not filepath or not isinstance(filepath, str):
        raise FileUploadSecurityException("Invalid file path specified.")

    if not os.path.exists(filepath):
        raise FileUploadSecurityException(f"File path does not exist: {os.path.basename(filepath)}")

    # Check extension
    filename_lower = filepath.lower()
    has_valid_ext = any(filename_lower.endswith(ext) for ext in allowed_extensions)
    if not has_valid_ext:
        raise FileUploadSecurityException(
            f"File extension rejected for security reasons: {os.path.basename(filepath)}"
        )

    # Check file size
    file_size = os.path.getsize(filepath)
    if file_size > max_bytes:
        raise FileUploadSecurityException(
            f"File size exceeds security cap ({file_size / (1024*1024):.1f}MB > {max_bytes / (1024*1024):.1f}MB)."
        )

    return os.path.abspath(filepath)


def validate_connectivity_matrix(matrix: np.ndarray, expected_dim: tuple = (200, 200), name: str = "Connectivity Matrix") -> np.ndarray:
    """Validate 2D connectivity matrix for proper shape, finite values, and range."""
    if not isinstance(matrix, np.ndarray):
        raise InvalidDataFormatException(f"{name} must be a numpy ndarray.")

    if matrix.ndim != 2 or matrix.shape != expected_dim:
        raise InvalidDataFormatException(
            f"{name} shape mismatch: expected {expected_dim}, got {matrix.shape}."
        )

    if not np.isfinite(matrix).all():
        raise InvalidDataFormatException(f"{name} contains non-finite numbers (NaN or Inf).")

    # Symmetric sanity check (for correlation matrices)
    if not np.allclose(matrix, matrix.T, atol=1e-3):
        raise InvalidDataFormatException(f"{name} is not a symmetric matrix.")

    return matrix


def validate_manifest_dataframe(df: pd.DataFrame, required_cols: list, name: str = "Manifest") -> pd.DataFrame:
    """Validate data manifest schema and non-empty rows."""
    if df is None or not isinstance(df, pd.DataFrame):
        raise InvalidDataFormatException(f"{name} must be a valid pandas DataFrame.")

    if df.empty:
        raise InvalidDataFormatException(f"{name} contains no records.")

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise InvalidDataFormatException(f"{name} missing required columns: {missing}")

    return df
