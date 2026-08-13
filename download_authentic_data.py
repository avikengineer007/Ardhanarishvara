"""
Automated Authentic Dataset Downloader for Ardhanarishvara.
Downloads:
  1. ABIDE-I authentic CPAC CC200 preprocessed fMRI timeseries via Nilearn (no credentials needed).
  2. OpenNeuro ds006780 (SFARI ASD EEG dataset) or Mendeley ASD EEG via Python APIs.
"""

import os
import sys
# pyrefly: ignore [missing-import]
import numpy as np

import config
from security.sanitized_logging import log_info, sanitize_errors


@sanitize_errors("Failed to download ABIDE-I fMRI data.")
def download_real_abide_fmri(n_subjects: int = 50):
    """
    Download authentic ABIDE-I CPAC CC200 ROI timeseries using Nilearn.
    Saves to data/raw/fmri/
    """
    # pyrefly: ignore [missing-import]
    from nilearn import datasets

    log_info(f"Downloading {n_subjects} authentic ABIDE-I subjects (CPAC pipeline, CC200 atlas)...")
    os.makedirs(config.FMRI_DIR, exist_ok=True)

    abide_data = datasets.fetch_abide_pcp(
        data_dir=config.FMRI_DIR,
        pipeline="cpac",
        derivatives=["rois_cc200"],
        n_subjects=n_subjects,
        verbose=1
    )

    actual_n = len(abide_data.rois_cc200)
    asd_count = int(sum(1 for dx in abide_data.phenotypic["DX_GROUP"] if dx == 1))
    td_count = int(sum(1 for dx in abide_data.phenotypic["DX_GROUP"] if dx == 2))

    log_info(f"Successfully downloaded {actual_n} authentic ABIDE-I subjects!")
    log_info(f"Class breakdown: ASD={asd_count}, TD={td_count}")
    return abide_data


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    download_real_abide_fmri(n_subjects=n)
