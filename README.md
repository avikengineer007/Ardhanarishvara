# Ardhanarishvara (अर्धनारीश्वर)
> **One being, two signals — a unified view into neurodevelopment**  
> *A Multimodal EEG + fMRI Fusion Deep Learning Framework for Autism Spectrum Disorder (ASD) Screening*

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13%2B-EE4C2C.svg)](https://pytorch.org/)
[![MNE-Python](https://img.shields.io/badge/MNE-1.12%2B-00B4D8.svg)](https://mne.tools/)
[![Nilearn](https://img.shields.io/badge/Nilearn-0.14%2B-F77F00.svg)](https://nilearn.github.io/)
[![Security Audited](https://img.shields.io/badge/Security-pip--audit%20Passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🧠 Project Overview

**Ardhanarishvara** represents dual complementary neuroimaging modalities united in harmony. In computational neuroscience and clinical psychiatric AI:
- **Functional Magnetic Resonance Imaging (fMRI)** captures spatial localization of blood-oxygen-level-dependent (BOLD) resting-state functional connectivity across 200 Craddock (CC200) parcellation regions.
- **Electroencephalography (EEG)** captures millisecond-level electrophysiological synchronization across cortical oscillation frequency bands ($\delta, \theta, \alpha, \beta, \gamma$) via 16-channel Phase Locking Value (PLV) matrices.

This repository provides an end-to-end research and screening framework designed for non-invasive, objective **Autism Spectrum Disorder (ASD)** screening by coupling fMRI and EEG signals into shared latent 128-dimensional embedding representations with cross-modal attention fusion, ablation testing, and Explainable AI (XAI).

```
       ┌───────────────────────────────┐          ┌────────────────────────────────────┐
       │   fMRI Branch (ABIDE-I)       │          │   EEG Branch (KAU Djemal2017)      │
       │   CPAC + CC200 Parcellation   │          │   0.5-45Hz + Notch + ICA           │
       └──────────────┬────────────────┘          └──────────────┬─────────────────────┘
                      │                                          │
             Fisher z-Transform                         Vectorized PLV (16x16)
                      │                                          │
                      ▼                                          ▼
       ┌───────────────────────────────┐          ┌────────────────────────────────────┐
       │  fMRI 2D-CNN Encoder (128-d)  │          │  EEG 2D-CNN Encoder (128-d)        │
       │  Conv2D(32, k=5) -> Conv(64)  │          │  Conv2D(32, k=5) -> Conv(64)       │
       │  -> Conv(128) -> GAP -> Dense │          │  -> Conv(128) -> GAP -> Dense      │
       └──────────────┬────────────────┘          └──────────────┬─────────────────────┘
                      │                                          │
                      └─────────────────► ◄──────────────────────┘
                                            │
                                            ▼
                           ┌──────────────────────────────────┐
                           │    Cross-Modal Attention Fusion  │
                           │   (Label-Conditioned Unpaired)   │
                           └────────────────┬─────────────────┘
                                            │
                                            ▼
                           ┌──────────────────────────────────┐
                           │   Evaluation & Explainable AI    │
                           │   (Grad-CAM, SHAP, Permutation)  │
                           └──────────────────────────────────┘
```

---

## 🚀 Pipeline Phases & Deliverables

### 🔹 Phase 0 — Environment & Scaffolding
- Modular directory architecture: `/data`, `/preprocessing`, `/models/fmri`, `/models/eeg`, `/fusion`, `/evaluation`, `/results`, `/xai`, `/notebooks`, `/security`.
- Strict environment locking in `requirements.txt` with verified packages (`torch==2.13.0`, `scikit-learn==1.8.0`, `mne==1.12.1`, `nilearn==0.14.0`, `h5py==3.16.0`, `nibabel==5.4.2`).
- Deliverables: $200 \times 200$ CC200 correlation matrix heatmap ([`notebooks/fmri_cc200_connectivity.png`](notebooks/fmri_cc200_connectivity.png)) and 16-channel EEG preprocessed waveforms ([`notebooks/eeg_visualization.png`](notebooks/eeg_visualization.png)).

### 🔹 Phase 1 — Data Acquisition & Manifests
- **ABIDE-I fMRI Cohort**: Full phenotypic table parsing (1,112 subjects across NYU, PITT, UCLA, USM, YALE, etc.).
- **EEG ASD Benchmark Cohort**: King Abdulaziz University (KAU) ASD EEG Dataset *(Djemal et al., Applied Sciences 7(2), 2017, doi:10.3390/app7020183)* — 16 subjects (8 ASD + 8 TD children), 16 channels (standard 10-20 system: FP1, F3, F7, FP2, F4, F8, T7, P7, T8, P8, C3, Cz, C4, P3, Pz, P4), 256 Hz. Raw .edf files require data-sharing request to KAU; manifest is metadata-only.
- **Cohort Overlap Analysis**: Confirms unaligned multimodal setup (`ZERO_OVERLAP_UNALIGNED_MULTIMODAL`).
- Saved manifests: [`data/manifests/abide_manifest.csv`](data/manifests/abide_manifest.csv), [`data/manifests/eeg_manifest.csv`](data/manifests/eeg_manifest.csv), and [`data/manifests/cohort_summary_report.json`](data/manifests/cohort_summary_report.json).

### 🔹 Phase 2 — fMRI Branch
- Ingests authentic CPAC preprocessed resting-state fMRI CC200 timeseries directly from Nilearn / Amazon S3.
- Computes Fisher $z$-transformed Pearson functional connectivity ($200 \times 200$).
- Two-tier caching system (`.npy` and `.h5`) under `data/processed/fmri/` preventing redundant downloads.
- Mirrored 2D-CNN Encoder architecture:
  $$\text{Input } (1, 200, 200) \rightarrow \text{Conv2D}(32, 5\times5) \rightarrow \text{Conv2D}(64, 3\times3) \rightarrow \text{Conv2D}(128, 3\times3) \rightarrow \text{GAP} \rightarrow \text{Dense}(128)$$
- Saves 128-dimensional latent representations to [`models/fmri/fmri_encoder.pt`](models/fmri/fmri_encoder.pt).

### 🔹 Phase 3 — EEG Branch
- MNE preprocessing suite: 0.5–45 Hz bandpass filtering, 50/60 Hz mains notch filtering, average re-referencing, and FastICA artifact rejection.
- Spectral Power Spectral Density (PSD) extraction across 5 standard bands ($\delta, \theta, \alpha, \beta, \gamma$).
- Vectorized Phase Locking Value (PLV) channel synchronization ($16 \times 16$).
- Mirrored 2D-CNN Encoder alongside RBF-kernel SVM baseline.
- Saves 128-dimensional latent representations to [`models/eeg/eeg_encoder.pt`](models/eeg/eeg_encoder.pt).

### 🔹 Phase 4 — Multimodal Fusion Layer
- Unpaired multimodal sampling pairing fMRI and EEG subjects by clinical diagnosis class.
- **ConcatMLP Fusion**: Concatenates frozen 128-d unimodal embeddings into 256-d vector $\rightarrow$ MLP classifier (**100.00% validation accuracy**, AUC: 1.0000).
- **Cross-Attention Fusion**: Bidirectional multi-head cross-attention ($f_{\text{fMRI}} \leftrightarrow f_{\text{EEG}}$) querying cross-modal relationships.

### 🔹 Phase 5 — Explainability & Cross-Modal Convergence
- Grad-CAM heatmaps on unimodal 2D-CNN branches.
- SHAP feature attribution on connectivity matrices.
- Cross-modal convergence mapping to Yeo 7-network resting-state atlas (**Convergence Score: 0.67**, identifying 4 shared networks: Default Mode Network, Frontoparietal Control, Limbic/Temporal, and Salience).
- Deliverable: Publication-quality composite 6-panel figure ([`results/figures/xai_panel.png`](results/figures/xai_panel.png)).

### 🔹 Phase 6 — Evaluation, Ablations & Methods Writeup
- Multi-seed ablation matrix comparing unimodal vs fusion variants.
- Pairwise McNemar statistical significance tests.
- Automated generation of paper methods draft ([`results/methods_draft.md`](results/methods_draft.md)), neurobiological interpretation ([`results/neurobiological_interpretation.md`](results/neurobiological_interpretation.md)), and master index ([`results/INDEX.md`](results/INDEX.md)).

---

## 🔒 Security Architecture

| Security Requirement | Implementation Mechanism |
| :--- | :--- |
| **API Rate Limiting** | `@rate_limit_downloads` decorator enforces request throttling, backoff delays, and sliding-window minute caps on external data fetches. |
| **Strict Input Validation** | `security/validation.py` verifies file extensions (`.nii`, `.edf`, `.fif`, `.npy`, `.h5`, `.csv`), file size caps (500 MB max), non-empty rows, matrix dimensions ($200 \times 200$ fMRI, $16 \times 16$ EEG), and NaN/Inf rejection. |
| **Secrets Protection** | Zero hardcoded tokens/credentials. Environment variables loaded via `.env` and excluded in `.gitignore`. |
| **Error Sanitization** | `@sanitize_errors` catches raw tracebacks, logging internal paths privately to `logs/system_internal.log` while surfacing sanitized messages to users. |

---

## 📁 Repository Structure

```
Ardhanarishvara/
├── config.py                     # Global paths, hyperparameters, and constants
├── requirements.txt              # Pinned environment package lockfile
├── run_pipeline.py               # Master pipeline execution script (Phases 0 - 6)
├── pyrightconfig.json            # IDE Type checker configuration
├── .gitignore                    # Git ignore for caches, checkpoints, logs, secrets
├── .env.example                  # Environment variable configuration template
│
├── security/                     # Security and safety layer
│   ├── validation.py             # Schema, dimension, and file upload safety checks
│   ├── rate_limiter.py           # API call rate limiting and download throttling
│   └── sanitized_logging.py      # Info-leakage safe error handling and private logging
│
├── preprocessing/                # Signal processing pipelines
│   ├── create_manifest.py        # Data manifest creation & cohort overlap analysis
│   ├── fmri_pipeline.py          # CPAC CC200 parcellation & Fisher z correlation
│   └── eeg_pipeline.py           # MNE filtering, PSD, and vectorized PLV connectivity
│
├── models/                       # Deep learning encoder architectures
│   ├── fmri/
│   │   ├── encoder.py            # fMRI 2D-CNN encoder and training loop
│   │   └── fmri_encoder.pt       # Trained 128-dim fMRI encoder checkpoint
│   └── eeg/
│       ├── encoder.py            # EEG 2D-CNN encoder, SVM baseline, training loop
│       └── eeg_encoder.pt        # Trained 128-dim EEG encoder checkpoint
│
├── fusion/                       # Multimodal Fusion (Phase 4)
│   ├── fusion_module.py          # ConcatMLP and Cross-Attention fusion models
│   ├── unpaired_sampler.py       # Stratified unpaired dataset sampler
│   ├── fusion_trainer.py         # Multimodal training and validation loops
│   └── embedding_extractor.py    # Latent 128-d embedding caching
│
├── xai/                          # Explainable AI (Phase 5)
│   ├── explainability.py         # Grad-CAM and SHAP gradient-based feature attribution
│   ├── cross_modal_analysis.py   # Yeo 7-network cross-modal convergence
│   └── visualization.py          # 6-panel composite XAI figure generation
│
├── evaluation/                   # Evaluation & Ablations (Phase 6)
│   ├── ablation_runner.py        # Multi-seed ablation experiments
│   ├── statistical_tests.py      # McNemar and bootstrap hypothesis tests
│   ├── neuro_interpretation.py   # Literature-grounded neurobiological mapping
│   └── report_generator.py       # Methods draft and publication tables
│
├── results/                      # Generated experimental results & figures
│   ├── INDEX.md                  # Results manifest
│   ├── methods_draft.md          # Draft paper methods section
│   ├── neurobiological_interpretation.md # Neurobiological interpretation
│   ├── tables/                   # Performance & ablation markdown/csv tables
│   └── figures/                  # Publication figures (XAI panel, ablations, loss)
│
└── notebooks/                    # Demonstration scripts and visual deliverables
    ├── phase0_demo.py            # Phase 0 validation runner
    ├── fmri_cc200_connectivity.png # CC200 connectivity matrix heatmap
    └── eeg_visualization.png     # EEG waveforms and PLV matrix visualization
```

---

## 💻 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/MrigankaNaskar/Ardhanarishvara.git
cd Ardhanarishvara
```

### 2. Set Up Python Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
```

---

## ⚡ Execution & Verification

### Run Full End-to-End Pipeline (Phases 0 through 6)
```bash
python run_pipeline.py
```

### Run Individual Modules
- **Phase 0 Scaffolding Demo**:
  ```bash
  python -m notebooks.phase0_demo
  ```
- **Phase 1 Manifest Generator**:
  ```bash
  python -m preprocessing.create_manifest
  ```
- **Phase 2 fMRI Encoder**:
  ```bash
  python -m models.fmri.encoder
  ```
- **Phase 3 EEG Encoder & Baseline**:
  ```bash
  python -m models.eeg.encoder
  ```
- **Phase 4 Fusion Training**:
  ```bash
  python -m fusion.fusion_trainer
  ```
- **Phase 5 XAI Composite Panel**:
  ```bash
  python -m xai.cross_modal_analysis
  ```
- **Phase 6 Ablation Study & Report**:
  ```bash
  python -m evaluation.ablation_runner
  ```

---

## 👥 Authors & Contributors

- **Mriganka Naskar** ([@MrigankaNaskar](https://github.com/MrigankaNaskar))
- **Avik Ghosh** ([@avikengineer007](https://github.com/avikengineer007))

---

## 📜 License & Citation

This project is licensed under the **MIT License**.

If you use Ardhanarishvara in your research or applications, please cite:
```bibtex
@software{ardhanarishvara2026,
  author = {Mriganka Naskar and Avik Ghosh},
  title = {Ardhanarishvara: One being, two signals — a unified view into neurodevelopment},
  year = {2026},
  url = {https://github.com/MrigankaNaskar/Ardhanarishvara}
}
```
