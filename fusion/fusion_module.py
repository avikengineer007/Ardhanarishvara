"""
Multimodal Fusion Architectures for Ardhanarishvara Phase 4.
Two fusion strategies for combining 128-dim fMRI and 128-dim EEG embeddings:
  1. ConcatMLPFusion   — Simple concatenation + MLP (fast, debuggable baseline)
  2. CrossAttentionFusion — Bidirectional cross-modal attention with interpretable weights
Both accept frozen encoder embeddings and produce ASD/TD predictions with confidence.
Also includes UnimodalClassifier for single-branch baseline comparison.
"""

import math
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.nn.functional as F

import config


class ConcatMLPFusion(nn.Module):
    """
    Concatenation + MLP Fusion for multimodal ASD classification.
    Architecture:
        fmri_embed(128) ⊕ eeg_embed(128) → concat(256)
        → Linear(256, 256) → BN → ReLU → Dropout(0.3)
        → Linear(256, 128) → BN → ReLU → Dropout(0.3)
        → Linear(128, 2) → logits
    """

    def __init__(self, fmri_dim: int = 128, eeg_dim: int = 128,
                 hidden_dim: int = 256, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.fmri_dim = fmri_dim
        self.eeg_dim = eeg_dim
        combined_dim = fmri_dim + eeg_dim

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, fmri_embed: torch.Tensor, eeg_embed: torch.Tensor):
        """
        Args:
            fmri_embed: (B, 128) fMRI embedding from frozen encoder
            eeg_embed:  (B, 128) EEG embedding from frozen encoder
        Returns:
            logits:       (B, 2)  class logits for ASD vs TD
            confidence:   (B,)    max softmax probability
            attn_weights: None    (not applicable for concat fusion)
        """
        combined = torch.cat([fmri_embed, eeg_embed], dim=1)  # (B, 256)
        logits = self.classifier(combined)

        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            confidence = probs.max(dim=1).values

        return logits, confidence, None


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional Cross-Modal Attention Fusion for multimodal ASD classification.
    Splits 128-dim embeddings into num_heads groups, applies cross-attention in both
    directions (fMRI → EEG and EEG → fMRI), and classifies the gated residual output.
    Returns interpretable per-head attention weights for XAI analysis.

    Architecture:
        fMRI embed(128) ──Q──► CrossAttn(fMRI→EEG) ──► gated residual ──┐
                                                                         ├─ concat(256) → MLP → logits
        EEG  embed(128) ──Q──► CrossAttn(EEG→fMRI) ──► gated residual ──┘
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 4,
                 hidden_dim: int = 256, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self._attn_scale = math.sqrt(self.head_dim)

        # --- Cross-attention projections: fMRI queries attending to EEG ---
        self.q_fmri = nn.Linear(embed_dim, embed_dim)
        self.k_eeg = nn.Linear(embed_dim, embed_dim)
        self.v_eeg = nn.Linear(embed_dim, embed_dim)

        # --- Cross-attention projections: EEG queries attending to fMRI ---
        self.q_eeg = nn.Linear(embed_dim, embed_dim)
        self.k_fmri = nn.Linear(embed_dim, embed_dim)
        self.v_fmri = nn.Linear(embed_dim, embed_dim)

        # Output projections
        self.out_fmri = nn.Linear(embed_dim, embed_dim)
        self.out_eeg = nn.Linear(embed_dim, embed_dim)

        # Layer normalization for attended outputs
        self.ln_fmri = nn.LayerNorm(embed_dim)
        self.ln_eeg = nn.LayerNorm(embed_dim)

        # Gating mechanism for residual connections
        self.gate_fmri = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
        self.gate_eeg = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())

        # MLP classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def _cross_attend(self, query, key, value, q_proj, k_proj, v_proj, out_proj):
        """
        Compute multi-head cross-attention between two modalities.
        Splits embedding into num_heads groups and computes per-head gated attention.

        Returns:
            output: (B, embed_dim) attended representation
            attn_weights: (B, num_heads) per-head attention strengths
        """
        B = query.size(0)

        # Project and reshape to (B, num_heads, head_dim)
        Q = q_proj(query).view(B, self.num_heads, self.head_dim)
        K = k_proj(key).view(B, self.num_heads, self.head_dim)
        V = v_proj(value).view(B, self.num_heads, self.head_dim)

        # Per-head attention: dot-product similarity → sigmoid gating
        attn_scores = (Q * K).sum(dim=-1, keepdim=True) / self._attn_scale  # (B, H, 1)
        attn_weights = torch.sigmoid(attn_scores)  # (B, H, 1)

        # Weighted value aggregation
        attended = attn_weights * V  # (B, H, head_dim)
        attended = attended.reshape(B, self.embed_dim)  # (B, embed_dim)
        output = out_proj(attended)

        return output, attn_weights.squeeze(-1)  # (B, embed_dim), (B, H)

    def forward(self, fmri_embed: torch.Tensor, eeg_embed: torch.Tensor):
        """
        Args:
            fmri_embed: (B, 128) fMRI embedding from frozen encoder
            eeg_embed:  (B, 128) EEG embedding from frozen encoder
        Returns:
            logits:       (B, 2)  class logits
            confidence:   (B,)    max softmax probability
            attn_weights: dict    {'fmri_to_eeg': (B, H), 'eeg_to_fmri': (B, H)}
        """
        # Bidirectional cross-attention
        fmri_attended, attn_f2e = self._cross_attend(
            fmri_embed, eeg_embed, eeg_embed,
            self.q_fmri, self.k_eeg, self.v_eeg, self.out_fmri
        )
        eeg_attended, attn_e2f = self._cross_attend(
            eeg_embed, fmri_embed, fmri_embed,
            self.q_eeg, self.k_fmri, self.v_fmri, self.out_eeg
        )

        # Gated residual: gate * attended + (1-gate) * original
        gate_f = self.gate_fmri(torch.cat([fmri_embed, fmri_attended], dim=1))
        fmri_fused = self.ln_fmri(gate_f * fmri_attended + (1.0 - gate_f) * fmri_embed)

        gate_e = self.gate_eeg(torch.cat([eeg_embed, eeg_attended], dim=1))
        eeg_fused = self.ln_eeg(gate_e * eeg_attended + (1.0 - gate_e) * eeg_embed)

        # Classify concatenated fused representations
        combined = torch.cat([fmri_fused, eeg_fused], dim=1)  # (B, 256)
        logits = self.classifier(combined)

        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            confidence = probs.max(dim=1).values

        attn_weights = {
            "fmri_to_eeg": attn_f2e.detach(),   # (B, num_heads)
            "eeg_to_fmri": attn_e2f.detach(),    # (B, num_heads)
        }

        return logits, confidence, attn_weights


class UnimodalClassifier(nn.Module):
    """
    Simple linear classifier for single-modality baseline comparison.
    Used to establish fMRI-only and EEG-only accuracy baselines.
    """

    def __init__(self, embed_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, embed: torch.Tensor):
        """
        Args:
            embed: (B, 128) single-modality embedding
        Returns:
            logits:     (B, 2) class logits
            confidence: (B,) max softmax probability
        """
        logits = self.classifier(embed)
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            confidence = probs.max(dim=1).values
        return logits, confidence
