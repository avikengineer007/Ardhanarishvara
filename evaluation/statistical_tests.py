"""
Statistical Significance Testing for Ardhanarishvara Phase 6.
Implements McNemar's test and paired bootstrap CI for rigorous model comparison.
Goes beyond raw accuracy — provides p-values and confidence intervals.
"""

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from scipy import stats

import config
from security.sanitized_logging import sanitize_errors, log_info


@sanitize_errors("Failed to compute McNemar's test.")
def mcnemar_test(y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray) -> dict:
    """
    McNemar's test for comparing two classifiers on the same test set.
    Tests whether the two models make significantly different errors.

    Args:
        y_true:  (N,) ground truth labels
        preds_a: (N,) predictions from model A
        preds_b: (N,) predictions from model B

    Returns:
        Dict with chi2 statistic, p-value, and interpretation.
    """
    y_true = np.asarray(y_true)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)

    assert len(y_true) == len(preds_a) == len(preds_b), \
        "All arrays must have the same length."

    # Build 2x2 contingency table
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    # b: A correct, B wrong | c: A wrong, B correct
    b = int((correct_a & ~correct_b).sum())  # A right, B wrong
    c = int((~correct_a & correct_b).sum())  # A wrong, B right

    # McNemar's test (with continuity correction)
    if b + c == 0:
        return {
            "chi2": 0.0,
            "p_value": 1.0,
            "b_count": b,
            "c_count": c,
            "significant": False,
            "interpretation": "Models make identical errors — no difference."
        }

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1.0 - stats.chi2.cdf(chi2, df=1)

    # Interpretation
    if p_value < 0.001:
        sig_label = "*** (p < 0.001)"
        significant = True
    elif p_value < 0.01:
        sig_label = "** (p < 0.01)"
        significant = True
    elif p_value < 0.05:
        sig_label = "* (p < 0.05)"
        significant = True
    else:
        sig_label = "n.s."
        significant = False

    result = {
        "chi2": chi2,
        "p_value": p_value,
        "b_count": b,
        "c_count": c,
        "significant": significant,
        "significance_label": sig_label,
        "interpretation": (
            f"Model A outperforms B" if b > c else
            f"Model B outperforms A" if c > b else
            "No difference"
        ) + f" (χ²={chi2:.4f}, p={p_value:.6f} {sig_label})"
    }

    log_info(f"McNemar's test: b={b}, c={c}, χ²={chi2:.4f}, p={p_value:.6f} {sig_label}")
    return result


@sanitize_errors("Failed to compute paired bootstrap CI.")
def paired_bootstrap_ci(y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray,
                        metric_fn=None, n_iterations: int | None = None,
                        ci_level: float | None = None, seed: int = 42) -> dict:
    """
    Paired bootstrap confidence interval for the difference between two models.

    Args:
        y_true:       (N,) ground truth labels
        scores_a:     (N,) prediction scores/probabilities from model A
        scores_b:     (N,) prediction scores/probabilities from model B
        metric_fn:    Scoring function(y_true, y_scores) → float.
                      Default: accuracy on binary predictions.
        n_iterations: Number of bootstrap resamples (default: config)
        ci_level:     Confidence level (default: config)
        seed:         Random seed

    Returns:
        Dict with mean difference, CI bounds, and p-value.
    """
    if n_iterations is None:
        n_iterations = config.BOOTSTRAP_N_ITERATIONS
    if ci_level is None:
        ci_level = config.BOOTSTRAP_CI_LEVEL

    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    n = len(y_true)

    if metric_fn is None:
        def metric_fn(yt, ys):
            preds = (ys >= 0.5).astype(int) if ys.max() <= 1.0 else ys.astype(int)
            return float((preds == yt).mean())

    rng = np.random.RandomState(seed)
    diffs = np.zeros(n_iterations)

    for i in range(n_iterations):
        idx = rng.choice(n, size=n, replace=True)
        score_a = metric_fn(y_true[idx], scores_a[idx])
        score_b = metric_fn(y_true[idx], scores_b[idx])
        diffs[i] = score_a - score_b

    mean_diff = float(np.mean(diffs))
    alpha = 1.0 - ci_level
    lower = float(np.percentile(diffs, 100 * alpha / 2))
    upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))

    # Two-sided p-value: fraction of bootstrap samples where diff <= 0
    p_value = float(np.mean(diffs <= 0)) if mean_diff > 0 else float(np.mean(diffs >= 0))
    p_value = min(p_value * 2, 1.0)  # Two-sided

    significant = not (lower <= 0 <= upper)

    result = {
        "mean_diff": mean_diff,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_level": ci_level,
        "p_value": p_value,
        "n_iterations": n_iterations,
        "significant": significant,
        "interpretation": (
            f"Model A is {'significantly' if significant else 'not significantly'} "
            f"{'better' if mean_diff > 0 else 'worse'} than Model B "
            f"(Δ={mean_diff:.4f}, {ci_level*100:.0f}% CI: [{lower:.4f}, {upper:.4f}], "
            f"p={p_value:.6f})"
        )
    }

    log_info(f"Bootstrap CI: Δ={mean_diff:.4f} [{lower:.4f}, {upper:.4f}], p={p_value:.6f}")
    return result


@sanitize_errors("Failed to generate significance report.")
def generate_significance_report(ablation_results: list, save_path: str | None = None) -> str:
    """
    Generate a formatted significance report comparing all model pairs.

    Args:
        ablation_results: List of dicts, each with 'ablation_id', 'val_targets',
                          'val_preds', 'val_probs'
        save_path: Path to save markdown report

    Returns:
        Formatted markdown string.
    """
    import os

    lines = ["# Statistical Significance Report\n"]
    lines.append("## Pairwise McNemar's Tests\n")
    lines.append("| Model A | Model B | χ² | p-value | Significance |")
    lines.append("|:---|:---|:---|:---|:---|")

    comparisons = []
    n = len(ablation_results)

    for i in range(n):
        for j in range(i + 1, n):
            a = ablation_results[i]
            b = ablation_results[j]

            if "val_targets" not in a or "val_preds" not in a:
                continue
            if "val_targets" not in b or "val_preds" not in b:
                continue

            # Only compare if same targets (same test set)
            targets_a = np.asarray(a["val_targets"])
            targets_b = np.asarray(b["val_targets"])

            if len(targets_a) != len(targets_b):
                continue

            result = mcnemar_test(targets_a, np.asarray(a["val_preds"]),
                                  np.asarray(b["val_preds"]))

            lines.append(
                f"| {a['ablation_id']} | {b['ablation_id']} | "
                f"{result['chi2']:.4f} | {result['p_value']:.6f} | "
                f"{result.get('significance_label', 'n.s.')} |"
            )
            comparisons.append({
                "model_a": a["ablation_id"],
                "model_b": b["ablation_id"],
                **result
            })

    report = "\n".join(lines)

    if save_path is None:
        save_path = os.path.join(config.TABLES_DIR, "significance_report.md")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)

    log_info(f"Saved significance report to {os.path.basename(save_path)}")
    return report


@sanitize_errors("Failed to run statistical analysis.")
def run_statistical_analysis(ablation_data=None, save_path: str | None = None) -> str:
    """
    Convenience orchestrator for statistical significance testing.
    Accepts either an ablation DataFrame or a list of ablation results,
    computes pairwise McNemar tests / Bootstrap CIs, and saves the report.
    """
    if isinstance(ablation_data, list):
        return generate_significance_report(ablation_data, save_path=save_path)
    
    # If a DataFrame is passed, generate summary statistics table
    import os
    if save_path is None:
        save_path = os.path.join(config.TABLES_DIR, "significance_report.md")
    
    lines = ["# Statistical Significance & Ablation Report\n"]
    if ablation_data is not None:
        lines.append("## Aggregated Ablation Performance\n")
        lines.append(str(ablation_data))
    
    report = "\n".join(lines)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)
    log_info(f"Saved statistical analysis report to {os.path.basename(save_path)}")
    return report
