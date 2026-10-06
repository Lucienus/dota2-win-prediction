"""Probability metrics and paired, match-level uncertainty estimates."""
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score, brier_score_loss


def calibration_bins(y, p, bins=10):
    result = []
    indices = np.minimum((p * bins).astype(int), bins - 1)
    for i in range(bins):
        mask = indices == i
        result.append({"lower": i / bins, "upper": (i + 1) / bins, "n": int(mask.sum()),
                       "mean_probability": float(p[mask].mean()) if mask.any() else None,
                       "observed_win_rate": float(y[mask].mean()) if mask.any() else None})
    return result


def probability_metrics(labels, probabilities):
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if len(y) == 0 or len(y) != len(p) or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid or empty probability vectors")
    pred = p >= 0.5
    bins = calibration_bins(y, p)
    ece = sum(row["n"] * abs(row["mean_probability"] - row["observed_win_rate"]) for row in bins if row["n"]) / len(y)
    return {"n": len(y), "radiant_win_rate": float(y.mean()), "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2 else None,
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)), "ece_10_equal_width_bins": float(ece), "calibration_bins": bins}


def paired_bootstrap(labels, probabilities_a, probabilities_b, repeats=2000, seed=731):
    """One observation per match at a fixed minute. CI conditional on these fitted models."""
    y, a, b = np.asarray(labels), np.asarray(probabilities_a), np.asarray(probabilities_b)
    if not len(y) or len(a) != len(y) or len(b) != len(y):
        raise ValueError("Paired vectors must be nonempty and equal length")
    differences = np.column_stack((((a >= .5) == y).astype(float) - ((b >= .5) == y), (a - y) ** 2 - (b - y) ** 2))
    rng = np.random.default_rng(seed)
    samples = np.empty((repeats, 2))
    for i in range(repeats):
        samples[i] = differences[rng.integers(0, len(y), len(y))].mean(0)
    return {name: {"difference_a_minus_b": float(differences[:, j].mean()),
                   "ci95": np.quantile(samples[:, j], [.025, .975]).tolist()}
            for j, name in enumerate(("accuracy", "brier"))}
