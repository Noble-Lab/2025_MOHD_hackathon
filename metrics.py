import numpy as np
from scipy.stats import spearmanr, pearsonr

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    resid = ((y_true - y_pred)**2).sum(axis=0)
    tot   = ((y_true - y_true.mean(axis=0))**2).sum(axis=0)
    valid = tot > 0
    if not np.any(valid):
        return float('nan')
    return float(1.0 - resid[valid].sum() / tot[valid].sum())

def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals = []
    for j in range(y_true.shape[1]):
        if np.std(y_true[:, j]) == 0: 
            continue
        rho, _ = spearmanr(y_true[:, j], y_pred[:, j])
        if not np.isnan(rho):
            vals.append(rho)
    return float(np.mean(vals)) if vals else float('nan')

def pearson_featurewise(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Pearson r separately for each feature (column),
    then return the mean across features with non-constant truth.
    """
    vals = []
    for j in range(y_true.shape[1]):
        if np.std(y_true[:, j]) == 0:
            continue
        r, _ = pearsonr(y_true[:, j], y_pred[:, j])
        if not np.isnan(r):
            vals.append(r)
    return float(np.mean(vals)) if vals else float("nan")

def pearson_flat(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Flatten both matrices and compute a single global Pearson r
    across all sample–feature entries.
    """
    r, _ = pearsonr(y_true.ravel(), y_pred.ravel())
    return float(r)

METRICS = {
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
    "spearman": spearman,
    "pearson_featurewise": pearson_featurewise,
    "pearson_flat": pearson_flat,
}