from __future__ import annotations

from typing import Dict

import numpy as np


def rmse(values: np.ndarray, axis=None):
    return np.sqrt(np.mean(np.square(values), axis=axis))


def compute_metrics(
    target: np.ndarray,
    mu: np.ndarray,
    obs_state: np.ndarray,
    true_state: np.ndarray,
    sigma: np.ndarray | None = None,
) -> Dict[str, float]:
    pred_error = mu - target
    corrected = obs_state - mu
    obs_error = obs_state - true_state
    corrected_error = corrected - true_state
    obs_norm = np.linalg.norm(obs_error, axis=1)
    corrected_norm = np.linalg.norm(corrected_error, axis=1)

    metrics = {
        "noise_rmse": float(rmse(pred_error)),
        "noise_rmse_dx": float(rmse(pred_error[:, 0])),
        "noise_rmse_dy": float(rmse(pred_error[:, 1])),
        "noise_rmse_vx": float(rmse(pred_error[:, 2])),
        "noise_rmse_vy": float(rmse(pred_error[:, 3])),
        "observed_rmse": float(rmse(obs_error)),
        "corrected_rmse": float(rmse(corrected_error)),
        "improvement_ratio": float(rmse(corrected_error) / max(rmse(obs_error), 1.0e-8)),
        "worsening_rate": float(np.mean(corrected_norm > obs_norm)),
    }
    if sigma is not None:
        var = np.square(np.maximum(sigma, 1.0e-8))
        nll = 0.5 * (np.square(target - mu) / var + np.log(var) + np.log(2.0 * np.pi))
        metrics["nll"] = float(np.mean(nll))
        metrics["predicted_sigma_mean"] = float(np.mean(sigma))
    else:
        metrics["nll"] = float("nan")
        metrics["predicted_sigma_mean"] = float("nan")
    return metrics

