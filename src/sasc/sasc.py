from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


@dataclass(frozen=True)
class SASCConfig:
    n_entropy_bins: int = 8
    support_margin_std: float = 0.25
    t_max_absolute: float = 1.45
    t_max_additive: float = 0.35
    max_extrapolation_slope: float = 0.75
    lower_fallback_to_source_ts: bool = True


def make_entropy_bins(
    train_df: pd.DataFrame,
    entropy_col: str = "entropy_mean",
    target_t_col: str = "source_bank_target_temperature",
    n_bins: int = 8,
) -> pd.DataFrame:
    """Make entropy bins used for the monotone isotonic SASC map."""
    required = [entropy_col, target_t_col]
    missing = [c for c in required if c not in train_df.columns]
    if missing:
        raise KeyError(f"Missing columns for entropy bins: {missing}")

    df = train_df[required].dropna().copy().sort_values(entropy_col).reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("At least two source-bank conditions are required")

    df["bin_id"] = pd.qcut(
        df[entropy_col],
        q=min(int(n_bins), len(df)),
        labels=False,
        duplicates="drop",
    )

    binned = (
        df.groupby("bin_id", as_index=False)
        .agg(
            entropy_bin_mean=(entropy_col, "mean"),
            entropy_bin_min=(entropy_col, "min"),
            entropy_bin_max=(entropy_col, "max"),
            target_T_bin_mean=(target_t_col, "mean"),
            target_T_bin_median=(target_t_col, "median"),
            target_T_bin_std=(target_t_col, "std"),
            n_bin=(target_t_col, "count"),
        )
        .sort_values("entropy_bin_mean")
        .reset_index(drop=True)
    )

    if len(binned) < 2:
        raise ValueError("Too few entropy bins for isotonic fit")
    return binned


def fit_sasc_model(
    train_df: pd.DataFrame,
    source_t: float,
    config: SASCConfig | None = None,
    entropy_col: str = "entropy_mean",
    target_t_col: str = "source_bank_target_temperature",
) -> dict:
    """Fit the frozen SASC/Frozen V3 model from source-bank conditions."""
    config = config or SASCConfig()
    source_t = float(source_t)
    if source_t <= 0:
        raise ValueError("source_t must be positive")

    binned = make_entropy_bins(
        train_df=train_df,
        entropy_col=entropy_col,
        target_t_col=target_t_col,
        n_bins=config.n_entropy_bins,
    )

    x_bin = binned["entropy_bin_mean"].to_numpy(dtype=float)
    y_bin = binned["target_T_bin_mean"].to_numpy(dtype=float)

    t_min = source_t
    t_max_safe = float(min(config.t_max_absolute, source_t + config.t_max_additive))
    y_safe = np.clip(y_bin, t_min, t_max_safe)

    iso = IsotonicRegression(
        increasing=True,
        out_of_bounds="clip",
        y_min=t_min,
        y_max=t_max_safe,
    )
    iso.fit(x_bin, y_safe)

    x_train = train_df[entropy_col].to_numpy(dtype=float)
    x_min = float(np.min(x_train))
    x_max = float(np.max(x_train))
    x_std = float(np.std(x_train, ddof=0))

    support_low = x_min - config.support_margin_std * x_std
    support_high = x_max + config.support_margin_std * x_std

    if len(x_bin) >= 2:
        dx = float(x_bin[-1] - x_bin[-2])
        dy = float(iso.predict([x_bin[-1]])[0] - iso.predict([x_bin[-2]])[0])
        raw_slope = 0.0 if abs(dx) < 1e-12 else dy / dx
    else:
        raw_slope = 0.0

    extrapolation_slope = float(np.clip(raw_slope, 0.0, config.max_extrapolation_slope))
    t_at_train_max = float(iso.predict([x_max])[0])

    return {
        "model": iso,
        "binned": binned,
        "config": asdict(config),
        "source_T": source_t,
        "T_min": t_min,
        "T_max_safe": t_max_safe,
        "entropy_train_min": x_min,
        "entropy_train_max": x_max,
        "entropy_train_mean": float(np.mean(x_train)),
        "entropy_train_std": x_std,
        "support_low": float(support_low),
        "support_high": float(support_high),
        "T_at_train_max": t_at_train_max,
        "raw_extrapolation_slope": float(raw_slope),
        "extrapolation_slope": extrapolation_slope,
        "n_train_conditions": int(len(train_df)),
        "n_bins_used": int(len(binned)),
    }


def predict_sasc_temperature(model_info: dict, entropy_value: float) -> dict:
    """Predict temperature and support-region label for one target stream."""
    entropy_value = float(entropy_value)
    iso = model_info["model"]
    cfg = model_info["config"]

    source_t = float(model_info["source_T"])
    t_min = float(model_info["T_min"])
    t_max_safe = float(model_info["T_max_safe"])
    support_low = float(model_info["support_low"])
    support_high = float(model_info["support_high"])
    entropy_train_max = float(model_info["entropy_train_max"])

    below_support = entropy_value < support_low
    above_support = entropy_value > support_high
    inside_support = not below_support and not above_support

    used_lower_fallback = False
    used_high_extrapolation = False

    if below_support and bool(cfg.get("lower_fallback_to_source_ts", True)):
        raw_t = source_t
        pred_t = source_t
        used_lower_fallback = True
    elif above_support:
        base_t = float(model_info["T_at_train_max"])
        slope = float(model_info["extrapolation_slope"])
        raw_t = base_t + slope * (entropy_value - entropy_train_max)
        pred_t = float(np.clip(raw_t, t_min, t_max_safe))
        used_high_extrapolation = True
    else:
        raw_t = float(iso.predict([entropy_value])[0])
        pred_t = float(np.clip(raw_t, t_min, t_max_safe))

    if used_lower_fallback:
        support_region = "lower_fallback"
    elif used_high_extrapolation:
        support_region = "high_extrapolation"
    else:
        support_region = "inside_support"

    return {
        "predicted_T": float(pred_t),
        "raw_predicted_T": float(raw_t),
        "support_region": support_region,
        "used_lower_fallback": bool(used_lower_fallback),
        "used_high_extrapolation": bool(used_high_extrapolation),
        "inside_support": bool(inside_support),
        "below_support": bool(below_support),
        "above_support": bool(above_support),
    }
