from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_ci_table(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
    n_boot: int = 2000,
    seed: int = 123,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for group_key, group in df.groupby(group_cols):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        base = {group_cols[i]: group_key[i] for i in range(len(group_cols))}
        for col in value_cols:
            values = group[col].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                mean = ci_low = ci_high = np.nan
            else:
                boot = []
                for _ in range(n_boot):
                    idx = rng.integers(0, len(values), size=len(values))
                    boot.append(float(np.mean(values[idx])))
                mean = float(np.mean(values))
                ci_low = float(np.percentile(boot, 2.5))
                ci_high = float(np.percentile(boot, 97.5))
            rows.append({**base, "quantity": col, "mean": mean, "ci95_low": ci_low, "ci95_high": ci_high, "n_rows": int(len(values)), "n_boot": int(n_boot)})
    return pd.DataFrame(rows)
