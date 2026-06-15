#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sasc import SASCConfig, fit_sasc_model, predict_sasc_temperature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SASC/Frozen V3 from descriptor CSV files.")
    parser.add_argument("--source-ts", required=True, help="CSV with backbone, seed, temperature columns")
    parser.add_argument("--source-bank", required=True, help="CSV with source-bank stream descriptors and fitted temperatures")
    parser.add_argument("--target-stream", required=True, help="CSV with target stream descriptors")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_ts = pd.read_csv(args.source_ts)
    source_bank = pd.read_csv(args.source_bank)
    target_stream = pd.read_csv(args.target_stream)

    if "temperature" not in source_ts.columns and "source_T" in source_ts.columns:
        source_ts = source_ts.rename(columns={"source_T": "temperature"})

    required_source_ts = {"backbone", "seed", "temperature"}
    required_bank = {"backbone", "seed", "corruption", "severity", "condition", "entropy_mean", "source_bank_target_temperature"}
    required_target = {"backbone", "seed", "corruption", "severity", "condition", "entropy_mean"}

    for name, df, req in [
        ("source_ts", source_ts, required_source_ts),
        ("source_bank", source_bank, required_bank),
        ("target_stream", target_stream, required_target),
    ]:
        missing = sorted(req - set(df.columns))
        if missing:
            raise KeyError(f"{name} missing required columns: {missing}")

    config = SASCConfig()
    prediction_rows = []
    model_rows = []
    bin_rows = []

    for _, ts_row in source_ts.iterrows():
        backbone = str(ts_row["backbone"])
        seed = int(ts_row["seed"])
        source_t = float(ts_row["temperature"])

        train_df = source_bank[(source_bank["backbone"] == backbone) & (source_bank["seed"].astype(int) == seed)].copy()
        test_df = target_stream[(target_stream["backbone"] == backbone) & (target_stream["seed"].astype(int) == seed)].copy()

        if len(train_df) == 0:
            raise ValueError(f"No source-bank rows for {backbone}, seed={seed}")
        if len(test_df) == 0:
            raise ValueError(f"No target-stream rows for {backbone}, seed={seed}")

        model_info = fit_sasc_model(train_df=train_df, source_t=source_t, config=config)

        model_rows.append({
            "backbone": backbone,
            "seed": seed,
            "source_T": source_t,
            "T_max_safe": model_info["T_max_safe"],
            "support_low": model_info["support_low"],
            "support_high": model_info["support_high"],
            "entropy_train_min": model_info["entropy_train_min"],
            "entropy_train_max": model_info["entropy_train_max"],
            "entropy_train_std": model_info["entropy_train_std"],
            "extrapolation_slope": model_info["extrapolation_slope"],
            "n_train_conditions": model_info["n_train_conditions"],
            "n_bins_used": model_info["n_bins_used"],
        })

        bins = model_info["binned"].copy()
        bins["backbone"] = backbone
        bins["seed"] = seed
        bin_rows.append(bins)

        for _, row in test_df.iterrows():
            pred = predict_sasc_temperature(model_info, float(row["entropy_mean"]))
            prediction_rows.append({
                "backbone": backbone,
                "seed": seed,
                "method": "sasc_frozen_v3",
                "corruption": row["corruption"],
                "severity": int(row["severity"]),
                "condition": row["condition"],
                "entropy_mean": float(row["entropy_mean"]),
                "source_T": source_t,
                **pred,
            })

    predictions = pd.DataFrame(prediction_rows)
    model_info = pd.DataFrame(model_rows)
    bins = pd.concat(bin_rows, ignore_index=True)

    predictions.to_csv(out_dir / "sasc_temperature_predictions.csv", index=False)
    model_info.to_csv(out_dir / "sasc_model_info.csv", index=False)
    bins.to_csv(out_dir / "sasc_entropy_bins.csv", index=False)

    by_backbone = (
        predictions.groupby(["backbone", "support_region"], as_index=False)
        .agg(count=("predicted_T", "count"), mean_predicted_T=("predicted_T", "mean"), mean_entropy=("entropy_mean", "mean"))
    )
    by_backbone["fraction"] = by_backbone["count"] / by_backbone.groupby("backbone")["count"].transform("sum")
    by_backbone.to_csv(out_dir / "sasc_support_summary_by_backbone.csv", index=False)

    by_severity = (
        predictions.groupby(["backbone", "severity", "support_region"], as_index=False)
        .agg(count=("predicted_T", "count"), mean_predicted_T=("predicted_T", "mean"), mean_entropy=("entropy_mean", "mean"))
    )
    by_severity["fraction"] = by_severity["count"] / by_severity.groupby(["backbone", "severity"])["count"].transform("sum")
    by_severity.to_csv(out_dir / "sasc_support_summary_by_backbone_severity.csv", index=False)

    manifest = {
        "outputs": [
            "sasc_temperature_predictions.csv",
            "sasc_model_info.csv",
            "sasc_entropy_bins.csv",
            "sasc_support_summary_by_backbone.csv",
            "sasc_support_summary_by_backbone_severity.csv",
        ],
        "target_labels_used_for_fit": False,
        "config": config.__dict__,
    }
    with open(out_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved SASC outputs to {out_dir}")


if __name__ == "__main__":
    main()
