#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Create compact paper tables from SASC outputs.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(args.predictions)

    support = (
        pred.groupby(["backbone", "support_region"], as_index=False)
        .agg(
            count=("predicted_T", "count"),
            mean_predicted_T=("predicted_T", "mean"),
            min_predicted_T=("predicted_T", "min"),
            max_predicted_T=("predicted_T", "max"),
            mean_entropy=("entropy_mean", "mean"),
        )
    )
    support["fraction"] = support["count"] / support.groupby("backbone")["count"].transform("sum")
    support.to_csv(out_dir / "table_support_regions_by_backbone.csv", index=False)

    severity = (
        pred.groupby(["backbone", "severity", "support_region"], as_index=False)
        .agg(count=("predicted_T", "count"), mean_predicted_T=("predicted_T", "mean"), mean_entropy=("entropy_mean", "mean"))
    )
    severity["fraction"] = severity["count"] / severity.groupby(["backbone", "severity"])["count"].transform("sum")
    severity.to_csv(out_dir / "table_support_regions_by_severity.csv", index=False)

    print(f"Saved compact tables to {out_dir}")


if __name__ == "__main__":
    main()
