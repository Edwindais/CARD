#!/usr/bin/env python3
"""Aggregate per-case ACDC-C and M&Ms calibration metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_GROUPS = {
    "ACDC-C": ("Clean", "Bias", "Motion", "Ghosting", "Spike"),
    "M&Ms": ("A", "B", "C", "D"),
}


def aggregate_display_blocks(per_case: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ("dice", "roi_ece", "roi_sce", "roi_ace", "roi_nll")
    required = {"display_block", "reporting_group", "case_id", *metric_columns}
    missing = sorted(required.difference(per_case.columns))
    if missing:
        raise ValueError(f"per-case metrics missing columns: {missing}")

    per_case = per_case.copy()
    for column in ("display_block", "reporting_group", "case_id"):
        values = per_case[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"per-case metrics contain an empty {column}")

    unexpected_blocks = sorted(set(per_case["display_block"].astype(str)).difference(EXPECTED_GROUPS))
    if unexpected_blocks:
        raise ValueError(f"per-case metrics contain unexpected display blocks: {unexpected_blocks}")

    for column in metric_columns:
        per_case[column] = pd.to_numeric(per_case[column], errors="coerce")
    if not np.isfinite(per_case[list(metric_columns)].to_numpy()).all():
        raise ValueError("per-case metrics contain non-finite values")

    duplicate = per_case.duplicated(
        subset=["display_block", "reporting_group", "case_id"], keep=False
    )
    if duplicate.any():
        raise ValueError("per-case metrics contain duplicate cases")

    rows = []
    for block, expected in EXPECTED_GROUPS.items():
        block_rows = per_case[per_case["display_block"] == block]
        found = set(block_rows["reporting_group"].astype(str))
        missing_groups = sorted(set(expected).difference(found))
        unexpected = sorted(found.difference(expected))
        if missing_groups:
            raise ValueError(f"{block} missing reporting groups: {missing_groups}")
        if unexpected:
            raise ValueError(f"{block} has unexpected reporting groups: {unexpected}")
        group_means = block_rows.groupby("reporting_group", as_index=False)[
            list(metric_columns)
        ].mean()
        rows.append({
            "display_block": block,
            "dice": float(group_means["dice"].mean()),
            "roi_nll": float(group_means["roi_nll"].mean()),
            "roi_ece_percent": float(group_means["roi_ece"].mean() * 100.0),
            "roi_sce_percent": float(group_means["roi_sce"].mean() * 100.0),
            "roi_ace_percent": float(group_means["roi_ace"].mean() * 100.0),
            "n_cases": int(block_rows["case_id"].nunique()),
            "n_reporting_groups": len(expected),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    per_case = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    summary = aggregate_display_blocks(per_case)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_case.to_csv(args.output_dir / "all_per_case_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
