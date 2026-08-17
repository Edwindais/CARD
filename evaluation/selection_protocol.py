"""Lightweight source-validation protocol checks for CARD tuple selection."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Protocol, Sequence


FORMAL_ROI_SAMPLE_MODE = "slice2d_kernel10_external"


class CaseWithId(Protocol):
    case_id: str


def load_protocol_manifest(path: Path, dataset: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) not in {1, 2}:
        raise ValueError(f"Unsupported source-validation manifest schema: {payload.get('schema_version')!r}")
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict) or dataset not in datasets:
        raise ValueError(f"Dataset {dataset!r} is missing from protocol manifest {path}")
    dataset_protocol = datasets[dataset]
    expected_ids = dataset_protocol.get("expected_case_ids")
    expected_settings = dataset_protocol.get("settings")
    if not isinstance(expected_ids, list) or not expected_ids:
        raise ValueError(f"Protocol manifest has no expected_case_ids for {dataset}")
    if not isinstance(expected_settings, list) or not expected_settings:
        raise ValueError(f"Protocol manifest has no settings for {dataset}")
    return payload, dataset_protocol


def validate_case_inventory(
    case_ids: Sequence[str],
    expected_ids: Sequence[str],
    context: str,
) -> list[str]:
    observed = [str(case_id) for case_id in case_ids]
    duplicates = sorted(case_id for case_id, count in Counter(observed).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate case IDs in {context}: {duplicates}")

    expected_set = {str(case_id) for case_id in expected_ids}
    observed_set = set(observed)
    missing = sorted(expected_set - observed_set)
    unexpected = sorted(observed_set - expected_set)
    if missing or unexpected:
        raise ValueError(
            f"Incomplete or contaminated {context}: missing={missing}, unexpected={unexpected}"
        )
    return observed


def validate_formal_cache(
    setting_cases: dict[str, list[CaseWithId]],
    settings: Sequence[str],
    dataset_protocol: dict[str, Any],
) -> dict[str, Any]:
    expected_settings = [str(value) for value in dataset_protocol["settings"]]
    if list(settings) != expected_settings:
        raise ValueError(
            "Formal selection settings do not match the locked protocol: "
            f"got={list(settings)}, expected={expected_settings}"
        )

    expected_ids = [str(value) for value in dataset_protocol["expected_case_ids"]]
    observed: dict[str, list[str]] = {}
    for setting in expected_settings:
        cases = setting_cases.get(setting, [])
        case_ids = [case.case_id for case in cases]
        observed[setting] = validate_case_inventory(
            case_ids,
            expected_ids,
            f"formal cache for {setting}",
        )
    return {
        "expected_case_ids": expected_ids,
        "observed_case_ids_by_setting": observed,
        "complete": True,
    }


def build_selection_result(
    *,
    dataset: str,
    selected: dict[str, Any],
    family: str,
    settings: Sequence[str],
    roi_dilation_kernel: int,
    run_kind: str,
    protocol_manifest: str | None,
    dataset_protocol: dict[str, Any] | None,
    cache_audit: dict[str, Any],
    cache_root: str | Path,
    roi_samples_per_case: int | None,
    num_bins: int,
    objective: str = "roi_nll",
) -> dict[str, Any]:
    formal = run_kind == "formal"
    if formal:
        if protocol_manifest is None or dataset_protocol is None:
            raise ValueError("Formal selection requires the locked source-validation manifest and dataset protocol")
        provenance = {
            "source_val_manifest": protocol_manifest,
            "status": "locked",
        }
    else:
        provenance = {"status": "scout_only"}

    return {
        "schema_version": 2,
        "formal_run": formal,
        "dataset": dataset,
        **selected,
        "cache_audit": cache_audit,
        "protocol": {
            "family": family,
            "settings": list(settings),
            "roi_sample_mode": FORMAL_ROI_SAMPLE_MODE,
            "roi_dilation_kernel": int(roi_dilation_kernel),
            "objective": {
                "roi_nll": "condition-equal mean source-validation ROI-NLL",
                "roi_ece": "condition-equal mean source-validation ROI-ECE",
            }[objective],
            "cache_root": str(cache_root),
            "run_kind": run_kind,
            "roi_samples_per_case": roi_samples_per_case,
            "num_bins": int(num_bins),
        },
        "provenance": provenance,
    }
