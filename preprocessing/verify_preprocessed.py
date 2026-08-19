#!/usr/bin/env python3
"""Verify the cardiac preprocessed-data contract."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from preprocess_data import _PAPER_TEST_PATIENT_ORDER, _PAPER_TRAIN_PATIENT_ORDER, _PAPER_VAL_PATIENT_ORDER
from preprocessing.preprocess_mnm import (
    EXPECTED_CENTRE_COUNTS,
    EXPECTED_RAW_SPLIT_COUNTS,
    EXPECTED_TEST_COUNT,
    PREALIGNED_MARKER_NAME,
    PREPROCESSING_TAG,
    TARGET_SPACING,
)


def read_split(dataset_root: Path) -> dict[str, list[str]]:
    split_path = dataset_root / "split_info.json"
    if split_path.is_file():
        return json.loads(split_path.read_text(encoding="utf-8"))
    return json.loads((dataset_root / "splits_final.json").read_text(encoding="utf-8"))[0]


def expected_acdc_split() -> dict[str, list[str]]:
    def phases(patient_ids):
        return [f"ACDC_{patient_id}_{phase}" for patient_id in patient_ids for phase in ("ED", "ES")]
    return {
        "train": phases(_PAPER_TRAIN_PATIENT_ORDER),
        "val": phases(_PAPER_VAL_PATIENT_ORDER),
        "test": phases(_PAPER_TEST_PATIENT_ORDER),
    }


def case_names(data_dir: Path) -> list[str]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Preprocessed data directory not found: {data_dir}")
    return sorted(path.stem for path in data_dir.glob("*.pkl"))


def assert_case_files(data_dir: Path, names: list[str], label: str) -> None:
    missing = [
        f"{name}{suffix}"
        for name in names
        for suffix in (".pkl", ".b2nd", "_seg.b2nd")
        if not (data_dir / f"{name}{suffix}").is_file()
    ]
    if missing:
        raise RuntimeError(f"{label} is missing {len(missing)} processed files; first: {missing[0]}")


def assert_split(actual: dict[str, list[str]], expected: dict[str, list[str]], label: str) -> None:
    for key in ("train", "val", "test"):
        if actual.get(key, []) != expected[key]:
            raise RuntimeError(f"{label} {key} split does not match the expected protocol")


def verify_mnm(preprocessed: Path) -> None:
    """Verify every M&Ms phase in the uniform held-out dataset."""
    preprocessed = preprocessed.resolve()
    mnm_root = preprocessed / "MNM"
    mnm_split = read_split(mnm_root)
    mnm_data = mnm_root / "nnUNetPlans_3d_fullres"
    marker_path = mnm_data / PREALIGNED_MARKER_NAME
    if not marker_path.is_file():
        raise RuntimeError(f"M&Ms preprocessing marker is missing: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("preprocessing") != PREPROCESSING_TAG:
        raise RuntimeError(f"M&Ms preprocessing marker is invalid: {marker_path}")
    if int(marker.get("case_count", -1)) != EXPECTED_TEST_COUNT:
        raise RuntimeError(f"M&Ms marker case count is invalid: {marker_path}")
    if marker.get("raw_split_counts") != EXPECTED_RAW_SPLIT_COUNTS:
        raise RuntimeError(f"M&Ms marker raw split counts are invalid: {marker_path}")
    if marker.get("roi_sample_mode") != "slice2d_kernel10_external":
        raise RuntimeError(f"M&Ms marker ROI mode is invalid: {marker_path}")
    mnm_names = case_names(mnm_data)
    if sorted(mnm_split.get("test", [])) != mnm_names or len(mnm_names) != EXPECTED_TEST_COUNT:
        raise RuntimeError(f"M&Ms expected {EXPECTED_TEST_COUNT} held-out phases, found {len(mnm_names)}")
    centre_counts = Counter(name.split("_")[1] for name in mnm_names)
    if dict(sorted(centre_counts.items())) != EXPECTED_CENTRE_COUNTS:
        raise RuntimeError(f"M&Ms centre counts do not match: {dict(centre_counts)}")
    assert_case_files(mnm_data, mnm_names, "M&Ms")
    for name in mnm_names:
        with (mnm_data / f"{name}.pkl").open("rb") as handle:
            properties = pickle.load(handle)
        if not np.allclose(properties["spacing"], TARGET_SPACING):
            raise RuntimeError(f"M&Ms {name} spacing does not match {TARGET_SPACING}")
        actual_spacing = properties.get("spatial_metadata", {}).get("actual_resampled_spacing")
        if actual_spacing is None or not np.allclose(actual_spacing, TARGET_SPACING):
            raise RuntimeError(f"M&Ms {name} actual spacing does not match {TARGET_SPACING}")
        if properties.get("preprocessing") != PREPROCESSING_TAG:
            raise RuntimeError(f"M&Ms preprocessing metadata is invalid for {name}")
        image = np.asarray(properties["data"], dtype=np.float32)
        labels = np.asarray(properties["seg"])
        if not np.isfinite(image).all():
            raise RuntimeError(f"M&Ms {name} contains non-finite image values")
        if not np.isclose(float(image.mean()), 0.0, atol=1e-5):
            raise RuntimeError(f"M&Ms {name} image is not zero-mean")
        if not np.isclose(float(image.std()), 1.0, atol=1e-5):
            raise RuntimeError(f"M&Ms {name} image is not unit-variance")
        if not set(np.unique(labels)).issubset({0, 1, 2, 3}):
            raise RuntimeError(f"M&Ms {name} contains invalid labels")
        if image.shape != labels.shape or tuple(properties.get("shape", ())) != image.shape:
            raise RuntimeError(f"M&Ms {name} image/label/metadata shapes disagree")


def verify_cardiac(work_root: Path) -> None:
    preprocessed = work_root.resolve() / "nnUNet_preprocessed"
    acdc_root = preprocessed / "Dataset002_ACDC"
    acdc_split = expected_acdc_split()
    assert_split(read_split(acdc_root), acdc_split, "ACDC")
    acdc_data = acdc_root / "nnUNetPlans_3d_fullres"
    acdc_names = case_names(acdc_data)
    if acdc_names != sorted(sum(acdc_split.values(), [])):
        raise RuntimeError(f"ACDC expected 200 processed phases, found {len(acdc_names)}")
    assert_case_files(acdc_data, acdc_names, "ACDC")

    verify_mnm(preprocessed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--mnm-only", action="store_true")
    args = parser.parse_args()
    if args.mnm_only:
        verify_mnm(args.work_root / "nnUNet_preprocessed")
        print("Preprocessed data verified: M&Ms")
    else:
        verify_cardiac(args.work_root)
        print("Preprocessed data verified: cardiac")


if __name__ == "__main__":
    main()
