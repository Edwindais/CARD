#!/usr/bin/env python3
"""Prepare labeled M&Ms cardiac phases for held-out CARD evaluation."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


TARGET_SPACING = (1.25, 1.25, 10.0)
EXPECTED_CENTRE_COUNTS = {"1": 190, "2": 148, "3": 102, "4": 150, "5": 100}
EXPECTED_RAW_SPLIT_COUNTS = {"train": 350, "val": 68, "test": 272}
EXPECTED_TEST_COUNT = 690
PREALIGNED_MARKER_NAME = "MNM_PREALIGNED.json"
PREPROCESSING_TAG = "uniform_resample_full_volume_zscore_acdc_aligned"


def zscore_normalize(image):
    """Apply whole-volume z-score normalization without intensity clipping."""
    import numpy as np

    image = np.asarray(image, dtype=np.float32)
    mean = float(image.mean())
    std = float(image.std())
    if std < 1e-8:
        raise ValueError("Cannot z-score a constant M&Ms volume")
    return ((image - mean) / std).astype(np.float32), {"mean": mean, "std": std}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--reuse-extracted", action="store_true")
    return parser.parse_args()


def _extract_patient(job: tuple[str, str, str, str]) -> tuple[str, list[str]]:
    import nibabel as nib
    import numpy as np

    split, image_path_text, centre, intermediate_root_text = job
    image_path = Path(image_path_text)
    intermediate_root = Path(intermediate_root_text)
    patient_id = image_path.name.removesuffix("_sa.nii.gz")
    label_path = image_path.with_name(f"{patient_id}_sa_gt.nii.gz")
    if not label_path.is_file():
        raise FileNotFoundError(f"M&Ms label not found: {label_path}")
    image_nii = nib.load(image_path)
    label_nii = nib.load(label_path)
    image = np.asarray(image_nii.dataobj)
    label = np.asarray(label_nii.dataobj)
    if image.shape != label.shape or image.ndim != 4:
        raise RuntimeError(
            f"Unexpected M&Ms image/label shape for {patient_id}: {image.shape}, {label.shape}"
        )
    outputs = []
    for frame in range(label.shape[-1]):
        if not np.any(label[..., frame] > 0):
            continue
        case_root = intermediate_root / str(centre) / f"{patient_id}_{frame}"
        case_root.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(image[..., frame], image_nii.affine), case_root / "image.nii.gz")
        nib.save(nib.Nifti1Image(label[..., frame], label_nii.affine), case_root / "seg.nii.gz")
        outputs.append(str(case_root))
    return split, outputs


def extract_phases(
    raw_root: Path,
    intermediate_root: Path,
    num_workers: int,
) -> dict[str, list[Path]]:
    import pandas as pd

    metadata_path = raw_root / "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"M&Ms metadata not found: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    centre_by_patient = dict(zip(metadata["External code"], metadata["Centre"], strict=False))
    split_directories = {"train": "Training", "val": "Validation", "test": "Testing"}
    images_by_split = {
        split: [
            path for path in sorted((raw_root / directory).glob("**/*_sa.nii.gz"))
            if not path.name.endswith("_sa_gt.nii.gz")
        ]
        for split, directory in split_directories.items()
    }
    if not any(images_by_split.values()):
        raise RuntimeError(f"No M&Ms 4D cardiac images found below {raw_root}")

    jobs = []
    for split, images in images_by_split.items():
        for image_path in images:
            patient_id = image_path.name.removesuffix("_sa.nii.gz")
            centre = centre_by_patient.get(patient_id)
            if centre is None:
                raise RuntimeError(f"M&Ms centre metadata missing for {patient_id}")
            jobs.append((split, str(image_path), str(centre), str(intermediate_root)))

    outputs = {split: [] for split in split_directories}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for index, (split, case_roots) in enumerate(executor.map(_extract_patient, jobs), start=1):
            outputs[split].extend(Path(path) for path in case_roots)
            if index % 25 == 0 or index == len(jobs):
                print(f"Extracted labeled phases from {index}/{len(jobs)} patients", flush=True)
    if not any(outputs.values()):
        raise RuntimeError("No labeled M&Ms cardiac phases were found")
    return outputs


def discover_extracted_phases(
    raw_root: Path,
    intermediate_root: Path,
) -> dict[str, list[Path]]:
    split_directories = {"train": "Training", "val": "Validation", "test": "Testing"}
    split_by_patient = {}
    for split, directory in split_directories.items():
        for image_path in (raw_root / directory).glob("**/*_sa.nii.gz"):
            if image_path.name.endswith("_sa_gt.nii.gz"):
                continue
            split_by_patient[image_path.name.removesuffix("_sa.nii.gz")] = split

    outputs = {split: [] for split in split_directories}
    for case_root in sorted(path for path in intermediate_root.glob("*/*") if path.is_dir()):
        if not (case_root / "image.nii.gz").is_file() or not (case_root / "seg.nii.gz").is_file():
            raise RuntimeError(f"Incomplete extracted M&Ms phase: {case_root}")
        patient_id = case_root.name.rsplit("_", 1)[0]
        split = split_by_patient.get(patient_id)
        if split is None:
            raise RuntimeError(f"Cannot map extracted M&Ms patient to a raw split: {patient_id}")
        outputs[split].append(case_root)
    return outputs


def _convert_case(job: tuple[str, str]) -> str:
    import blosc2
    import nibabel as nib
    import numpy as np
    from nibabel.processing import resample_from_to

    case_root_text, output_dir_text = job
    case_root = Path(case_root_text)
    output_dir = Path(output_dir_text)
    image_path = case_root / "image.nii.gz"
    label_path = case_root / "seg.nii.gz"
    image_nii = nib.load(image_path)
    label_nii = nib.load(label_path)
    original_spacing = nib.affines.voxel_sizes(image_nii.affine)
    target_affine = image_nii.affine.copy()
    target_affine[:3, :3] = (
        image_nii.affine[:3, :3]
        / original_spacing[np.newaxis, :]
        * np.asarray(TARGET_SPACING)[np.newaxis, :]
    )
    target_shape = tuple(
        int(value)
        for value in np.maximum(
            1,
            np.ceil(np.asarray(image_nii.shape) * original_spacing / np.asarray(TARGET_SPACING)),
        )
    )
    target = (target_shape, target_affine)
    image_resampled = resample_from_to(image_nii, target, order=1, mode="nearest")
    label_resampled = resample_from_to(label_nii, target, order=0, mode="nearest")
    image_array = image_resampled.get_fdata(dtype=np.float32)
    label_array = label_resampled.get_fdata().astype(np.uint8)
    actual_spacing = tuple(float(value) for value in nib.affines.voxel_sizes(target_affine))
    slice_axis = int(np.argmax(actual_spacing))
    if slice_axis != 2:
        image_array = np.moveaxis(image_array, slice_axis, 2)
        label_array = np.moveaxis(label_array, slice_axis, 2)
    transform_applied = "none" if slice_axis == 2 else f"moveaxis_{slice_axis}to2"

    # Match the ACDC checkpoint's in-plane orientation once during preprocessing.
    image_array = np.rot90(np.flip(image_array, axis=1), k=1, axes=(0, 1))
    label_array = np.rot90(np.flip(label_array, axis=1), k=1, axes=(0, 1))

    # M&Ms: 1=LV, 2=MYO, 3=RV. ACDC: 1=RV, 2=MYO, 3=LV.
    label_one = label_array == 1
    label_three = label_array == 3
    label_array[label_one] = 3
    label_array[label_three] = 1

    image_array = np.ascontiguousarray(image_array)
    label_array = np.ascontiguousarray(label_array)
    image_array, normalization_params = zscore_normalize(image_array)
    image_array = np.ascontiguousarray(image_array)
    centre = case_root.parent.name
    case_name = f"MNM_{centre}_{case_root.name}"
    properties = {
        "data": image_array,
        "seg": label_array,
        "spacing": list(TARGET_SPACING),
        "spatial_metadata": {
            "orientation": f"SAX_D{slice_axis}",
            "original_shape": tuple(int(value) for value in image_nii.shape),
            "original_orient": "".join(nib.aff2axcodes(image_nii.affine)),
            "original_spacing": tuple(float(value) for value in original_spacing),
            "resampled_shape": target_shape,
            "resampled_orient": "".join(nib.aff2axcodes(target_affine)),
            "actual_resampled_spacing": actual_spacing,
            "slice_axis_detected": slice_axis,
            "reoriented_shape": image_array.shape,
            "transform_applied": f"{transform_applied}+flip_width+rot90_ccw",
            "label_mapping": "mnm_1_lv_3_rv_to_acdc_1_rv_3_lv",
            "img_path": str(image_path),
            "seg_path": str(label_path),
        },
        "shape": image_array.shape,
        "preprocessing": PREPROCESSING_TAG,
        "normalization_params": normalization_params,
    }
    with (output_dir / f"{case_name}.pkl").open("wb") as handle:
        pickle.dump(properties, handle)
    blosc2.save_array(image_array, str(output_dir / f"{case_name}.b2nd"), mode="w")
    blosc2.save_array(label_array, str(output_dir / f"{case_name}_seg.b2nd"), mode="w")
    return case_name


def convert_cases(
    case_roots: list[Path],
    output_dir: Path,
    num_workers: int,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(str(case_root), str(output_dir)) for case_root in case_roots]
    case_names = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for index, case_name in enumerate(executor.map(_convert_case, jobs), start=1):
            case_names.append(case_name)
            if index % 50 == 0 or index == len(jobs):
                print(f"Converted {index}/{len(jobs)} phases", flush=True)
    return sorted(case_names)


def main() -> None:
    args = parse_args()
    intermediate_root = args.work_root.resolve() / "mnm_resampled"
    output_root = args.work_root.resolve() / "nnUNet_preprocessed" / "MNM"
    output_dir = output_root / "nnUNetPlans_3d_fullres"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"M&Ms output is not empty: {output_dir}. Use a new work root.")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")
    print(f"M&Ms preprocessing workers: {args.num_workers}", flush=True)
    if args.reuse_extracted:
        cases_by_split = discover_extracted_phases(
            args.raw_root.resolve(), intermediate_root
        )
        print(
            f"Reusing {sum(len(cases) for cases in cases_by_split.values())} extracted phases",
            flush=True,
        )
    else:
        cases_by_split = extract_phases(
            args.raw_root.resolve(), intermediate_root, args.num_workers
        )
    raw_split = {
        split_name: convert_cases(case_roots, output_dir, args.num_workers)
        for split_name, case_roots in cases_by_split.items()
    }
    actual_counts = {key: len(value) for key, value in raw_split.items()}
    if actual_counts != EXPECTED_RAW_SPLIT_COUNTS:
        raise RuntimeError(f"M&Ms raw split counts do not match: {actual_counts}")
    test_cases = sorted(case for cases in raw_split.values() for case in cases)
    if len(test_cases) != EXPECTED_TEST_COUNT:
        raise RuntimeError(f"M&Ms expected {EXPECTED_TEST_COUNT} cases, found {len(test_cases)}")
    (output_root / "split_info.json").write_text(
        json.dumps({"test": test_cases}, indent=2) + "\n"
    )
    (output_dir / PREALIGNED_MARKER_NAME).write_text(
        json.dumps(
            {
                "schema_version": "mnm_prealigned_v1",
                "preprocessing": PREPROCESSING_TAG,
                "case_count": EXPECTED_TEST_COUNT,
                "raw_split_counts": EXPECTED_RAW_SPLIT_COUNTS,
                "centre_counts": EXPECTED_CENTRE_COUNTS,
                "spacing_mm": list(TARGET_SPACING),
                "orientation_aligned": True,
                "labels_mapped_to_acdc": True,
                "roi_sample_mode": "slice2d_kernel10_external",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"M&Ms preprocessing complete: {output_dir}")


if __name__ == "__main__":
    main()
