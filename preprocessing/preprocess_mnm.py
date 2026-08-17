#!/usr/bin/env python3
"""Prepare labeled M&Ms cardiac phases for held-out CARD evaluation."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


TARGET_SPACING = (1.25, 1.25, 10.0)
EXPECTED_CENTRE_COUNTS = {"1": 190, "2": 148, "3": 102, "4": 150, "5": 100}
EXPECTED_SPLIT_COUNTS = {"train": 482, "val": 103, "test": 105}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    return parser.parse_args()


def resample_image(image, spacing, interpolator):
    import numpy as np
    import SimpleITK as sitk

    transform = sitk.ResampleImageFilter()
    transform.SetInterpolator(interpolator)
    transform.SetOutputDirection(image.GetDirection())
    transform.SetOutputOrigin(image.GetOrigin())
    transform.SetUseNearestNeighborExtrapolator(True)
    transform.SetOutputSpacing(spacing)
    scale = np.asarray(image.GetSpacing()) / np.asarray(spacing)
    transform.SetSize([int(value + 1) for value in np.asarray(image.GetSize()) * scale])
    return transform.Execute(image)


def resample_label(label, spacing, reference):
    import numpy as np
    import SimpleITK as sitk

    source = sitk.GetArrayFromImage(label)
    output = None
    reference_label = None
    for index, value in enumerate(np.unique(source)):
        binary = sitk.GetImageFromArray((source == value).astype(np.float32))
        binary.CopyInformation(label)
        reference_label = resample_image(binary, spacing, sitk.sitkLinear)
        resampled = np.rint(sitk.GetArrayFromImage(reference_label)) * value
        if index == 0:
            output = resampled
        else:
            output[resampled == value] = value
    result = sitk.GetImageFromArray(output)
    result.CopyInformation(reference)
    return result


def extract_phases(raw_root: Path, intermediate_root: Path) -> dict[str, list[Path]]:
    import nibabel as nib
    import numpy as np
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

    outputs = {split: [] for split in split_directories}
    for split, images in images_by_split.items():
        for image_path in images:
            patient_id = image_path.name.removesuffix("_sa.nii.gz")
            label_path = image_path.with_name(f"{patient_id}_sa_gt.nii.gz")
            if not label_path.is_file():
                raise FileNotFoundError(f"M&Ms label not found: {label_path}")
            centre = centre_by_patient.get(patient_id)
            if centre is None:
                raise RuntimeError(f"M&Ms centre metadata missing for {patient_id}")
            image_nii = nib.load(image_path)
            label_nii = nib.load(label_path)
            image = image_nii.get_fdata()
            label = label_nii.get_fdata()
            if image.shape != label.shape or image.ndim != 4:
                raise RuntimeError(
                    f"Unexpected M&Ms image/label shape for {patient_id}: {image.shape}, {label.shape}"
                )
            for frame in range(label.shape[-1]):
                if not np.any(label[..., frame] > 0):
                    continue
                case_root = intermediate_root / str(centre) / f"{patient_id}_{frame}"
                case_root.mkdir(parents=True, exist_ok=True)
                nib.save(nib.Nifti1Image(image[..., frame], image_nii.affine), case_root / "image.nii.gz")
                nib.save(nib.Nifti1Image(label[..., frame], label_nii.affine), case_root / "seg.nii.gz")
                outputs[split].append(case_root)
    if not any(outputs.values()):
        raise RuntimeError("No labeled M&Ms cardiac phases were found")
    return outputs


def convert_cases(case_roots: list[Path], output_dir: Path) -> list[str]:
    import blosc2
    import nibabel as nib
    import numpy as np
    import SimpleITK as sitk

    output_dir.mkdir(parents=True, exist_ok=True)
    case_names = []
    for case_root in case_roots:
        image_path = case_root / "image.nii.gz"
        label_path = case_root / "seg.nii.gz"
        image = sitk.ReadImage(str(image_path))
        label = sitk.ReadImage(str(label_path))
        image_resampled = resample_image(image, TARGET_SPACING, sitk.sitkLinear)
        label_resampled = resample_label(label, TARGET_SPACING, image_resampled)
        sitk.WriteImage(image_resampled, str(image_path))
        sitk.WriteImage(label_resampled, str(label_path))

        image_nii = nib.load(image_path)
        label_nii = nib.load(label_path)
        image_array = np.ascontiguousarray(image_nii.get_fdata().astype(np.float32))
        label_array = np.ascontiguousarray(label_nii.get_fdata().astype(np.uint8))
        centre = case_root.parent.name
        case_name = f"MNM_{centre}_{case_root.name}"
        properties = {
            "data": image_array,
            "seg": label_array,
            "spacing": tuple(float(value) for value in image_nii.header.get_zooms()[:3]),
            "shape": image_array.shape,
            "patient_id": case_root.name,
            "vendor": centre,
        }
        with (output_dir / f"{case_name}.pkl").open("wb") as handle:
            pickle.dump(properties, handle)
        blosc2.save_array(image_array, str(output_dir / f"{case_name}.b2nd"), mode="w")
        blosc2.save_array(label_array, str(output_dir / f"{case_name}_seg.b2nd"), mode="w")
        case_names.append(case_name)
    return sorted(case_names)


def main() -> None:
    args = parse_args()
    intermediate_root = args.work_root.resolve() / "mnm_resampled"
    output_root = args.work_root.resolve() / "nnUNet_preprocessed" / "MNM"
    output_dir = output_root / "nnUNetPlans_3d_fullres"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"M&Ms output is not empty: {output_dir}. Use a new work root.")
    cases_by_split = extract_phases(args.raw_root.resolve(), intermediate_root)
    split = {
        split_name: convert_cases(case_roots, output_dir)
        for split_name, case_roots in cases_by_split.items()
    }
    actual_counts = {key: len(value) for key, value in split.items()}
    if actual_counts != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(f"M&Ms split counts do not match the expected protocol: {actual_counts}")
    (output_root / "split_info.json").write_text(json.dumps(split, indent=2) + "\n")
    print(f"M&Ms preprocessing complete: {output_dir}")


if __name__ == "__main__":
    main()
