#!/usr/bin/env python3
"""
ACDC preprocessing pipeline for this repository.

Pipeline steps:
1. Convert ACDC-style raw NIfTI into nnUNet raw folder layout.
2. Run `nnUNetv2_plan_and_preprocess`.
3. Write `splits_final.json`.

Notes:
- Supports official ACDC layout (`training/patientXXX/...`) and flat sample layout
  (`patientXXX_frameYY.nii.gz` files directly under `acdc_root`).
- By default, conversion normalizes each image volume to [0, 1] and rewrites header
  spacing to 1mm isotropic to match repository preprocessing convention.
"""

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np


ACDC_LABELS = {
    "background": 0,
    "RV": 1,
    "MYO": 2,
    "LV": 3,
}

# Paper split: these 20 patients form the fixed ACDC-C test set.
PAPER_TEST_PATIENT_IDS = {
    "001", "007", "008", "011", "013",
    "022", "024", "033", "052", "059",
    "064", "065", "066", "068", "075",
    "080", "081", "083", "084", "093",
}

# Patient order used by the evaluation protocol. TorchIO corruption seeds depend on
# case position, so the fixed-test ordering is part of the evaluation protocol.
_PAPER_TRAIN_PATIENT_ORDER = (
    "099", "038", "050", "100", "058", "021", "049", "020", "072", "040",
    "060", "089", "004", "056", "098", "096", "031", "018", "094", "047",
    "048", "055", "097", "074", "043", "041", "063", "037", "095", "054",
    "026", "088", "032", "069", "006", "071", "012", "073", "061", "017",
    "025", "010", "057", "029", "051", "005", "036", "046", "062", "034",
    "076", "092", "070", "077", "067", "003", "091", "016", "014", "044",
    "042", "090", "053", "027", "035", "086", "023", "009", "079", "015",
)  # 70 patients
_PAPER_VAL_PATIENT_ORDER = (
    "028", "085", "082", "087", "019", "030", "078", "045", "002", "039",
)  # 10 patients
_PAPER_TEST_PATIENT_ORDER = (
    "011", "013", "084", "033", "093", "022", "068", "024", "083", "081",
    "080", "001", "007", "066", "008", "065", "075", "064", "059", "052",
)  # 20 patients


def dataset_name(dataset_id: int) -> str:
    return f"Dataset{dataset_id:03d}_ACDC"


def create_dataset_json(output_dir: Path, num_training: int, name: str) -> None:
    dataset_info = {
        "channel_names": {"0": "MRI"},
        "labels": ACDC_LABELS,
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "name": name,
        "description": "ACDC Cardiac MRI Segmentation",
        "reference": "https://www.creatis.insa-lyon.fr/Challenge/acdc/",
        "licence": "CC-BY-SA 4.0",
        "release": "1.0",
    }
    with output_dir.joinpath("dataset.json").open("w") as f:
        json.dump(dataset_info, f, indent=2)


def parse_info_cfg(info_path: Path) -> Tuple[Optional[int], Optional[int]]:
    ed_frame, es_frame = None, None
    if not info_path.exists():
        return ed_frame, es_frame

    with info_path.open("r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if ":" in line:
                key, value = line.split(":", 1)
            elif "=" in line:
                key, value = line.split("=", 1)
            else:
                continue
            key = key.strip().upper()
            value = value.strip()
            if not value.isdigit():
                continue
            if key == "ED":
                ed_frame = int(value)
            elif key == "ES":
                es_frame = int(value)
    return ed_frame, es_frame


def normalize_and_reheader_nifti(
    src_path: Path,
    dst_path: Path,
    is_label: bool,
    normalize_image: bool,
    force_isotropic_header: bool,
) -> None:
    nii = nib.load(str(src_path))
    arr = nii.get_fdata().astype(np.float32)

    if is_label:
        out_arr = np.rint(arr).astype(np.uint8)
    else:
        out_arr = arr
        if normalize_image:
            min_val = float(out_arr.min())
            max_val = float(out_arr.max())
            if max_val > min_val:
                out_arr = (out_arr - min_val) / (max_val - min_val)
            else:
                out_arr = np.zeros_like(out_arr, dtype=np.float32)
        out_arr = out_arr.astype(np.float32)

    if force_isotropic_header:
        out_nii = nib.Nifti1Image(out_arr, np.eye(4, dtype=np.float32))
        out_nii.header.set_zooms((1.0, 1.0, 1.0))
    else:
        out_nii = nib.Nifti1Image(out_arr, nii.affine, header=nii.header)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_nii, str(dst_path))


def maybe_adapt_flat_raw_layout(acdc_root: Path) -> Tuple[Path, Optional[Path]]:
    training_dir = acdc_root / "training"
    if training_dir.exists():
        return acdc_root, None

    flat_files = sorted(acdc_root.glob("patient*_frame*.nii.gz"))
    if not flat_files:
        return acdc_root, None

    grouped: Dict[str, Dict[int, Dict[str, Path]]] = {}
    pattern = re.compile(r"^(patient\d+)_frame(\d+)(?:(_gt))?\.nii\.gz$")
    for f in flat_files:
        m = pattern.match(f.name)
        if m is None:
            raise ValueError(f"Invalid flat ACDC filename: {f.name}")
        patient_name, frame_str, gt_tag = m.group(1), m.group(2), m.group(3)
        frame_id = int(frame_str)
        if patient_name not in grouped:
            grouped[patient_name] = {}
        if frame_id not in grouped[patient_name]:
            grouped[patient_name][frame_id] = {}
        key = "gt" if gt_tag else "img"
        grouped[patient_name][frame_id][key] = f

    for patient_name, frame_dict in grouped.items():
        for frame_id, content in frame_dict.items():
            if set(content) != {"img", "gt"}:
                raise FileNotFoundError(
                    f"Flat ACDC frame requires matching image and label: {patient_name}_frame{frame_id:02d}"
                )

    tmp_root = Path(tempfile.mkdtemp(prefix="acdc_flat_adapt_", dir=str(acdc_root.parent)))
    out_training = tmp_root / "training"
    out_training.mkdir(parents=True, exist_ok=True)

    for patient_name, frame_dict in sorted(grouped.items()):
        patient_out = out_training / patient_name
        patient_out.mkdir(parents=True, exist_ok=True)

        complete_frames = sorted(frame_dict)

        # For flat examples without Info.cfg: use first and last available frame as ED/ES.
        ed_frame = complete_frames[0]
        es_frame = complete_frames[-1]

        for frame_id in complete_frames:
            src_img = frame_dict[frame_id]["img"]
            src_gt = frame_dict[frame_id]["gt"]
            frame_str = f"{frame_id:02d}"
            shutil.copy2(src_img, patient_out / f"{patient_name}_frame{frame_str}.nii.gz")
            shutil.copy2(src_gt, patient_out / f"{patient_name}_frame{frame_str}_gt.nii.gz")

        with patient_out.joinpath("Info.cfg").open("w") as f:
            f.write(f"ED: {ed_frame}\n")
            f.write(f"ES: {es_frame}\n")

    print(f"Detected flat raw layout. Adapted to temporary ACDC structure: {tmp_root}")
    return tmp_root, tmp_root


def convert_acdc_to_nnunet(
    acdc_root: Path,
    nnunet_raw_dir: Path,
    dataset_id: int,
    normalize_image: bool,
    force_isotropic_header: bool,
    use_fixed_test_split: bool,
) -> Path:
    dname = dataset_name(dataset_id)
    dataset_dir = nnunet_raw_dir / dname
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    labels_ts = dataset_dir / "labelsTs"

    for d in (images_tr, labels_tr, images_ts, labels_ts):
        d.mkdir(parents=True, exist_ok=True)

    training_dir = acdc_root / "training"
    # Cases used for train/val split JSON (excludes held-out test IDs).
    train_cases: List[str] = []
    # Cases physically written to imagesTr/labelsTr (nnUNet integrity count).
    nnunet_train_cases: List[str] = []
    test_cases: List[str] = []

    if not training_dir.is_dir():
        raise FileNotFoundError(f"ACDC training directory not found: {training_dir}")

    patient_dirs = sorted(d for d in training_dir.iterdir() if d.is_dir() and d.name.startswith("patient"))
    if not patient_dirs:
        raise FileNotFoundError(f"No ACDC patient directories found in {training_dir}")

    expected_patient_ids = set(
        _PAPER_TRAIN_PATIENT_ORDER + _PAPER_VAL_PATIENT_ORDER + _PAPER_TEST_PATIENT_ORDER
    )
    unknown_patient_ids = sorted(
        patient_dir.name.replace("patient", "")
        for patient_dir in patient_dirs
        if patient_dir.name.replace("patient", "") not in expected_patient_ids
    )
    if unknown_patient_ids:
        raise ValueError(f"ACDC patient IDs are not covered by the paper split: {unknown_patient_ids}")

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name.replace("patient", "")
        ed_frame, es_frame = parse_info_cfg(patient_dir / "Info.cfg")
        if ed_frame is None or es_frame is None:
            raise ValueError(f"Missing ED or ES frame in {patient_dir / 'Info.cfg'}")

        for phase, frame_num in (("ED", ed_frame), ("ES", es_frame)):
            frame_str = f"{frame_num:02d}"
            img_file = patient_dir / f"{patient_dir.name}_frame{frame_str}.nii.gz"
            gt_file = patient_dir / f"{patient_dir.name}_frame{frame_str}_gt.nii.gz"
            if not img_file.is_file():
                raise FileNotFoundError(f"Missing ACDC image: {img_file}")
            if not gt_file.is_file():
                raise FileNotFoundError(f"Missing ACDC label: {gt_file}")

            case_name = f"ACDC_{patient_id}_{phase}"
            is_split_test = patient_id in PAPER_TEST_PATIENT_IDS
            put_in_test = use_fixed_test_split and is_split_test
            image_out = images_ts if put_in_test else images_tr
            label_out = labels_ts if put_in_test else labels_tr
            normalize_and_reheader_nifti(
                img_file,
                image_out / f"{case_name}_0000.nii.gz",
                is_label=False,
                normalize_image=normalize_image,
                force_isotropic_header=force_isotropic_header,
            )
            normalize_and_reheader_nifti(
                gt_file,
                label_out / f"{case_name}.nii.gz",
                is_label=True,
                normalize_image=False,
                force_isotropic_header=force_isotropic_header,
            )
            if put_in_test:
                test_cases.append(case_name)
            else:
                nnunet_train_cases.append(case_name)
                if is_split_test:
                    test_cases.append(case_name)
                else:
                    train_cases.append(case_name)
    print(f"Processed {len(nnunet_train_cases)} nnUNet training cases from {len(patient_dirs)} patients")

    create_dataset_json(dataset_dir, len(nnunet_train_cases), dname)

    # Build split lists using hardcoded patient orderings.
    # The exact ordering matters because TorchIO corruption seed depends on
    # case_idx, which is determined by list position in splits_final.json.
    def _ordered_cases(patient_order, available_cases):
        available = {c for c in available_cases}
        result = []
        for pid in patient_order:
            for phase in ("ED", "ES"):
                name = f"ACDC_{pid}_{phase}"
                if name in available:
                    result.append(name)
        return result

    all_non_test = set(train_cases)
    train_only_cases = _ordered_cases(_PAPER_TRAIN_PATIENT_ORDER, all_non_test)
    val_cases = _ordered_cases(_PAPER_VAL_PATIENT_ORDER, all_non_test)
    test_ordered = _ordered_cases(_PAPER_TEST_PATIENT_ORDER, test_cases)

    with dataset_dir.joinpath("split_info.json").open("w") as f:
        json.dump({"train": train_only_cases, "val": val_cases, "test": test_ordered}, f, indent=2)
    print(f"Split: train={len(train_only_cases)}, val={len(val_cases)}, test={len(test_ordered)}")
    return dataset_dir


def run_nnunet_preprocessing(dataset_id: int) -> None:
    print("Running nnUNet preprocessing")
    cmd = ["nnUNetv2_plan_and_preprocess", "-d", f"{dataset_id:03d}", "--verify_dataset_integrity"]
    subprocess.run(cmd, check=True)
    print("nnUNet preprocessing complete")


def _overwrite_b2nd(path: Path, arr: np.ndarray) -> None:
    import blosc2

    if path.exists():
        path.unlink()
    blosc2.asarray(np.ascontiguousarray(arr)).save(str(path), mode="w")


def _preprocess_one_fixed_test_case(img_path: Path, seg_path: Path, out_dir: Path) -> str:
    """
    Fixed-test preprocessing used by the evaluation protocol:
    - no nonzero-cropping
    - z-score with foreground mask (img > 0)
    - output shape (1, D, H, W), transpose order (2, 1, 0)
    - class_locations empty, no bbox field
    """
    case_name = img_path.name.replace("_0000.nii.gz", "")

    nii = nib.load(str(img_path))
    img = nii.get_fdata().astype(np.float32)
    seg = nib.load(str(seg_path)).get_fdata().astype(np.int16)
    spacing = [np.float32(s) for s in nii.header.get_zooms()[:3]]

    fg = img > 0
    if np.any(fg):
        mean = float(img[fg].mean())
        std = float(img[fg].std())
        if std < 1e-8:
            std = 1.0
    else:
        mean = 0.0
        std = 1.0

    img_norm = ((img - mean) / std).transpose(2, 1, 0)[None, ...].astype(np.float32)
    seg_np = seg.transpose(2, 1, 0)[None, ...].astype(np.int16)

    _overwrite_b2nd(out_dir / f"{case_name}.b2nd", img_norm)
    _overwrite_b2nd(out_dir / f"{case_name}_seg.b2nd", seg_np)

    props = {
        "sitk_stuff": {
            "spacing": spacing,
            "origin": [0, 0, 0],
            "direction": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        },
        "spacing": spacing,
        "original_spacing": spacing,
        "shape_before_cropping": tuple(img_norm.shape[1:]),
        "shape_after_cropping_and_before_resampling": tuple(img_norm.shape[1:]),
        "class_locations": {},
    }
    with (out_dir / f"{case_name}.pkl").open("wb") as f:
        pickle.dump(props, f)
    return case_name


def preprocess_fixed_test_cases(dataset_id: int) -> None:
    """Generate preprocessed outputs for the fixed ACDC-C test split."""
    nnunet_raw = os.environ.get("nnUNet_raw")
    nnunet_preprocessed = os.environ.get("nnUNet_preprocessed")
    if not nnunet_raw or not nnunet_preprocessed:
        raise RuntimeError("nnUNet_raw and nnUNet_preprocessed must be set")

    dname = dataset_name(dataset_id)
    raw_dir = Path(nnunet_raw) / dname
    img_ts = raw_dir / "imagesTs"
    seg_ts = raw_dir / "labelsTs"
    out_dir = Path(nnunet_preprocessed) / dname / "nnUNetPlans_3d_fullres"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(img_ts.glob("*.nii.gz"))
    if not img_files:
        raise FileNotFoundError(f"No fixed-test images found in {img_ts}")

    done = 0
    for img_path in img_files:
        case = img_path.name.replace("_0000.nii.gz", "")
        seg_path = seg_ts / f"{case}.nii.gz"
        if not seg_path.exists():
            raise FileNotFoundError(f"Missing fixed-test label: {seg_path}")
        _preprocess_one_fixed_test_case(img_path, seg_path, out_dir)
        done += 1
    print(f"Fixed-test preprocessing complete: {done} case(s)")


def canonicalize_pkl_direction(dataset_id: int) -> None:
    """Normalize SimpleITK direction metadata during ACDC preprocessing."""
    nnunet_preprocessed = os.environ.get("nnUNet_preprocessed")
    if not nnunet_preprocessed:
        raise RuntimeError("nnUNet_preprocessed must be set")

    pre_dir = Path(nnunet_preprocessed) / dataset_name(dataset_id) / "nnUNetPlans_3d_fullres"
    if not pre_dir.exists():
        return

    fixed = 0
    target_dir = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    for pkl_path in sorted(pre_dir.glob("ACDC_*.pkl")):
        with pkl_path.open("rb") as f:
            props = pickle.load(f)

        sitk = props.get("sitk_stuff")
        if not isinstance(sitk, dict):
            continue

        cur = sitk.get("direction")
        if tuple(float(x) for x in cur) == target_dir:
            continue

        # Keep spacing/origin values, only canonicalize direction convention.
        sitk["direction"] = target_dir
        with pkl_path.open("wb") as f:
            pickle.dump(props, f)
        fixed += 1

    if fixed > 0:
        print(f"Canonicalized sitk direction in {fixed} pkl file(s).")


def create_splits_final(dataset_id: int) -> None:
    nnunet_raw = os.environ.get("nnUNet_raw")
    nnunet_preprocessed = os.environ.get("nnUNet_preprocessed")
    if not nnunet_raw or not nnunet_preprocessed:
        raise RuntimeError("nnUNet_raw and nnUNet_preprocessed environment variables must be set")

    dname = dataset_name(dataset_id)
    split_info_path = Path(nnunet_raw) / dname / "split_info.json"
    with split_info_path.open("r") as f:
        split_info = json.load(f)

    split_record = {"train": split_info["train"], "val": split_info["val"]}
    if "test" in split_info:
        split_record["test"] = split_info["test"]
    splits = [split_record]

    preprocessed_dataset_dir = Path(nnunet_preprocessed) / dname
    output_path = preprocessed_dataset_dir / "splits_final.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(splits, f, indent=4)

    # Mirror split_info into preprocessed root so inference can resolve test split
    # without depending on an external nnUNet_raw path.
    split_info_out = preprocessed_dataset_dir / "split_info.json"
    with split_info_out.open("w") as f:
        json.dump(split_info, f, indent=2)

    print(f"Created splits_final.json at {output_path}")
    print(f"Created split_info.json at {split_info_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ACDC data for training")
    parser.add_argument(
        "--acdc_root",
        type=str,
        required=True,
        help="Path to ACDC root. Supports official layout or flat sample layout.",
    )
    parser.add_argument(
        "--dataset_id",
        type=int,
        default=2,
        help="nnUNet dataset ID (default: 2 -> Dataset002_ACDC)",
    )
    parser.add_argument(
        "--skip_nnunet",
        action="store_true",
        help="Skip nnUNet preprocessing (only convert to raw format)",
    )
    parser.add_argument(
        "--no_unit_intensity",
        action="store_true",
        help="Disable per-volume min-max normalization to [0,1] during conversion.",
    )
    parser.add_argument(
        "--no_isotropic_header",
        action="store_true",
        help="Disable 1mm isotropic header rewrite during conversion.",
    )
    parser.add_argument(
        "--disable_fixed_test_preprocess",
        action="store_true",
        help="Disable preprocessing of the fixed ACDC-C test split.",
    )
    args = parser.parse_args()

    nnunet_raw = os.environ.get("nnUNet_raw")
    nnunet_preprocessed = os.environ.get("nnUNet_preprocessed")
    if not nnunet_raw or not nnunet_preprocessed:
        raise RuntimeError("nnUNet_raw and nnUNet_preprocessed must be set")

    acdc_root = Path(args.acdc_root).resolve()
    adapted_root, temp_root = maybe_adapt_flat_raw_layout(acdc_root)

    print(f"Input ACDC root: {acdc_root}")
    if adapted_root != acdc_root:
        print(f"Adapted ACDC root: {adapted_root}")
    print(f"nnUNet_raw: {nnunet_raw}")
    print(f"nnUNet_preprocessed: {nnunet_preprocessed}")
    print(f"min-max normalize to [0,1]: {not args.no_unit_intensity}")
    print(f"force 1mm isotropic header: {not args.no_isotropic_header}")
    print()

    try:
        print("[Step 1/3] Converting ACDC to nnUNet raw format...")
        convert_acdc_to_nnunet(
            acdc_root=adapted_root,
            nnunet_raw_dir=Path(nnunet_raw),
            dataset_id=args.dataset_id,
            normalize_image=not args.no_unit_intensity,
            force_isotropic_header=not args.no_isotropic_header,
            use_fixed_test_split=not args.disable_fixed_test_preprocess,
        )

        if not args.skip_nnunet:
            print("\n[Step 2/3] Running nnUNet preprocessing...")
            run_nnunet_preprocessing(args.dataset_id)

            if not args.disable_fixed_test_preprocess:
                preprocess_fixed_test_cases(args.dataset_id)

            canonicalize_pkl_direction(args.dataset_id)
        else:
            print("\n[Step 2/3] Skipping nnUNet preprocessing (--skip_nnunet)")

        print("\n[Step 3/3] Creating splits_final.json...")
        create_splits_final(args.dataset_id)
    finally:
        if temp_root is not None and temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)

    print(f"Preprocessing complete: {Path(nnunet_preprocessed) / dataset_name(args.dataset_id)}")


if __name__ == "__main__":
    main()
