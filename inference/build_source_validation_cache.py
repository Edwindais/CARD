#!/usr/bin/env python3
"""Generate CARD parameter-selection caches from source-validation cases."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from dataset.acdc_nnunet_dataloader import ACDC_nnUNet_Dataset, data_collate
from dataset.acdc_c_torchio_dataset import ACDC_C_TorchIO_Dataset
from dataset.source_validation_augmentation import SourceValidationAugmentationDataset, load_conditions
from evaluation.selection_protocol import load_protocol_manifest, validate_case_inventory
from inference.reproduce_results import build_diffusion


FAMILY_TO_DATASET = {"cardiac": "acdc"}
FAMILY_CONFIG = {
    "cardiac": ("Dataset002_ACDC", ACDC_nnUNet_Dataset, 4, 256),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=sorted(FAMILY_CONFIG))
    parser.add_argument("--preprocessed-root", required=True, type=Path)
    parser.add_argument("--primary-checkpoint", required=True, type=Path)
    parser.add_argument("--reference-checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--condition", choices=list(load_conditions()))
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=PACKAGE_ROOT / "metadata" / "card_source_val_manifest.json",
    )
    return parser.parse_args()


def plans_dir(dataset_root: Path) -> Path:
    for name in ("nnUNetPlans_3d_fullres", "nnUNetPlans_2d"):
        candidate = dataset_root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No nnUNet plans directory in {dataset_root}")


def set_inference_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dataset(family: str, preprocessed_root: Path, condition: str | None = None, seed: int = 42):
    dataset_name, dataset_class, _, image_size = FAMILY_CONFIG[family]
    root = plans_dir(preprocessed_root / dataset_name)
    common = {
        "root_dir": str(root),
        "mode": "val",
        "stage_two": False,
        "num_foreground_classes": 3,
        "predict_background_sdf": False,
        "categorical_use_background": True,
        "split_config": {"roi_z": 1, "roi_y": image_size, "roi_x": image_size},
    }
    if family == "cardiac" and condition is not None:
        config = load_conditions()[condition]
        return ACDC_C_TorchIO_Dataset(
            **common,
            corruption_type="Clean" if condition == "clean" else "SourceValidationAugmentation",
            severity=1,
            seed=seed,
            photometric_preset=config.get("preset") or "source_validation_low",
            photometric_ops="contrast,gamma,noise",
        )
    return dataset_class(
        **common,
    )


def case_and_slice(name: str) -> tuple[str, int]:
    case_id, slice_text = name.rsplit("_z", 1)
    return case_id, int(slice_text)


def validate_source_dataset(dataset, dataset_protocol: dict) -> list[str]:
    base_dataset = dataset.dataset if isinstance(dataset, SourceValidationAugmentationDataset) else dataset
    case_ids = [Path(file_name).stem for file_name in base_dataset.file_names]
    return validate_case_inventory(
        case_ids,
        dataset_protocol["expected_case_ids"],
        "source-validation dataset",
    )


def save_case(output_dir: Path, case_id: str, slices: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]) -> None:
    ordered = sorted(slices, key=lambda item: item[0])
    final_logits = np.stack([item[1] for item in ordered], axis=0).astype(np.float32)
    agg_js = np.stack([item[2] for item in ordered], axis=0).astype(np.float32)
    gt_label = np.stack([item[3] for item in ordered], axis=0).astype(np.int64)
    meta = np.asarray(json.dumps({"case_id": case_id}))
    np.savez_compressed(
        output_dir / f"{case_id}.npz",
        final_logits=final_logits,
        agg_js=agg_js,
        gt_label=gt_label,
        meta=meta,
    )


def generate_setting(
    primary,
    reference,
    dataset,
    condition: str,
    seed: int,
    output_dir: Path,
    args: argparse.Namespace,
    spec: dict,
) -> None:
    set_inference_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=data_collate,
    )
    device = next(primary.parameters()).device
    current_case = None
    current_slices: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["segmentation"]
            if images.ndim == 5:
                images = images.squeeze(2)
            if labels.ndim == 5:
                labels = labels.squeeze(2)
            output = primary.sample_full_tac(
                channel_cond=images,
                reference_denoise_fn=reference.denoise_fn,
                primary_cond_scale=spec["primary_scale"],
                reference_cond_scale=spec["reference_scale"],
                t_min=1.0,
                t_max=1.0,
                w_b=0.0,
                w_k=1.0,
            )
            logits = output["final_logits"].float().cpu().numpy()
            agg_js = output["tac"].squeeze(1).float().cpu().numpy()
            labels_np = labels.cpu().numpy()
            for index, name in enumerate(batch["name"]):
                case_id, slice_index = case_and_slice(name)
                if current_case is not None and case_id != current_case:
                    save_case(output_dir, current_case, current_slices)
                    current_slices = []
                current_case = case_id
                current_slices.append(
                    (slice_index, logits[index], agg_js[index], np.squeeze(labels_np[index]))
                )
    if current_case is not None:
        save_case(output_dir, current_case, current_slices)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for source-validation cache generation")
    dataset_key = FAMILY_TO_DATASET[args.family]
    _, protocol = load_protocol_manifest(args.protocol_manifest, dataset_key)
    conditions = list(load_conditions())
    if len(conditions) != len(protocol["settings"]):
        raise RuntimeError("Source-validation conditions and protocol settings differ")
    spec = {
        "primary_type": "base",
        "reference_type": "mini",
        "primary_scale": 1.0,
        "reference_scale": 0.8,
    }
    _, _, num_classes, _ = FAMILY_CONFIG[args.family]
    device = torch.device(args.device)
    runs = list(zip(conditions, protocol["settings"], protocol["seeds"], strict=True))
    if args.condition:
        runs = [run for run in runs if run[0] == args.condition]
    primary = None
    reference = None
    inventory_checked = False
    for condition, setting, seed in runs:
        dataset = make_dataset(args.family, args.preprocessed_root, condition, int(seed))
        if not inventory_checked:
            validate_source_dataset(dataset, protocol)
            inventory_checked = True
            primary = build_diffusion(num_classes, spec["primary_type"], args.primary_checkpoint, device)
            reference = build_diffusion(num_classes, spec["reference_type"], args.reference_checkpoint, device)
        if args.max_cases is not None:
            base_dataset = dataset.dataset if isinstance(dataset, SourceValidationAugmentationDataset) else dataset
            base_dataset.file_names = base_dataset.file_names[: args.max_cases]
            base_dataset.slice_mapping = []
            base_dataset._scan_slices()
        generate_setting(
            primary,
            reference,
            dataset,
            condition,
            int(seed),
            args.output_root / setting,
            args,
            spec,
        )
    print(f"Source-validation cache complete: {args.output_root}")


if __name__ == "__main__":
    main()
