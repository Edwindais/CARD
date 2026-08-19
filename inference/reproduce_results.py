#!/usr/bin/env python3
"""Run CARD inference on ACDC-C or M&Ms and write per-case ROI metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchio
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from dataset.acdc_c_torchio_dataset import ACDC_C_TorchIO_Dataset
from dataset.acdc_nnunet_dataloader import data_collate
from dataset.mnm_nnunet_dataloader import MNM_nnUNet_Dataset
from model.ddpm import CategoricalDiffusion
from train import create_denoising_model
from utils.canonical_roi_metrics import (
    calibration_from_roi_arrays,
    roi_arrays_from_tensors_slice2d_kernel10_external,
)


BLOCKS = ("acdc-c", "mnm")


def set_inference_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_torchio_version(version: str | None = None) -> None:
    actual = version or torchio.__version__
    if actual != "1.2.0":
        raise RuntimeError(f"ACDC-C evaluation requires TorchIO 1.2.0, found {actual}.")


def require_mnm_preprocessed_contract(root: Path) -> None:
    marker = root / "MNM_PREALIGNED.json"
    if not marker.is_file():
        raise RuntimeError(f"M&Ms preprocessing contract marker not found: {marker}")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("preprocessing") != "uniform_resample_full_volume_zscore_acdc_aligned":
        raise RuntimeError(f"Invalid M&Ms preprocessing contract: {marker}")
    if int(payload.get("case_count", -1)) != 690:
        raise RuntimeError(f"Invalid M&Ms case count in {marker}")
    if payload.get("roi_sample_mode") != "slice2d_kernel10_external":
        raise RuntimeError(f"Invalid M&Ms ROI contract in {marker}")


def _plans_dir(dataset_root: Path) -> Path:
    for name in ("nnUNetPlans_3d_fullres", "nnUNetPlans_2d"):
        candidate = dataset_root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No nnUNet preprocessed plans directory in {dataset_root}")


def _model_cfg(num_classes: int, model_type: str):
    return OmegaConf.create({
        "model": {
            "denoising_fn": "swin_unetr", "swinunetr_type": model_type,
            "swinunetr_kwargs": {}, "total_input_channels": num_classes + 1,
            "diffusion_num_channels": num_classes, "cond_channels": 1,
        }
    })


def load_model_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["ema"] if "ema" in payload else payload
    if not isinstance(state, dict):
        raise ValueError(f"Invalid CARD model weights: {checkpoint}")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    if "null_emb_xt" not in state:
        raise ValueError(f"Invalid CARD model weights: {checkpoint}")
    return state


def build_diffusion(num_classes: int, model_type: str, checkpoint: Path, device: torch.device):
    state = load_model_state(checkpoint)
    image_size = int(state["null_emb_xt"].shape[-1])
    dataset_cfg = OmegaConf.create({"diffusion_depth_size": 1})
    denoiser = create_denoising_model(_model_cfg(num_classes, model_type), dataset_cfg)
    diffusion = CategoricalDiffusion(
        denoiser, image_size=image_size, num_frames=1, num_classes=num_classes,
        cond_channels=1, timesteps=50, objective="categorical",
        aux_loss_type="dice_ce", aux_loss_weight=1.0, aux_dice_weight=0.5,
        aux_ce_weight=0.5, kl_loss_weight=0.1, final_step_mode="majority",
        condition_mask_ratio=0.2, sample_mode="one_hot", inference_stochastic=False,
    )
    diffusion.load_state_dict(state, strict=True)
    diffusion.to(device).eval()
    return diffusion


def _case_and_slice(name: str) -> tuple[str, int]:
    case_id, z_text = name.rsplit("_z", 1)
    return case_id, int(z_text)


def _dice(pred: np.ndarray, label: np.ndarray, num_classes: int) -> float:
    values = []
    for class_id in range(1, num_classes):
        p = pred == class_id
        y = label == class_id
        denom = int(p.sum() + y.sum())
        if denom:
            values.append(2.0 * float(np.logical_and(p, y).sum()) / denom)
    return float(np.mean(values)) if values else 1.0


def _finish_case(case_id: str, slices: list[tuple[int, np.ndarray, np.ndarray]], meta: dict, output_dir: Path):
    ordered = sorted(slices, key=lambda item: item[0])
    probs = np.stack([item[1] for item in ordered], axis=1)  # C,D,H,W
    labels = np.stack([item[2] for item in ordered], axis=0)  # D,H,W
    prob_tensor = torch.from_numpy(probs).unsqueeze(0)
    label_tensor = torch.from_numpy(labels).unsqueeze(0)
    if meta["roi_sample_mode"] != "slice2d_kernel10_external":
        raise RuntimeError(f"Unsupported ROI sample mode: {meta['roi_sample_mode']}")
    probs_roi, labels_roi = roi_arrays_from_tensors_slice2d_kernel10_external(
        prob_tensor, label_tensor, roi_dilation_kernel=10, dtype=np.float16,
    )
    metrics = calibration_from_roi_arrays(probs_roi, labels_roi, num_bins=10)
    pred = probs.argmax(axis=0)
    public_case_id = f"{meta['scanner']}__{case_id}" if meta.get("scanner") else case_id
    safe_group = str(meta["reporting_group"]).replace("/", "_").replace("\\", "_").replace(" ", "_")
    sample_dir = output_dir / "roi_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / f"{safe_group}__{public_case_id}__roi_kernel10.npz"
    np.savez_compressed(sample_path, probs_roi=probs_roi, labels_roi=labels_roi)
    return {
        **meta, "case_id": public_case_id, "dice": _dice(pred, labels, probs.shape[0]),
        **metrics, "roi_sample_path": str(sample_path),
    }


def infer_dataset(diffusion, reference_diffusion, dataset, meta: dict, tuple_params: dict, args):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=data_collate)
    rows, current_case, current_slices = [], None, []
    device = next(diffusion.parameters()).device
    with torch.inference_mode():
        for batch in loader:
            images = batch.get("image", batch.get("img")).to(device)
            labels = batch.get("segmentation", batch.get("mask"))
            if images.ndim == 5:
                images = images.squeeze(2)
            if labels.ndim == 5:
                labels = labels.squeeze(2)
            output = diffusion.sample_full_tac(
                channel_cond=images,
                reference_denoise_fn=reference_diffusion.denoise_fn,
                primary_cond_scale=args.primary_cond_scale,
                reference_cond_scale=args.reference_cond_scale,
                t_min=tuple_params["t_min"],
                t_max=tuple_params["t_max"],
                w_b=tuple_params["w_b"], w_k=tuple_params["w_k"],
            )
            probs = output["probabilities"]
            probs_np = probs.float().cpu().numpy()
            labels_np = labels.cpu().numpy()
            names = batch["name"]
            for index, name in enumerate(names):
                case_id, z_index = _case_and_slice(name)
                if current_case is not None and case_id != current_case:
                    rows.append(_finish_case(current_case, current_slices, meta, args.output_dir))
                    current_slices = []
                    if args.max_cases and len(rows) >= args.max_cases:
                        break
                current_case = case_id
                label_slice = np.squeeze(labels_np[index]).astype(np.int64)
                current_slices.append((z_index, probs_np[index], label_slice))
            if args.max_cases and len(rows) >= args.max_cases:
                break
    if current_slices and (not args.max_cases or len(rows) < args.max_cases):
        rows.append(_finish_case(current_case, current_slices, meta, args.output_dir))
    return rows


def make_jobs(args):
    size = args.image_size
    common = dict(stage_two=False, predict_background_sdf=False, categorical_use_background=True,
                  split_config={"roi_z": 1, "roi_y": size, "roi_x": size})
    if args.block == "acdc-c":
        os.environ["nnUNet_preprocessed"] = str(args.cardiac_root)
        root = _plans_dir(args.cardiac_root / "Dataset002_ACDC")
        for condition in ("Clean", "Bias", "Motion", "Ghosting", "Spike"):
            yield ACDC_C_TorchIO_Dataset(root_dir=str(root), mode="test", corruption_type=condition, severity=1, seed=42, **common), {"display_block": "ACDC-C", "reporting_group": condition, "roi_sample_mode": "slice2d_kernel10_external"}
    elif args.block == "mnm":
        for vendor in "ABCD":
            yield MNM_nnUNet_Dataset(str(args.mnm_root), vendor, args.max_cases), {"display_block": "M&Ms", "reporting_group": vendor, "roi_sample_mode": "slice2d_kernel10_external"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", required=True, choices=BLOCKS)
    parser.add_argument("--cardiac-root", type=Path)
    parser.add_argument("--mnm-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--primary-checkpoint", required=True, type=Path)
    parser.add_argument("--reference-checkpoint", required=True, type=Path)
    parser.add_argument("--primary-model-type", default="base")
    parser.add_argument(
        "--reference-model-type",
        default="mini",
        help="SwinUNETR size used by the reference checkpoint.",
    )
    parser.add_argument(
        "--tuples-file",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    if args.block == "acdc-c":
        require_torchio_version()
    elif args.block == "mnm":
        require_mnm_preprocessed_contract(args.mnm_root)
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for CARD inference")

    family = "acdc"
    set_inference_seed()
    tuple_params = json.loads(args.tuples_file.read_text())[family]
    num_classes = 4
    device = torch.device(args.device)
    primary_model = build_diffusion(
        num_classes, args.primary_model_type, args.primary_checkpoint, device
    )
    reference_model = build_diffusion(
        num_classes,
        args.reference_model_type,
        args.reference_checkpoint,
        device,
    )
    if primary_model.image_size != reference_model.image_size:
        raise RuntimeError(
            "Primary/reference model image-size mismatch: "
            f"{primary_model.image_size} != {reference_model.image_size}"
        )
    args.image_size = int(primary_model.image_size)
    args.primary_cond_scale = 0.8
    args.reference_cond_scale = 0.8
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, meta in make_jobs(args):
        rows.extend(
            infer_dataset(
                primary_model,
                reference_model,
                dataset,
                meta,
                tuple_params,
                args,
            )
        )
    output = args.output_dir / "per_case_metrics.csv"
    if not rows:
        raise RuntimeError("Inference produced no cases")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] {output} cases={len(rows)}")


if __name__ == "__main__":
    main()
