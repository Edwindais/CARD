#!/usr/bin/env python3
"""Select CARD four-parameter tuples from source-validation caches.

This script runs the CARD parameter-selection stage.  It does **not**
run model inference.  It consumes source-validation cache files produced by the
training/evaluation pipeline:

```
source_val_cache/<setting>/*.npz
  required arrays: final_logits, agg_js, gt_label
```
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("select_card_parameters.py requires torch") from exc

from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.canonical_roi_metrics import dilated_foreground_roi_external2d  # noqa: E402
from evaluation.selection_protocol import (  # noqa: E402
    build_selection_result,
    load_protocol_manifest,
    validate_formal_cache,
)


DEFAULT_T_MIN_GRID = (2.0, 3.5, 5.0, 7.0)
DEFAULT_T_MAX_GRID = (7.5, 10.0, 15.0, 22.5, 30.0)
DEFAULT_W_B_GRID = (0.004, 0.006, 0.008, 0.010, 0.012, 0.016, 0.020, 0.040)
DEFAULT_W_K_GRID = (0.0010, 0.00125, 0.0015, 0.00175, 0.0020, 0.0025, 0.0035, 0.0050, 0.0100)
DEFAULT_BOUNDS = ((1.0, 10.0), (5.0, 30.0), (0.001, 0.080), (0.001, 0.020))
DATASET_SETTINGS = {
    "acdc": {
        "clean": "acdc_id",
        "prefix": "acdc_photo",
    },
}
DEFAULT_FAMILY = "source_validation"
SOURCE_VALIDATION_SUFFIXES = ("low", "medium", "response_noise_heavy")


@dataclass(frozen=True)
class Params:
    t_min: float
    t_max: float
    w_b: float
    w_k: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.t_min, self.t_max, self.w_b, self.w_k


@dataclass
class CacheCase:
    setting: str
    case_id: str
    logits_roi: torch.Tensor  # (N,C)
    js_roi: torch.Tensor      # (N,)
    labels_roi: torch.Tensor  # (N,)
    n_roi: int
    path: str
    meta: dict[str, Any]


def parse_float_list(raw: str | None, default: Sequence[float]) -> tuple[float, ...]:
    if raw is None:
        return tuple(float(x) for x in default)
    values = tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    if not values:
        raise ValueError(f"empty float list: {raw!r}")
    return values


def parse_bounds(raw: str) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    if len(parts) != 4:
        raise ValueError("--bounds requires four low,high pairs separated by ';'")
    parsed = []
    for part in parts:
        lo_hi = [float(x) for x in part.split(",") if x.strip()]
        if len(lo_hi) != 2:
            raise ValueError(f"invalid bounds pair: {part!r}")
        lo, hi = lo_hi
        if lo >= hi:
            raise ValueError(f"invalid bounds pair: {part!r}")
        parsed.append((lo, hi))
    return parsed[0], parsed[1], parsed[2], parsed[3]


def validate_params(values: Iterable[float]) -> Params:
    vals = tuple(float(v) for v in values)
    if len(vals) != 4:
        raise ValueError(f"expected four params, got {vals}")
    t_min, t_max, w_b, w_k = vals
    if t_max - t_min < 1e-3:
        raise ValueError(f"T_max must be greater than T_min, got {vals}")
    if min(vals) <= 0:
        raise ValueError(f"all params must be positive, got {vals}")
    return Params(t_min=t_min, t_max=t_max, w_b=w_b, w_k=w_k)


def default_settings(dataset: str) -> list[str]:
    cfg = DATASET_SETTINGS[dataset]
    return [
        cfg["clean"],
        *[f"{cfg['prefix']}_{DEFAULT_FAMILY}_{suffix}" for suffix in SOURCE_VALIDATION_SUFFIXES],
    ]


def load_meta(raw: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "meta" not in raw.files:
        return {}
    value = raw["meta"]
    text = value.item() if value.shape == () else str(value)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"raw_meta": text}


def deterministic_subsample(n: int, max_samples: int | None, seed: int) -> np.ndarray | slice:
    if max_samples is None or max_samples <= 0 or n <= max_samples:
        return slice(None)
    rng = np.random.default_rng(int(seed))
    return rng.choice(n, size=int(max_samples), replace=False)


def normalize_selection_roi_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"kernel10", "dilate10", "slice2d_kernel10", "slice2d_kernel10_external"}:
        return "kernel10"
    raise ValueError(f"Unsupported ROI mode {mode!r}; use kernel10.")


def roi_samples_slice2d_kernel10(
    logits: torch.Tensor,
    agg_js: torch.Tensor,
    gt: torch.Tensor,
    *,
    roi_dilation_kernel: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if gt.ndim == 4 and gt.shape[1] == 1:
        labels = gt[:, 0].long()
    elif gt.ndim == 3:
        labels = gt.long()
    else:
        raise ValueError(f"gt_label must be (D,1,H,W) or (D,H,W), got {tuple(gt.shape)}")
    if agg_js.ndim == 4 and agg_js.shape[1] == 1:
        agg_js = agg_js.squeeze(1)
    if logits.ndim != 4:
        raise ValueError(f"final_logits must be (D,C,H,W), got {tuple(logits.shape)}")
    if tuple(labels.shape) != (logits.shape[0], logits.shape[2], logits.shape[3]):
        raise ValueError(f"gt_label/logit shape mismatch: labels={tuple(labels.shape)}, logits={tuple(logits.shape)}")
    if tuple(agg_js.shape) != tuple(labels.shape):
        raise ValueError(f"agg_js/logit shape mismatch: agg_js={tuple(agg_js.shape)}, labels={tuple(labels.shape)}")

    roi_mask = dilated_foreground_roi_external2d(
        labels,
        kernel_size=roi_dilation_kernel,
        ignore_index=-1,
    )
    valid = (labels != -1) & roi_mask
    labels_valid = labels[valid].long()
    logits_valid = logits.permute(0, 2, 3, 1)[valid].contiguous()
    js_valid = agg_js[valid].contiguous()
    return logits_valid, js_valid, labels_valid


def load_cache_cases(
    cache_root: Path,
    settings: Sequence[str],
    *,
    roi_mode: str,
    device: torch.device,
    roi_dilation_kernel: int,
    max_cases_per_setting: int | None,
    roi_samples_per_case: int | None,
    seed: int,
    require_case_id_meta: bool,
) -> dict[str, list[CacheCase]]:
    out: dict[str, list[CacheCase]] = {}
    for setting_idx, setting in enumerate(settings):
        files = sorted((cache_root / setting).glob("*.npz"))
        if max_cases_per_setting and max_cases_per_setting > 0:
            files = files[: int(max_cases_per_setting)]
        if not files:
            raise FileNotFoundError(f"No cache files found in {cache_root / setting}")
        cases = []
        for idx, path in enumerate(files):
            raw = np.load(path, allow_pickle=False)
            required = {"final_logits", "agg_js", "gt_label"}
            missing = required.difference(raw.files)
            if missing:
                raise KeyError(f"{path} missing arrays: {sorted(missing)}")
            logits = torch.from_numpy(raw["final_logits"].astype(np.float32)).to(device)
            agg_js = torch.from_numpy(raw["agg_js"].astype(np.float32)).to(device)
            gt = torch.from_numpy(raw["gt_label"].astype(np.int64)).to(device)
            if agg_js.ndim == 4 and agg_js.shape[1] == 1:
                agg_js = agg_js.squeeze(1)
            normalized_roi_mode = normalize_selection_roi_mode(roi_mode)
            if normalized_roi_mode == "kernel10":
                logits_roi, js_roi, labels_roi = roi_samples_slice2d_kernel10(
                    logits,
                    agg_js,
                    gt,
                    roi_dilation_kernel=roi_dilation_kernel,
                )
            else:
                raise ValueError(f"Unsupported roi_mode {roi_mode!r}")
            valid_labels = (labels_roi >= 0) & (labels_roi < logits_roi.shape[1])
            logits_roi = logits_roi[valid_labels]
            js_roi = js_roi[valid_labels]
            labels_roi = labels_roi[valid_labels]
            take = deterministic_subsample(
                int(labels_roi.numel()),
                roi_samples_per_case,
                seed=seed + 1009 * setting_idx + 9176 * idx,
            )
            if isinstance(take, np.ndarray):
                take = torch.as_tensor(take, device=logits_roi.device, dtype=torch.long)
            logits_roi = logits_roi[take].contiguous()
            js_roi = js_roi[take].contiguous()
            labels_roi = labels_roi[take].contiguous()
            meta = load_meta(raw)
            if require_case_id_meta and not meta.get("case_id"):
                raise ValueError(
                    f"Formal selection requires meta.case_id in every cache file; missing in {path}"
                )
            cases.append(
                CacheCase(
                    setting=setting,
                    case_id=str(meta.get("case_id", path.stem)),
                    logits_roi=logits_roi,
                    js_roi=js_roi,
                    labels_roi=labels_roi,
                    n_roi=int(labels_roi.numel()),
                    path=str(path),
                    meta=meta,
                )
            )
        out[setting] = cases
        print(f"[load-cache] {setting}: {len(cases)} cases", flush=True)
    return out


def apply_tac_logits(logits_roi: torch.Tensor, js_roi: torch.Tensor, params: Params) -> torch.Tensor:
    weight = torch.sigmoid((js_roi - params.w_b) / params.w_k)
    teff = params.t_min + weight * (params.t_max - params.t_min)
    return F.softmax(logits_roi / teff.unsqueeze(1), dim=1)


@torch.inference_mode()
def nll_for_cases(cases: Sequence[CacheCase], params: Params) -> tuple[float, int]:
    total_nll = 0.0
    total_n = 0
    for case in cases:
        if case.labels_roi.numel() == 0:
            continue
        weight = torch.sigmoid((case.js_roi - params.w_b) / params.w_k)
        teff = params.t_min + weight * (params.t_max - params.t_min)
        total_nll += float(F.cross_entropy(case.logits_roi / teff.unsqueeze(1), case.labels_roi.long(), reduction="sum").item())
        total_n += int(case.labels_roi.numel())
    if total_n == 0:
        raise RuntimeError("No valid ROI samples")
    return total_nll / total_n, total_n


@torch.inference_mode()
def calibration_metrics_for_cases(cases: Sequence[CacheCase], params: Params, num_bins: int) -> dict[str, float]:
    probs_all = []
    labels_all = []
    for case in cases:
        if case.labels_roi.numel() == 0:
            continue
        probs_all.append(apply_tac_logits(case.logits_roi, case.js_roi, params))
        labels_all.append(case.labels_roi.long())
    if not probs_all:
        return {"roi_ece": 0.0, "roi_sce": 0.0, "roi_ace": 0.0, "roi_nll": 0.0, "n_roi_pixels": 0}
    probs = torch.cat(probs_all, dim=0).detach().cpu().numpy().astype(np.float64)
    labels = torch.cat(labels_all, dim=0).detach().cpu().numpy().astype(np.int64)
    n = int(labels.shape[0])
    pred = probs.argmax(axis=1)
    conf = probs[np.arange(n), pred]
    correct = pred == labels
    edges = np.linspace(0.0, 1.0, int(num_bins) + 1)

    ece = 0.0
    for b in range(int(num_bins)):
        in_bin = (conf >= edges[b]) & (conf <= edges[b + 1]) if b == 0 else (conf > edges[b]) & (conf <= edges[b + 1])
        if in_bin.any():
            ece += float(in_bin.mean()) * abs(float(correct[in_bin].mean()) - float(conf[in_bin].mean()))

    sce_terms = []
    ace_terms = []
    for cls in range(probs.shape[1]):
        cls_probs = probs[:, cls]
        cls_true = (labels == cls).astype(np.float64)
        cls_sce = 0.0
        for b in range(int(num_bins)):
            in_bin = (cls_probs >= edges[b]) & (cls_probs <= edges[b + 1]) if b == 0 else (cls_probs > edges[b]) & (cls_probs <= edges[b + 1])
            if in_bin.any():
                cls_sce += float(in_bin.mean()) * abs(float(cls_true[in_bin].mean()) - float(cls_probs[in_bin].mean()))
        sce_terms.append(cls_sce)

        order = np.argsort(cls_probs)
        chunks = np.array_split(order, int(num_bins))
        vals = []
        for chunk in chunks:
            if chunk.size:
                vals.append(abs(float(cls_true[chunk].mean()) - float(cls_probs[chunk].mean())))
        ace_terms.append(float(np.mean(vals)) if vals else 0.0)

    nll = -np.log(np.clip(probs[np.arange(n), labels], 1e-8, 1.0)).mean()
    return {
        "roi_ece": float(ece),
        "roi_sce": float(np.mean(sce_terms)),
        "roi_ace": float(np.mean(ace_terms)),
        "roi_nll": float(nll),
        "n_roi_pixels": int(n),
    }


def roi_ece_for_cases(cases: Sequence[CacheCase], params: Params, num_bins: int) -> float:
    """Compute ROI-ECE with the same bin convention as the reported metric."""

    counts = torch.zeros(int(num_bins), dtype=torch.float64, device=cases[0].logits_roi.device)
    confidence_sums = torch.zeros_like(counts)
    correct_sums = torch.zeros_like(counts)
    boundaries = torch.linspace(0.0, 1.0, int(num_bins) + 1, device=counts.device)[1:-1]
    for case in cases:
        if case.labels_roi.numel() == 0:
            continue
        probs = apply_tac_logits(case.logits_roi, case.js_roi, params)
        confidence, prediction = probs.max(dim=1)
        bin_ids = torch.bucketize(confidence, boundaries, right=False)
        counts.scatter_add_(0, bin_ids, torch.ones_like(confidence, dtype=torch.float64))
        confidence_sums.scatter_add_(0, bin_ids, confidence.to(torch.float64))
        correct_sums.scatter_add_(0, bin_ids, prediction.eq(case.labels_roi).to(torch.float64))
    total = counts.sum()
    if total.item() == 0:
        raise RuntimeError("No valid ROI samples")
    nonempty = counts > 0
    accuracy = correct_sums[nonempty] / counts[nonempty]
    mean_confidence = confidence_sums[nonempty] / counts[nonempty]
    return float(((counts[nonempty] / total) * (accuracy - mean_confidence).abs()).sum().item())


def objective_for_settings(
    setting_cases: dict[str, list[CacheCase]],
    settings: Sequence[str],
    params: Params,
    *,
    objective: str = "roi_nll",
    num_bins: int = 15,
) -> float:
    if objective == "roi_nll":
        values = [nll_for_cases(setting_cases[name], params)[0] for name in settings]
    elif objective == "roi_ece":
        values = [roi_ece_for_cases(setting_cases[name], params, num_bins) for name in settings]
    else:
        raise ValueError(f"Unsupported objective: {objective!r}")
    return float(np.mean(values))


def iter_grid(
    t_min_grid: Sequence[float],
    t_max_grid: Sequence[float],
    w_b_grid: Sequence[float],
    w_k_grid: Sequence[float],
):
    for t_min in t_min_grid:
        for t_max in t_max_grid:
            if t_max - t_min < 1e-3:
                continue
            for w_b in w_b_grid:
                for w_k in w_k_grid:
                    yield validate_params((t_min, t_max, w_b, w_k))


def boundary_hit(params: Params, bounds: Sequence[tuple[float, float]], tol: float = 1e-6) -> bool:
    return any(abs(v - lo) <= tol or abs(v - hi) <= tol for v, (lo, hi) in zip(params.as_tuple(), bounds))


def optimize(
    setting_cases: dict[str, list[CacheCase]],
    settings: Sequence[str],
    *,
    t_min_grid: Sequence[float],
    t_max_grid: Sequence[float],
    w_b_grid: Sequence[float],
    w_k_grid: Sequence[float],
    bounds: Sequence[tuple[float, float]],
    top_k: int,
    maxiter: int,
    ftol: float,
    no_refine: bool,
    num_bins: int,
    objective: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    objective_column = {"roi_nll": "objective_nll", "roi_ece": "objective_ece"}[objective]
    rows = []
    started = time.time()
    for idx, params in enumerate(iter_grid(t_min_grid, t_max_grid, w_b_grid, w_k_grid)):
        objective_value = objective_for_settings(
            setting_cases,
            settings,
            params,
            objective=objective,
            num_bins=num_bins,
        )
        rows.append({
            "stage": "grid",
            "init_rank": idx,
            "t_min": params.t_min,
            "t_max": params.t_max,
            "w_b": params.w_b,
            "w_k": params.w_k,
            objective_column: objective_value,
            "boundary_hit": boundary_hit(params, bounds),
        })
        if (idx + 1) % 100 == 0:
            print(f"[grid] {idx + 1} candidates", flush=True)
    grid = pd.DataFrame(rows).sort_values(objective_column, ascending=True).reset_index(drop=True)
    refined_rows = []
    if not no_refine:
        def safe_obj(values: np.ndarray) -> float:
            try:
                params = validate_params(values)
            except ValueError:
                return 1e9
            val = objective_for_settings(
                setting_cases,
                settings,
                params,
                objective=objective,
                num_bins=num_bins,
            )
            return val if math.isfinite(val) else 1e9

        for rank, row in grid.head(int(top_k)).iterrows():
            init = validate_params((row.t_min, row.t_max, row.w_b, row.w_k))
            t0 = time.time()
            result = minimize(
                safe_obj,
                x0=np.array(init.as_tuple(), dtype=np.float64),
                method="SLSQP",
                bounds=bounds,
                options={"maxiter": int(maxiter), "ftol": float(ftol)},
            )
            params = validate_params(result.x if result.success else init.as_tuple())
            refined_rows.append({
                "stage": "slsqp",
                "init_rank": int(rank),
                "t_min": params.t_min,
                "t_max": params.t_max,
                "w_b": params.w_b,
                "w_k": params.w_k,
                objective_column: objective_for_settings(
                    setting_cases,
                    settings,
                    params,
                    objective=objective,
                    num_bins=num_bins,
                ),
                "boundary_hit": boundary_hit(params, bounds),
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "nit": int(result.nit),
                "nfev": int(result.nfev),
                "elapsed_sec": time.time() - t0,
            })
            print(f"[slsqp] {rank + 1}/{min(top_k, len(grid))}: {refined_rows[-1][objective_column]:.6f}", flush=True)
    refined = pd.DataFrame(refined_rows).sort_values(objective_column, ascending=True).reset_index(drop=True) if refined_rows else pd.DataFrame()
    best_row = refined.iloc[0].to_dict() if not refined.empty and float(refined.iloc[0][objective_column]) <= float(grid.iloc[0][objective_column]) else grid.iloc[0].to_dict()
    best_params = validate_params((best_row["t_min"], best_row["t_max"], best_row["w_b"], best_row["w_k"]))
    per_setting = []
    for setting in settings:
        metrics = calibration_metrics_for_cases(setting_cases[setting], best_params, num_bins=num_bins)
        per_setting.append({"setting": setting, **metrics})
    selected = {
        "selected": {
            "stage": str(best_row["stage"]),
            "objective": objective,
            "params": {
                "t_min": best_params.t_min,
                "t_max": best_params.t_max,
                "w_b": best_params.w_b,
                "w_k": best_params.w_k,
            },
            objective_column: objective_for_settings(
                setting_cases,
                settings,
                best_params,
                objective=objective,
                num_bins=num_bins,
            ),
            "boundary_hit": boundary_hit(best_params, bounds),
        },
        "per_setting": per_setting,
        "elapsed_sec": time.time() - started,
    }
    return grid, refined, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dataset", choices=sorted(DATASET_SETTINGS), required=True)
    parser.add_argument(
        "--run-kind",
        choices=["formal", "scout"],
        default="formal",
        help="Formal runs require the complete source-validation cache; scout runs may use a subset.",
    )
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=None,
        help="Locked source-validation manifest. Required for formal selection.",
    )
    parser.add_argument("--settings", default=None, help="Comma-separated cache setting names. Default is the locked CARD setting list.")
    parser.add_argument(
        "--selection-roi-mode",
        default="kernel10",
        help="Source-validation ROI mode. Default/kernel10 uses exact per-slice 2D kernel-10 GT dilation.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--roi-dilation-kernel", type=int, default=10)
    parser.add_argument("--max-cases-per-setting", type=int, default=None)
    parser.add_argument("--roi-samples-per-case", type=int, default=None)
    parser.add_argument("--num-bins", type=int, default=15)
    parser.add_argument(
        "--objective",
        choices=["roi_nll", "roi_ece"],
        default="roi_nll",
        help="Source-validation objective. Formal selection uses ROI-NLL.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--ftol", type=float, default=1e-9)
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--t-min-grid", default=None)
    parser.add_argument("--t-max-grid", default=None)
    parser.add_argument("--w-b-grid", default=None)
    parser.add_argument("--w-k-grid", default=None)
    parser.add_argument("--bounds", default="1.0,10.0;5.0,30.0;0.001,0.080;0.001,0.020")
    parser.add_argument("--seed", type=int, default=20260522)
    args = parser.parse_args()

    if args.run_kind == "formal":
        if args.protocol_manifest is None:
            raise ValueError("--protocol-manifest is required for formal selection")
        if args.max_cases_per_setting not in (None, 0):
            raise ValueError("Formal selection forbids --max-cases-per-setting")
        if args.roi_samples_per_case not in (None, 0):
            raise ValueError("Formal selection forbids ROI-pixel subsampling; omit --roi-samples-per-case")
        if args.objective != "roi_nll":
            raise ValueError("Formal selection uses --objective roi_nll; use --run-kind scout for ROI-ECE analysis")

    if args.settings:
        settings = [x.strip() for x in args.settings.split(",") if x.strip()]
    else:
        settings = default_settings(args.dataset)

    dataset_protocol: dict[str, Any] | None = None
    if args.protocol_manifest is not None:
        _, dataset_protocol = load_protocol_manifest(args.protocol_manifest, args.dataset)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selection_roi_mode = normalize_selection_roi_mode(args.selection_roi_mode)
    setting_cases = load_cache_cases(
        args.cache_root,
        settings,
        roi_mode=selection_roi_mode,
        device=device,
        roi_dilation_kernel=args.roi_dilation_kernel,
        max_cases_per_setting=args.max_cases_per_setting,
        roi_samples_per_case=args.roi_samples_per_case,
        seed=args.seed,
        require_case_id_meta=args.run_kind == "formal",
    )

    cache_audit: dict[str, Any]
    if args.run_kind == "formal":
        assert dataset_protocol is not None
        cache_audit = validate_formal_cache(setting_cases, settings, dataset_protocol)
    else:
        cache_audit = {
            "complete": False,
            "observed_case_ids_by_setting": {
                setting: [case.case_id for case in setting_cases[setting]] for setting in settings
            },
        }

    grid, refined, selected = optimize(
        setting_cases,
        settings,
        t_min_grid=parse_float_list(args.t_min_grid, DEFAULT_T_MIN_GRID),
        t_max_grid=parse_float_list(args.t_max_grid, DEFAULT_T_MAX_GRID),
        w_b_grid=parse_float_list(args.w_b_grid, DEFAULT_W_B_GRID),
        w_k_grid=parse_float_list(args.w_k_grid, DEFAULT_W_K_GRID),
        bounds=parse_bounds(args.bounds),
        top_k=args.top_k,
        maxiter=args.maxiter,
        ftol=args.ftol,
        no_refine=args.no_refine,
        num_bins=args.num_bins,
        objective=args.objective,
    )
    grid.to_csv(args.out_dir / "coarse_grid_results.csv", index=False)
    if not refined.empty:
        refined.to_csv(args.out_dir / "slsqp_refined_results.csv", index=False)
    with (args.out_dir / "selected_tuple_by_setting.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["setting", "roi_nll", "roi_ece", "roi_sce", "roi_ace", "n_roi_pixels"])
        writer.writeheader()
        writer.writerows(selected["per_setting"])
    protocol_manifest = None
    if args.protocol_manifest is not None:
        resolved_manifest = args.protocol_manifest.resolve()
        try:
            protocol_manifest = resolved_manifest.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            protocol_manifest = str(resolved_manifest)
    manifest = build_selection_result(
        dataset=args.dataset,
        selected=selected,
        family=DEFAULT_FAMILY,
        settings=settings,
        roi_dilation_kernel=args.roi_dilation_kernel,
        run_kind=args.run_kind,
        protocol_manifest=protocol_manifest,
        dataset_protocol=dataset_protocol,
        cache_audit=cache_audit,
        cache_root=args.cache_root,
        roi_samples_per_case=args.roi_samples_per_case,
        num_bins=args.num_bins,
        objective=args.objective,
    )
    (args.out_dir / "selected_tuple.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["selected"]["params"], indent=2), flush=True)
    print(f"[done] {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
