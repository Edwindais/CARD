"""ROI calibration helpers used by CARD.

Metric contract:
  - Default ROI: foreground GT is dilated independently in each 2D
    H/W slice with kernel size 10; background inside this ROI is retained.
  - ROI-ECE: top-label / predicted-class confidence, fixed-width bins.
  - ROI-SCE: class-wise static calibration error, fixed-width bins.
  - ROI-ACE: class-wise adaptive calibration error, equal-count bins.
  - ROI-NLL: mean ``-log p_true`` over valid ROI pixels/voxels.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # Array-only metric utilities do not require torch.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None or F is None:
        raise ModuleNotFoundError(
            "torch is required for tensor ROI extraction/dilation; "
            "NumPy-based metric utilities remain available without torch."
        )


def _as_label_tensor(labels: torch.Tensor) -> torch.Tensor:
    _require_torch()
    labels = labels.detach()
    if labels.ndim in (4, 5) and labels.shape[1] == 1:
        return labels[:, 0].long()
    return labels.long()


def dilated_foreground_roi_external2d(
    labels: torch.Tensor,
    *,
    kernel_size: int = 10,
    ignore_index: int = -1,
) -> torch.Tensor:
    """Return the exact per-slice 2D kernel ROI used by CARD metrics.

    ``labels`` may be ``(B,H,W)`` or ``(B,1,H,W)``.  For even kernels such as
    10, padding is asymmetric (4 pixels before and 5 after), matching the
    reported evaluation path.  Volumes should be flattened into independent
    2D slices before calling this helper.
    """
    _require_torch()
    label_spatial = _as_label_tensor(labels)
    if label_spatial.ndim != 3:
        raise ValueError(
            "2D kernel ROI expects slice labels with shape (B,H,W) or (B,1,H,W), "
            f"got {tuple(labels.shape)}"
        )
    if kernel_size <= 0:
        return label_spatial.ne(0) & label_spatial.ne(ignore_index)

    mask = (label_spatial.ne(0) & label_spatial.ne(ignore_index)).float().unsqueeze(1)
    pad_total = int(kernel_size) - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded = F.pad(mask, (pad_beg, pad_end, pad_beg, pad_end))
    dilated = F.max_pool2d(padded, kernel_size=int(kernel_size), stride=1, padding=0)
    if dilated.shape[2:] != mask.shape[2:]:
        raise RuntimeError(
            "2D kernel ROI produced an unexpected shape: "
            f"got {tuple(dilated.shape)}, expected {tuple(mask.shape)}"
        )
    return dilated[:, 0].gt(0.5)


def roi_arrays_from_tensors_slice2d_kernel10_external(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    roi_dilation_kernel: int = 10,
    ignore_index: int = -1,
    max_pixels: Optional[int] = None,
    seed: int = 0,
    dtype: np.dtype = np.float16,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(probs_roi, labels_roi)`` with per-slice 2D kernel dilation.

    ``probabilities`` is expected as ``(B,C,H,W)`` or ``(B,C,D,H,W)`` and labels
    as ``(B,1,...)`` / ``(B,...)``.  Returned probabilities have shape ``(N,C)``
    and labels have shape ``(N,)``.
    """
    _require_torch()
    if probabilities.ndim not in (4, 5):
        raise ValueError(f"Expected probabilities with 4/5 dims, got {probabilities.shape}")

    if probabilities.ndim == 4:
        bsz, channels, height, width = probabilities.shape
        probs_2d = probabilities.detach()
        labels_2d = _as_label_tensor(labels)
        if labels_2d.shape[0] != bsz or labels_2d.shape[1:] != (height, width):
            raise ValueError(
                "Label/probability spatial shape mismatch for 2D kernel ROI: "
                f"probs={tuple(probabilities.shape)}, labels={tuple(labels_2d.shape)}"
            )
    else:
        bsz, channels, depth, height, width = probabilities.shape
        label_spatial = labels.detach()
        if label_spatial.ndim == 5 and label_spatial.shape[1] == 1:
            label_spatial = label_spatial[:, 0]
        if label_spatial.ndim != 4:
            raise ValueError(
                "5D probabilities require labels with shape (B,D,H,W) or (B,1,D,H,W), "
                f"got {tuple(labels.shape)}"
            )
        label_spatial = label_spatial.long()
        if label_spatial.shape[0] != bsz or label_spatial.shape[1:] != (depth, height, width):
            raise ValueError(
                "Label/probability spatial shape mismatch for 2D kernel ROI: "
                f"probs={tuple(probabilities.shape)}, labels={tuple(label_spatial.shape)}"
            )
        probs_2d = probabilities.detach().permute(0, 2, 1, 3, 4).reshape(bsz * depth, channels, height, width)
        labels_2d = label_spatial.detach().reshape(bsz * depth, height, width)

    roi_mask = dilated_foreground_roi_external2d(
        labels_2d,
        kernel_size=roi_dilation_kernel,
        ignore_index=ignore_index,
    ).to(probabilities.device)
    probs_spatial = probs_2d.detach().float().permute(0, 2, 3, 1)
    labels_spatial = labels_2d.to(probabilities.device).long()

    probs_roi = probs_spatial[roi_mask]
    labels_roi = labels_spatial[roi_mask]
    valid = labels_roi.ne(ignore_index) & labels_roi.ge(0) & labels_roi.lt(probs_roi.shape[-1])
    probs_roi = probs_roi[valid]
    labels_roi = labels_roi[valid]

    if max_pixels is not None and max_pixels > 0 and probs_roi.shape[0] > max_pixels:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        perm = torch.randperm(probs_roi.shape[0], generator=generator)[: int(max_pixels)]
        probs_roi = probs_roi.cpu()[perm]
        labels_roi = labels_roi.cpu()[perm]
    else:
        probs_roi = probs_roi.cpu()
        labels_roi = labels_roi.cpu()

    return probs_roi.numpy().astype(dtype, copy=False), labels_roi.numpy().astype(np.int16, copy=False)


def calibration_from_roi_arrays(
    probs_roi: np.ndarray,
    labels_roi: np.ndarray,
    *,
    num_bins: int = 10,
    eps: float = 1e-8,
) -> dict[str, float | int]:
    """Compute ROI-ECE/SCE/ACE/NLL from saved ROI arrays."""
    probs = np.asarray(probs_roi, dtype=np.float64)
    labels = np.asarray(labels_roi, dtype=np.int64).reshape(-1)
    if probs.ndim != 2:
        raise ValueError(f"Expected probs_roi shape (N,C), got {probs.shape}")
    valid = (labels >= 0) & (labels < probs.shape[1])
    probs = probs[valid]
    labels = labels[valid]
    n = int(labels.shape[0])
    if n == 0:
        return {"roi_ece": 0.0, "roi_sce": 0.0, "roi_ace": 0.0, "roi_nll": 0.0, "n_roi_pixels": 0}

    pred = probs.argmax(axis=1)
    conf = probs[np.arange(n), pred]
    correct = pred == labels
    edges = np.linspace(0.0, 1.0, int(num_bins) + 1)

    ece = 0.0
    for b in range(int(num_bins)):
        in_bin = (conf >= edges[b]) & (conf <= edges[b + 1]) if b == 0 else (conf > edges[b]) & (conf <= edges[b + 1])
        prop = float(in_bin.mean())
        if prop > 0.0:
            ece += abs(float(correct[in_bin].mean()) - float(conf[in_bin].mean())) * prop

    class_errors = []
    for cls in range(probs.shape[1]):
        cls_probs = probs[:, cls]
        cls_targets = labels == cls
        cls_error = 0.0
        for b in range(int(num_bins)):
            in_bin = (cls_probs >= edges[b]) & (cls_probs <= edges[b + 1]) if b == 0 else (cls_probs > edges[b]) & (cls_probs <= edges[b + 1])
            prop = float(in_bin.mean())
            if prop > 0.0:
                cls_error += abs(float(cls_targets[in_bin].mean()) - float(cls_probs[in_bin].mean())) * prop
        class_errors.append(cls_error)
    sce = float(np.mean(class_errors)) if class_errors else 0.0

    adaptive_bins = min(int(num_bins), n)
    ace_class_errors = []
    if adaptive_bins > 0:
        boundaries = np.linspace(0, n, adaptive_bins + 1)
        for cls in range(probs.shape[1]):
            cls_probs = probs[:, cls]
            cls_targets = labels == cls
            order = np.argsort(cls_probs)
            sorted_probs = cls_probs[order]
            sorted_targets = cls_targets[order]
            bin_errors = []
            for b in range(adaptive_bins):
                start = int(np.floor(boundaries[b]))
                end = int(np.floor(boundaries[b + 1]))
                if b == adaptive_bins - 1:
                    end = n
                if end <= start:
                    continue
                bin_errors.append(abs(float(sorted_targets[start:end].mean()) - float(sorted_probs[start:end].mean())))
            if bin_errors:
                ace_class_errors.append(float(np.mean(bin_errors)))
    ace = float(np.mean(ace_class_errors)) if ace_class_errors else 0.0

    true_probs = np.clip(probs[np.arange(n), labels], eps, 1.0)
    nll = float((-np.log(true_probs)).mean())
    return {
        "roi_ece": float(ece),
        "roi_sce": float(sce),
        "roi_ace": float(ace),
        "roi_nll": nll,
        "n_roi_pixels": n,
    }
