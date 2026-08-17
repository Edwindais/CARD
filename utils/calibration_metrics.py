"""Shared calibration metrics for 2D/3D medical segmentation."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _squeeze_channel_if_present(tensor: torch.Tensor, *, prob_ndim: int) -> torch.Tensor:
    """Return labels/predictions/masks as ``(B, *spatial)`` tensors."""
    if tensor.ndim == prob_ndim and tensor.shape[1] == 1:
        return tensor.squeeze(1)
    if tensor.ndim == prob_ndim - 1:
        return tensor
    if tensor.ndim == prob_ndim and tensor.shape[1] != 1:
        # Some callers may accidentally pass one-hot labels.  Fail loudly
        # instead of silently flattening the class axis as spatial data.
        raise ValueError(
            "Expected labels/predictions/masks with a singleton channel "
            f"dimension or no channel dimension, got shape={tuple(tensor.shape)}"
        )
    raise ValueError(
        "Expected labels/predictions/masks to have probability ndim or "
        f"probability ndim - 1, got shape={tuple(tensor.shape)}"
    )


def _flatten_inputs(
    predictions: Optional[torch.Tensor],
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_background: bool,
    ignore_index: int,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten segmentation tensors to ``(N, C)``, ``(N,)``, ``(N,)``.

    Args:
        predictions: Optional hard labels.  If omitted, ``argmax(probabilities)``
            is used.
        probabilities: Softmax probabilities with shape ``(B,C,H,W)`` or
            ``(B,C,D,H,W)``.
        labels: Ground-truth labels with shape ``(B,1,...)`` or ``(B,...)``.
        ignore_background: When true, valid voxels with label 0 are removed.
        ignore_index: Label value ignored by metrics.
        valid_mask: Optional boolean ROI/valid mask with shape ``(B,1,...)`` or
            ``(B,...)``.  Background inside the ROI is retained unless
            ``ignore_background`` is true.
    """
    if probabilities.ndim not in (4, 5):
        raise ValueError(
            "probabilities must be 4D (B,C,H,W) or 5D (B,C,D,H,W), "
            f"got shape={tuple(probabilities.shape)}"
        )
    if probabilities.shape[1] < 2:
        raise ValueError("calibration metrics require at least two classes")

    device = probabilities.device
    num_classes = int(probabilities.shape[1])
    spatial_order = (0, *range(2, probabilities.ndim), 1)
    probs_flat = probabilities.permute(*spatial_order).reshape(-1, num_classes)

    labels_no_channel = _squeeze_channel_if_present(labels, prob_ndim=probabilities.ndim)
    labels_flat = labels_no_channel.reshape(-1).to(device=device)

    if predictions is None:
        pred_no_channel = torch.argmax(probabilities, dim=1)
    else:
        pred_no_channel = _squeeze_channel_if_present(predictions, prob_ndim=probabilities.ndim)
    pred_flat = pred_no_channel.reshape(-1).to(device=device)

    mask_flat = torch.ones_like(labels_flat, dtype=torch.bool, device=device)
    mask_flat &= labels_flat != ignore_index
    if valid_mask is not None:
        valid_no_channel = _squeeze_channel_if_present(valid_mask.to(device=device), prob_ndim=probabilities.ndim)
        mask_flat &= valid_no_channel.reshape(-1).bool()
    if ignore_background:
        mask_flat &= labels_flat != 0

    if mask_flat.sum() == 0:
        empty_probs = probs_flat[:0]
        empty_labels = labels_flat[:0].long()
        empty_preds = pred_flat[:0].long()
        return empty_probs, empty_labels, empty_preds

    labels_selected = labels_flat[mask_flat].long()
    pred_selected = pred_flat[mask_flat].long()
    if labels_selected.min().item() < 0 or labels_selected.max().item() >= num_classes:
        raise ValueError(
            f"labels contain values outside [0, {num_classes - 1}] after masking: "
            f"min={int(labels_selected.min())}, max={int(labels_selected.max())}"
        )
    if pred_selected.min().item() < 0 or pred_selected.max().item() >= num_classes:
        raise ValueError(
            f"predictions contain values outside [0, {num_classes - 1}] after masking: "
            f"min={int(pred_selected.min())}, max={int(pred_selected.max())}"
        )

    return probs_flat[mask_flat], labels_selected, pred_selected


def _uniform_bin_edges(num_bins: int, device: torch.device) -> torch.Tensor:
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins}")
    return torch.linspace(0.0, 1.0, num_bins + 1, device=device)


def _bin_mask(values: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, *, is_first: bool) -> torch.Tensor:
    if is_first:
        return values.ge(lower.item()) & values.le(upper.item())
    return values.gt(lower.item()) & values.le(upper.item())


def expected_calibration_error(
    predictions: torch.Tensor,
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 10,
    ignore_background: bool = True,
    ignore_index: int = -1,
    valid_mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute Expected Calibration Error (ECE) for 2D or 3D segmentation predictions.

    ECE measures the difference between prediction confidence and actual accuracy.
    Lower ECE indicates better calibrated predictions.

    Args:
        predictions: Predicted class labels - (B, 1, H, W) for 2D or (B, 1, D, H, W) for 3D
        probabilities: Softmax probabilities - (B, C, H, W) for 2D or (B, C, D, H, W) for 3D
        labels: Ground truth labels - (B, 1, H, W) for 2D or (B, 1, D, H, W) for 3D
        num_bins: Number of bins for calibration histogram (default: 15)
        ignore_background: Whether to exclude background class (class 0) from ECE
        ignore_index: Label value to ignore (e.g., -1 for padding)

    Returns:
        ece: Expected Calibration Error as a float
    """
    device = probabilities.device
    probs_flat, labels_flat, pred_flat = _flatten_inputs(
        predictions,
        probabilities,
        labels,
        ignore_background=ignore_background,
        ignore_index=ignore_index,
        valid_mask=valid_mask,
    )
    if labels_flat.numel() == 0:
        return 0.0

    # Confidence is the probability assigned to the predicted class.
    confidences = probs_flat[torch.arange(len(pred_flat), device=device), pred_flat]
    accuracies = pred_flat.eq(labels_flat)

    bin_boundaries = _uniform_bin_edges(num_bins, device)
    ece = torch.tensor(0.0, device=device)
    for idx, (bin_lower, bin_upper) in enumerate(zip(bin_boundaries[:-1], bin_boundaries[1:])):
        in_bin = _bin_mask(confidences, bin_lower, bin_upper, is_first=(idx == 0))
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece.item())


# Keep the old function name as an alias for backward compatibility
expected_calibration_error_3d = expected_calibration_error


def static_calibration_error(
    predictions: Optional[torch.Tensor],
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 10,
    ignore_background: bool = True,
    ignore_index: int = -1,
    valid_mask: Optional[torch.Tensor] = None,
) -> float:
    """Compute class-wise Static Calibration Error (SCE).

    SCE bins each class probability uniformly, compares mean confidence with
    the empirical frequency of that class in the bin, weights by bin occupancy,
    then averages across evaluated classes.  This makes multi-class calibration
    explicit instead of reducing everything to the predicted class only.
    """
    device = probabilities.device
    probs_flat, labels_flat, _ = _flatten_inputs(
        predictions,
        probabilities,
        labels,
        ignore_background=ignore_background,
        ignore_index=ignore_index,
        valid_mask=valid_mask,
    )
    if labels_flat.numel() == 0:
        return 0.0

    num_classes = probs_flat.shape[1]
    class_ids = range(1, num_classes) if ignore_background else range(num_classes)
    bin_boundaries = _uniform_bin_edges(num_bins, device)

    per_class_errors = []
    for cls in class_ids:
        cls_probs = probs_flat[:, cls]
        cls_targets = labels_flat.eq(cls).float()
        cls_error = torch.tensor(0.0, device=device)
        for idx, (bin_lower, bin_upper) in enumerate(zip(bin_boundaries[:-1], bin_boundaries[1:])):
            in_bin = _bin_mask(cls_probs, bin_lower, bin_upper, is_first=(idx == 0))
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                avg_confidence = cls_probs[in_bin].mean()
                avg_accuracy = cls_targets[in_bin].mean()
                cls_error += torch.abs(avg_confidence - avg_accuracy) * prop_in_bin
        per_class_errors.append(cls_error)

    if not per_class_errors:
        return 0.0
    return float(torch.stack(per_class_errors).mean().item())


def adaptive_calibration_error(
    predictions: Optional[torch.Tensor],
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 10,
    ignore_background: bool = True,
    ignore_index: int = -1,
    valid_mask: Optional[torch.Tensor] = None,
) -> float:
    """Compute class-wise Adaptive Calibration Error (ACE).

    ACE uses equal-count bins per class probability instead of fixed-width
    confidence bins.  Empty inputs return 0.0 so ROI metrics remain well-defined
    for empty foreground/ROI masks.
    """
    probs_flat, labels_flat, _ = _flatten_inputs(
        predictions,
        probabilities,
        labels,
        ignore_background=ignore_background,
        ignore_index=ignore_index,
        valid_mask=valid_mask,
    )
    if labels_flat.numel() == 0:
        return 0.0

    device = probabilities.device
    num_classes = probs_flat.shape[1]
    class_ids = range(1, num_classes) if ignore_background else range(num_classes)
    n_samples = int(labels_flat.numel())
    n_bins = min(int(num_bins), n_samples)
    if n_bins <= 0:
        return 0.0

    per_class_errors = []
    for cls in class_ids:
        cls_probs = probs_flat[:, cls]
        cls_targets = labels_flat.eq(cls).float()
        order = torch.argsort(cls_probs)
        sorted_probs = cls_probs[order]
        sorted_targets = cls_targets[order]
        boundaries = torch.linspace(0, n_samples, n_bins + 1, device=device)
        bin_errors = []
        for idx in range(n_bins):
            start = int(torch.floor(boundaries[idx]).item())
            end = int(torch.floor(boundaries[idx + 1]).item())
            if idx == n_bins - 1:
                end = n_samples
            if end <= start:
                continue
            avg_confidence = sorted_probs[start:end].mean()
            avg_accuracy = sorted_targets[start:end].mean()
            bin_errors.append(torch.abs(avg_confidence - avg_accuracy))
        if bin_errors:
            per_class_errors.append(torch.stack(bin_errors).mean())

    if not per_class_errors:
        return 0.0
    return float(torch.stack(per_class_errors).mean().item())


def calibration_errors(
    predictions: Optional[torch.Tensor],
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 10,
    ignore_background: bool = True,
    ignore_index: int = -1,
    valid_mask: Optional[torch.Tensor] = None,
) -> dict:
    """Return canonical ``ECE/SCE/ACE`` metrics for the same valid mask."""
    if predictions is None:
        predictions = torch.argmax(probabilities, dim=1, keepdim=True)
    return {
        "ECE": expected_calibration_error(
            predictions,
            probabilities,
            labels,
            num_bins=num_bins,
            ignore_background=ignore_background,
            ignore_index=ignore_index,
            valid_mask=valid_mask,
        ),
        "SCE": static_calibration_error(
            predictions,
            probabilities,
            labels,
            num_bins=num_bins,
            ignore_background=ignore_background,
            ignore_index=ignore_index,
            valid_mask=valid_mask,
        ),
        "ACE": adaptive_calibration_error(
            predictions,
            probabilities,
            labels,
            num_bins=num_bins,
            ignore_background=ignore_background,
            ignore_index=ignore_index,
            valid_mask=valid_mask,
        ),
    }


def dilate_foreground_mask(
    labels: torch.Tensor,
    kernel_size: int = 10,
    ignore_index: int = -1,
    spatial_dims: Optional[int] = None,
) -> torch.Tensor:
    """Create a dilated foreground ROI mask from labels.

    The returned mask has shape ``(B,1,...)`` and can be passed as
    ``valid_mask`` to the calibration functions.  Background inside the
    dilated ROI is intentionally retained by setting ``ignore_background=False``
    in the caller.
    """
    if spatial_dims is None:
        if labels.ndim == 5:
            spatial_dims = 3
        elif labels.ndim == 3:
            spatial_dims = 2
        elif labels.ndim == 4:
            spatial_dims = 2 if labels.shape[1] == 1 else 3
        else:
            raise ValueError(f"Unsupported label shape for ROI dilation: {tuple(labels.shape)}")

    if spatial_dims == 2:
        # 2D labels: (B,H,W) or (B,1,H,W)
        mask = labels if labels.ndim == 4 else labels.unsqueeze(1)
        pool = F.max_pool2d
    elif spatial_dims == 3:
        # 3D labels: (B,D,H,W) or (B,1,D,H,W)
        mask = labels if labels.ndim == 5 else labels.unsqueeze(1)
        pool = F.max_pool3d
    else:
        raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")

    if kernel_size <= 0:
        return mask.ne(0) & mask.ne(ignore_index)

    mask = (mask.ne(0) & mask.ne(ignore_index)).float()
    padding = kernel_size // 2
    dilated = pool(mask, kernel_size=kernel_size, stride=1, padding=padding)
    # Even-sized kernels can increase shape by one.  Crop/pad via nearest
    # interpolation to preserve exact shape.
    if dilated.shape[2:] != mask.shape[2:]:
        mode = "nearest"
        dilated = F.interpolate(dilated, size=mask.shape[2:], mode=mode)
    return dilated > 0.5


def roi_calibration_errors(
    predictions: Optional[torch.Tensor],
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 10,
    roi_dilation_kernel: int = 10,
    ignore_index: int = -1,
) -> dict:
    """Compute ``ECE/SCE/ACE`` inside a dilated foreground ROI.

    Background voxels inside the ROI are included.
    """
    roi_mask = dilate_foreground_mask(
        labels,
        kernel_size=roi_dilation_kernel,
        ignore_index=ignore_index,
        spatial_dims=probabilities.ndim - 2,
    )
    return calibration_errors(
        predictions,
        probabilities,
        labels,
        num_bins=num_bins,
        ignore_background=False,
        ignore_index=ignore_index,
        valid_mask=roi_mask,
    )


def expected_calibration_error_batch(
    predictions: torch.Tensor,
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 15,
    ignore_background: bool = True,
    ignore_index: int = -1
) -> torch.Tensor:
    """
    Compute ECE for each sample in a batch independently.

    Args:
        predictions: Predicted class labels - (B, 1, H, W) for 2D or (B, 1, D, H, W) for 3D
        probabilities: Softmax probabilities - (B, C, H, W) for 2D or (B, C, D, H, W) for 3D
        labels: Ground truth labels - (B, 1, H, W) for 2D or (B, 1, D, H, W) for 3D
        num_bins: Number of bins for calibration
        ignore_background: Whether to exclude background class
        ignore_index: Label value to ignore (e.g., -1 for padding)

    Returns:
        ece_per_sample: ECE for each sample in batch (B,)
    """
    batch_size = predictions.shape[0]
    ece_values = []

    for i in range(batch_size):
        ece = expected_calibration_error(
            predictions[i:i+1],
            probabilities[i:i+1],
            labels[i:i+1],
            num_bins=num_bins,
            ignore_background=ignore_background,
            ignore_index=ignore_index
        )
        ece_values.append(ece)

    return torch.tensor(ece_values)
