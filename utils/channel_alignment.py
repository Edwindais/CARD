"""Pre-flight channel alignment check for CARD categorical training/eval.

Catches ghost-class mismatches: model head dim, cfg.dataset
foreground count, and GT label range must agree before training/eval
proceeds. Runs in <1s and aborts startup with a clear message on mismatch.

Usage::

    from utils.channel_alignment import assert_channel_alignment
    sample = next(iter(train_dataloader))  # any one batch
    assert_channel_alignment(cfg, model, sample, role="train")
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch

# Keys that may carry the segmentation/categorical label tensor across
# the various dataloaders in this repo. Searched in priority order.
_CANDIDATE_MASK_KEYS: Sequence[str] = (
    "categorical_mask",
    "segmentation",
    "mask",
    "label",
    "gt",
    "y",
)

# Likely names of the final 1x1 convolution that maps features -> class
# logits. We check shape[0] (out-channels) of the first match.
_FINAL_CONV_PATTERNS: Sequence[str] = (
    "denoise_fn.out.conv.conv.weight",
    "denoise_fn.final_conv.weight",
    "denoise_fn.out_conv.weight",
    "out.conv.conv.weight",
    "final_conv.weight",
)


def _gt_max_value(sample: Any) -> Optional[int]:
    """Return the maximum integer label found in a batch's mask tensor.

    Ignores values < 0 (nnUNet uses -1 as ignore_index). Returns ``None``
    if no recognisable mask tensor is found.
    """
    tensor = _extract_mask_tensor(sample)
    if tensor is None:
        return None
    t = tensor.detach()
    if t.numel() == 0:
        return None
    if t.dtype.is_floating_point:
        t = t.long()
    valid = t[t >= 0]
    if valid.numel() == 0:
        return None
    return int(valid.max().item())


def _extract_mask_tensor(sample: Any) -> Optional[torch.Tensor]:
    if isinstance(sample, torch.Tensor):
        return sample
    if isinstance(sample, Mapping):
        for key in _CANDIDATE_MASK_KEYS:
            if key in sample and isinstance(sample[key], torch.Tensor):
                return sample[key]
        return None
    if isinstance(sample, (list, tuple)):
        for item in sample:
            t = _extract_mask_tensor(item)
            if t is not None:
                return t
    return None


def _find_final_conv_out_channels(model: torch.nn.Module) -> Optional[int]:
    state = model.state_dict()
    for pattern in _FINAL_CONV_PATTERNS:
        if pattern in state and state[pattern].ndim >= 2:
            return int(state[pattern].shape[0])
    # Fallback: scan for any param whose name endswith one of the patterns'
    # suffixes (covers wrappers that prepend prefixes).
    for name, tensor in state.items():
        if tensor.ndim < 2:
            continue
        for pat in _FINAL_CONV_PATTERNS:
            if name.endswith(pat):
                return int(tensor.shape[0])
    return None


def assert_channel_alignment(
    cfg,
    model: torch.nn.Module,
    sample: Optional[Any],
    role: str = "train",
) -> None:
    """Fail-fast cross-check between cfg, model head, and GT labels.

    Validates four invariants for categorical-diffusion runs:
      1. ``cfg.dataset.num_foreground_classes`` is a positive integer.
      2. ``cfg.model.out_dim == num_foreground_classes + int(use_bg)``.
      3. The model's final-conv out-channel count matches (2).
      4. The GT mask's max label is ``<= num_foreground_classes``.

    Gaussian/SDF runs only check (1) and skip the categorical-specific
    invariants. Pass ``sample=None`` to skip the GT-label check (e.g. on
    eval entrypoints where loading a batch is expensive).
    """
    diffusion_type = str(cfg.model.get("diffusion_type", "gaussian")).lower()
    dataset_cfg = cfg.dataset

    n_fg_raw = dataset_cfg.get("num_foreground_classes", None)
    if n_fg_raw is None or int(n_fg_raw) <= 0:
        raise ValueError(
            "cfg.dataset.num_foreground_classes must be a positive integer. "
            f"Got {n_fg_raw!r}. Add it explicitly to "
            f"config/dataset/{dataset_cfg.get('name', '<dataset>')}.yaml."
        )
    n_fg = int(n_fg_raw)

    use_bg = bool(dataset_cfg.get("categorical_use_background", True))
    expected_out = n_fg + (1 if use_bg else 0)

    cfg_out = int(cfg.model.get("out_dim", -1))
    head_out = _find_final_conv_out_channels(model)
    gt_max = _gt_max_value(sample) if sample is not None else None

    is_categorical = diffusion_type == "categorical"

    if is_categorical:
        if cfg_out != expected_out:
            raise ValueError(
                f"cfg.model.out_dim={cfg_out} but expected "
                f"{expected_out} (num_foreground_classes={n_fg} + "
                f"int(use_bg={use_bg})). Did prepare_cfg run?"
            )
        if head_out is not None and head_out != expected_out:
            raise ValueError(
                f"Model final-conv out-channels={head_out} mismatches "
                f"expected={expected_out}. Likely loading a checkpoint "
                "with a different class count."
            )

    if gt_max is not None and gt_max > n_fg:
        raise ValueError(
            f"GT label max={gt_max} exceeds num_foreground_classes={n_fg}. "
            f"Either the config under-counts classes or the dataset has "
            f"unexpected labels."
        )

    dataset_name = dataset_cfg.get("name", "<unknown>")
    print(
        f"[CONFIG-CHECK:{role}] dataset={dataset_name} "
        f"diffusion={diffusion_type} n_fg={n_fg} use_bg={use_bg} "
        f"cfg.out_dim={cfg_out} head_out={head_out} gt_max={gt_max} OK"
    )
