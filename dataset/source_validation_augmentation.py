"""Dataset-agnostic source-validation intensity augmentation."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE_ROOT / "metadata" / "mri_like_augmentation_presets.json"


def load_conditions() -> dict[str, dict]:
    return json.loads(CONFIG_PATH.read_text())["conditions"]


def apply_intensity_augmentation(image: torch.Tensor, config: dict, seed: int) -> torch.Tensor:
    if config.get("preset") is None:
        return image
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        output = image.float()
        contrast = torch.empty((), dtype=output.dtype).uniform_(*config["contrast"])
        mean = output.mean()
        output = (output - mean) * contrast + mean

        gamma = torch.empty((), dtype=output.dtype).uniform_(*config["gamma_range"])
        intensity_min = output.min()
        intensity_range = output.max() - intensity_min + 1e-5
        shifted = output - intensity_min + 1e-5
        output = intensity_range * torch.pow(shifted / intensity_range, gamma) + intensity_min
        output = output + torch.randn_like(output) * float(config["noise_std"])
    return output.clamp_(-3.5, 4.0).to(dtype=image.dtype)


class SourceValidationAugmentationDataset(Dataset):
    def __init__(self, dataset: Dataset, condition: str, seed: int):
        conditions = load_conditions()
        if condition not in conditions:
            raise ValueError(f"Unknown source-validation condition: {condition}")
        self.dataset = dataset
        self.config = conditions[condition]
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        item = dict(self.dataset[index])
        case_index, slice_index = self.dataset.slice_mapping[index]
        item_seed = self.seed + hash((case_index, slice_index)) % (2**31)
        item["img"] = apply_intensity_augmentation(
            item["img"], self.config, item_seed
        )
        return item
