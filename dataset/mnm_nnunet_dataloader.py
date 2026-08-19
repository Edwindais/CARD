"""Load preprocessed M&Ms cases and group them by scanner vendor."""

from __future__ import annotations

import os
import pickle

import numpy as np
import torch

from dataset.acdc_nnunet_dataloader import ACDC_nnUNet_Dataset


VENDOR_MAPPING = {"A": ("1", "5"), "B": ("2",), "C": ("3",), "D": ("4", "6")}


class MNM_nnUNet_Dataset(ACDC_nnUNet_Dataset):
    case_prefix = "MNM_"
    dataset_label = "M&Ms"

    def __init__(self, root_dir: str, target_vendor: str, max_cases: int | None = None):
        if target_vendor not in VENDOR_MAPPING:
            raise ValueError(f"Unknown M&Ms vendor: {target_vendor}")
        self.target_vendor = target_vendor
        self.max_cases = max_cases
        super().__init__(
            root_dir=root_dir,
            mode="test",
            num_foreground_classes=3,
            categorical_use_background=True,
            split_config={"roi_z": 1, "roi_y": 256, "roi_x": 256},
        )
    def get_file_names(self):
        case_paths = super().get_file_names()
        vendor_ids = VENDOR_MAPPING[self.target_vendor]
        selected = [path for path in case_paths if os.path.basename(path).split("_")[1] in vendor_ids]
        if not selected:
            raise RuntimeError(f"No M&Ms cases found for vendor {self.target_vendor}")
        if self.max_cases is not None:
            stratified = []
            for centre in vendor_ids:
                centre_cases = [
                    path for path in selected
                    if os.path.basename(path).split("_")[1] == centre
                ]
                if centre_cases:
                    stratified.append(centre_cases[0])
            stratified.extend(path for path in selected if path not in stratified)
            selected = stratified[: self.max_cases]
        return selected

    def load_nnUNet_data(self, pkl_file: str):
        with open(pkl_file, "rb") as handle:
            properties = pickle.load(handle)
        if "data" not in properties:
            return super().load_nnUNet_data(pkl_file)
        data = np.asarray(properties["data"], dtype=np.float32)
        seg = np.asarray(properties["seg"], dtype=np.float32)
        if data.ndim == 3:
            data = np.transpose(data, (2, 0, 1))[None]
            seg = np.transpose(seg, (2, 0, 1))[None]
        return torch.from_numpy(data), torch.from_numpy(seg), properties
