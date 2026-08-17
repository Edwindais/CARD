import glob
import json
import math
import os
import pickle
from collections import Counter
from collections.abc import Mapping
from typing import Dict, Optional

import blosc2
import numpy as np
import torch
import torchio as tio
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def _to_plain_dict(value, name: str) -> dict:
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return dict(value)


def _build_transforms(target_shape, training: bool, stage_two: bool):
    transforms = [tio.CropOrPad(target_shape=target_shape)]
    if training:
        transforms.extend([
            tio.RandomNoise(std=(0, 0.2 if stage_two else 0.15), p=0.5, include=["image"]),
            tio.RandomGamma(log_gamma=(-0.5, 0.5) if stage_two else (-0.4, 0.4), p=0.5, include=["image"]),
            tio.RandomAffine(
                scales=(0.85, 1.15) if stage_two else (0.9, 1.1),
                degrees=(8, 8, 8) if stage_two else (5, 5, 5),
                translation=(6, 6, 6) if stage_two else (4, 4, 4),
                image_interpolation="linear",
                label_interpolation="nearest",
                p=0.5,
            ),
        ])
        if stage_two:
            transforms.append(tio.RandomElasticDeformation(
                num_control_points=7,
                max_displacement=7.5,
                image_interpolation="linear",
                p=0.25,
            ))
    transforms.append(tio.Lambda(lambda tensor: tensor.clamp(-3.5, 4.0), include=["image"]))
    return tio.Compose(transforms)


class ACDC_nnUNet_Dataset(Dataset):
    case_prefix = "ACDC_"
    dataset_label = "ACDC"

    def __init__(
        self,
        root_dir,
        mode='train',
        stage_two=False,
        num_foreground_classes=3,  # RV, MYO, LV
        predict_background_sdf=False,
        categorical_use_background=True,
        split_config: Optional[Dict] = None,
    ):
        if not root_dir or not os.path.isdir(root_dir):
            raise RuntimeError(
                f"{self.dataset_label} preprocessed data not found at {root_dir}"
            )
        self.preprocessed_dir = os.fspath(root_dir)
        self.nnunet_preprocessed_dir = self.preprocessed_dir
        self.mode = mode
        self.stage_two = bool(stage_two)
        self.num_foreground_classes = int(num_foreground_classes)
        self.predict_background_sdf = bool(predict_background_sdf)
        self.categorical_use_background = bool(categorical_use_background)

        # Override split_config defaults if not provided to ensure 2D patch size
        split_config = _to_plain_dict(split_config, 'split_config')

        # Default target shape for 2D: (1, 256, 256)
        # roi_z=1 (Pseudo-3D), roi_y=256, roi_x=256
        split_config.setdefault('roi_z', 1)
        split_config.setdefault('roi_y', 256)
        split_config.setdefault('roi_x', 256)

        self.split_config = split_config
        self.target_shape = (
            int(split_config["roi_z"]),
            int(split_config["roi_y"]),
            int(split_config["roi_x"]),
        )
        self.transforms = _build_transforms(
            self.target_shape,
            training=mode == "train",
            stage_two=self.stage_two,
        )
        self.file_names = self.get_file_names()

        # Flatten slices mapping for ALL modes (train, val, AND test)
        # This ensures test mode also iterates over individual slices
        self.slice_mapping = []
        self._scan_slices()

    def get_file_names(self):
        """Get the configured dataset's case files and apply its split."""
        data_dir = self.nnunet_preprocessed_dir
        all_pkls = glob.glob(os.path.join(data_dir, f"{self.case_prefix}*.pkl"))
        if not all_pkls:
            raise RuntimeError(
                f"No {self.case_prefix}*.pkl files found for {self.dataset_label} in {data_dir}"
            )

        # Map basename -> full path
        case_map = {os.path.basename(p).replace('.pkl', ''): p for p in all_pkls}

        # 2. Load splits
        # PRIORITY: Check if a specific split file is provided in config
        custom_split = self.split_config.get('json_path') if self.split_config else None

        dataset_root = os.path.dirname(data_dir)
        splits_file = custom_split if custom_split else os.path.join(dataset_root, 'splits_final.json')
        test_split_file = os.path.join(dataset_root, 'split_info.json')

        # Determine target keys based on mode
        target_keys = []

        if os.path.exists(splits_file):
            # Load the main split file (this handles train/val AND test_A/B/C/D if present)
            with open(splits_file, 'r') as f:
                splits_data = json.load(f)

            # Handle list of folds (take fold 0) or direct dict
            if isinstance(splits_data, list):
                fold_data = splits_data[0]
            else:
                fold_data = splits_data

            # Try to get keys for current mode
            # Support 'test_A' directly in keys, or map 'test' -> 'validation' etc.
            if self.mode in fold_data:
                target_keys = fold_data[self.mode]
            elif self.mode == 'train':
                 target_keys = fold_data.get('train', [])
            elif self.mode == 'val':
                 target_keys = fold_data.get('val', fold_data.get('validation', []))
            elif self.mode == 'test':
                 target_keys = fold_data.get('test', [])

        # Test cases are stored beside the preprocessed dataset.
        if not target_keys and self.mode == 'test':
            if os.path.exists(test_split_file):
                with open(test_split_file, 'r') as f:
                    split_data = json.load(f)
                target_keys = split_data.get('test', [])
            if not target_keys:
                raise RuntimeError(f"No test split found in {dataset_root}")

        # Resolve the selected split to case files.
        if target_keys:
            duplicate_keys = sorted(
                key for key, count in Counter(target_keys).items() if count > 1
            )
            if duplicate_keys:
                raise RuntimeError(
                    f"Split {splits_file} contains duplicate cases for mode {self.mode!r}: {duplicate_keys[:5]}"
                )
            missing_keys = [key for key in target_keys if key not in case_map]
            if missing_keys:
                raise RuntimeError(
                    f"Split {splits_file} references missing cases for mode {self.mode!r}: {missing_keys[:5]}"
                )
            return [case_map[key] for key in target_keys]

        raise RuntimeError(f"No keys found for mode {self.mode}. Checked {splits_file}")

    def _scan_slices(self):
        """Scan all volumes to build a flat list of (case_idx, slice_idx)."""
        print(f"Scanning {len(self.file_names)} cases for 2D slices...")
        count = 0
        for i, pkl_file in enumerate(self.file_names):
            b2nd_file = pkl_file.replace('.pkl', '.b2nd')
            with open(pkl_file, 'rb') as f:
                props = pickle.load(f)

            shape = props.get('shape', props.get('shape_after_preprocessing', props.get('size')))
            if shape is None:
                dparams = {'nthreads': 1}
                mmap_kwargs = {} if os.name == "nt" else {'mmap_mode': 'r'}
                shape = blosc2.open(
                    urlpath=b2nd_file,
                    mode='r',
                    dparams=dparams,
                    **mmap_kwargs,
                ).shape

            if len(shape) == 3:
                depth = min(shape)
            elif len(shape) == 4:
                depth = shape[1]
            else:
                raise ValueError(f"Unsupported preprocessed shape for {pkl_file}: {shape}")

            for z in range(depth):
                self.slice_mapping.append((i, z))
            count += depth
        print(f"Flattening complete: {len(self.file_names)} volumes -> {count} slices.")

    def load_nnUNet_data(self, pkl_file):
        """Load one nnUNet case from its metadata and Blosc2 arrays."""
        with open(pkl_file, "rb") as handle:
            properties = pickle.load(handle)

        def load_array(path):
            dparams = {"nthreads": 1}
            mmap_kwargs = {} if os.name == "nt" else {"mmap_mode": "r"}
            packed = blosc2.open(urlpath=path, mode="r", dparams=dparams, **mmap_kwargs)
            content = packed[:]
            if isinstance(content, bytes):
                dtype = np.float32 if packed.typesize == 4 else np.uint8
                array = np.frombuffer(content, dtype=dtype)
                stored_shape = tuple(getattr(packed, "shape", ()))
                if stored_shape and math.prod(stored_shape) == array.size:
                    array = array.reshape(stored_shape)
                return array
            return np.asarray(content)

        data = load_array(pkl_file.replace(".pkl", ".b2nd")).astype(np.float32)
        seg = load_array(pkl_file.replace(".pkl", "_seg.b2nd")).astype(np.float32)
        shape = properties.get("shape", properties.get("shape_after_preprocessing"))
        if shape:
            expected_shape = tuple(int(value) for value in shape)
            expected_size = math.prod(expected_shape)
            if data.ndim == 1:
                if data.size != expected_size:
                    raise ValueError(f"Image size {data.size} does not match {expected_shape}")
                data = data.reshape(expected_shape)
            if seg.ndim == 1:
                if seg.size != expected_size:
                    raise ValueError(f"Segmentation size {seg.size} does not match {expected_shape}")
                seg = seg.reshape(expected_shape)
        if data.ndim == 3:
            data = data[None]
        if seg.ndim == 3:
            seg = seg[None]
        return torch.from_numpy(data).float(), torch.from_numpy(seg).float(), properties

    def __len__(self):
        return len(self.slice_mapping)

    def __getitem__(self, index):
        """Get 2D slice for all modes (train, val, test)."""
        case_idx, slice_idx = self.slice_mapping[index]
        pkl_file = self.file_names[case_idx]

        # 1. Load data manually to extract slice efficiently
        img, mask, properties = self.load_nnUNet_data(pkl_file)

        # 2. Extract Slice: (C, D, H, W) -> (C, 1, H, W)
        img_slice = img[:, slice_idx:slice_idx+1, :, :]
        mask_slice = mask[:, slice_idx:slice_idx+1, :, :]

        # 3. Apply Transforms
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=img_slice),
            label=tio.LabelMap(tensor=mask_slice)
        )

        if self.transforms is not None:
            transformed = self.transforms(subject)
            img = transformed.image.data
            mask = transformed.label.data
        else:
            img = img_slice
            mask = mask_slice

        # 4. Post-processing (Categorical)
        # Logic adapted from TS_nnUNet_Dataset.__getitem__
        labels_out = mask.long()

        # Handle background class
        if not self.categorical_use_background:
            labels_out = labels_out - 1

        # Squeeze channel dim: (1, 1, H, W) -> (1, H, W)
        # Note: CategoricalDiffusion expects (B, 1, H, W) or (B, H, W) input.
        # But 'label' implies standard segmentation map.
        # Let's keep (1, H, W).
        if labels_out.ndim == 4 and labels_out.shape[0] == 1:
             labels_out = labels_out.squeeze(0)

        return {
            'img': img,   # (1, 1, 256, 256)
            'mask': labels_out, # (1, 256, 256)
            'name': os.path.basename(pkl_file).replace('.pkl', '') + f"_z{slice_idx:03d}",
            'mask_sdf': torch.zeros_like(img), # Placeholder matching (C, 1, H, W)
            'categorical_mask': labels_out, # Required for categorical diffusion
            'affine': torch.eye(4) # Required for collate
        }


def data_collate(batch):
    """Collate function for ACDC dataset batches."""
    return {
        "image": torch.stack([item["img"] for item in batch]).float(),
        "mask_sdf": torch.stack([item["mask_sdf"] for item in batch]).float(),
        "segmentation": torch.stack([item["mask"] for item in batch]),
        "categorical_mask": torch.stack([item["categorical_mask"] for item in batch]).long(),
        "name": [item["name"] for item in batch],
        "affine": torch.stack([torch.as_tensor(item["affine"]).float() for item in batch]),
    }


def get_loader(
    cfg,
    mode='train',
    is_distributed=False,
    stage_two=False,
    root_override=None,
    dataset_class=ACDC_nnUNet_Dataset,
    default_num_foreground_classes=3,
):
    """Create a dataloader for ACDC nnUNet dataset.

    Args:
        cfg: Dataset configuration (OmegaConf or dict)
        mode: 'train', 'val', or 'test'
        is_distributed: Whether to use DistributedSampler
        stage_two: Whether to use enhanced augmentations
        root_override: Optional path to override root_dir
        dataset_class: Dataset adapter class to instantiate
        default_num_foreground_classes: Foreground-class default for the adapter

    Returns:
        DataLoader instance
    """
    if mode not in {'train', 'val', 'test'}:
        raise ValueError(f"Unsupported loader mode: {mode!r}")
    root_dir = root_override or cfg.get('root_dir', '')
    if mode == 'val' and root_override is None:
        root_dir = cfg.get('val_root_dir', root_dir)

    if mode == 'train':
        batch_size = cfg.get('batch_size', 16)
        num_workers = cfg.get('num_workers', 8)
    else:
        batch_size = cfg.get('test_batch_size', cfg.get('batch_size', 16))
        num_workers = cfg.get('val_num_workers', cfg.get('num_workers', 8))

    print(f'{dataset_class.dataset_label} {mode} loader - batch_size: {batch_size}')

    split_config = _to_plain_dict(cfg.get('split_config'), 'split_config')
    split_config['roi_x'] = cfg.get('roi_x', 256)
    split_config['roi_y'] = cfg.get('roi_y', 256)
    split_config['roi_z'] = cfg.get('roi_z', 1)

    dataset = dataset_class(
        root_dir=root_dir,
        mode=mode,
        stage_two=stage_two,
        num_foreground_classes=int(cfg.get('num_foreground_classes', default_num_foreground_classes)),
        predict_background_sdf=bool(cfg.get('predict_background_sdf', False)),
        categorical_use_background=bool(cfg.get('categorical_use_background', True)),
        split_config=split_config,
    )

    if is_distributed:
        sampler = DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = mode == 'train'

    drop_last = mode == 'train'
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=data_collate,
    )
    return dataloader
