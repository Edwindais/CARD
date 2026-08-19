"""
ACDC-C Dataset with TorchIO On-the-Fly Artifacts.

This dataloader applies MRI artifacts using TorchIO transforms at runtime,
following standard TorchIO-based corruption methodology for OOD evaluation.
It also supports the CARD source-validation response/noise augmentation.
"""

import os
import torch
import numpy as np
import torchio as tio
from dataset.acdc_nnunet_dataloader import ACDC_nnUNet_Dataset


# Artifact configurations for severity levels
ARTIFACT_CONFIGS = {
    'Bias': {
        0: {'coefficients': 0.5},
        1: {'coefficients': 1.0},
        2: {'coefficients': 1.5},
    },
    'Motion': {
        0: {'degrees': 10, 'translation': 10, 'num_transforms': 2},
        1: {'degrees': 20, 'translation': 20, 'num_transforms': 3},
        2: {'degrees': 30, 'translation': 30, 'num_transforms': 5},
    },
    'Ghosting': {
        0: {'num_ghosts': (4, 8), 'intensity': (0.5, 1.5)},
        1: {'num_ghosts': (8, 15), 'intensity': (1.0, 2.0)},
        2: {'num_ghosts': (15, 25), 'intensity': (2.0, 4.0)},
    },
    'Spike': {
        0: {'num_spikes': 2, 'intensity': (0.5, 1.0)},
        1: {'num_spikes': 5, 'intensity': (1.0, 2.0)},
        2: {'num_spikes': 10, 'intensity': (2.0, 4.0)},
    },
}

# Map artifact names to TorchIO transform classes
ARTIFACT_TRANSFORMS = {
    'Bias': tio.RandomBiasField,
    'Motion': tio.RandomMotion,
    'Ghosting': tio.RandomGhosting,
    'Spike': tio.RandomSpike,
}

# Source-validation response/noise augmentation. This protocol is independent of the
# ACDC-C Bias/Motion/Ghosting/Spike transforms above.
SOURCE_VALIDATION_AUGMENTATION = 'SourceValidationAugmentation'

PHOTOMETRIC_PRESETS = {
    'source_validation_low': {
        'gamma_range': (0.9, 1.1),
        'noise_std': 0.01,
        'clip_pm1': False,
        'contrast': (0.9, 1.1),
    },
    'source_validation_medium': {
        'gamma_range': (0.85, 1.18),
        'noise_std': 0.03,
        'clip_pm1': False,
        'contrast': (0.85, 1.18),
    },
    'source_validation_response_noise_heavy': {
        'gamma_range': (0.85, 1.18),
        'noise_std': 0.04,
        'clip_pm1': False,
        'contrast': (0.85, 1.18),
    },
}

PHOTOMETRIC_OPS = {'contrast', 'gamma', 'noise'}


class ACDC_C_TorchIO_Dataset(ACDC_nnUNet_Dataset):
    """
    ACDC dataset with TorchIO-based on-the-fly OOD corruption.

    Inherits from ACDC_nnUNet_Dataset (clean data) and applies
    MRI artifacts using TorchIO transforms with reproducible seeds.
    """

    def __init__(
        self,
        root_dir=None,
        mode='test',
        corruption_type='Bias',
        severity=1,
        seed=42,
        stage_two=False,
        num_foreground_classes=3,
        predict_background_sdf=False,
        categorical_use_background=True,
        split_config=None,
        photometric_preset='source_validation_medium',
        photometric_ops='all',
        **kwargs
    ):
        self.corruption_type = corruption_type
        self.severity = float(severity)
        self.base_seed = seed
        self.photometric_preset = photometric_preset
        self.photometric_ops = self._parse_photometric_ops(photometric_ops)

        self.photometric_config = None

        # Handle 'Clean' (no corruption)
        if corruption_type == 'Clean':
            self.artifact_transform = None
        elif corruption_type == SOURCE_VALIDATION_AUGMENTATION:
            if self.severity != 1.0:
                raise ValueError(
                    f"Severity {self.severity} not available for {corruption_type}; "
                    "Source-validation augmentation uses severity=1 only."
            )
            self.artifact_transform = None
            self.photometric_config = self._build_photometric_config(
                photometric_preset,
                self.photometric_ops,
            )
        else:
            if corruption_type not in ARTIFACT_CONFIGS:
                raise ValueError(f"Unknown corruption_type: {corruption_type}. "
                               f"Choose from: {list(ARTIFACT_CONFIGS.keys())}, "
                               f"{SOURCE_VALIDATION_AUGMENTATION}, or 'Clean'")
            if self.severity not in ARTIFACT_CONFIGS[corruption_type]:
                raise ValueError(f"Severity {self.severity} not available for {corruption_type}.")

            artifact_params = ARTIFACT_CONFIGS[corruption_type][self.severity]
            transform_class = ARTIFACT_TRANSFORMS[corruption_type]
            self.artifact_transform = transform_class(**artifact_params, p=1.0)


        # Initialize parent class (loads clean ACDC data)
        super().__init__(
            root_dir=root_dir,
            mode=mode,
            stage_two=stage_two,
            num_foreground_classes=num_foreground_classes,
            predict_background_sdf=predict_background_sdf,
            categorical_use_background=categorical_use_background,
            split_config=split_config,
        )

        # Force scan slices for 2D evaluation (parent only scans for train/val)
        if mode == 'test':
            if not hasattr(self, 'slice_mapping') or len(self.slice_mapping) == 0:
                self._scan_slices()
        self._apply_scout_subset(
            max_cases=kwargs.get('max_cases', None),
            case_offset=kwargs.get('case_offset', 0),
            max_slices_per_case=kwargs.get('max_slices_per_case', None),
            max_slices=kwargs.get('max_slices', None),
        )

    def __len__(self):
        return len(self.slice_mapping)

    @staticmethod
    def _optional_int(value):
        if value is None or value == "":
            return None
        return int(value)

    def _apply_scout_subset(
        self,
        *,
        max_cases=None,
        case_offset=0,
        max_slices_per_case=None,
        max_slices=None,
    ):
        """Optionally restrict source-val evaluation for fast scouting.

        Full source-validation runs leave these fields unset.  Smaller values
        can be passed through Hydra for a deterministic diagnostic subset
        before running the complete source-validation cache.
        """
        max_cases = self._optional_int(max_cases)
        case_offset = int(case_offset or 0)
        max_slices_per_case = self._optional_int(max_slices_per_case)
        max_slices = self._optional_int(max_slices)
        if max_cases is None and max_slices_per_case is None and max_slices is None and case_offset == 0:
            return

        if max_cases is not None or case_offset:
            start = max(0, case_offset)
            stop = None if max_cases is None else start + max_cases
            self.file_names = self.file_names[start:stop]

        # Rebuild slice_mapping after any case filtering, then optionally
        # retain only the first K slices per selected case.
        self.slice_mapping = []
        self._scan_slices()
        if max_slices_per_case is not None:
            per_case_counts = {}
            filtered = []
            for case_idx, slice_idx in self.slice_mapping:
                used = per_case_counts.get(case_idx, 0)
                if used < max_slices_per_case:
                    filtered.append((case_idx, slice_idx))
                    per_case_counts[case_idx] = used + 1
            self.slice_mapping = filtered
        if max_slices is not None:
            self.slice_mapping = self.slice_mapping[:max_slices]

    def __getitem__(self, index):
        # Get clean data from parent
        case_idx, slice_idx = self.slice_mapping[index]
        pkl_file = self.file_names[case_idx]

        # Load clean volumes
        img, mask, properties = self.load_nnUNet_data(pkl_file)

        if mask.ndim == 3:
            mask = mask.unsqueeze(0)

        # Structural ACDC-C artifacts use one deterministic seed per full case
        # volume; source-validation photometric augmentation keeps the legacy
        # (case_idx, slice_idx) seed below.
        artifact_mode = self.artifact_transform is not None
        item_seed = (
            self.base_seed + case_idx
            if artifact_mode
            else self.base_seed + hash((case_idx, slice_idx)) % (2**31)
        )
        torch.manual_seed(item_seed)
        np.random.seed(item_seed % (2**31))

        # Apply TorchIO artifact to the FULL volume (before slicing)
        affine = torch.eye(4)

        if self.artifact_transform is not None:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=img, affine=affine),
                label=tio.LabelMap(tensor=mask, affine=affine)
            )
            corrupted_subject = self.artifact_transform(subject)
            img_corrupted = corrupted_subject.image.data
        else:
            img_corrupted = img

        # Extract 2D slice
        img_slice = img_corrupted[:, slice_idx:slice_idx+1, :, :]
        mask_slice = mask[:, slice_idx:slice_idx+1, :, :]

        if self.photometric_config is not None:
            photometric_seed = self.base_seed + hash((case_idx, slice_idx)) % (2**31)
            torch.manual_seed(photometric_seed)
            np.random.seed(photometric_seed % (2**31))
            img_slice = self._apply_source_validation_augmentation(img_slice, self.photometric_config)

        # Apply any additional transforms from parent
        if self.transforms is not None:
            slice_subject = tio.Subject(
                image=tio.ScalarImage(tensor=img_slice, affine=affine),
                label=tio.LabelMap(tensor=mask_slice, affine=affine)
            )
            transformed = self.transforms(slice_subject)
            img_slice = transformed.image.data
            mask_slice = transformed.label.data

        # Prepare outputs
        mask_categorical = mask_slice.long()
        if not self.categorical_use_background:
            mask_categorical = mask_categorical - 1

        if mask_categorical.ndim == 4 and mask_categorical.shape[0] == 1:
            mask_categorical = mask_categorical.squeeze(0)

        basename = os.path.basename(pkl_file).replace('.pkl', '')
        name = f"{basename}_z{slice_idx:03d}"

        return {
            'img': img_slice,
            'mask': mask_slice,
            'name': name,
            'mask_sdf': torch.zeros_like(img_slice),
            'categorical_mask': mask_categorical,
            'affine': affine,
            'case_name': basename,
            'slice_index': slice_idx,
            'num_slices': img.shape[1]
        }

    @staticmethod
    def _uniform_scalar(low, high, ref_tensor):
        return torch.empty((), dtype=ref_tensor.dtype, device=ref_tensor.device).uniform_(low, high)

    @staticmethod
    def _parse_photometric_ops(ops):
        if ops is None or ops == 'all':
            return set(PHOTOMETRIC_OPS)
        if isinstance(ops, str):
            tokens = [item.strip() for item in ops.replace('+', ',').split(',') if item.strip()]
        else:
            tokens = [str(item).strip() for item in ops if str(item).strip()]

        parsed = set()
        for token in tokens:
            if token == 'all':
                parsed.update(PHOTOMETRIC_OPS)
            elif token in PHOTOMETRIC_OPS:
                parsed.add(token)
            else:
                raise ValueError(
                    f"Unknown photometric op: {token}. "
                    f"Choose from {sorted(PHOTOMETRIC_OPS | {'all'})}"
                )
        if not parsed:
            raise ValueError("photometric_ops resolved to an empty operation set")
        return parsed

    @classmethod
    def _build_photometric_config(cls, preset, ops):
        if preset not in PHOTOMETRIC_PRESETS:
            raise ValueError(
                f"Unknown photometric_preset: {preset}. "
                f"Choose from {sorted(PHOTOMETRIC_PRESETS)}"
            )
        cfg = dict(PHOTOMETRIC_PRESETS[preset])
        cfg['ops'] = set(ops)
        return cfg

    @classmethod
    def _apply_source_validation_augmentation(cls, img, cfg):
        """Apply the source-validation contrast/gamma/noise augmentation."""

        out = img.float()
        ops = set(cfg.get('ops', PHOTOMETRIC_OPS))

        if 'contrast' in ops:
            cmin, cmax = cfg['contrast']
            contrast = cls._uniform_scalar(cmin, cmax, out)
            img_mean = out.mean()
            out = (out - img_mean) * contrast + img_mean

        if 'gamma' in ops:
            gmin, gmax = cfg['gamma_range']
            gamma = cls._uniform_scalar(gmin, gmax, out)
            intensity_min = out.min()
            intensity_range = out.max() - intensity_min + 1e-5
            out = out - intensity_min + 1e-5
            out = intensity_range * torch.pow(out / intensity_range, gamma)
            out = out + intensity_min

        if 'noise' in ops:
            out = out + torch.randn_like(out) * float(cfg['noise_std'])
            if cfg.get('clip_pm1', False):
                out = torch.clamp(out, -1.0, 1.0)
        return out.to(dtype=img.dtype)
