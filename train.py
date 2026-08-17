#!/usr/bin/env python3
"""Train primary or reference diffusion segmentation models."""

import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from model.ddpm import CategoricalDiffusion, Trainer
from utils.segmentation_callback import SegmentationVisualizationCallback
from utils.channel_alignment import assert_channel_alignment
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
import importlib
from model.denoisers import TimeAwareSwinUNETR


def _resolve_to_dict(value):
    """Convert optional DictConfig to a plain dictionary."""
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def require_cuda_device(gpu_index):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    torch.cuda.init()
    torch.cuda.set_device(gpu_index)
    return torch.device(f"cuda:{gpu_index}")


def _get_swinunetr_config(swin_type: str = 'base'):
    """Get SwinUNETR configuration based on model size."""
    configs = {
        'mini': {
            'feature_size': 12,
            'depths': [1, 1, 1, 1],
            'num_heads': [3, 6, 12, 24],
        },
        'tiny': {
            'feature_size': 12,
            'depths': [2, 2, 2, 2],
            'num_heads': [3, 6, 12, 24],
        },
        'small': {
            'feature_size': 24,
            'depths': [2, 2, 2, 2],
            'num_heads': [3, 6, 12, 24],
        },
        'base': {
            'feature_size': 48,
            'depths': [2, 2, 18, 2],
            'num_heads': [6, 12, 24, 48],
        },
        'large': {
            'feature_size': 60,
            'depths': [2, 2, 18, 2],
            'num_heads': [6, 12, 24, 48],
        }
    }
    return configs.get(swin_type, configs['base'])


class MONAIModelWrapper(torch.nn.Module):
    """Wrapper for MONAI models to match the expected interface."""

    def __init__(self, model, *, cond_channels: int = 0):
        super().__init__()
        self.model = model
        self.cond_channels = int(cond_channels)

    def forward(self, x, t, **kwargs):
        output = self.model(x)
        if isinstance(output, list):
            return output[-1]
        return output

    def forward_with_cond_scale(self, x, t, *, cond_scale: float = 1.0, **kwargs):
        logits = self.forward(x, t, **kwargs)
        if cond_scale == 1.0 or self.cond_channels <= 0:
            return logits
        if x.shape[1] < self.cond_channels:
            raise ValueError('Input tensor has fewer channels than configured conditioning channels')
        uncond_x = x.clone()
        uncond_x[:, -self.cond_channels:, ...] = 0
        null_logits = self.forward(uncond_x, t, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale


def _ensure_cfg_wrapper(model: torch.nn.Module, cond_channels: int) -> torch.nn.Module:
    if hasattr(model, 'forward_with_cond_scale'):
        return model
    return MONAIModelWrapper(model, cond_channels=cond_channels)


def create_denoising_model(cfg: DictConfig, dataset_cfg: DictConfig):
    """Create a denoising backbone model based on configuration.

    Args:
        cfg: Full configuration (must contain cfg.model).
        dataset_cfg: Dataset configuration.

    Returns:
        A model instance with `forward_with_cond_scale` method.
    """
    name = str(cfg.model.denoising_fn).lower()
    if name not in {"swinunetr", "swin_unetr"}:
        raise ValueError("This package trains only the cardiac SwinUNETR denoiser")
    total_in = cfg.model.total_input_channels
    out_channels = cfg.model.diffusion_num_channels
    swin_type = cfg.model.get('swinunetr_type', 'base')
    defaults = {
        'spatial_dims': 2,
        'in_channels': total_in,
        'out_channels': out_channels,
        'patch_size': 2,
        'window_size': [7, 7],
        'use_checkpoint': False,
        'use_v2': True,
        'cond_channels': cfg.model.cond_channels,
        **_get_swinunetr_config(swin_type),
    }
    defaults.update(_resolve_to_dict(cfg.model.get('swinunetr_kwargs', {})) or {})
    model = TimeAwareSwinUNETR(**defaults)

    return _ensure_cfg_wrapper(model, cfg.model.cond_channels)


@hydra.main(config_path='config', config_name='joint_cfg', version_base=None)
def run(cfg: DictConfig):
    """Main training entry point."""
    with open_dict(cfg):
        cfg.model.gpus = 0
        cfg.model.save_and_sample_every = cfg.model.get('save_and_sample_every', 250)
        cfg.model.gradient_accumulate_every = cfg.model.get('gradient_accumulate_every', 1)

        if 'gpus_override' in cfg.model:
            cfg.model.gpus = cfg.model.gpus_override
        if 'train_num_steps_override' in cfg.model:
            cfg.model.train_num_steps = cfg.model.train_num_steps_override
        if 'save_and_sample_every_override' in cfg.model:
            cfg.model.save_and_sample_every = cfg.model.save_and_sample_every_override
        if 'results_folder_postfix_override' in cfg.model:
            cfg.model.results_folder_postfix = cfg.model.results_folder_postfix_override
        if 'tensorboard_dir_override' in cfg.model:
            cfg.model.tensorboard_dir = cfg.model.tensorboard_dir_override

        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix)

        os.makedirs(cfg.model.results_folder, exist_ok=True)
        config_save_path = os.path.join(cfg.model.results_folder, 'config.yaml')
        with open(config_save_path, 'w') as f:
            OmegaConf.save(cfg, f)
        print(f"Configuration saved to: {config_save_path}")

        if cfg.model.diffusion_type != "categorical":
            raise ValueError("This package supports categorical diffusion only")

        down_factor = max(1, cfg.dataset.get('pixel_unshuffle_factor', 1))
        image_channels = cfg.dataset.get('image_channels', 1)
        mask_classes = cfg.dataset.get('mask_classes', 0)

        if mask_classes <= 0:
            raise ValueError("cfg.dataset.mask_classes must be a positive integer")

        spatial_dims = cfg.get('spatial_dims', 3)
        channels_per_voxel = down_factor ** spatial_dims
        cond_channels = image_channels * channels_per_voxel
        cfg.model.cond_channels = cond_channels

        categorical_use_background = bool(cfg.dataset.get('categorical_use_background', True))
        cfg.dataset.categorical_use_background = categorical_use_background
        cfg.model.categorical_final_step_mode = cfg.model.get('categorical_final_step_mode', 'majority')
        categorical_foreground_classes = int(cfg.dataset.get('num_foreground_classes', mask_classes))
        categorical_num_classes = categorical_foreground_classes + (1 if categorical_use_background else 0)
        cfg.dataset.num_foreground_classes = categorical_foreground_classes
        cfg.model.categorical_num_classes = categorical_num_classes
        cfg.model.categorical_use_background = categorical_use_background
        cfg.model.diffusion_num_channels = categorical_num_classes
        cfg.model.total_input_channels = categorical_num_classes + cond_channels
        cfg.model.out_dim = categorical_num_classes
        cfg.model.objective = 'categorical'

    device = require_cuda_device(cfg.model.gpus)
    print(f"Using CUDA device: {cfg.model.gpus}")

    # Initialize model and diffusion process
    model = create_denoising_model(cfg, cfg.dataset).to(device)
    print(f"Using denoising backbone: {cfg.model.denoising_fn}")

    aux_loss_type = cfg.model.get('categorical_aux_loss_type', 'none')
    aux_loss_weight = cfg.model.get('categorical_aux_loss_weight', 0.0)
    aux_dice_weight = cfg.model.get('categorical_aux_dice_weight', 1.0)
    aux_ce_weight = cfg.model.get('categorical_aux_ce_weight', 1.0)
    aux_boundary_weight = cfg.model.get('categorical_aux_boundary_weight', 0.1)
    kl_loss_weight = cfg.model.get('categorical_kl_loss_weight', 1.0)
    condition_mask_ratio = cfg.model.get('cfg_condition_mask_ratio', 0.1)
    sample_mode = cfg.model.get('sample_mode', 'one_hot')
    train_stochastic = cfg.model.get('train_stochastic', True)
    inference_stochastic = cfg.model.get('inference_stochastic', False)
    noise_std = cfg.model.get('noise_std', 0.0)

    diffusion = CategoricalDiffusion(
        model,
        image_size=cfg.dataset.diffusion_img_size,
        num_frames=cfg.dataset.diffusion_depth_size,
        num_classes=cfg.model.diffusion_num_channels,
        cond_channels=cfg.model.cond_channels,
        timesteps=cfg.model.timesteps,
        objective=cfg.model.get('objective', 'categorical'),
        aux_loss_type=aux_loss_type,
        aux_loss_weight=aux_loss_weight,
        aux_dice_weight=aux_dice_weight,
        aux_ce_weight=aux_ce_weight,
        aux_boundary_weight=aux_boundary_weight,
        kl_loss_weight=kl_loss_weight,
        final_step_mode=cfg.model.categorical_final_step_mode,
        condition_mask_ratio=condition_mask_ratio,
        sample_mode=sample_mode,
        train_stochastic=train_stochastic,
        inference_stochastic=inference_stochastic,
        noise_std=noise_std,
    ).to(device)

    # Load dataset
    dataset = importlib.import_module(f'dataset.{cfg.dataset.name}_dataloader')
    train_dataloader = dataset.get_loader(cfg.dataset, mode='train', stage_two=False)
    val_root_override = cfg.dataset.get('val_root_dir')
    val_dataloader = None
    if val_root_override:
        val_dataloader = dataset.get_loader(
            cfg.dataset, mode='val', root_override=val_root_override
        )

    _sample_for_check = next(iter(train_dataloader))
    assert_channel_alignment(cfg, model, _sample_for_check, role='train')

    # Visualization callback
    vis_slices_cfg = cfg.model.get('vis_slices')
    vis_slices = None
    if vis_slices_cfg is not None:
        if isinstance(vis_slices_cfg, (list, tuple)):
            vis_slices = list(vis_slices_cfg)
        else:
            try:
                vis_slices = list(vis_slices_cfg)
            except TypeError:
                vis_slices = [int(vis_slices_cfg)]

    viz_callback = SegmentationVisualizationCallback(
        save_root=cfg.model.results_folder,
        slices=vis_slices,
        num_slices=cfg.model.get('vis_num_slices', 3),
        max_samples=cfg.model.get('vis_max_samples', 2),
        overlay_alpha=cfg.model.get('vis_overlay_alpha', 0.45),
        sampler=cfg.model.get('sampler', 'categorical'),
        sampling_steps=cfg.model.get('sampling_steps'),
        final_step_mode=cfg.model.get('categorical_final_step_mode', 'majority'),
    )

    # Create trainer
    trainer = Trainer(
        diffusion,
        cfg=cfg,
        dataset=train_dataloader,
        validation_loader=val_dataloader,
        save_and_sample_every=cfg.model.save_and_sample_every,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        results_folder=cfg.model.results_folder,
        lr_decay=True,
        test_callback=viz_callback,
        log_dir=cfg.model.get('tensorboard_dir'),
        max_grad_norm=cfg.model.get('max_grad_norm', 2.0),
        cfg_enabled=True,
        p_uncond=condition_mask_ratio,
    )

    if cfg.model.load_milestone:
        trainer.load(cfg.model.load_milestone, map_location='cpu', strict=False)
        if cfg.model.get('reset_scheduler_on_load', False) and trainer.scheduler is not None:
            trainer.scheduler.total_steps = cfg.model.train_num_steps
            trainer.scheduler.last_step = trainer.step
            lrs = trainer.scheduler.get_lr()
            for param_group, lr in zip(trainer.opt.param_groups, lrs):
                param_group['lr'] = lr
            print(
                "Reset scheduler after checkpoint load: "
                f"step={trainer.step}, total_steps={trainer.scheduler.total_steps}, "
                f"lr={lrs[0]:.2e}"
            )

    trainer.train()


if __name__ == '__main__':
    run()
