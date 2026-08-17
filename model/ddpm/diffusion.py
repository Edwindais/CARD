"""Training utilities for the categorical cardiac diffusion model."""

import copy
import gc
import math
from pathlib import Path

import numpy as np
import torch
from monai.metrics import HausdorffDistanceMetric
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from utils.metrics_seg import DiceMetric as SegDiceMetric


def exists(value):
    return value is not None


def noop(*args, **kwargs):
    pass


def cycle(loader):
    while True:
        yield from loader


class EMA:
    def __init__(self, beta):
        self.beta = beta

    def update_model_average(self, moving_average_model, current_model):
        for current_params, average_params in zip(
            current_model.parameters(), moving_average_model.parameters()
        ):
            average_params.data = self.update_average(
                average_params.data, current_params.data
            )

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        cfg,
        dataset=None,
        *,
        ema_decay=0.995,
        train_lr=1e-4,
        train_num_steps=100000,
        gradient_accumulate_every=1,
        amp=False,
        step_start_ema=200,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder='./results',
        max_grad_norm=2.0,
        is_distributed=False,
        rank=0,
        lr_decay=True,
        warmup_steps=1000,
        min_lr=None,
        test_callback=None,
        cfg_enabled=False,
        p_uncond=0.1,
        validation_loader=None,
        log_dir=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        if save_and_sample_every is None or save_and_sample_every <= 0:
            raise ValueError("save_and_sample_every must be a positive integer")
        self.save_and_sample_every = save_and_sample_every

        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.cfg = cfg
        self.diffusion_type = getattr(cfg.model, 'diffusion_type', 'categorical').lower()
        if self.diffusion_type != 'categorical':
            raise ValueError("This release supports only categorical diffusion training")
        self.categorical_use_background = bool(getattr(cfg.dataset, 'categorical_use_background', True))
        self.categorical_ignore_index = int(getattr(cfg.model, 'categorical_ignore_index', -1))
        self.dl = cycle(dataset)
        self.val_dl = validation_loader

        # Distributed setup
        self.is_distributed = is_distributed
        self.rank = rank
        self.lr_decay = lr_decay

        # Metric helpers
        self.num_classes = int(getattr(cfg.dataset, 'mask_classes', 0))
        self.metric_num_classes = max(1, self.num_classes + 1)
        self.dice_metric = SegDiceMetric(self.metric_num_classes)
        self.best_val_dice = -float('inf')
        self.last_val_metrics = None
        self.eval_sampler = str(getattr(cfg.model, 'sampler', 'categorical')).lower()
        self.eval_sampling_steps = getattr(cfg.model, 'sampling_steps', None)
        self.eval_final_step_mode = getattr(cfg.model, 'categorical_final_step_mode', 'majority')
        self.max_val_batches = getattr(cfg.model, 'max_val_batches', None)  # None means use all batches
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr,
                        weight_decay=1e-5, betas=(0.9, 0.99))

        # Initialize learning rate scheduler with warm-up
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr if min_lr is not None else train_lr * 0.1

        if lr_decay:
            self.scheduler = WarmupCosineAnnealingLR(
                self.opt,
                warmup_steps=warmup_steps,
                total_steps=train_num_steps,
                min_lr=self.min_lr
            )
        else:
            self.scheduler = None

        self.step = 0
        self.initial_lr = train_lr
        self.train_num_steps = train_num_steps

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.max_grad_norm = max_grad_norm

        if results_folder is None:
            results_folder = './results'
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        log_dir_path = Path(log_dir) if log_dir is not None else self.results_folder / 'tensorboard'
        log_dir_path.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir_path
        if not self.is_distributed or self.rank == 0:
            self.writer = SummaryWriter(log_dir=str(log_dir_path))
        else:
            self.writer = None

        # Add test callback
        self.test_callback = test_callback

        self.cfg_enabled = cfg_enabled
        self.p_uncond = p_uncond
        self.best_checkpoint_path = None
        self.best_val_step = None

        # Print scheduler configuration
        if not self.is_distributed or (self.is_distributed and self.rank == 0):
            if self.scheduler is not None:
                if warmup_steps > 0:
                    print(f"Initialized training with warm-up: {warmup_steps} steps, target lr: {train_lr:.2e}, min lr: {self.min_lr:.2e}")
                else:
                    print(f"Initialized training with cosine annealing (no warmup): target lr: {train_lr:.2e}, min lr: {self.min_lr:.2e}")
            else:
                print(f"Initialized training with constant lr: {train_lr:.2e}")

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone, is_best_dice=False):
        # Only save on rank 0 in distributed mode
        if self.is_distributed and self.rank != 0:
            return

        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None
        }

        # Save latest checkpoint
        latest_path = str(self.results_folder / f'model-{milestone}.pt')
        torch.save(data, latest_path)

        # Save best Dice checkpoint
        if is_best_dice:
            best_dice_path = str(self.results_folder / 'model-best-dice.pt')
            torch.save(data, best_dice_path)
            if not self.is_distributed or self.rank == 0:
                print(f"Saved best Dice checkpoint: {best_dice_path}")

    def load(self, milestone, map_location=None, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1])
                              for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(
                all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        if map_location:
            data = torch.load(milestone, map_location=map_location)
        else:
            data = torch.load(milestone)

        if not self.is_distributed:
            data['model'] = {k.replace('module.', ''): v for k, v in data['model'].items()}
            data['ema'] = {k.replace('module.', ''): v for k, v in data['ema'].items()}
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.step = data['step']
        self.scaler.load_state_dict(data['scaler'])
        if self.scheduler is not None and data.get('scheduler') is not None:
            self.scheduler.load_state_dict(data['scheduler'])

    def _prepare_callback_batch(self, image, label, data, prediction=None):
        callback_batch = {
            'image': image.detach().cpu().clone(),
            'segmentation': label.detach().cpu().clone()
        }

        if prediction is not None:
            callback_batch['prediction'] = prediction.detach().cpu().clone()

        if 'name' in data:
            callback_batch['name'] = list(data['name'])

        if 'affine' in data and data['affine'] is not None:
            callback_batch['affine'] = data['affine'].detach().cpu().clone()

        return callback_batch

    def _forward_diffusion(self, image, mask_sdf, label, categorical_mask, **kwargs):
        if categorical_mask is None:
            raise ValueError("categorical_mask must be provided for categorical diffusion")
        return self.model(
            image,
            mask_sdf,
            categorical_mask=categorical_mask,
            label=label,
            current_step=self.step,
            **kwargs,
        )

    def _invoke_test_callback(self, milestone, callback_batch, reason):
        if (self.test_callback is None or callback_batch is None or
                (self.is_distributed and self.rank != 0)):
            return

        self.test_callback.on_checkpoint(
            ema_model=self.ema_model,
            batch=callback_batch,
            milestone=milestone,
            results_folder=str(self.results_folder),
            reason=reason
        )

    def _log_train_metrics(self, loss_value: float, lr_value: float) -> None:
        if self.writer is None:
            return
        self.writer.add_scalar('train/loss', loss_value, self.step)
        self.writer.add_scalar('train/lr', lr_value, self.step)

    def _log_validation_visuals(self, callback_batch, global_step: int) -> None:
        if self.writer is None or callback_batch is None:
            return

        image = callback_batch['image'][0, 0].float()

        def _normalize_slice(slice_tensor: torch.Tensor) -> torch.Tensor:
            slice_min = slice_tensor.min()
            slice_max = slice_tensor.max()
            denom = (slice_max - slice_min).clamp(min=1e-6)
            return ((slice_tensor - slice_min) / denom).unsqueeze(0)

        # Handle 2D (H, W) and 3D (D, H, W) images
        if image.ndim == 2:
            # 2D mode: image is already (H, W)
            image_slice = _normalize_slice(image)
            slice_idx = 0
        else:
            # 3D mode: image is (D, H, W)
            depth = int(image.shape[0])
            slice_idx = max(0, depth // 2)
            image_slice = _normalize_slice(image[slice_idx])

        self.writer.add_image('val/image', image_slice, global_step)

        denom = max(1, self.num_classes)
        seg = callback_batch['segmentation'][0, 0].float()
        if seg.ndim == 2:
            # 2D mode
            gt_slice = seg.unsqueeze(0) / denom
        else:
            # 3D mode
            gt_slice = seg[slice_idx].unsqueeze(0) / denom
        self.writer.add_image('val/gt', gt_slice, global_step)

        pred = callback_batch.get('prediction')
        if pred is not None:
            pred_tensor = pred[0, 0].float()
            if pred_tensor.ndim == 2:
                pred_slice = pred_tensor.unsqueeze(0) / denom
            else:
                pred_slice = pred_tensor[slice_idx].unsqueeze(0) / denom
            self.writer.add_image('val/pred', pred_slice, global_step)

    def _log_validation_metrics(self, metrics, global_step: int) -> None:
        """Log validation metrics to TensorBoard."""
        if self.writer is None or (self.is_distributed and self.rank != 0):
            return

        for key, value in metrics.items():
            if key not in ('callback_batch', 'ckpt_batch') and value is not None:
                if not (isinstance(value, float) and math.isnan(value)):
                    self.writer.add_scalar(f'val/{key}', value, global_step)

        self._log_validation_visuals(metrics.get('callback_batch'), global_step)

    def _compute_dice(self, pred_seg: torch.Tensor, gt_seg: torch.Tensor):
        if pred_seg is None or gt_seg is None:
            return None

        # Handle both 2D (4D tensor) and 3D (5D tensor) inputs
        if pred_seg.ndim == 4:
            # 2D mode: (B, 1, H, W) -> add dummy depth dim for dice metric
            pred_perm = pred_seg.unsqueeze(-1)  # (B, 1, H, W, 1)
            gt_perm = gt_seg.unsqueeze(-1)
        else:
            # 3D mode: (B, 1, D, H, W) -> permute to (B, 1, H, W, D)
            pred_perm = pred_seg.permute(0, 1, 3, 4, 2)
            gt_perm = gt_seg.permute(0, 1, 3, 4, 2)

        dice_scores = self.dice_metric(pred_perm, gt_perm)
        if dice_scores.numel() == 0:
            return None
        # Exclude background channel (index 0) when possible
        if dice_scores.size(1) > 1:
            dice_scores = dice_scores[:, 1:]
        dice_value = torch.nanmean(dice_scores)
        if torch.isnan(dice_value):
            return None
        return float(dice_value.item())

    def _compute_hd95(self, pred_seg: torch.Tensor, gt_seg: torch.Tensor):
        if pred_seg is None or gt_seg is None:
            return None

        num_classes = self.metric_num_classes
        if torch.any(pred_seg < 0) or torch.any(pred_seg >= num_classes):
            values = pred_seg.unique().detach().cpu().tolist()
            raise RuntimeError(
                f"Class values in predictions must be in [0, {num_classes - 1}], got {values}"
            )

        ignore_index = getattr(self, 'categorical_ignore_index', -1)
        valid_gt = gt_seg != ignore_index
        if torch.any(valid_gt & ((gt_seg < 0) | (gt_seg >= num_classes))):
            values = gt_seg.unique().detach().cpu().tolist()
            raise RuntimeError(
                f"Class values in ground truth must be in [0, {num_classes - 1}] "
                f"or equal ignore_index={ignore_index}, got {values}"
            )
        gt_seg = gt_seg.masked_fill(~valid_gt, 0)

        # Handle both 2D (4D tensor) and 3D (5D tensor) inputs
        if pred_seg.ndim == 4:
            # 2D mode: (B, 1, H, W) -> squeeze -> (B, H, W) -> one_hot -> (B, H, W, K) -> permute -> (B, K, H, W)
            pred_one_hot = F.one_hot(pred_seg.squeeze(1).long(), num_classes=num_classes)
            pred_one_hot = pred_one_hot.permute(0, 3, 1, 2).float()  # (B, K, H, W)
            gt_one_hot = F.one_hot(gt_seg.squeeze(1).long(), num_classes=num_classes)
            gt_one_hot = gt_one_hot.permute(0, 3, 1, 2).float()
            # Add dummy depth dimension for MONAI metric
            pred_one_hot = pred_one_hot.unsqueeze(-1)  # (B, K, H, W, 1)
            gt_one_hot = gt_one_hot.unsqueeze(-1)
        else:
            # 3D mode: (B, 1, D, H, W)
            pred_one_hot = F.one_hot(pred_seg.squeeze(1).long(), num_classes=num_classes)
            pred_one_hot = pred_one_hot.permute(0, 4, 1, 2, 3).float()
            gt_one_hot = F.one_hot(gt_seg.squeeze(1).long(), num_classes=num_classes)
            gt_one_hot = gt_one_hot.permute(0, 4, 1, 2, 3).float()

        metric = HausdorffDistanceMetric(
            include_background=False,
            percentile=95.0,
            reduction="none"
        )
        values = metric(y_pred=pred_one_hot, y=gt_one_hot)
        metric.reset()
        values = values.reshape(-1)
        valid = ~torch.isnan(values)
        if valid.any():
            return float(values[valid].mean().item())
        return None

    def _evaluate_validation(self):
        if self.val_dl is None or (self.is_distributed and self.rank != 0):
            return None

        max_val_batches = getattr(self, 'max_val_batches', None)

        if not self.is_distributed or self.rank == 0:
            scope = max_val_batches if max_val_batches is not None else "all"
            print(f"Validation step {self.step}: sampler={self.eval_sampler}, batches={scope}")

        device = next(self.model.parameters()).device
        was_training = self.model.training
        was_ema_training = self.ema_model.training
        self.model.eval()
        self.ema_model.eval()

        total_loss = 0.0
        total_batches = 0
        dice_values: list[float] = []
        hd95_values: list[float] = []
        callback_batch = None
        ckpt_batch = None

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dl):
                if max_val_batches is not None and batch_idx >= max_val_batches:
                    break

                image = batch['image'].to(device, non_blocking=True)
                mask_sdf = batch.get('mask_sdf')
                if mask_sdf is not None:
                    mask_sdf = mask_sdf.to(device, non_blocking=True)
                label = batch['segmentation'].to(device, non_blocking=True)
                categorical_mask = batch.get('categorical_mask')
                if categorical_mask is not None:
                    categorical_mask = categorical_mask.to(device, non_blocking=True)
                else:
                    raise ValueError('categorical_mask is required for categorical diffusion validation')

                # Handle 2D mode: squeeze depth dimension if depth=1
                is_2d_mode = image.ndim == 5 and image.shape[2] == 1
                if is_2d_mode:
                    image = image.squeeze(2)
                    if mask_sdf is not None:
                        mask_sdf = mask_sdf.squeeze(2)
                    label = label.squeeze(2)
                    if categorical_mask is not None:
                        categorical_mask = categorical_mask.squeeze(2)

                val_loss = self._forward_diffusion(
                    image,
                    mask_sdf,
                    label,
                    categorical_mask,
                    prob_focus_present=0.,
                    focus_present_mask=None,
                    null_cond_prob=self.p_uncond if self.cfg_enabled else 0.0,
                )

                total_loss += val_loss.item()
                total_batches += 1

                preds_seg = self.ema_model.sample(
                    batch_size=image.shape[0],
                    proc=False,
                    channel_cond=image,
                    sampler=self.eval_sampler,
                    num_steps=self.eval_sampling_steps,
                    final_step_mode=self.eval_final_step_mode,
                )
                dice_val = self._compute_dice(preds_seg, label)
                if dice_val is not None and not math.isnan(dice_val):
                    dice_values.append(dice_val)

                hd95_val = self._compute_hd95(preds_seg, label)
                if hd95_val is not None and not math.isnan(hd95_val):
                    hd95_values.append(hd95_val)

                if callback_batch is None:
                    callback_batch = self._prepare_callback_batch(
                        image, label, batch, prediction=preds_seg
                    )
                    ckpt_mask = categorical_mask.detach().clone()
                    ckpt_batch = (
                        image.detach().clone(),
                        ckpt_mask,
                        label.detach().clone()
                    )

        if was_training:
            self.model.train()
        if was_ema_training:
            self.ema_model.train()

        if total_batches == 0:
            return None

        avg_loss = total_loss / total_batches
        mean_dice = float(np.mean(dice_values)) if dice_values else float('nan')
        mean_hd95 = float(np.mean(hd95_values)) if hd95_values else float('nan')

        if not self.is_distributed or self.rank == 0:
            eval_coverage = f"({total_batches}/{total_batches} batches)" if max_val_batches is None else f"({total_batches} batch(es), limited)"
            print(f"\nValidation complete {eval_coverage}: loss={avg_loss:.4f}, dice={mean_dice:.4f}, hd95={mean_hd95:.2f}\n")

        return {
            'loss': avg_loss,
            'dice': mean_dice,
            'hd95': mean_hd95,
            'callback_batch': callback_batch,
            'ckpt_batch': ckpt_batch,
        }

    def _handle_validation_results(self, metrics, global_step: int) -> None:
        if metrics is None:
            return

        self.last_val_metrics = metrics
        self._log_validation_metrics(metrics, global_step)

        dice_value = metrics.get('dice')
        # Track best Dice
        if dice_value is not None and not math.isnan(dice_value):
            if dice_value > self.best_val_dice:
                self.best_val_dice = dice_value
                self.best_val_step = global_step
                if self.writer is not None:
                    self.writer.add_scalar('val/best_dice', dice_value, global_step)
                if not self.is_distributed or self.rank == 0:
                    print(f'Validation Dice improved to {dice_value:.4f} at step {global_step}')
                self.save('best', is_best_dice=True)
                self.best_checkpoint_path = str(self.results_folder / 'model-best.pt')
                if not self.is_distributed or self.rank == 0:
                    print(f'Saved new best Dice checkpoint to {self.best_checkpoint_path}')

    def _run_checkpoint(self, milestone, callback_batch, image, mask, label, reason):
        self.save(milestone)

        self._invoke_test_callback(milestone, callback_batch, reason)

        self.model.test(
            image,
            mask,
            label=label,
            milestone=milestone,
        )
        gc.collect()
        torch.cuda.empty_cache()

    def train(
        self,
        prob_focus_present=0.,
        focus_present_mask=None,
        log_fn=noop
    ):
        assert callable(log_fn)
        device = next(self.model.parameters()).device

        while self.step < self.train_num_steps:
            # Get current learning rate from scheduler or use initial lr
            if self.scheduler is not None:
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.initial_lr

            for i in range(self.gradient_accumulate_every):
                data = next(self.dl)
                image = data['image'].to(device, non_blocking=True)
                mask_sdf = data.get('mask_sdf')
                if mask_sdf is not None:
                    mask_sdf = mask_sdf.to(device, non_blocking=True)

                label = data['segmentation'].to(device, non_blocking=True)
                categorical_mask = data.get('categorical_mask')
                if categorical_mask is not None:
                    categorical_mask = categorical_mask.to(device, non_blocking=True)
                else:
                    raise ValueError('categorical_mask is required for categorical diffusion training')

                # Handle 2D mode: squeeze depth dimension if depth=1
                is_2d_mode = image.ndim == 5 and image.shape[2] == 1
                if is_2d_mode:
                    image = image.squeeze(2)  # (B, C, 1, H, W) -> (B, C, H, W)
                    if mask_sdf is not None:
                        mask_sdf = mask_sdf.squeeze(2)
                    label = label.squeeze(2)
                    if categorical_mask is not None:
                        categorical_mask = categorical_mask.squeeze(2)

                with autocast(enabled=self.amp):
                    loss = self._forward_diffusion(
                        image,
                        mask_sdf,
                        label,
                        categorical_mask,
                        prob_focus_present=prob_focus_present,
                        focus_present_mask=focus_present_mask,
                        null_cond_prob=self.p_uncond if self.cfg_enabled else 0.0,
                    )

                    self.scaler.scale(
                        loss / self.gradient_accumulate_every).backward()

                # Only print from rank 0 when in distributed mode
                if i == self.gradient_accumulate_every - 1 and self.step % 20 == 0:
                    if not self.is_distributed or (self.is_distributed and self.rank == 0):
                        # Add warm-up phase indicator
                        warmup_info = ""
                        if self.scheduler is not None and hasattr(self.scheduler, 'warmup_steps'):
                            if self.step < self.scheduler.warmup_steps:
                                warmup_progress = self.step / self.scheduler.warmup_steps * 100
                                warmup_info = f", warmup: {warmup_progress:.1f}%"

                        print(f'{self.step}: {loss.item():.4f}, lr: {current_lr:.2e}{warmup_info}, config: {str(self.results_folder).split("/")[-1]}')

            log = {'loss': loss.item(), 'lr': current_lr}
            callback_batch = self._prepare_callback_batch(image, label, data)

            # Add warm-up information to log
            if self.scheduler is not None and hasattr(self.scheduler, 'warmup_steps'):
                log['warmup_phase'] = self.step < self.scheduler.warmup_steps
                if log['warmup_phase']:
                    log['warmup_progress'] = self.step / self.scheduler.warmup_steps

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # Update learning rate using cosine annealing scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            if (self.step + 1) % self.save_and_sample_every == 0:
                completed_steps = self.step + 1
                step_milestone = completed_steps // self.save_and_sample_every

                val_metrics = None
                checkpoint_callback = callback_batch
                checkpoint_image = image
                checkpoint_mask = categorical_mask
                checkpoint_label = label

                if self.val_dl is not None and (not self.is_distributed or self.rank == 0):
                    val_metrics = self._evaluate_validation()
                    self._handle_validation_results(val_metrics, completed_steps)
                    if val_metrics is not None:
                        cb = val_metrics.get('callback_batch')
                        if cb is not None:
                            checkpoint_callback = cb
                        ckpt_batch = val_metrics.get('ckpt_batch')
                        if ckpt_batch is not None:
                            checkpoint_image, checkpoint_mask, checkpoint_label = ckpt_batch
                        log['val_loss'] = val_metrics.get('loss')
                        log['val_dice'] = val_metrics.get('dice')
                        log['val_hd95'] = val_metrics.get('hd95')
                        if self.best_val_step is not None:
                            log['best_dice'] = self.best_val_dice
                            log['best_checkpoint_step'] = self.best_val_step
                            log['best_checkpoint_path'] = self.best_checkpoint_path

                self._run_checkpoint(
                    milestone=step_milestone,
                    callback_batch=checkpoint_callback,
                    image=checkpoint_image,
                    mask=checkpoint_mask,
                    label=checkpoint_label,
                    reason='step'
                )

                if self.best_val_step is not None:
                    log.setdefault('best_dice', self.best_val_dice)
                    log.setdefault('best_checkpoint_step', self.best_val_step)
                    log.setdefault('best_checkpoint_path', self.best_checkpoint_path)

            log_fn(log)
            self._log_train_metrics(log['loss'], log['lr'])
            self.step += 1

        self.save('final')
        print(f'training completed: {self.results_folder / "model-final.pt"}')
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

# warm-up learning rate scheduler

class WarmupCosineAnnealingLR:
    """
    Learning rate scheduler with linear warm-up followed by cosine annealing.
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0, last_step=-1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.last_step = last_step

        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]

        # Initialize scheduler
        self.step()

    def get_lr(self):
        if self.warmup_steps > 0 and self.last_step < self.warmup_steps:
            # Linear warm-up
            warmup_factor = (self.last_step + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            # When warmup_steps = 0, use the full range for cosine annealing
            if self.warmup_steps == 0:
                progress = self.last_step / self.total_steps
            else:
                progress = (self.last_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)  # Clamp to [0, 1]

            lrs = []
            for base_lr in self.base_lrs:
                lr = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                lrs.append(lr)
            return lrs

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def step(self):
        self.last_step += 1
        lrs = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr

    def state_dict(self):
        return {
            'last_step': self.last_step,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr': self.min_lr,
            'base_lrs': self.base_lrs
        }

    def load_state_dict(self, state_dict):
        self.last_step = state_dict['last_step']
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.min_lr = state_dict['min_lr']
        self.base_lrs = state_dict['base_lrs']
