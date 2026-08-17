import logging
import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical


logger = logging.getLogger(__name__)

EPS = 1e-12  # Avoid denormals; use consistent epsilon across operations


def exists(x):
    return x is not None


class OneHotCategoricalBCHW(OneHotCategorical):
    """OneHot categorical distribution working with channel-first tensors."""

    def __init__(
        self,
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        if probs is not None and probs.ndim < 2:
            raise ValueError("`probs` must have at least 2 dimensions (B, C, ...)")
        if logits is not None and logits.ndim < 2:
            raise ValueError("`logits` must have at least 2 dimensions (B, C, ...)")

        probs_last = self._channels_last(probs) if probs is not None else None
        logits_last = self._channels_last(logits) if logits is not None else None
        super().__init__(probs=probs_last, logits=logits_last, validate_args=False)

    @property
    def probs(self) -> torch.Tensor:
        """Return probabilities in channel-first format (B, C, ...)."""
        return self._channels_second(super().probs)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        sample = super().sample(sample_shape)
        return self._channels_second(sample)

    def max_prob_sample(self) -> torch.Tensor:
        """Sample deterministically by selecting the most probable class per voxel."""
        probs = self.probs  # (B, C, D, H, W)
        # argmax over class dimension (dim=1) to get (B, D, H, W)
        max_indices = probs.argmax(dim=1)
        # Convert to one-hot: (B, D, H, W) → (B, D, H, W, C)
        one_hot = F.one_hot(max_indices, num_classes=probs.shape[1])
        # Convert back to channel-first: (B, D, H, W, C) → (B, C, D, H, W)
        return self._channels_second(one_hot.float())

    def prob_sample(self) -> torch.Tensor:
        """Return probabilities in channel-first layout (self.probs is already channel-first)."""
        return self.probs

    @staticmethod
    def _channels_last(arr: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if arr is None:
            return None
        if arr.ndim == 2:
            return arr
        dims = list(range(arr.ndim))
        perm = [0] + dims[2:] + [1]
        return arr.permute(perm)

    @staticmethod
    def _channels_second(arr: torch.Tensor) -> torch.Tensor:
        if arr.ndim == 2:
            return arr
        dims = list(range(arr.ndim))
        perm = [0, arr.ndim - 1] + dims[1:-1]
        return arr.permute(perm)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0, 0.999)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    b = t.shape[0]
    out = a.gather(0, t)
    return out.reshape(b, *([1] * (len(x_shape) - 1)))


def js_divergence_map(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute per-voxel Jensen-Shannon divergence between two probability distributions.

    Returns the full spatial map used by dual-model calibration guidance.

    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M) where M = 0.5 * (P + Q)

    Args:
        p: First probability distribution (B, K, D, H, W) or (B, K, H, W)
           Must be valid probabilities summing to 1 along dim=1
        q: Second probability distribution, same shape as p
        eps: Small constant for numerical stability (avoid log(0))

    Returns:
        Per-voxel JS divergence map (B, 1, D, H, W) or (B, 1, H, W)
        Values are in [0, log(2)] ≈ [0, 0.693]
    """
    # Clamp to avoid numerical issues
    p = p.clamp(eps, 1 - eps)
    q = q.clamp(eps, 1 - eps)

    # Midpoint distribution
    m = 0.5 * (p + q)
    m = m.clamp(eps, 1 - eps)

    # KL divergences: KL(P || M) = sum_k P_k * log(P_k / M_k)
    kl_pm = (p * (p / m).log()).sum(dim=1, keepdim=True)  # (B, 1, ...)
    kl_qm = (q * (q / m).log()).sum(dim=1, keepdim=True)  # (B, 1, ...)

    # JS divergence per voxel
    js_map = 0.5 * (kl_pm + kl_qm)  # (B, 1, ...)

    return js_map.clamp_min(0)  # Ensure non-negative (numerical stability)


class CategoricalDiffusion(nn.Module):
    """Categorical cardiac segmentation diffusion with CARD calibration sampling."""

    def __init__(
        self,
        denoise_fn: nn.Module,
        *,
        image_size: int,
        num_frames: int,
        num_classes: int,
        cond_channels: int = 0,
        timesteps: int = 1000,
        objective: str = 'categorical',
        aux_loss_type: str = 'none',
        aux_loss_weight: float = 0.0,
        aux_dice_weight: float = 1.0,
        aux_ce_weight: float = 1.0,
        aux_boundary_weight: float = 0.1,
        kl_loss_weight: float = 1.0,
        final_step_mode: str = 'majority',
        condition_mask_ratio: float = 0.1,
        xt_mask_ratio: float = 0.0,
        sample_mode: str = 'one_hot',
        train_stochastic: bool = True,
        inference_stochastic: bool = False,
        noise_std: float = 0.0,
        inference_debug: bool = False,
    ) -> None:
        super().__init__()
        if objective != 'categorical':
            raise ValueError(f"Unsupported objective '{objective}' for categorical diffusion")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1")
        if aux_loss_type not in {'none', 'l1', 'l2', 'dice_ce'}:
            raise ValueError("aux_loss_type must be one of {'none', 'l1', 'l2', 'dice_ce'}")
        if kl_loss_weight < 0.0:
            raise ValueError('kl_loss_weight must be non-negative')
        if aux_dice_weight < 0.0 or aux_ce_weight < 0.0:
            raise ValueError('auxiliary Dice/CE weights must be non-negative')
        if aux_boundary_weight < 0.0:
            raise ValueError('aux_boundary_weight must be non-negative')
        if not 0.0 <= float(condition_mask_ratio) <= 1.0:
            raise ValueError('condition_mask_ratio must be between 0 and 1 inclusive')
        if not 0.0 <= float(xt_mask_ratio) <= 1.0:
            raise ValueError('xt_mask_ratio must be between 0 and 1 inclusive')

        self.denoise_fn = denoise_fn
        self.image_size = image_size
        self.num_frames = num_frames
        self.num_classes = int(num_classes)
        self.cond_channels = int(cond_channels)
        self.num_timesteps = int(timesteps)
        self.aux_loss_type = aux_loss_type
        self.aux_loss_weight = float(aux_loss_weight)
        self.aux_dice_weight = float(aux_dice_weight)
        self.aux_ce_weight = float(aux_ce_weight)
        self.aux_boundary_weight = float(aux_boundary_weight)
        self.kl_loss_weight = float(kl_loss_weight)
        self.eps = EPS
        self.condition_mask_ratio = float(condition_mask_ratio)
        self.xt_mask_ratio = float(xt_mask_ratio)

        # Unified sample mode: used for both training and inference intermediate steps
        # 'one_hot': One-hot representation (discrete categorical)
        # 'prob_flow': Probability flow (soft distribution)
        self.sample_mode = str(sample_mode)

        # Training sampling behavior (only applies when sample_mode='one_hot')
        # True: Use stochastic sampling (diverse, prevents mode collapse)
        # False: Use deterministic majority (stable, reproducible)
        self.train_stochastic = bool(train_stochastic)

        # Inference sampling behavior (only applies when sample_mode='one_hot')
        # True: Use stochastic sampling (diverse outputs, good for ensemble/uncertainty)
        # False: Use deterministic majority (stable, reproducible results)
        self.inference_stochastic = bool(inference_stochastic)

        self.noise_std = float(noise_std)
        self.inference_debug = bool(inference_debug)

        # Validate sample mode
        if self.sample_mode not in {'one_hot', 'prob_flow'}:
            raise ValueError(f"sample_mode must be 'one_hot' or 'prob_flow', got '{sample_mode}'")

        if self.noise_std < 0.0:
            raise ValueError(f"noise_std must be non-negative, got {noise_std}")

        betas = cosine_beta_schedule(self.num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas', alphas.float())
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float())

        # Learnable null embeddings for classifier-free guidance style training
        # 2D mode (num_frames=1): Use 4D tensors (1, K/C, H, W)
        # 3D mode (num_frames>1): Use 5D tensors (1, K/C, D, H, W)
        if num_frames == 1:
            # 2D mode: 4D tensors
            self.null_emb_xt = nn.Parameter(torch.zeros(1, num_classes, image_size, image_size))
            if cond_channels > 0:
                self.null_emb_cond = nn.Parameter(torch.zeros(1, cond_channels, image_size, image_size))
            else:
                self.register_parameter('null_emb_cond', None)
        else:
            # 3D mode: 5D tensors
            self.null_emb_xt = nn.Parameter(torch.zeros(1, num_classes, num_frames, image_size, image_size))
            if cond_channels > 0:
                self.null_emb_cond = nn.Parameter(torch.zeros(1, cond_channels, num_frames, image_size, image_size))
            else:
                self.register_parameter('null_emb_cond', None)

        self.final_step_mode = self._validate_final_step_mode(final_step_mode)

    @staticmethod
    def _validate_final_step_mode(mode: str) -> str:
        allowed = {'sample', 'majority', 'argmax', 'prob', 'confidence'}
        if mode not in allowed:
            raise ValueError(f"final_step_mode must be one of {sorted(allowed)}, got '{mode}'")
        return mode

    def _concat_with_cond(self, tensor: torch.Tensor, channel_cond: Optional[torch.Tensor]) -> torch.Tensor:
        if self.cond_channels == 0:
            return tensor
        if channel_cond is None:
            raise ValueError("channel_cond must be provided when cond_channels > 0")
        return torch.cat((tensor, channel_cond), dim=1)

    def _call_denoise_fn(self, model_input: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Call denoise_fn with 2D mode handling.

        For 2D mode (num_frames=1), squeeze depth dimension before calling
        denoise_fn and unsqueeze after to maintain consistent tensor shapes.

        Supports cond_scale by using forward_with_cond_scale if available.
        """
        # Check if 2D mode: num_frames=1 and input is 4D (already squeezed)
        # or 5D with depth=1 (needs squeeze)
        is_2d_mode = self.num_frames == 1
        needs_squeeze = is_2d_mode and model_input.ndim == 5 and model_input.shape[2] == 1

        if needs_squeeze:
            model_input = model_input.squeeze(2)  # (B, C, 1, H, W) -> (B, C, H, W)

        # Check if cond_scale is provided and use appropriate forward method
        cond_scale = kwargs.pop('cond_scale', 1.0)
        if hasattr(self.denoise_fn, 'forward_with_cond_scale') and cond_scale != 1.0:
            pred_logits = self.denoise_fn.forward_with_cond_scale(model_input, t, cond_scale=cond_scale, **kwargs)
        else:
            pred_logits = self.denoise_fn(model_input, t, **kwargs)

        # Restore depth dimension for 2D mode if needed (some functions expect 5D output)
        if needs_squeeze and pred_logits.ndim == 4:
            pred_logits = pred_logits.unsqueeze(2)  # (B, C, H, W) -> (B, C, 1, H, W)

        return pred_logits

    def _call_denoise_fn_external(
        self,
        external_model: nn.Module,
        model_input: torch.Tensor,
        t: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Call an external denoise_fn with 2D mode handling.

        Same logic as _call_denoise_fn but for a different model (e.g., d1 model
        for dual-model calibration guidance).
        """
        is_2d_mode = self.num_frames == 1
        needs_squeeze = is_2d_mode and model_input.ndim == 5 and model_input.shape[2] == 1

        if needs_squeeze:
            model_input = model_input.squeeze(2)

        cond_scale = kwargs.pop('cond_scale', 1.0)
        if hasattr(external_model, 'forward_with_cond_scale') and cond_scale != 1.0:
            pred_logits = external_model.forward_with_cond_scale(model_input, t, cond_scale=cond_scale, **kwargs)
        else:
            pred_logits = external_model(model_input, t, **kwargs)

        if needs_squeeze and pred_logits.ndim == 4:
            pred_logits = pred_logits.unsqueeze(2)

        return pred_logits


    def _indices_to_one_hot(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert class indices to one-hot encoding.

        Supports both 2D mode and 3D mode:
        - 2D mode: (B, 1, H, W) or (B, H, W) -> (B, K, H, W)
        - 3D mode: (B, 1, D, H, W) or (B, D, H, W) -> (B, K, D, H, W)

        Args:
            indices: Class indices tensor

        Returns:
            One-hot encoded tensor
        """
        indices = indices.long()

        # Detect mode based on input dimensions
        # 2D mode: (B, 1, H, W) or (B, H, W) - 4D or 3D tensor
        # 3D mode: (B, 1, D, H, W) or (B, D, H, W) - 5D or 4D tensor
        is_2d_mode = self.num_frames == 1

        if is_2d_mode:
            # 2D mode: squeeze channel dim if present
            if indices.ndim == 4 and indices.shape[1] == 1:
                indices = indices.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
            # Clamp indices
            indices_clamped = indices.clamp(min=0, max=self.num_classes - 1)
            # Convert to one-hot: (B, H, W) -> (B, H, W, K) -> (B, K, H, W)
            one_hot = F.one_hot(indices_clamped, num_classes=self.num_classes)
            one_hot = one_hot.movedim(-1, 1).contiguous().float()
        else:
            # 3D mode: handle 4D or 5D input
            if indices.ndim == 4:
                indices = indices.unsqueeze(1)  # (B, D, H, W) -> (B, 1, D, H, W)
            # Clamp indices
            indices_squeezed = indices.squeeze(1)  # (B, D, H, W)
            indices_clamped = indices_squeezed.clamp(min=0, max=self.num_classes - 1)
            # Convert to one-hot: (B, D, H, W) -> (B, D, H, W, K) -> (B, K, D, H, W)
            one_hot = F.one_hot(indices_clamped, num_classes=self.num_classes)
            one_hot = one_hot.movedim(-1, 1).contiguous().float()

        return one_hot

    def _build_sampling_timesteps(self, steps: Optional[int], device: torch.device) -> torch.Tensor:
        total = int(self.num_timesteps)
        if steps is None or steps <= 0:
            steps = total
        steps = min(int(steps), total)
        if steps == total:
            return torch.arange(total - 1, -1, -1, device=device, dtype=torch.long)

        step_vals = torch.linspace(0, total - 1, steps=steps, device=device)
        step_vals = torch.unique(torch.round(step_vals).long())

        if step_vals.numel() == 0:
            step_vals = torch.tensor([0, total - 1], device=device, dtype=torch.long)
        if step_vals[0].item() != 0:
            step_vals = torch.cat((torch.tensor([0], device=device, dtype=torch.long), step_vals))
        if step_vals[-1].item() != total - 1:
            step_vals = torch.cat((step_vals[:-1], torch.tensor([total - 1], device=device, dtype=torch.long)))
        step_vals = torch.unique(step_vals)
        indices = torch.flip(step_vals, dims=(0,))
        if indices[-1].item() != 0:
            indices = torch.cat((indices, torch.zeros(1, device=device, dtype=torch.long)))
        return indices

    def _categorical_step(self, pred_x0: torch.Tensor, target_t: int, eta: float = 0.0, use_probabilistic_forward: bool = False, sample_mode: str = None, stochastic: bool = None) -> torch.Tensor:
        """
        Map a predicted clean categorical state to the requested noise level.

        Args:
            pred_x0: Predicted clean image (B, K, D, H, W)
            target_t: Target timestep
            eta: Stochasticity parameter (0.0 = deterministic, 1.0 = stochastic)
            use_probabilistic_forward: Legacy parameter for backward compatibility
            sample_mode: 'prob_flow' (deterministic) or 'one_hot' (stochastic). Uses self.sample_mode if None.
            stochastic: For one_hot mode: True=sample, False=majority. Uses self.inference_stochastic if None.

        Returns:
            xt at target_t
        """
        if target_t < 0:
            logger.debug(f"final categorical target {target_t}: returning predicted x0")
            return pred_x0

        # Use unified sample_mode if not specified
        if sample_mode is None:
            sample_mode = self.sample_mode

        # Use inference_stochastic if stochastic not explicitly specified (this is inference code)
        if stochastic is None:
            stochastic = self.inference_stochastic

        device = pred_x0.device
        batch = pred_x0.shape[0]
        t_tensor = torch.full((batch,), target_t, device=device, dtype=torch.long)

        # Compute base forward diffusion: q(x_t | x_0) = alpha_bar * x_0 + (1 - alpha_bar) * uniform
        alpha_bar = extract(self.alphas_cumprod, t_tensor, pred_x0.shape)
        uniform = torch.full_like(pred_x0, 1.0 / self.num_classes)
        xt_mean = alpha_bar * pred_x0 + (1.0 - alpha_bar) * uniform

        # Normalize to valid probability distribution
        xt_mean = xt_mean.clamp_min(self.eps)
        xt_mean = xt_mean / xt_mean.sum(dim=1, keepdim=True).clamp_min(self.eps)

        # Apply stochasticity based on sample_mode and stochastic flag
        if sample_mode == 'one_hot' or use_probabilistic_forward:
            dist = OneHotCategoricalBCHW(probs=xt_mean)
            if stochastic:
                # Stochastic: sample from categorical distribution
                xt = dist.sample()
                # Optionally combine with eta-based stochasticity
                if eta > 0.0:
                    xt_sample2 = dist.sample()
                    xt = (1.0 - eta) * xt + eta * xt_sample2
            else:
                # Deterministic: use majority vote
                xt = dist.max_prob_sample()
        elif sample_mode == 'prob_flow':
            # Deterministic: use probability flow
            if eta > 0.0:
                # Add eta-based noise for slight stochasticity
                dist = OneHotCategoricalBCHW(probs=xt_mean)
                xt_sample = dist.sample()
                xt = (1.0 - eta) * xt_mean + eta * xt_sample
            else:
                # Pure deterministic
                xt = xt_mean
        else:
            raise ValueError(f"Unknown sample_mode: {sample_mode}. Must be 'one_hot' or 'prob_flow'.")

        # Final normalization
        xt = xt.clamp_min(self.eps)
        xt = xt / xt.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return xt

    def _add_noise_and_normalize(self, probs: torch.Tensor, noise_std: float = None) -> torch.Tensor:
        """
        Add Gaussian noise to probability distribution and renormalize.

        Args:
            probs: Probability distribution (B, K, D, H, W)
            noise_std: Standard deviation of Gaussian noise (uses self.noise_std if None)

        Returns:
            Noised and renormalized probability distribution
        """
        if noise_std is None:
            noise_std = self.noise_std

        if noise_std <= 0.0:
            return probs

        # Add Gaussian noise
        noise = torch.randn_like(probs) * noise_std
        noised_probs = probs + noise

        # Clamp to non-negative
        noised_probs = noised_probs.clamp(min=self.eps)

        # Renormalize to sum to 1
        noised_probs = noised_probs / noised_probs.sum(dim=1, keepdim=True).clamp_min(self.eps)

        return noised_probs

    def q_xt_given_x0(self, x0: torch.Tensor, t: torch.Tensor, use_mean: bool = False, sample_mode: str = None, noise_std: float = None) -> OneHotCategoricalBCHW:
        """
        Forward diffusion process: q(x_t | x_0)

        Args:
            x0: Clean data (B, K, D, H, W)
            t: Timestep (B,)
            use_mean: Deprecated, kept for backward compatibility
            sample_mode: 'one_hot' or 'prob_flow'. If None, uses self.sample_mode
            noise_std: Noise standard deviation for augmentation. If None, uses self.noise_std

        Returns:
            Categorical distribution at timestep t
        """
        # Default to unified sample_mode if not specified
        if sample_mode is None:
            sample_mode = self.sample_mode
        if noise_std is None:
            noise_std = self.noise_std

        alpha_bar_t = extract(self.alphas_cumprod, t, x0.shape)
        probs = alpha_bar_t * x0 + (1.0 - alpha_bar_t) / self.num_classes

        if noise_std > 0.0:
            probs = self._add_noise_and_normalize(probs, noise_std)
        else:
            probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(self.eps)

        return OneHotCategoricalBCHW(probs=probs)

    def theta_post(self, xt: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha_t = extract(self.alphas, t, xt.shape)
        alpha_bar_tm1 = extract(self.alphas_cumprod_prev, t, xt.shape)
        term1 = alpha_t * xt + (1.0 - alpha_t) / self.num_classes
        term2 = alpha_bar_tm1 * x0 + (1.0 - alpha_bar_tm1) / self.num_classes
        theta = term1 * term2
        theta = theta / theta.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return theta

    def theta_post_prob(self, xt: torch.Tensor, theta_x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha_t = extract(self.alphas, t, xt.shape)
        alpha_bar_tm1 = extract(self.alphas_cumprod_prev, t, xt.shape)

        theta_xt_xtm1 = alpha_t * xt + (1.0 - alpha_t) / self.num_classes

        spatial_shape = xt.shape[2:]
        eye = torch.eye(self.num_classes, device=xt.device, dtype=xt.dtype)
        eye = eye.view(1, self.num_classes, self.num_classes, *([1] * len(spatial_shape)))

        alpha_bar_tm1_exp = alpha_bar_tm1.unsqueeze(1)
        theta_xtm1_x0 = alpha_bar_tm1_exp * eye + (1.0 - alpha_bar_tm1_exp) / self.num_classes

        aux = theta_xt_xtm1.unsqueeze(2) * theta_xtm1_x0
        aux_sum = aux.sum(dim=1, keepdim=True).clamp_min(self.eps)
        theta_xtm1_xtx0 = aux / aux_sum

        theta = (theta_xtm1_xtx0 * theta_x0.unsqueeze(1)).sum(dim=2)

        # Use log-space computation for numerical stability (avoids double normalization)
        log_theta = torch.log(theta.clamp_min(self.eps))
        log_theta = log_theta - torch.logsumexp(log_theta, dim=1, keepdim=True)
        theta = torch.exp(log_theta)

        return theta

    def _create_null_embeddings(
        self,
        xt: torch.Tensor,
        channel_cond: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Create learnable null embeddings for xt and channel_cond.

        Supports both 2D mode (4D tensors: B, K/C, H, W) and
        3D mode (5D tensors: B, K/C, D, H, W).

        Args:
            xt: Current categorical distribution - 4D (B, K, H, W) or 5D (B, K, D, H, W)
            channel_cond: Input image conditioning - 4D or 5D

        Returns:
            Tuple of (null_xt, null_channel_cond)
        """
        B = xt.shape[0]

        # 2D mode: 4D tensors (num_frames=1)
        if self.num_frames == 1:
            null_xt_logits = self.null_emb_xt.expand(B, -1, -1, -1)  # (B, K, H, W)
        else:
            # 3D mode: 5D tensors
            null_xt_logits = self.null_emb_xt.expand(B, -1, -1, -1, -1)  # (B, K, D, H, W)

        # Apply softmax to get valid probability distribution
        null_xt = F.softmax(null_xt_logits, dim=1)

        # Null embedding for channel_cond
        null_channel_cond = None
        if channel_cond is not None and self.null_emb_cond is not None:
            B_c = channel_cond.shape[0]
            if self.num_frames == 1:
                null_channel_cond = self.null_emb_cond.expand(B_c, -1, -1, -1)  # (B, C, H, W)
            else:
                null_channel_cond = self.null_emb_cond.expand(B_c, -1, -1, -1, -1)  # (B, C, D, H, W)

        return null_xt, null_channel_cond

    def _apply_dual_masking(
        self,
        xt: torch.Tensor,
        channel_cond: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply conditional masking to either xt OR channel_cond, but never both simultaneously.

        This implements classifier-free guidance style training where we randomly drop
        either the noisy state (xt) or the input image (channel_cond) to teach the model
        to denoise with and without each conditioning signal.

        Masking strategy:
        1. With probability xt_mask_ratio: mask xt (replace with uniform distribution)
        2. With probability condition_mask_ratio: mask channel_cond (replace with zeros)
        3. Never mask both at the same time (this would provide no information)
        4. At least one of {xt, channel_cond} is always provided

        Args:
            xt: Current noisy categorical distribution (B, K, D, H, W)
            channel_cond: Input image conditioning (B, C, D, H, W) or None

        Returns:
            Tuple of (masked_xt, masked_channel_cond)
        """
        # If both mask ratios are zero, return inputs as-is
        if self.xt_mask_ratio <= 0.0 and self.condition_mask_ratio <= 0.0:
            return xt, channel_cond

        if channel_cond is not None and channel_cond.ndim < 2:
            raise ValueError('channel_cond must have at least two dimensions (B, C, ...)')
        if xt.ndim < 2:
            raise ValueError('xt must have at least two dimensions (B, K, ...)')

        batch = xt.shape[0]
        device = xt.device

        # Create null embeddings
        null_xt, null_channel_cond = self._create_null_embeddings(xt, channel_cond)

        # Determine which samples get which masking
        # Total probability budget: xt_mask_ratio + condition_mask_ratio
        total_mask_prob = self.xt_mask_ratio + self.condition_mask_ratio

        if total_mask_prob <= 0.0:
            return xt, channel_cond

        # Generate random values for each sample in batch
        rand_vals = torch.rand((batch,), device=device)

        # Samples with rand < xt_mask_ratio → mask xt
        # Samples with xt_mask_ratio <= rand < total_mask_prob → mask channel_cond
        # Samples with rand >= total_mask_prob → mask nothing

        mask_xt = rand_vals < self.xt_mask_ratio
        mask_cond = (rand_vals >= self.xt_mask_ratio) & (rand_vals < total_mask_prob)

        # Apply masking to xt
        if mask_xt.any():
            mask_xt_view = mask_xt.view(batch, *([1] * (xt.ndim - 1)))
            xt = torch.where(mask_xt_view, null_xt, xt)

        # Apply masking to channel_cond
        if channel_cond is not None and mask_cond.any():
            mask_cond_view = mask_cond.view(batch, *([1] * (channel_cond.ndim - 1)))
            channel_cond = torch.where(mask_cond_view, null_channel_cond, channel_cond)

        return xt, channel_cond

    def _kl_div(self, true_theta: torch.Tensor, pred_theta: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between true and predicted posteriors."""
        kl = true_theta * (torch.log(true_theta.clamp_min(self.eps)) - torch.log(pred_theta.clamp_min(self.eps)))
        kl = kl.sum(dim=1)  # Sum over classes
        return kl.mean()  # Mean over batch and spatial dimensions

    def _boundary_loss(self, pred_x0: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """Compute boundary-focused loss using morphological gradients.

        This loss emphasizes the boundary regions of the segmentation by:
        1. Computing morphological gradients to identify boundaries
        2. Applying higher weights to boundary voxels
        3. Computing weighted cross-entropy on boundary regions

        Args:
            pred_x0: Predicted x0 probabilities, shape (B, K, D, H, W)
            x0: Target x0 one-hot, shape (B, K, D, H, W)

        Returns:
            Scalar boundary loss
        """
        eps = 1e-6

        # Convert one-hot to class indices for boundary detection
        # Shape: (B, 1, D, H, W)
        target_indices = torch.argmax(x0, dim=1, keepdim=True)

        # Compute boundaries using morphological gradient (dilation - erosion)
        # Use max pooling for dilation and -max(-x) for erosion
        kernel_size = 3
        padding = kernel_size // 2

        # Max pooling = dilation
        dilated = F.max_pool3d(
            target_indices.float(),
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )

        # Erosion = -max_pool(-x)
        eroded = -F.max_pool3d(
            -target_indices.float(),
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )

        # Morphological gradient: dilation - erosion
        # Non-zero values indicate boundary regions
        boundary_mask = (dilated - eroded) > 0  # Shape: (B, 1, D, H, W)
        boundary_weight = boundary_mask.float()  # 1.0 at boundaries, 0.0 elsewhere

        # Add small weight to non-boundary regions to avoid zero gradients
        boundary_weight = boundary_weight + 0.1  # Now: 1.1 at boundaries, 0.1 elsewhere

        # Compute weighted cross-entropy
        # pred_x0: (B, K, D, H, W), x0: (B, K, D, H, W)
        log_pred = pred_x0.clamp_min(eps).log()
        ce_per_voxel = -(x0 * log_pred).sum(dim=1, keepdim=True)  # (B, 1, D, H, W)

        # Weight by boundary mask
        weighted_ce = ce_per_voxel * boundary_weight

        # Normalize by sum of weights to get meaningful scale
        loss = weighted_ce.sum() / boundary_weight.sum().clamp_min(1.0)

        return loss

    def _dice_ce_loss(self, pred_x0: torch.Tensor, x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Dice + Cross-Entropy loss.

        Returns:
            Tuple of (combined_loss, dice_loss, ce_loss)
        """
        eps = 1e-6

        # Flatten spatial dimensions: (B, K, D, H, W) -> (B, K, D*H*W)
        pred_flat = pred_x0.view(pred_x0.shape[0], pred_x0.shape[1], -1)
        target_flat = x0.view(x0.shape[0], x0.shape[1], -1)

        # Dice loss (per class)
        intersection = (pred_flat * target_flat).sum(dim=-1)
        pred_sum = pred_flat.sum(dim=-1)
        target_sum = target_flat.sum(dim=-1)

        dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
        present = (target_sum > 0.0).to(dtype=pred_flat.dtype)
        dice_denom = present.sum().clamp_min(1.0)
        dice_loss = ((1.0 - dice) * present).sum() / dice_denom

        # Cross-entropy loss
        log_pred = pred_flat.clamp_min(self.eps).log()
        ce = -(target_flat * log_pred).sum(dim=1)  # Sum over classes
        ce_loss = ce.mean()  # Mean over batch and voxels

        combined = self.aux_dice_weight * dice_loss + self.aux_ce_weight * ce_loss
        return combined, dice_loss, ce_loss

    def _auxiliary_loss(self, pred_x0: torch.Tensor, x0: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Compute auxiliary loss (L1, L2, Dice+CE) with optional boundary loss.

        Returns:
            Tuple of (total_aux_loss, loss_dict) where loss_dict contains individual components
        """
        total_aux_loss = torch.zeros((), device=pred_x0.device)
        loss_dict = {}

        # Main auxiliary loss (Dice+CE, L1, L2, or asy-focal CE + Dice)
        if self.aux_loss_type != 'none' and self.aux_loss_weight > 0.0:
            if self.aux_loss_type == 'dice_ce':
                combined, dice_loss, ce_loss = self._dice_ce_loss(pred_x0, x0)
                main_aux = combined
                loss_dict['dice'] = dice_loss.item()
                loss_dict['ce'] = ce_loss.item()
            elif self.aux_loss_type == 'l1':
                aux = torch.abs(pred_x0 - x0)
                aux = aux.sum(dim=1)  # Sum over classes
                main_aux = aux.mean()  # Mean over batch and spatial dimensions
                loss_dict['l1'] = main_aux.item()
            else:  # l2
                aux = (pred_x0 - x0) ** 2
                aux = aux.sum(dim=1)  # Sum over classes
                main_aux = aux.mean()  # Mean over batch and spatial dimensions
                loss_dict['l2'] = main_aux.item()

            total_aux_loss = total_aux_loss + self.aux_loss_weight * main_aux

        # Boundary loss (always computed if weight > 0, independent of aux_loss_type)
        if self.aux_boundary_weight > 0.0:
            boundary_loss = self._boundary_loss(pred_x0, x0)
            loss_dict['boundary'] = boundary_loss.item()
            total_aux_loss = total_aux_loss + self.aux_boundary_weight * boundary_loss

        return total_aux_loss, loss_dict

    def p_losses(
        self,
        mask_indices: torch.Tensor,
        channel_cond: Optional[torch.Tensor] = None,
        *,
        cond: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        null_cond_prob: float = 0.0,
        prob_focus_present: float = 0.0,
        focus_present_mask: Optional[torch.Tensor] = None,
        **extra_kwargs,
    ) -> torch.Tensor:
        x0 = self._indices_to_one_hot(mask_indices)

        batch_size = mask_indices.shape[0]
        device = mask_indices.device
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)

        # Use unified sample_mode for diversity control
        xt_dist = self.q_xt_given_x0(x0, t, sample_mode=self.sample_mode, noise_std=self.noise_std)
        if self.sample_mode == 'one_hot':
            # For one-hot mode, train_stochastic controls sampling behavior
            if self.train_stochastic:
                xt = xt_dist.sample()  # Stochastic: diverse training
            else:
                xt = xt_dist.max_prob_sample()  # Deterministic: stable training
        elif self.sample_mode == 'prob_flow':
            xt = xt_dist.probs  # Probability flow: always deterministic
        else:
            raise ValueError(f"Unknown sample_mode: {self.sample_mode}")

        # Apply dual masking: mask either xt OR channel_cond, but never both
        xt_masked, channel_cond_masked = self._apply_dual_masking(xt, channel_cond)

        model_input = self._concat_with_cond(xt_masked, channel_cond_masked)

        # Use unified helper for 2D mode handling
        pred_logits = self._call_denoise_fn(
            model_input,
            t,
            cond=cond,
            mask=mask,
            null_cond_prob=null_cond_prob,
            prob_focus_present=prob_focus_present,
            focus_present_mask=focus_present_mask,
        )

        pred_x0 = F.softmax(pred_logits, dim=1)

        true_theta = self.theta_post(xt_masked, x0, t)
        pred_theta = self.theta_post_prob(xt_masked, pred_x0, t)

        loss = self._kl_div(true_theta, pred_theta)
        aux_loss, _ = self._auxiliary_loss(pred_x0, x0)
        total_loss = self.kl_loss_weight * loss + aux_loss

        return total_loss

    def forward(
        self,
        image: torch.Tensor,
        mask_sdf: Optional[torch.Tensor] = None,
        *,
        categorical_mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        null_cond_prob: float = 0.0,
        prob_focus_present: float = 0.0,
        focus_present_mask: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        **extra_kwargs,
    ) -> torch.Tensor:
        _ = label  # unused but accepted for API compatibility
        extra_kwargs.pop('label', None)
        kwargs = {
            'cond': cond,
            'mask': mask,
            'null_cond_prob': null_cond_prob,
            'prob_focus_present': prob_focus_present,
            'focus_present_mask': focus_present_mask,
        }
        return self.p_losses(
            categorical_mask,
            channel_cond=image,
            **kwargs,
        )

    @torch.inference_mode()
    def test(self, y, x, label, milestone):
        return

    @torch.inference_mode()
    def categorical_sample_loop(
        self,
        shape: tuple[int, ...],
        *,
        cond: Optional[torch.Tensor] = None,
        cond_scale: float = 1.0,
        channel_cond: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        sampling_steps: Optional[int] = None,
        proc: bool = True,
        final_step_mode: Optional[str] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if channel_cond is None and self.cond_channels > 0:
            raise ValueError("channel_cond is required for conditional sampling")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        device = self.betas.device
        batch_size = shape[0]
        dtype = channel_cond.dtype if channel_cond is not None else torch.float32
        probs = torch.full(shape, 1.0 / self.num_classes, device=device, dtype=dtype)
        step_list = self._build_sampling_timesteps(sampling_steps, device=device).tolist()
        iterator = enumerate(step_list)
        if proc:
            from tqdm import tqdm
            iterator = enumerate(tqdm(step_list, desc="categorical sampling", leave=False))

        for index, timestep in iterator:
            t = torch.full((batch_size,), int(timestep), device=device, dtype=torch.long)
            model_input = self._concat_with_cond(
                probs,
                channel_cond.to(dtype=probs.dtype) if channel_cond is not None else None,
            )
            logits = self._call_denoise_fn(
                model_input,
                t,
                cond=cond,
                cond_scale=cond_scale,
                mask=mask,
            )
            pred_x0 = F.softmax(logits.float() / temperature, dim=1).to(logits.dtype)
            next_t = int(step_list[index + 1]) if index + 1 < len(step_list) else -1
            probs = self._categorical_step(pred_x0, next_t)

        mode = self._validate_final_step_mode(final_step_mode or self.final_step_mode)
        if mode == "confidence":
            return probs
        distribution = OneHotCategoricalBCHW(probs=probs)
        if mode == "sample":
            return distribution.sample()
        return distribution.max_prob_sample()

    @torch.inference_mode()
    def _sample_card(
        self,
        *,
        channel_cond: torch.Tensor,
        reference_denoise_fn: nn.Module,
        primary_cond_scale: float,
        reference_cond_scale: float,
        t_min: float,
        t_max: float,
        w_b: float,
        w_k: float,
        num_steps: int,
        proc: bool,
        final_step_mode: str,
    ) -> dict[str, torch.Tensor]:
        """Run the five-state categorical reverse sampler with CARD calibration."""
        if w_k <= 0:
            raise ValueError("w_k must be positive")
        batch_size = channel_cond.shape[0]
        shape = (
            (batch_size, self.num_classes, self.image_size, self.image_size)
            if self.num_frames == 1
            else (batch_size, self.num_classes, self.num_frames, self.image_size, self.image_size)
        )
        probs = torch.full(
            shape,
            1.0 / self.num_classes,
            device=self.betas.device,
            dtype=channel_cond.dtype,
        )
        step_list = self._build_sampling_timesteps(num_steps, device=self.betas.device).tolist()
        iterator = enumerate(step_list)
        if proc:
            from tqdm import tqdm
            iterator = enumerate(tqdm(step_list, desc="categorical CARD sampling", leave=False))

        js_history = []
        final_logits = None
        for index, timestep in iterator:
            t = torch.full((batch_size,), int(timestep), device=probs.device, dtype=torch.long)
            model_input = self._concat_with_cond(probs, channel_cond.to(dtype=probs.dtype))
            final_logits = self._call_denoise_fn(
                model_input, t, cond_scale=primary_cond_scale, cond=None, mask=None
            )
            reference_logits = self._call_denoise_fn_external(
                reference_denoise_fn,
                model_input,
                t,
                cond_scale=reference_cond_scale,
                cond=None,
                mask=None,
            )
            primary_probs = F.softmax(final_logits.float(), dim=1).to(final_logits.dtype)
            reference_probs = F.softmax(reference_logits.float(), dim=1).to(reference_logits.dtype)
            js_map = js_divergence_map(primary_probs, reference_probs)
            js_history.append(js_map)

            weight = torch.sigmoid((js_map - w_b) / w_k)
            step_temperature = t_min + weight * (t_max - t_min)
            pred_x0 = F.softmax(final_logits.float() / step_temperature, dim=1).to(final_logits.dtype)
            if index + 1 < len(step_list):
                probs = self._categorical_step(pred_x0, int(step_list[index + 1]))

        if final_logits is None:
            raise RuntimeError("CARD sampler produced no reverse steps")
        tac = torch.stack(js_history, dim=0).mean(dim=0)
        weight = torch.sigmoid((tac - w_b) / w_k)
        temperature = t_min + weight * (t_max - t_min)
        probabilities = F.softmax(final_logits.float() / temperature, dim=1).to(final_logits.dtype)
        distribution = OneHotCategoricalBCHW(probs=probabilities)
        mode = self._validate_final_step_mode(final_step_mode)
        if mode == "sample":
            categorical = distribution.sample()
        else:
            categorical = distribution.max_prob_sample()
        return {
            "prediction": categorical.argmax(dim=1, keepdim=True),
            "probabilities": probabilities,
            "final_logits": final_logits.detach(),
            "tac": tac.detach(),
        }

    def sample_full_tac(
        self,
        *,
        channel_cond: torch.Tensor,
        reference_denoise_fn: nn.Module,
        primary_cond_scale: float,
        reference_cond_scale: float,
        t_min: float,
        t_max: float,
        w_b: float,
        w_k: float,
        num_steps: int = 5,
        proc: bool = False,
        final_step_mode: str = "majority",
    ) -> dict[str, torch.Tensor]:
        return self._sample_card(
            channel_cond=channel_cond,
            reference_denoise_fn=reference_denoise_fn,
            primary_cond_scale=primary_cond_scale,
            reference_cond_scale=reference_cond_scale,
            t_min=t_min,
            t_max=t_max,
            w_b=w_b,
            w_k=w_k,
            num_steps=num_steps,
            proc=proc,
            final_step_mode=final_step_mode,
        )

    @torch.inference_mode()
    def sample(
        self,
        batch_size: int = 16,
        cond: Optional[torch.Tensor] = None,
        cond_scale: float = 1.0,
        proc: bool = True,
        mask: Optional[torch.Tensor] = None,
        channel_cond: Optional[torch.Tensor] = None,
        sampler: str = "categorical",
        num_steps: Optional[int] = None,
        final_step_mode: Optional[str] = None,
        return_prob: bool = False,
        temperature: float = 1.0,
        **unsupported,
    ) -> torch.Tensor:
        """Run the deterministic categorical reverse sampler used during training validation."""
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise TypeError(f"Unsupported categorical sampling options: {names}")
        if sampler != "categorical":
            raise ValueError(f"sampler must be 'categorical', got {sampler!r}")
        if channel_cond is None and self.cond_channels > 0:
            raise ValueError("channel_cond is required for conditional sampling")

        shape = (
            (batch_size, self.num_classes, self.image_size, self.image_size)
            if self.num_frames == 1
            else (batch_size, self.num_classes, self.num_frames, self.image_size, self.image_size)
        )
        mode = self._validate_final_step_mode(final_step_mode or self.final_step_mode)
        sampled = self.categorical_sample_loop(
            shape,
            cond=cond,
            cond_scale=cond_scale,
            channel_cond=channel_cond,
            mask=mask,
            sampling_steps=num_steps,
            proc=proc,
            final_step_mode=mode,
            temperature=temperature,
        )
        if mode == "confidence":
            probabilities = sampled
            predictions = probabilities.argmax(dim=1, keepdim=True)
            return (predictions, probabilities) if return_prob else probabilities

        predictions = sampled.argmax(dim=1, keepdim=True)
        if not return_prob:
            return predictions
        probabilities = self.categorical_sample_loop(
            shape,
            cond=cond,
            cond_scale=cond_scale,
            channel_cond=channel_cond,
            mask=mask,
            sampling_steps=num_steps,
            proc=False,
            final_step_mode="confidence",
            temperature=temperature,
        )
        return predictions, probabilities
