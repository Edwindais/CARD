from __future__ import annotations

import torch
from torch import nn

from monai.networks.nets import SwinUNETR

from .film import SinusoidalPosEmb, FiLMHead, apply_film


class TimeAwareSwinUNETR(SwinUNETR):
    """SwinUNETR variant with FiLM-based timestep conditioning.

    Supports both 2D and 3D inputs via spatial_dims parameter.
    For 2D inputs (spatial_dims=2), expects shape (B, C, H, W).
    For 3D inputs (spatial_dims=3), expects shape (B, C, D, H, W).
    """

    def __init__(
        self,
        *,
        time_embed_dim=None,
        film_hidden_dim=None,
        cond_channels: int = 0,
        **super_kwargs,
    ) -> None:
        feature_size = super_kwargs.get('feature_size', 24)
        self.spatial_dims = int(super_kwargs.get('spatial_dims', 3))
        super().__init__(**super_kwargs)

        self.cond_channels = int(cond_channels)
        base_dim = int(feature_size)
        emb_input_dim = max(32, base_dim)
        self.time_dim = time_embed_dim or base_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(emb_input_dim),
            nn.Linear(emb_input_dim, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        channels = [
            base_dim,
            base_dim,
            base_dim * 2,
            base_dim * 4,
            base_dim * 8,
            base_dim * 16,
            base_dim * 8,
            base_dim * 4,
            base_dim * 2,
            base_dim,
            base_dim,
        ]
        head_names = [
            'enc0',
            'vit0',
            'vit1',
            'vit2',
            'vit3',
            'vit4',
            'dec3',
            'dec2',
            'dec1',
            'dec0',
            'out',
        ]
        if len(channels) != len(head_names):
            raise RuntimeError('FiLM head configuration mismatch')
        self.film_heads = nn.ModuleDict({
            name: FiLMHead(self.time_dim, ch, hidden_dim=film_hidden_dim)
            for name, ch in zip(head_names, channels)
        })

    def _compute_time_emb(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim == 0:
            time = time.unsqueeze(0)
        if time.shape[0] != time.numel():
            time = time.reshape(-1)
        return self.time_mlp(time.float())

    def forward(self, x_in: torch.Tensor, time: torch.Tensor, **_) -> torch.Tensor:
        if time.shape[0] != x_in.shape[0]:
            raise ValueError('time tensor batch dimension must match input batch size')
        time_emb = self._compute_time_emb(time)

        # For 2D mode (spatial_dims=2): input should be (B, C, H, W) - 4D tensor
        # For 3D mode (spatial_dims=3): input should be (B, C, D, H, W) - 5D tensor
        # MONAI SwinUNETR handles both cases correctly based on spatial_dims

        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(x_in.shape[2:])

        hidden_states_out = self.swinViT(x_in, self.normalize)

        enc0 = self.encoder1(x_in)
        enc0 = apply_film(enc0, *self.film_heads['enc0'](time_emb))

        hs0 = apply_film(hidden_states_out[0], *self.film_heads['vit0'](time_emb))
        enc1 = self.encoder2(hs0)

        hs1 = apply_film(hidden_states_out[1], *self.film_heads['vit1'](time_emb))
        enc2 = self.encoder3(hs1)

        hs2 = apply_film(hidden_states_out[2], *self.film_heads['vit2'](time_emb))
        enc3 = self.encoder4(hs2)

        hs3 = apply_film(hidden_states_out[3], *self.film_heads['vit3'](time_emb))
        hs4 = apply_film(hidden_states_out[4], *self.film_heads['vit4'](time_emb))

        dec4 = self.encoder10(hs4)

        dec3 = self.decoder5(dec4, hs3)
        dec3 = apply_film(dec3, *self.film_heads['dec3'](time_emb))

        dec2 = self.decoder4(dec3, enc3)
        dec2 = apply_film(dec2, *self.film_heads['dec2'](time_emb))

        dec1 = self.decoder3(dec2, enc2)
        dec1 = apply_film(dec1, *self.film_heads['dec1'](time_emb))

        dec0 = self.decoder2(dec1, enc1)
        dec0 = apply_film(dec0, *self.film_heads['dec0'](time_emb))

        out = self.decoder1(dec0, enc0)
        out = apply_film(out, *self.film_heads['out'](time_emb))

        logits = self.out(out)

        return logits

    def forward_with_cond_scale(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        *,
        cond_scale: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        logits = self.forward(x, time, **kwargs)
        if cond_scale == 1.0 or self.cond_channels <= 0:
            return logits
        if x.shape[1] < self.cond_channels:
            raise ValueError('Input tensor has fewer channels than configured conditioning channels')
        uncond_x = x.clone()
        uncond_x[:, -self.cond_channels:, ...] = 0
        null_logits = self.forward(uncond_x, time, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale
