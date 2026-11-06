from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int, base: int, dropout: float) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            [
                ConvBlock(in_channels, base),
                ConvBlock(base, base * 2),
                ConvBlock(base * 2, base * 4, dropout),
                ConvBlock(base * 4, base * 8, dropout),
            ]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for idx, stage in enumerate(self.stages):
            if idx:
                x = self.pool(x)
            x = stage(x)
            features.append(x)
        return features


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        heads: int,
        layers: int,
        max_tokens: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.position = nn.Parameter(torch.zeros(1, max_tokens, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids) + self.position[:, : input_ids.shape[1]]
        x = self.encoder(x, src_key_padding_mask=~mask)
        weights = mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.norm(pooled)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([self.reduce(x), skip], dim=1))


class MultiModalUNet(nn.Module):
    """Dual Sentinel encoders with Transformer text conditioning and U-Net decoding."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 10,
        base: int = 32,
        text_dim: int = 128,
        text_heads: int = 4,
        text_layers: int = 2,
        max_tokens: int = 96,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.s1_encoder = ImageEncoder(3, base, dropout)
        self.s2_encoder = ImageEncoder(5, base, dropout)
        channels = [base, base * 2, base * 4, base * 8]
        self.fuse = nn.ModuleList([ConvBlock(c * 2, c, dropout if i > 1 else 0.0) for i, c in enumerate(channels)])
        self.text_encoder = TextEncoder(vocab_size, text_dim, text_heads, text_layers, max_tokens, dropout)
        self.film = nn.Linear(text_dim, channels[-1] * 2)
        self.text_to_bottleneck = nn.Linear(text_dim, channels[-1])
        self.up2 = UpBlock(channels[3], channels[2], channels[2], dropout)
        self.up1 = UpBlock(channels[2], channels[1], channels[1], dropout)
        self.up0 = UpBlock(channels[1], channels[0], channels[0], dropout)
        self.head = nn.Conv2d(channels[0], num_classes, 1)

    def forward(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        input_ids: torch.Tensor,
        text_mask: torch.Tensor,
        osm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        s1_features = self.s1_encoder(s1)
        s2_features = self.s2_encoder(s2)
        fused = [layer(torch.cat([a, b], dim=1)) for layer, a, b in zip(self.fuse, s1_features, s2_features)]
        text = self.text_encoder(input_ids, text_mask)
        gamma, beta = self.film(text).chunk(2, dim=1)
        bottleneck = fused[3]
        bottleneck = bottleneck * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        bottleneck = bottleneck + self.text_to_bottleneck(text)[:, :, None, None]
        x = self.up2(bottleneck, fused[2])
        x = self.up1(x, fused[1])
        x = self.up0(x, fused[0])
        return self.head(x)
