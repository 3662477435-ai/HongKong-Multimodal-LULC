from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import VGG16_BN_Weights, ResNet50_Weights, Swin_T_Weights, resnet50, swin_t, vgg16_bn
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.segmentation import fcn_resnet50

from .advanced_model import ConvGNAct
from .model import TextEncoder


def _rgb_adapter(channels: int) -> nn.Conv2d:
    adapter = nn.Conv2d(channels, 3, 1)
    with torch.no_grad():
        adapter.weight.zero_()
        adapter.bias.zero_()
        # S2 order: B1, B2(blue), B3(green), B4(red), ...
        adapter.weight[0, 3, 0, 0] = 1.0
        adapter.weight[1, 2, 0, 0] = 1.0
        adapter.weight[2, 1, 0, 0] = 1.0
    return adapter


def _expand_pretrained_conv(
    conv: nn.Conv2d,
    input_channels: int,
) -> nn.Conv2d:
    """Expand an RGB convolution without a learned 48->3 information bottleneck.

    The RGB kernel mean is copied to every input channel and scaled by
    3 / input_channels so that the expected response magnitude stays close to
    the pretrained three-channel layer.  The whole expanded layer remains
    trainable.
    """
    expanded = nn.Conv2d(
        input_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        mean_kernel = conv.weight.mean(dim=1, keepdim=True)
        expanded.weight.copy_(
            mean_kernel.expand(-1, input_channels, -1, -1)
            * (3.0 / float(input_channels))
        )
        if conv.bias is not None and expanded.bias is not None:
            expanded.bias.copy_(conv.bias)
    return expanded


class DilatedConvGNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        groups = min(16, max(1, out_channels // 2))
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, 3, padding=dilation,
                dilation=dilation, bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )


class CNNBaseline(nn.Module):
    """Plain spatial CNN without pooling, skip connections or attention."""

    def __init__(self, input_channels: int, num_classes: int = 14, pretrained: bool = False) -> None:
        super().__init__()
        del pretrained
        self.network = nn.Sequential(
            DilatedConvGNAct(input_channels, 64, 1),
            DilatedConvGNAct(64, 96, 1),
            DilatedConvGNAct(96, 96, 2),
            DilatedConvGNAct(96, 96, 4),
            DilatedConvGNAct(96, 64, 2),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FCNNBaseline(nn.Module):
    """Compact FCN-8s-style encoder with additive multi-scale score skips."""

    def __init__(self, input_channels: int, num_classes: int = 14, pretrained: bool = False) -> None:
        super().__init__()
        del pretrained
        self.enc1 = nn.Sequential(ConvGNAct(input_channels, 64), ConvGNAct(64, 64))
        self.enc2 = nn.Sequential(ConvGNAct(64, 128), ConvGNAct(128, 128))
        self.enc3 = nn.Sequential(ConvGNAct(128, 256), ConvGNAct(256, 256))
        self.bottleneck = nn.Sequential(
            ConvGNAct(256, 384), nn.Dropout2d(0.1), ConvGNAct(384, 384)
        )
        self.pool = nn.MaxPool2d(2)
        self.score2 = nn.Conv2d(128, num_classes, 1)
        self.score3 = nn.Conv2d(256, num_classes, 1)
        self.score4 = nn.Conv2d(384, num_classes, 1)

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        return e1, e2, e3, b

    def logits_from_features(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        input_size: tuple[int, int],
    ) -> torch.Tensor:
        _, e2, e3, b = features
        score = F.interpolate(
            self.score4(b), size=e3.shape[-2:], mode="bilinear", align_corners=False
        ) + self.score3(e3)
        score = F.interpolate(
            score, size=e2.shape[-2:], mode="bilinear", align_corners=False
        ) + self.score2(e2)
        return F.interpolate(score, size=input_size, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits_from_features(self.encode(x), x.shape[-2:])


class TransformerBaseline(nn.Module):
    """Patch-token Transformer segmentation baseline with a light decoder."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int = 14,
        pretrained: bool = False,
        embed_dim: int = 128,
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        del pretrained
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(
            input_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.position = nn.Parameter(torch.zeros(1, embed_dim, 32, 32))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.Sequential(
            ConvGNAct(embed_dim, 96),
            ConvGNAct(96, 64),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, num_classes, 1),
        )
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self.patch_embed(x)
        position = F.interpolate(
            self.position, size=features.shape[-2:], mode="bilinear", align_corners=False
        )
        features = features + position
        batch, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        tokens = self.norm(self.transformer(tokens))
        features = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        logits = self.decoder(features)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


class FCNResNet50Baseline(nn.Module):
    """Paper comparator: FCN with a ResNet-50 backbone."""

    def __init__(self, input_channels: int, num_classes: int = 14, pretrained: bool = True) -> None:
        super().__init__()
        self.adapter = _rgb_adapter(input_channels)
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.model = fcn_resnet50(
            weights=None,
            weights_backbone=weights,
            num_classes=num_classes,
            aux_loss=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.adapter(x))["out"]


class FCNResNet50DirectBaseline(nn.Module):
    """FCN-ResNet50 whose pretrained first convolution directly accepts 48 channels."""

    def __init__(self, input_channels: int, num_classes: int = 14, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.model = fcn_resnet50(
            weights=None,
            weights_backbone=weights,
            num_classes=num_classes,
            aux_loss=False,
        )
        self.model.backbone.conv1 = _expand_pretrained_conv(
            self.model.backbone.conv1, input_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBNAct(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class VGGUNetTransformerBaseline(nn.Module):
    """Paper comparator: VGG16 encoder, bottleneck Transformer and UNet decoder."""

    def __init__(self, input_channels: int, num_classes: int = 14, pretrained: bool = True) -> None:
        super().__init__()
        self.adapter = _rgb_adapter(input_channels)
        weights = VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        features = vgg16_bn(weights=weights).features
        self.enc1 = features[:7]
        self.enc2 = features[7:14]
        self.enc3 = features[14:24]
        self.enc4 = features[24:34]
        self.enc5 = features[34:44]
        layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.dec4 = DecoderBlock(512, 512, 512)
        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNAct(64, 32),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        batch, channels, height, width = e5.shape
        tokens = e5.flatten(2).transpose(1, 2)
        e5 = self.transformer(tokens).transpose(1, 2).reshape(batch, channels, height, width)
        d4 = self.dec4(e5, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.final(d1)


class VGGUNetTransformerDirectBaseline(VGGUNetTransformerBaseline):
    """VGG-UNet-Transformer with a trainable direct multichannel first layer."""

    def __init__(self, input_channels: int, num_classes: int = 14, pretrained: bool = True) -> None:
        super().__init__(
            input_channels=input_channels, num_classes=num_classes, pretrained=pretrained
        )
        # Remove the 48->3 adapter and expand VGG's pretrained first convolution.
        self.adapter = nn.Identity()
        self.enc1[0] = _expand_pretrained_conv(self.enc1[0], input_channels)


class _ASPP(nn.Module):
    def __init__(self, channels: int, out_channels: int = 256) -> None:
        super().__init__()
        def conv(kernel: int, dilation: int) -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(channels, out_channels, kernel, padding=dilation if kernel > 1 else 0, dilation=dilation, bias=False),
                nn.GroupNorm(16, out_channels), nn.GELU(),
            )
        self.branches = nn.ModuleList([conv(1, 1), conv(3, 6), conv(3, 12), conv(3, 18)])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, out_channels, 1, bias=False), nn.GroupNorm(16, out_channels), nn.GELU())
        self.project = nn.Sequential(nn.Conv2d(out_channels * 5, out_channels, 1, bias=False), nn.GroupNorm(16, out_channels), nn.GELU(), nn.Dropout2d(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        items = [branch(x) for branch in self.branches]
        pooled = F.interpolate(self.pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.project(torch.cat([*items, pooled], dim=1))


class DeepLabV3PlusResNet50Direct48(nn.Module):
    """Direct-48-channel DeepLabV3+-ResNet50 (OS=16, ASPP and decoder)."""
    def __init__(self, input_channels: int, num_classes: int = 8, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights, replace_stride_with_dilation=[False, False, True])
        backbone.conv1 = _expand_pretrained_conv(backbone.conv1, input_channels)
        self.backbone = create_feature_extractor(backbone, return_nodes={"layer1": "low", "layer4": "high"})
        self.aspp = _ASPP(2048)
        self.low = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False), nn.GroupNorm(12, 48), nn.GELU())
        self.decoder = nn.Sequential(ConvGNAct(304, 256), ConvGNAct(256, 256), nn.Conv2d(256, num_classes, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        features = self.backbone(x)
        high = self.aspp(features["high"])
        high = F.interpolate(high, size=features["low"].shape[-2:], mode="bilinear", align_corners=False)
        logits = self.decoder(torch.cat([high, self.low(features["low"])], dim=1))
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


class _UPerHead(nn.Module):
    def __init__(self, channels: tuple[int, int, int, int], out_channels: int = 256, num_classes: int = 8) -> None:
        super().__init__()
        self.lateral = nn.ModuleList([nn.Sequential(nn.Conv2d(c, out_channels, 1, bias=False), nn.GroupNorm(16, out_channels), nn.GELU()) for c in channels])
        self.ppm_scales = (1, 2, 3, 6)
        self.ppm = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(scale), nn.Conv2d(out_channels, out_channels, 1, bias=False), nn.GroupNorm(16, out_channels), nn.GELU()) for scale in self.ppm_scales])
        self.ppm_bottleneck = ConvGNAct(out_channels * 5, out_channels)
        self.fpn = nn.ModuleList([ConvGNAct(out_channels, out_channels) for _ in range(3)])
        self.fuse = nn.Sequential(ConvGNAct(out_channels * 4, out_channels), nn.Dropout2d(0.1), nn.Conv2d(out_channels, num_classes, 1))

    def forward(self, features: list[torch.Tensor], size: tuple[int, int]) -> torch.Tensor:
        lateral = [layer(feature) for layer, feature in zip(self.lateral, features)]
        top = lateral[-1]
        pooled = [F.interpolate(module(top), size=top.shape[-2:], mode="bilinear", align_corners=False) for module in self.ppm]
        top = self.ppm_bottleneck(torch.cat([top, *pooled], dim=1))
        outputs = [None, None, None, top]
        for idx in range(2, -1, -1):
            top = lateral[idx] + F.interpolate(top, size=lateral[idx].shape[-2:], mode="bilinear", align_corners=False)
            outputs[idx] = self.fpn[idx](top)
        fused = torch.cat([F.interpolate(item, size=outputs[0].shape[-2:], mode="bilinear", align_corners=False) for item in outputs], dim=1)
        return F.interpolate(self.fuse(fused), size=size, mode="bilinear", align_corners=False)


class SwinTUPerNetDirect48(nn.Module):
    """Swin-T (patch 4/window 7) with a UPerNet-style PPM-FPN decoder."""
    def __init__(self, input_channels: int, num_classes: int = 8, pretrained: bool = True) -> None:
        super().__init__()
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        self.swin = swin_t(weights=weights)
        self.swin.features[0][0] = _expand_pretrained_conv(self.swin.features[0][0], input_channels)
        self.decoder = _UPerHead((96, 192, 384, 768), num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        values = []
        current = x
        for index, module in enumerate(self.swin.features):
            current = module(current)
            if index in (1, 3, 5, 7):
                values.append(current.permute(0, 3, 1, 2).contiguous())
        return self.decoder(values, size)


class SegFormerB1Direct48(nn.Module):
    """ImageNet-pretrained MiT-B1 with its standard lightweight MLP decoder."""
    def __init__(self, input_channels: int, num_classes: int = 8, pretrained: bool = True) -> None:
        super().__init__()
        try:
            from transformers import SegformerForSemanticSegmentation
        except ImportError as exc:
            raise ImportError("SegFormer-B1 requires transformers; install it in the training environment.") from exc
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/mit-b1", num_labels=num_classes, ignore_mismatched_sizes=True,
                local_files_only=True,
            )
        else:
            from transformers import SegformerConfig
            self.model = SegformerForSemanticSegmentation(SegformerConfig(num_labels=num_classes))
        first = self.model.segformer.encoder.patch_embeddings[0].proj
        self.model.segformer.encoder.patch_embeddings[0].proj = _expand_pretrained_conv(first, input_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=x, return_dict=True).logits
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


class MultiModalEarlyFusionComparator(nn.Module):
    """Neutral early-fusion front-end for same-data architecture comparisons.

    Every dense modality is used at native 10 m alignment.  OSM text is
    encoded once per patch and broadcast as additional spatial channels.  The
    same front-end is used for FCN and VGG comparators, so their difference is
    attributable to the segmentation backbone rather than input availability.
    """

    def __init__(
        self,
        backbone: str,
        vocab_size: int,
        s2_channels: int,
        s1_channels: int,
        urban_channels: int,
        road_channels: int,
        osm_channels: int = 20,
        text_dim: int = 128,
        text_channels: int = 16,
        text_heads: int = 4,
        text_layers: int = 2,
        max_tokens: int = 96,
        dropout: float = 0.1,
        pretrained: bool = True,
        use_text: bool = True,
        num_classes: int = 14,
    ) -> None:
        super().__init__()
        self.s1_channels = s1_channels
        self.urban_channels = urban_channels
        self.road_channels = road_channels
        self.osm_channels = osm_channels
        self.use_text = use_text
        self.text_encoder = (
            TextEncoder(vocab_size, text_dim, text_heads, text_layers, max_tokens, dropout)
            if use_text else None
        )
        self.text_projection = nn.Linear(text_dim, text_channels) if use_text else None
        dense_channels = (
            s2_channels + s1_channels + urban_channels + road_channels
            + osm_channels + (text_channels if use_text else 0)
        )
        self.fusion = nn.Sequential(
            ConvGNAct(dense_channels, 96, 1),
            nn.Dropout2d(dropout),
            ConvGNAct(96, 48),
        )
        constructors = {
            "cnn": CNNBaseline,
            "fcnn": FCNNBaseline,
            "transformer": TransformerBaseline,
            "fcn_resnet50": FCNResNet50Baseline,
            "vgg_unet_transformer": VGGUNetTransformerBaseline,
            "fcn_resnet50_direct48": FCNResNet50DirectBaseline,
            "vgg_unet_transformer_direct48": VGGUNetTransformerDirectBaseline,
            "segformer_b1_direct48": SegFormerB1Direct48,
            "deeplabv3plus_resnet50_direct48": DeepLabV3PlusResNet50Direct48,
            "swin_t_upernet_direct48": SwinTUPerNetDirect48,
        }
        self.segmenter = constructors[backbone](
            48, num_classes=num_classes, pretrained=pretrained
        )

    def fuse_modalities(
        self,
        s2: torch.Tensor,
        s1: torch.Tensor,
        urban: torch.Tensor,
        roads: torch.Tensor,
        osm: torch.Tensor,
        input_ids: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        parts = [s2]
        if self.s1_channels:
            parts.append(s1)
        if self.urban_channels:
            parts.append(urban)
        if self.road_channels:
            parts.append(roads)
        if self.osm_channels:
            parts.append(osm)
        if self.use_text:
            if self.text_encoder is None or self.text_projection is None:
                raise RuntimeError("Text modules are missing")
            text = self.text_projection(self.text_encoder(input_ids, text_mask))
            text = text[:, :, None, None].expand(-1, -1, s2.shape[-2], s2.shape[-1])
            parts.append(text)
        return self.fusion(torch.cat(parts, dim=1))

    def forward(
        self,
        s2: torch.Tensor,
        s1: torch.Tensor,
        urban: torch.Tensor,
        roads: torch.Tensor,
        osm: torch.Tensor,
        input_ids: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        fused = self.fuse_modalities(
            s2, s1, urban, roads, osm, input_ids, text_mask
        )
        return self.segmenter(fused)
