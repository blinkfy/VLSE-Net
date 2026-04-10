from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_PUBLIC_DIR = Path(__file__).resolve().parent
if str(_PUBLIC_DIR) not in sys.path:
    sys.path.insert(0, str(_PUBLIC_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from DSConv_pro import DSConv_pro
from feature_renorm import FeatureReNormalization2d

class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGatedSkip(nn.Module):
    """Lightweight channel + spatial gating for skip features.

    The decoder feature acts as the gating signal. We first upsample decoder
    features to the skip resolution, then compute:
    - channel gate: from global pooled decoder+skip descriptors
    - spatial gate: from local concatenated decoder+skip features
    The final gated skip is: skip * g_c * g_s
    """

    def __init__(self, gate_channels: int, skip_channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, skip_channels // max(1, reduction))
        self.channel_gate = nn.Sequential(
            nn.Linear(gate_channels + skip_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, skip_channels),
        )
        self.spatial_gate = nn.Conv2d(gate_channels + skip_channels, 1, kernel_size=1, bias=True)

        # Residual gating (identity-init): weight = 1 + tanh(alpha) * delta, where delta in [-1, 1].
        self.alpha_c = nn.Parameter(torch.tensor(0.0))
        self.alpha_s = nn.Parameter(torch.tensor(0.0))

    def forward(self, gate_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        if gate_feat.shape[-2:] != skip_feat.shape[-2:]:
            gate_feat = F.interpolate(gate_feat, size=skip_feat.shape[-2:], mode="bilinear", align_corners=False)

        gate_pool = F.adaptive_avg_pool2d(gate_feat, output_size=1).flatten(1)
        skip_pool = F.adaptive_avg_pool2d(skip_feat, output_size=1).flatten(1)
        channel_raw = self.channel_gate(torch.cat([gate_pool, skip_pool], dim=1))
        channel_delta = torch.sigmoid(channel_raw) * 2.0 - 1.0
        alpha_c = torch.tanh(self.alpha_c)
        channel_weight = (1.0 + alpha_c * channel_delta).unsqueeze(-1).unsqueeze(-1)

        spatial_raw = self.spatial_gate(torch.cat([gate_feat, skip_feat], dim=1))
        spatial_delta = torch.sigmoid(spatial_raw) * 2.0 - 1.0
        alpha_s = torch.tanh(self.alpha_s)
        spatial_weight = 1.0 + alpha_s * spatial_delta
        return skip_feat * channel_weight * spatial_weight


def _choose_gn_groups(num_channels: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if num_channels % g == 0:
            return g
    return 1


def make_gn(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """Stable GroupNorm helper for small-batch training."""
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class DecoderTextAdapter(nn.Module):
    """Lightweight text-conditioned residual refinement for decoder features."""

    def __init__(
        self,
        channels: int,
        text_dim: int,
        *,
        alpha_init: float = 0.1,
    ) -> None:
        super().__init__()
        gn_groups = _choose_gn_groups(channels)
        self.delta = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(gn_groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.to_gamma_beta = nn.Sequential(
            nn.Linear(text_dim, channels * 2),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        delta = self.delta(x)
        gamma_beta = self.to_gamma_beta(text_feat)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        delta = delta * (1.0 + gamma) + beta
        return x + self.alpha * delta


class TriBranchDirectionalRefine(nn.Module):
    """Directional refinement plugin: std 3x3 + horizontal/vertical DSConv branches."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        reduction: int = 4,
        max_res_scale: float = 0.30,
        alpha_init: float = -2.0,
        extend_scope: float = 1.0,
    ) -> None:
        super().__init__()
        if DSConv_pro is None:
            raise ImportError("DSConv_pro is required for TriBranchDirectionalRefine.")

        hidden = max(channels // reduction, 16)
        gate_hidden = max(hidden // 2, 8)
        self.max_res_scale = float(max_res_scale)

        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            make_gn(hidden),
            nn.LeakyReLU(inplace=True),
        )
        self.std_branch = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            make_gn(hidden),
            nn.LeakyReLU(inplace=True),
        )

        self.ds_h = DSConv_pro(hidden, hidden, kernel_size=kernel_size, extend_scope=extend_scope, morph=0)
        self.ds_v = DSConv_pro(hidden, hidden, kernel_size=kernel_size, extend_scope=extend_scope, morph=1)
        self.branch_h = nn.Sequential(
            make_gn(hidden),
            nn.LeakyReLU(inplace=True),
        )
        self.branch_v = nn.Sequential(
            make_gn(hidden),
            nn.LeakyReLU(inplace=True),
        )

        self.gate = nn.Sequential(
            nn.Conv2d(hidden * 3, gate_hidden, kernel_size=1, bias=False),
            make_gn(gate_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(gate_hidden, 3, kernel_size=1, bias=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            make_gn(hidden),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            make_gn(channels),
        )

        # Channel-wise residual scaling keeps early training perturbation small.
        self.alpha = nn.Parameter(torch.full((1, channels, 1, 1), float(alpha_init), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        z = self.reduce(x)

        zs = self.std_branch(z)
        zh = self.branch_h(self.ds_h(z))
        zv = self.branch_v(self.ds_v(z))

        gate_logits = self.gate(torch.cat([zs, zh, zv], dim=1))
        gate = torch.softmax(gate_logits, dim=1)
        z = gate[:, 0:1] * zs + gate[:, 1:2] * zh + gate[:, 2:3] * zv
        z = self.fuse(z)

        scale = self.max_res_scale * torch.sigmoid(self.alpha)
        return identity + scale * z


class MultiScaleContext(nn.Module):
    """Lightweight multi-scale context block for bottleneck only."""

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        max_res_scale: float = 0.15,
        beta_init: float = -2.2,
    ) -> None:
        super().__init__()
        hidden = max(channels // reduction, 32)
        self.max_res_scale = float(max_res_scale)

        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            make_gn(hidden),
            nn.LeakyReLU(inplace=True),
        )

        def _dw_branch(dilation: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(
                    hidden,
                    hidden,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    groups=hidden,
                    bias=False,
                ),
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                make_gn(hidden),
                nn.LeakyReLU(inplace=True),
            )

        self.b1 = _dw_branch(1)
        self.b2 = _dw_branch(2)
        self.b3 = _dw_branch(3)

        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * 4, hidden, kernel_size=1, bias=False),
            make_gn(hidden),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            make_gn(channels),
        )
        self.beta = nn.Parameter(torch.full((1, channels, 1, 1), float(beta_init), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        z = self.reduce(x)
        z1 = self.b1(z)
        z2 = self.b2(z)
        z3 = self.b3(z)
        z = self.fuse(torch.cat([z, z1, z2, z3], dim=1))
        scale = self.max_res_scale * torch.sigmoid(self.beta)
        return identity + scale * z


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.conv = ConvBNReLU(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(x)
        down = self.pool(feat)
        return feat, down


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_skip_attention: bool = False,
        skip_attention_reduction: int = 8,
    ) -> None:
        super().__init__()
        self.skip_gate = (
            AttentionGatedSkip(in_channels, skip_channels, reduction=skip_attention_reduction)
            if use_skip_attention
            else None
        )
        self.conv = ConvBNReLU(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if self.skip_gate is not None:
            skip = self.skip_gate(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class CLIPTextEncoder(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "RN50",
        freeze: bool = True,
        normalize: bool = True,
        download_root: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.clip_model, _ = clip.load(
            clip_model_name,
            device="cpu",
            jit=False,
            download_root=download_root,
        )
        self.normalize = normalize
        self.text_dim = int(self.clip_model.text_projection.shape[1])
        if freeze:
            self.freeze_parameters()

    def freeze_parameters(self) -> None:
        for param in self.clip_model.transformer.parameters():
            param.requires_grad = False
        for param in self.clip_model.token_embedding.parameters():
            param.requires_grad = False
        self.clip_model.positional_embedding.requires_grad = False
        self.clip_model.ln_final.weight.requires_grad = False
        self.clip_model.ln_final.bias.requires_grad = False
        self.clip_model.text_projection.requires_grad = False

    def unfreeze_parameters(self) -> None:
        for param in self.clip_model.transformer.parameters():
            param.requires_grad = True
        for param in self.clip_model.token_embedding.parameters():
            param.requires_grad = True
        self.clip_model.positional_embedding.requires_grad = True
        self.clip_model.ln_final.weight.requires_grad = True
        self.clip_model.ln_final.bias.requires_grad = True
        self.clip_model.text_projection.requires_grad = True

    def encode_tokens(self, prompts: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = clip.tokenize(prompts, truncate=True).to(device)
        x = self.clip_model.token_embedding(tokens).type(self.clip_model.dtype)
        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).float()

        eot_indices = tokens.argmax(dim=-1)
        global_features = x[torch.arange(x.shape[0], device=device), eot_indices] @ self.clip_model.text_projection.float()
        token_features = x @ self.clip_model.text_projection.float()

        if self.normalize:
            global_features = F.normalize(global_features, dim=-1)
            token_features = F.normalize(token_features, dim=-1)
        return global_features, token_features

    def forward(self, prompts: List[str], device: torch.device) -> torch.Tensor:
        text_features, _ = self.encode_tokens(prompts, device)
        return text_features


class TextConditioning(nn.Module):
    def __init__(
        self,
        text_dim: int,
        feature_dim: int,
        hidden_dim: int = 256,
        residual: bool = True,
        feature_renorm: str = "none",  # {none, match_input_stats, match_input_std}
        renorm_detach_stats: bool = True,
        renorm_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.residual = residual
        if feature_renorm not in {"none", "match_input_stats", "match_input_std"}:
            raise ValueError(f"Unsupported feature_renorm: {feature_renorm}")
        self.renorm = (
            FeatureReNormalization2d(
                mode=feature_renorm,
                detach_stats=renorm_detach_stats,
                alpha=renorm_alpha,
            )
            if feature_renorm != "none"
            else None
        )
        self.to_gamma_beta = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim * 2),
        )

    def forward(self, visual_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.to_gamma_beta(text_feat)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        if self.residual:
            out = visual_feat * (1.0 + gamma) + beta
        else:
            out = visual_feat * gamma + beta
        if self.renorm is not None:
            out = self.renorm(out, visual_feat)
        return out


class RegionTextAlignmentLoss(nn.Module):
    """Lightweight region-level alignment between mask-pooled visual features and text features."""

    def __init__(self, visual_dim: int, text_dim: int, proj_dim: int = 256, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def masked_pool(self, feat_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = F.interpolate(mask.float(), size=feat_map.shape[-2:], mode="nearest")
        mask_sum = mask.flatten(2).sum(dim=-1).clamp_min(self.eps)
        pooled = (feat_map * mask).flatten(2).sum(dim=-1) / mask_sum
        return pooled

    def forward(self, feat_map: torch.Tensor, text_feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        region_feat = self.masked_pool(feat_map, mask)
        region_feat = self.visual_proj(region_feat)
        text_feat = self.text_proj(text_feat)
        region_feat = F.normalize(region_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)
        return (1.0 - (region_feat * text_feat).sum(dim=-1)).mean()


class TextCrossAttention(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.text_to_visual = nn.Linear(text_dim, visual_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=visual_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(text_dim, visual_dim),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(visual_dim, visual_dim),
            nn.Dropout(proj_dropout),
        )

    def forward(self, visual_feat: torch.Tensor, text_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = visual_feat.shape
        visual_tokens = visual_feat.flatten(2).transpose(1, 2)
        visual_tokens = self.visual_norm(visual_tokens)

        text_tokens = self.text_norm(text_tokens)
        text_tokens = self.text_to_visual(text_tokens)

        attn_out, attn_weights = self.attn(
            query=visual_tokens,
            key=text_tokens,
            value=text_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        gate = self.gate(text_tokens.mean(dim=1)).unsqueeze(1)
        attn_out = self.out_proj(attn_out) * gate
        fused = visual_tokens + attn_out
        fused = fused.transpose(1, 2).reshape(b, c, h, w)
        return fused, attn_weights


class LSCM(nn.Module):
    """Language-driven Semantic Calibration Module (LSCM)."""

    def __init__(
        self,
        *,
        encoder_channels: Sequence[int],
        bottleneck_channels: int,
        clip_model_name: str,
        freeze_text: bool,
        text_hidden_dim: int,
        alignment_proj_dim: int,
        condition_bottleneck: bool,
        condition_decoder: bool,
        download_root: Optional[str],
        text_spatial_mode: str,
        cross_attn_heads: int,
        cross_attn_dropout: float,
        cross_attn_proj_dropout: float,
        multi_scale_fusion: bool,
        use_decoder_text_adapter: bool,
        feature_renorm: str,
        renorm_detach_stats: bool,
        renorm_alpha: float,
    ) -> None:
        super().__init__()
        if len(encoder_channels) != 4:
            raise ValueError("encoder_channels must contain exactly 4 levels for this implementation.")

        c1, c2, c3, c4 = encoder_channels
        self.text_encoder = CLIPTextEncoder(
            clip_model_name=clip_model_name,
            freeze=freeze_text,
            download_root=download_root,
        )
        text_dim = self.text_encoder.text_dim
        self.text_dim = int(text_dim)
        self.text_context_length = int(getattr(self.text_encoder.clip_model, "context_length", 77))

        self._multi_scale_fusion = bool(multi_scale_fusion)
        self._condition_bottleneck = bool(condition_bottleneck)
        self._condition_decoder = bool(condition_decoder)
        self._use_decoder_text_adapter = bool(use_decoder_text_adapter)

        cond_text_dim_b = text_dim
        cond_text_dim4 = text_dim
        cond_text_dim3 = text_dim
        cond_text_dim2 = text_dim
        cond_text_dim1 = text_dim

        self.bottleneck_condition = (
            TextConditioning(
                text_dim=cond_text_dim_b,
                feature_dim=bottleneck_channels,
                hidden_dim=text_hidden_dim,
                feature_renorm=feature_renorm,
                renorm_detach_stats=renorm_detach_stats,
                renorm_alpha=renorm_alpha,
            )
            if self._condition_bottleneck
            else None
        )

        self.dec4_condition = (
            TextConditioning(
                text_dim=cond_text_dim4,
                feature_dim=c4,
                hidden_dim=text_hidden_dim,
                feature_renorm=feature_renorm,
                renorm_detach_stats=renorm_detach_stats,
                renorm_alpha=renorm_alpha,
            )
            if self._condition_decoder
            else None
        )
        self.dec3_condition = (
            TextConditioning(
                text_dim=cond_text_dim3,
                feature_dim=c3,
                hidden_dim=text_hidden_dim,
                feature_renorm=feature_renorm,
                renorm_detach_stats=renorm_detach_stats,
                renorm_alpha=renorm_alpha,
            )
            if self._condition_decoder
            else None
        )
        self.dec2_condition = (
            TextConditioning(
                text_dim=cond_text_dim2,
                feature_dim=c2,
                hidden_dim=text_hidden_dim,
                feature_renorm=feature_renorm,
                renorm_detach_stats=renorm_detach_stats,
                renorm_alpha=renorm_alpha,
            )
            if self._condition_decoder
            else None
        )

        self.skip4_condition = (
            TextConditioning(
                text_dim=cond_text_dim4,
                feature_dim=c4,
                hidden_dim=text_hidden_dim,
                feature_renorm=feature_renorm,
                renorm_detach_stats=renorm_detach_stats,
                renorm_alpha=renorm_alpha,
            )
            if self._multi_scale_fusion
            else None
        )
        self.skip3_condition = (
            TextConditioning(
                text_dim=cond_text_dim3,
                feature_dim=c3,
                hidden_dim=text_hidden_dim,
                feature_renorm=feature_renorm,
                renorm_detach_stats=renorm_detach_stats,
                renorm_alpha=renorm_alpha,
            )
            if self._multi_scale_fusion
            else None
        )
        self.skip2_condition = (
            TextConditioning(
                text_dim=cond_text_dim2,
                feature_dim=c2,
                hidden_dim=text_hidden_dim,
                feature_renorm=feature_renorm,
                renorm_detach_stats=renorm_detach_stats,
                renorm_alpha=renorm_alpha,
            )
            if self._multi_scale_fusion
            else None
        )
        self.skip1_condition = (
            TextConditioning(
                text_dim=cond_text_dim1,
                feature_dim=c1,
                hidden_dim=text_hidden_dim,
                feature_renorm=feature_renorm,
                renorm_detach_stats=renorm_detach_stats,
                renorm_alpha=renorm_alpha,
            )
            if self._multi_scale_fusion
            else None
        )
        self.dec1_condition = (
            TextConditioning(
                text_dim=cond_text_dim1,
                feature_dim=c1,
                hidden_dim=text_hidden_dim,
                feature_renorm=feature_renorm,
                renorm_detach_stats=renorm_detach_stats,
                renorm_alpha=renorm_alpha,
            )
            if (self._multi_scale_fusion and self._condition_decoder)
            else None
        )

        self.dec4_text_adapter = None
        self.dec3_text_adapter = None
        self.dec2_text_adapter = None
        if self._use_decoder_text_adapter:
            self.dec4_text_adapter = DecoderTextAdapter(channels=c4, text_dim=cond_text_dim4)
            self.dec3_text_adapter = DecoderTextAdapter(channels=c3, text_dim=cond_text_dim3)
            self.dec2_text_adapter = DecoderTextAdapter(channels=c2, text_dim=cond_text_dim2)

        self.region_alignment = RegionTextAlignmentLoss(
            visual_dim=bottleneck_channels,
            text_dim=text_dim,
            proj_dim=alignment_proj_dim,
        )

        self.text_spatial_mode = str(text_spatial_mode)
        self.cross_attn = None
        if self.text_spatial_mode == "cross_attention":
            self.cross_attn = TextCrossAttention(
                visual_dim=bottleneck_channels,
                text_dim=text_dim,
                num_heads=cross_attn_heads,
                attn_dropout=cross_attn_dropout,
                proj_dropout=cross_attn_proj_dropout,
            )

    def __repr__(self) -> str:
        # Hide internal composition to keep a clean “paper-level” presentation.
        return f"{self.__class__.__name__}()"

    def _prepare_prompts(self, prompts: List[str], batch_size: int) -> List[str]:
        if len(prompts) == batch_size:
            return prompts
        if len(prompts) == 1:
            return prompts * batch_size
        raise ValueError(f"Expected 1 prompt or {batch_size} prompts, but got {len(prompts)}.")

    def _build_nonsemantic_text_features(
        self,
        batch_size: int,
        device: torch.device,
        mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mode == "none":
            text_feat = torch.zeros(batch_size, self.text_dim, device=device, dtype=torch.float32)
            text_token_feat = torch.zeros(
                batch_size,
                self.text_context_length,
                self.text_dim,
                device=device,
                dtype=torch.float32,
            )
            return text_feat, text_token_feat

        # mode == "random"
        text_feat = torch.randn(batch_size, self.text_dim, device=device, dtype=torch.float32)
        text_feat = F.normalize(text_feat, dim=-1)
        text_token_feat = torch.randn(
            batch_size,
            self.text_context_length,
            self.text_dim,
            device=device,
            dtype=torch.float32,
        )
        text_token_feat = F.normalize(text_token_feat, dim=-1)
        return text_feat, text_token_feat

    def encode(
        self,
        *,
        prompts: Optional[List[str]],
        text_input_mode: str,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mode = str(text_input_mode).strip().lower().replace("-", "_")
        if mode in {"raw", "shuffle_text", "cluster_center", "raw_plus_center", "cluster_topk", "raw_plus_topk"}:
            prompts = prompts or ["pore"]
            prompts = self._prepare_prompts(prompts, batch_size)
            text_feat, text_token_feat = self.text_encoder.encode_tokens(prompts, device)
        elif mode in {"random", "none"}:
            text_feat, text_token_feat = self._build_nonsemantic_text_features(batch_size, device, mode)
        else:
            raise ValueError(
                "Unsupported text_input_mode: "
                f"{text_input_mode}. Expected one of raw/shuffle_text/cluster_center/raw_plus_center/cluster_topk/raw_plus_topk/random/none"
            )

        # current implementation: all scales share the same global text feature
        text_b = text_feat
        text_s4 = text_feat
        text_s3 = text_feat
        text_s2 = text_feat
        text_s1 = text_feat
        return text_feat, text_token_feat, text_b, text_s4, text_s3, text_s2, text_s1

    def calibrate_skips(
        self,
        *,
        skip1: torch.Tensor,
        skip2: torch.Tensor,
        skip3: torch.Tensor,
        skip4: torch.Tensor,
        text_s1: torch.Tensor,
        text_s2: torch.Tensor,
        text_s3: torch.Tensor,
        text_s4: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.skip1_condition is not None:
            skip1 = self.skip1_condition(skip1, text_s1)
        if self.skip2_condition is not None:
            skip2 = self.skip2_condition(skip2, text_s2)
        if self.skip3_condition is not None:
            skip3 = self.skip3_condition(skip3, text_s3)
        if self.skip4_condition is not None:
            skip4 = self.skip4_condition(skip4, text_s4)
        return skip1, skip2, skip3, skip4

    def calibrate_bottleneck(self, x: torch.Tensor, text_b: torch.Tensor) -> torch.Tensor:
        if self.bottleneck_condition is not None:
            x = self.bottleneck_condition(x, text_b)
        return x

    def cross_modal_interaction(
        self, x: torch.Tensor, text_token_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cross_attn_weights = None
        if self.cross_attn is not None:
            x, cross_attn_weights = self.cross_attn(x, text_token_feat)
        return x, cross_attn_weights

    def calibrate_dec4(self, x: torch.Tensor, text_s4: torch.Tensor) -> torch.Tensor:
        if self.dec4_condition is not None:
            x = self.dec4_condition(x, text_s4)
        if self.dec4_text_adapter is not None:
            x = self.dec4_text_adapter(x, text_s4)
        return x

    def calibrate_dec3(self, x: torch.Tensor, text_s3: torch.Tensor) -> torch.Tensor:
        if self.dec3_condition is not None:
            x = self.dec3_condition(x, text_s3)
        if self.dec3_text_adapter is not None:
            x = self.dec3_text_adapter(x, text_s3)
        return x

    def calibrate_dec2(self, x: torch.Tensor, text_s2: torch.Tensor) -> torch.Tensor:
        if self.dec2_condition is not None:
            x = self.dec2_condition(x, text_s2)
        if self.dec2_text_adapter is not None:
            x = self.dec2_text_adapter(x, text_s2)
        return x

    def calibrate_dec1(self, x: torch.Tensor, text_s1: torch.Tensor) -> torch.Tensor:
        if self.dec1_condition is not None:
            x = self.dec1_condition(x, text_s1)
        return x

    def region_alignment_loss(
        self,
        *,
        logits: torch.Tensor,
        aligned_feat: torch.Tensor,
        text_feat: torch.Tensor,
        region_mask: torch.Tensor,
        region_mask_mode: str,
        region_mask_blend_alpha: float,
        region_use_hard_pred: bool,
    ) -> torch.Tensor:
        pred_mask = torch.sigmoid(logits)
        if region_use_hard_pred:
            pred_mask = (pred_mask >= 0.5).float()
        blend_alpha = float(max(0.0, min(1.0, region_mask_blend_alpha)))
        if region_mask_mode == "pred":
            pooled_mask = pred_mask
        elif region_mask_mode == "blend":
            pooled_mask = (1.0 - blend_alpha) * region_mask.float() + blend_alpha * pred_mask
        else:
            pooled_mask = region_mask.float()
        return self.region_alignment(aligned_feat, text_feat, pooled_mask)


class ASRM(nn.Module):
    """Anisotropic Structure Refinement Module (ASRM)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        encoder_channels: Sequence[int],
        bottleneck_channels: int,
        use_skip_attention: bool,
        skip_attention_reduction: int,
        use_directional_refine: bool,
        directional_kernel_size: int,
        directional_max_res_scale: float,
        directional_alpha_init: float,
        directional_extend_scope: float,
        use_bottleneck_context: bool,
        bottleneck_context_max_res_scale: float,
        bottleneck_context_beta_init: float,
        use_decoder_directional_refine: bool,
    ) -> None:
        super().__init__()
        if len(encoder_channels) != 4:
            raise ValueError("encoder_channels must contain exactly 4 levels for this implementation.")

        c1, c2, c3, c4 = encoder_channels
        self.enc1 = EncoderBlock(in_channels, c1)
        self.enc2 = EncoderBlock(c1, c2)
        self.enc3 = EncoderBlock(c2, c3)
        self.enc4 = EncoderBlock(c3, c4)

        self.bottleneck = ConvBNReLU(c4, bottleneck_channels)

        self.dec4 = DecoderBlock(
            bottleneck_channels,
            c4,
            c4,
            use_skip_attention=use_skip_attention,
            skip_attention_reduction=skip_attention_reduction,
        )
        self.dec3 = DecoderBlock(
            c4,
            c3,
            c3,
            use_skip_attention=use_skip_attention,
            skip_attention_reduction=skip_attention_reduction,
        )
        self.dec2 = DecoderBlock(
            c3,
            c2,
            c2,
            use_skip_attention=use_skip_attention,
            skip_attention_reduction=skip_attention_reduction,
        )
        self.dec1 = DecoderBlock(
            c2,
            c1,
            c1,
            use_skip_attention=use_skip_attention,
            skip_attention_reduction=skip_attention_reduction,
        )
        self.seg_head = nn.Sequential(
            ConvBNReLU(c1, c1),
            nn.Conv2d(c1, num_classes, kernel_size=1),
        )

        self.refine_skip4 = nn.Identity()
        self.refine_bottleneck = nn.Identity()
        self.dec4_refine = nn.Identity()
        self.bottleneck_context = nn.Identity()
        if bool(use_directional_refine):
            self.refine_skip4 = TriBranchDirectionalRefine(
                channels=c4,
                kernel_size=directional_kernel_size,
                max_res_scale=directional_max_res_scale,
                alpha_init=directional_alpha_init,
                extend_scope=directional_extend_scope,
            )
            self.refine_bottleneck = TriBranchDirectionalRefine(
                channels=bottleneck_channels,
                kernel_size=directional_kernel_size,
                max_res_scale=directional_max_res_scale,
                alpha_init=directional_alpha_init,
                extend_scope=directional_extend_scope,
            )
        if bool(use_bottleneck_context):
            self.bottleneck_context = MultiScaleContext(
                channels=bottleneck_channels,
                max_res_scale=bottleneck_context_max_res_scale,
                beta_init=bottleneck_context_beta_init,
            )
        if bool(use_decoder_directional_refine):
            self.dec4_refine = TriBranchDirectionalRefine(
                channels=c4,
                kernel_size=directional_kernel_size,
                max_res_scale=directional_max_res_scale,
                alpha_init=directional_alpha_init,
                extend_scope=directional_extend_scope,
            )

    def __repr__(self) -> str:
        # Hide internal composition to keep a clean “paper-level” presentation.
        return f"{self.__class__.__name__}()"

    def forward(
        self,
        x: torch.Tensor,
        *,
        lscm: LSCM,
        text_feat: torch.Tensor,
        text_token_feat: torch.Tensor,
        text_b: torch.Tensor,
        text_s4: torch.Tensor,
        text_s3: torch.Tensor,
        text_s2: torch.Tensor,
        text_s1: torch.Tensor,
        region_mask: Optional[torch.Tensor],
        region_mask_mode: str,
        region_mask_blend_alpha: float,
        region_use_hard_pred: bool,
        return_aux: bool,
        return_attention: bool,
        return_features: bool,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)
        skip4 = self.refine_skip4(skip4)

        skip1, skip2, skip3, skip4 = lscm.calibrate_skips(
            skip1=skip1,
            skip2=skip2,
            skip3=skip3,
            skip4=skip4,
            text_s1=text_s1,
            text_s2=text_s2,
            text_s3=text_s3,
            text_s4=text_s4,
        )

        x = self.bottleneck(x)
        x = lscm.calibrate_bottleneck(x, text_b)

        x = self.refine_bottleneck(x)
        x = self.bottleneck_context(x)

        x, cross_attn_weights = lscm.cross_modal_interaction(x, text_token_feat)
        aligned_feat = x

        x = self.dec4(x, skip4)
        x = self.dec4_refine(x)
        x = lscm.calibrate_dec4(x, text_s4)

        x = self.dec3(x, skip3)
        x = lscm.calibrate_dec3(x, text_s3)

        x = self.dec2(x, skip2)
        x = lscm.calibrate_dec2(x, text_s2)

        x = self.dec1(x, skip1)
        x = lscm.calibrate_dec1(x, text_s1)

        decoder_feat = x
        logits = self.seg_head(x)

        if return_aux or return_attention or return_features:
            aux: dict[str, torch.Tensor] = {"logits": logits}
            if region_mask is not None:
                aux["alignment_loss"] = lscm.region_alignment_loss(
                    logits=logits,
                    aligned_feat=aligned_feat,
                    text_feat=text_feat,
                    region_mask=region_mask,
                    region_mask_mode=region_mask_mode,
                    region_mask_blend_alpha=region_mask_blend_alpha,
                    region_use_hard_pred=region_use_hard_pred,
                )
            if cross_attn_weights is not None:
                aux["cross_attn_weights"] = cross_attn_weights
            if return_features:
                aux["text_feat"] = text_feat.detach()
                aux["text_skip1_feat"] = text_s1.detach()
                aux["bottleneck_feat"] = aligned_feat.detach()
                aux["decoder_feat"] = decoder_feat.detach()
            return aux
        return logits


class VLSENet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        encoder_channels: Sequence[int] = (64, 128, 256, 512),
        bottleneck_channels: int = 1024,
        clip_model_name: str = "RN50",
        freeze_text: bool = True,
        text_hidden_dim: int = 256,
        alignment_proj_dim: int = 256,
        condition_bottleneck: bool = True,
        condition_decoder: bool = True,
        download_root: Optional[str] = None,
        text_spatial_mode: str = "cross_attention",  # {none, cross_attention}
        cross_attn_heads: int = 8,
        cross_attn_dropout: float = 0.0,
        cross_attn_proj_dropout: float = 0.0,
        use_skip_attention: bool = True,
        skip_attention_reduction: int = 8,
        multi_scale_fusion: bool = True,
        use_decoder_text_adapter: bool = True,
        use_directional_refine: bool = True,
        directional_kernel_size: int = 5,
        directional_max_res_scale: float = 0.30,
        directional_alpha_init: float = -2.0,
        directional_extend_scope: float = 1.0,
        use_bottleneck_context: bool = True,
        bottleneck_context_max_res_scale: float = 0.15,
        bottleneck_context_beta_init: float = -2.2,
        use_decoder_directional_refine: bool = True,
        feature_renorm: str = "match_input_std",  # {none, match_input_stats, match_input_std}
        renorm_detach_stats: bool = True,
        renorm_alpha: float = 1.0,
    ) -> None:
        super().__init__()

        self.lscm = LSCM(
            encoder_channels=encoder_channels,
            bottleneck_channels=bottleneck_channels,
            clip_model_name=clip_model_name,
            freeze_text=freeze_text,
            text_hidden_dim=text_hidden_dim,
            alignment_proj_dim=alignment_proj_dim,
            condition_bottleneck=condition_bottleneck,
            condition_decoder=condition_decoder,
            download_root=download_root,
            text_spatial_mode=text_spatial_mode,
            cross_attn_heads=cross_attn_heads,
            cross_attn_dropout=cross_attn_dropout,
            cross_attn_proj_dropout=cross_attn_proj_dropout,
            multi_scale_fusion=multi_scale_fusion,
            use_decoder_text_adapter=use_decoder_text_adapter,
            feature_renorm=feature_renorm,
            renorm_detach_stats=renorm_detach_stats,
            renorm_alpha=renorm_alpha,
        )
        self.asrm = ASRM(
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_channels=encoder_channels,
            bottleneck_channels=bottleneck_channels,
            use_skip_attention=use_skip_attention,
            skip_attention_reduction=skip_attention_reduction,
            use_directional_refine=use_directional_refine,
            directional_kernel_size=directional_kernel_size,
            directional_max_res_scale=directional_max_res_scale,
            directional_alpha_init=directional_alpha_init,
            directional_extend_scope=directional_extend_scope,
            use_bottleneck_context=use_bottleneck_context,
            bottleneck_context_max_res_scale=bottleneck_context_max_res_scale,
            bottleneck_context_beta_init=bottleneck_context_beta_init,
            use_decoder_directional_refine=use_decoder_directional_refine,
        )

    def forward(
        self,
        x: torch.Tensor,
        prompts: Optional[List[str]] = None,
        text_input_mode: str = "raw",
        region_mask: Optional[torch.Tensor] = None,
        region_mask_mode: str = "gt",
        region_mask_blend_alpha: float = 0.5,
        region_use_hard_pred: bool = False,
        return_aux: bool = False,
        return_attention: bool = False,
        return_features: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        (
            text_feat,
            text_token_feat,
            text_b,
            text_s4,
            text_s3,
            text_s2,
            text_s1,
        ) = self.lscm.encode(
            prompts=prompts,
            text_input_mode=text_input_mode,
            batch_size=batch_size,
            device=x.device,
        )

        return self.asrm(
            x,
            lscm=self.lscm,
            text_feat=text_feat,
            text_token_feat=text_token_feat,
            text_b=text_b,
            text_s4=text_s4,
            text_s3=text_s3,
            text_s2=text_s2,
            text_s1=text_s1,
            region_mask=region_mask,
            region_mask_mode=region_mask_mode,
            region_mask_blend_alpha=region_mask_blend_alpha,
            region_use_hard_pred=region_use_hard_pred,
            return_aux=return_aux,
            return_attention=return_attention,
            return_features=return_features,
        )

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        # Backward compatibility: accept legacy checkpoints saved before introducing `lscm.` / `asrm.` prefixes.
        has_prefix = any(
            isinstance(k, str) and (k.startswith("lscm.") or k.startswith("asrm.")) for k in state_dict.keys()
        )
        if has_prefix:
            return super().load_state_dict(state_dict, strict=strict)

        lscm_prefixes = (
            "text_encoder",
            "bottleneck_condition",
            "dec4_condition",
            "dec3_condition",
            "dec2_condition",
            "dec1_condition",
            "skip4_condition",
            "skip3_condition",
            "skip2_condition",
            "skip1_condition",
            "dec4_text_adapter",
            "dec3_text_adapter",
            "dec2_text_adapter",
            "region_alignment",
            "cross_attn",
        )
        asrm_prefixes = (
            "enc1",
            "enc2",
            "enc3",
            "enc4",
            "bottleneck",
            "dec4",
            "dec3",
            "dec2",
            "dec1",
            "seg_head",
            "refine_skip4",
            "refine_bottleneck",
            "dec4_refine",
            "bottleneck_context",
        )
        remapped: dict = {}
        for k, v in state_dict.items():
            if isinstance(k, str) and k.startswith(lscm_prefixes):
                remapped[f"lscm.{k}"] = v
            elif isinstance(k, str) and k.startswith(asrm_prefixes):
                remapped[f"asrm.{k}"] = v
            else:
                remapped[k] = v
        return super().load_state_dict(remapped, strict=strict)

class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        probs = probs.flatten(1)
        targets = targets.flatten(1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        dice_loss = 1.0 - dice.mean()
        return bce + dice_loss

def example_training_step(device: Optional[torch.device] = None, image_size: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VLSENet(num_classes=1, freeze_text=True).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    images = torch.randn(2, 3, image_size, image_size, device=device)
    masks = torch.randint(0, 2, (2, 1, image_size, image_size), device=device).float()
    prompts = ["pore", "rock pore"]

    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(images, prompts, region_mask=masks, region_mask_mode="blend", region_mask_blend_alpha=0.2, return_aux=True)
    logits = outputs["logits"]
    loss = criterion(logits, masks) + 0.05 * outputs["alignment_loss"]
    loss.backward()
    optimizer.step()
    return logits.detach(), loss.detach()


def _quick_forward_check() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VLSENet(num_classes=1).to(device)
    model.eval()
    with torch.no_grad():
        images = torch.randn(2, 3, 224, 224, device=device)
        logits = model(images, ["pore", "rock pore"])
    print("Text-Guided U-Net forward pass ok. Logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    _quick_forward_check()
    _, loss = example_training_step(image_size=224)
    print("Text-Guided U-Net example training loss:", float(loss))
