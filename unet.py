""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
import torch.nn.functional as F
from DSConv_pro import DSConv_pro

def make_gn(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class TriBranchDirectionalRefine(nn.Module):
    """Directional refine plugin: std 3x3 + horizontal/vertical DSConv branches."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        reduction: int = 4,
        max_res_scale: float = 0.30,
        alpha_init: float = -2.0,
        extend_scope: float = 1.0,
        use_adaptive_structural_fusion: bool = True,
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

        # Channel-wise residual scaling keeps perturbation bounded early in training.
        self.alpha = nn.Parameter(torch.full((1, channels, 1, 1), float(alpha_init), dtype=torch.float32))
        self.use_adaptive_structural_fusion = bool(use_adaptive_structural_fusion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        z = self.reduce(x)

        zs = self.std_branch(z)
        zh = self.branch_h(self.ds_h(z))
        zv = self.branch_v(self.ds_v(z))

        if self.use_adaptive_structural_fusion:
            gate_logits = self.gate(torch.cat([zs, zh, zv], dim=1))
            gate = torch.softmax(gate_logits, dim=1)
            z = gate[:, 0:1] * zs + gate[:, 1:2] * zh + gate[:, 2:3] * zv
        else:
            # w/o Adaptive Structural Fusion: keep 3 branches, use fixed uniform fusion.
            z = (zs + zh + zv) / 3.0
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


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        def _conv(in_ch: int, out_ch: int) -> nn.Module:
            return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)

        self.double_conv = nn.Sequential(
            _conv(in_channels, mid_channels),
            make_gn(mid_channels),
            nn.LeakyReLU(inplace=True),
            _conv(mid_channels, out_channels),
            make_gn(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class AttentionGatedSkip(nn.Module):
    """Lightweight channel + spatial gating for skip features."""

    def __init__(self, gate_channels: int, skip_channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, skip_channels // max(1, reduction))
        self.channel_gate = nn.Sequential(
            nn.Linear(gate_channels + skip_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, skip_channels),
        )
        self.spatial_gate = nn.Conv2d(gate_channels + skip_channels, 1, kernel_size=1, bias=True)

        # Residual gating (identity-init): weight = 1 + tanh(alpha) * delta.
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


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels,
        out_channels,
        bilinear=True,
        *,
        use_skip_attention: bool = False,
        skip_attention_reduction: int = 8
    ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            gate_channels = in_channels // 2
            skip_channels = in_channels // 2
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            gate_channels = in_channels // 2
            skip_channels = in_channels // 2

        self.skip_gate = (
            AttentionGatedSkip(gate_channels, skip_channels, reduction=skip_attention_reduction)
            if use_skip_attention
            else None
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        if self.skip_gate is not None:
            x2 = self.skip_gate(x1, x2)
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        bilinear=False,
        use_directional_refine: bool = False,
        use_adaptive_structural_fusion: bool = True,
        directional_kernel_size: int = 5,
        directional_max_res_scale: float = 0.30,
        directional_alpha_init: float = -2.0,
        directional_extend_scope: float = 1.0,
        use_bottleneck_context: bool = False,
        bottleneck_context_max_res_scale: float = 0.15,
        bottleneck_context_beta_init: float = -2.2,
        use_decoder_directional_refine: bool = False,
        decoder_directional_kernel_size: int = 3,
        use_skip_attention: bool = False,
        skip_attention_reduction: int = 8,
    ):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (
            Up(
                1024,
                512 // factor,
                bilinear,
                use_skip_attention=use_skip_attention,
                skip_attention_reduction=skip_attention_reduction
            )
        )
        self.up2 = (
            Up(
                512,
                256 // factor,
                bilinear,
                use_skip_attention=use_skip_attention,
                skip_attention_reduction=skip_attention_reduction
            )
        )
        self.up3 = (
            Up(
                256,
                128 // factor,
                bilinear,
                use_skip_attention=use_skip_attention,
                skip_attention_reduction=skip_attention_reduction
            )
        )
        self.up4 = (
            Up(
                128,
                64,
                bilinear,
                use_skip_attention=use_skip_attention,
                skip_attention_reduction=skip_attention_reduction
            )
        )
        self.outc = (OutConv(64, n_classes))

        self.refine_skip4 = nn.Identity()
        self.refine_bottleneck = nn.Identity()
        self.dec4_refine = nn.Identity()
        self.bottleneck_context = nn.Identity()

        if bool(use_directional_refine):
            self.refine_skip4 = TriBranchDirectionalRefine(
                channels=512,
                kernel_size=directional_kernel_size,
                max_res_scale=directional_max_res_scale,
                alpha_init=directional_alpha_init,
                extend_scope=directional_extend_scope,
                use_adaptive_structural_fusion=use_adaptive_structural_fusion,
            )
            self.refine_bottleneck = TriBranchDirectionalRefine(
                channels=1024 // factor,
                kernel_size=directional_kernel_size,
                max_res_scale=directional_max_res_scale,
                alpha_init=directional_alpha_init,
                extend_scope=directional_extend_scope,
                use_adaptive_structural_fusion=use_adaptive_structural_fusion,
            )
        if bool(use_bottleneck_context):
            self.bottleneck_context = MultiScaleContext(
                channels=1024 // factor,
                max_res_scale=bottleneck_context_max_res_scale,
                beta_init=bottleneck_context_beta_init,
            )
        if bool(use_decoder_directional_refine):
            # Keep decoder-side refine conservative, following V3 where deep decoder
            # enhancement is weaker than skip4/bottleneck to avoid over-perturbation.
            self.dec4_refine = TriBranchDirectionalRefine(
                channels=512 // factor,
                kernel_size=int(decoder_directional_kernel_size),
                max_res_scale=0.20,
                alpha_init=-2.2,
                extend_scope=directional_extend_scope,
                use_adaptive_structural_fusion=use_adaptive_structural_fusion,
            )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.refine_skip4(x4)
        x5 = self.down4(x4)
        x5 = self.refine_bottleneck(x5)
        x5 = self.bottleneck_context(x5)
        x = self.up1(x5, x4)
        x = self.dec4_refine(x)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
