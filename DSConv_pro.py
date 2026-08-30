# -*- coding: utf-8 -*-
"""
Dynamic Snake Convolution Module (Pro Version)
An improved version of snake convolution based on DSCNet paper, implemented using grid_stample+einops.
Optimized: Automatically detects devices from input tensors without manually passing device parameters.
"""
import torch
from torch import nn
import einops


class DSConv_pro(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 9,
        extend_scope: float = 1.0,
        morph: int = 0,
        if_offset: bool = True,
    ):
        """
        Dynamic snake convolution

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            kernel_size: Snake convolution kernel size (default 9)
            extend_scope: Offset extension range (default 1.0)
            morph: convolution kernel morphology, 0=along x-axis, 1=along y-axis
            if_offset: Whether to enable deformation, False indicates a standard convolution kernel
        """
        super().__init__()

        if morph not in (0, 1):
            raise ValueError("morph should be 0 or 1.")

        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset

        self.gn_offset = nn.GroupNorm(kernel_size, 2 * kernel_size)
        self.gn = nn.GroupNorm(out_channels // 4, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size, 3, padding=1)

        self.dsc_conv_x = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

    def forward(self, input: torch.Tensor):
        # Predict offset map, range [-1,1]
        offset = self.offset_conv(input)
        offset = self.gn_offset(offset)
        offset = self.tanh(offset)

        # Serpentine deformable convolution
        y_coordinate_map, x_coordinate_map = get_coordinate_map_2D(
            offset=offset,
            morph=self.morph,
            extend_scope=self.extend_scope,
        )
        deformed_feature = get_interpolated_feature(
            input,
            y_coordinate_map,
            x_coordinate_map,
        )

        if self.morph == 0:
            output = self.dsc_conv_x(deformed_feature)
        elif self.morph == 1:
            output = self.dsc_conv_y(deformed_feature)

        output = self.gn(output)
        output = self.relu(output)

        return output


def get_coordinate_map_2D(
    offset: torch.Tensor,
    morph: int,
    extend_scope: float = 1.0,
):
    """
    Calculate 2D coordinate mapping based on the offset
    Args: 
        offset: The offset predicted by the network [B, 2*K, W, H] 
        morph: The shape of the convolution kernel, 0=along x-axis, 1=along y-axis 
        extend_scope: The range of offset expansion
    Returns: 
        y_coordinate_map: y-axis coordinate mapping [B, K_H * H, K_W * W] 
        x_coordinate_map: x-axis coordinate mapping [B, K_H * H, K_W * W]
    """
    if morph not in (0, 1):
        raise ValueError("morph should be 0 or 1.")

    batch_size, _, width, height = offset.shape
    kernel_size = offset.shape[1] // 2
    center = kernel_size // 2
    device = offset.device

    y_offset_, x_offset_ = torch.split(offset, kernel_size, dim=1)

    y_center_ = torch.arange(0, width, dtype=torch.float32, device=device)
    y_center_ = einops.repeat(y_center_, "w -> k w h", k=kernel_size, h=height)

    x_center_ = torch.arange(0, height, dtype=torch.float32, device=device)
    x_center_ = einops.repeat(x_center_, "h -> k w h", k=kernel_size, w=width)

    if morph == 0:
        y_spread_ = torch.zeros([kernel_size], device=device)
        x_spread_ = torch.linspace(-center, center, kernel_size, device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        y_offset_ = einops.rearrange(y_offset_, "b k w h -> k b w h")
        y_offset_new_ = y_offset_.detach().clone()

        y_offset_new_[center] = 0
        for index in range(1, center + 1):
            y_offset_new_[center + index] = (
                y_offset_new_[center + index - 1] + y_offset_[center + index]
            )
            y_offset_new_[center - index] = (
                y_offset_new_[center - index + 1] + y_offset_[center - index]
            )

        y_offset_new_ = einops.rearrange(y_offset_new_, "k b w h -> b k w h")
        y_new_ = y_new_.add(y_offset_new_.mul(extend_scope))

        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b (w k) h")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b (w k) h")

    elif morph == 1:
        y_spread_ = torch.linspace(-center, center, kernel_size, device=device)
        x_spread_ = torch.zeros([kernel_size], device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        x_offset_ = einops.rearrange(x_offset_, "b k w h -> k b w h")
        x_offset_new_ = x_offset_.detach().clone()

        x_offset_new_[center] = 0
        for index in range(1, center + 1):
            x_offset_new_[center + index] = (
                x_offset_new_[center + index - 1] + x_offset_[center + index]
            )
            x_offset_new_[center - index] = (
                x_offset_new_[center - index + 1] + x_offset_[center - index]
            )

        x_offset_new_ = einops.rearrange(x_offset_new_, "k b w h -> b k w h")
        x_new_ = x_new_.add(x_offset_new_.mul(extend_scope))

        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b w (h k)")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b w (h k)")

    return y_coordinate_map, x_coordinate_map


def get_interpolated_feature(
    input_feature: torch.Tensor,
    y_coordinate_map: torch.Tensor,
    x_coordinate_map: torch.Tensor,
    interpolate_mode: str = "bilinear",
):
    """
    Interpolate and sample the feature map based on coordinate mapping
    Args: 
        input_feature: feature map to be interpolated [B, C, H, W] 
        y_coordinate_map: y-axis coordinate mapping [B, K_H * H, K_W * W] 
        x_coordinate_map: x-axis coordinate mapping [B, K_H * H, K_W * W] 
        interpolate_mode: interpolation mode 'bilinear' or 'bicubic'
    Return: 
        interpolated_feature: interpolated feature map [B, C, K_H * H, K_W * W]
    """
    if interpolate_mode not in ("bilinear", "bicubic"):
        raise ValueError("interpolate_mode should be 'bilinear' or 'bicubic'.")

    y_max = input_feature.shape[-2] - 1
    x_max = input_feature.shape[-1] - 1

    y_coordinate_map_ = _coordinate_map_scaling(y_coordinate_map, origin=[0, y_max])
    x_coordinate_map_ = _coordinate_map_scaling(x_coordinate_map, origin=[0, x_max])

    y_coordinate_map_ = torch.unsqueeze(y_coordinate_map_, dim=-1)
    x_coordinate_map_ = torch.unsqueeze(x_coordinate_map_, dim=-1)

    # Grid shape: [B, H, W, 2], where the last dimension stores [x, y].
    grid = torch.cat([x_coordinate_map_, y_coordinate_map_], dim=-1)

    interpolated_feature = nn.functional.grid_sample(
        input=input_feature,
        grid=grid,
        mode=interpolate_mode,
        padding_mode="zeros",
        align_corners=True,
    )

    return interpolated_feature


def _coordinate_map_scaling(
    coordinate_map: torch.Tensor,
    origin: list,
    target: list = [-1, 1],
):
    """
    Scale the coordinate mapping from the original range to the target range (default [-1, 1]), adapting to grid_sample
    Args: 
        coordinate_map: The coordinate mapping to be scaled 
        origin: The original value range [min, max] 
        target: The target value range, default [-1, 1]
    Return: 
        coordinate_map_scaled: scaled coordinate mapping
    """
    min, max = origin
    a, b = target

    coordinate_map_scaled = torch.clamp(coordinate_map, min, max)

    scale_factor = (b - a) / (max - min)
    coordinate_map_scaled = a + scale_factor * (coordinate_map_scaled - min)

    return coordinate_map_scaled
