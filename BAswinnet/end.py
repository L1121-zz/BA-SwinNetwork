import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from mmseg.models.backbones import SwinTransformer


# 边界感知模块 - 增强小目标边界
class BoundaryAwareModule(nn.Module):
    """Boundary-aware module to enhance small object segmentation."""

    def __init__(self, in_channels, reduction=16):
        super(BoundaryAwareModule, self).__init__()
        # 边界检测分支
        self.edge_detect = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2, groups=in_channels)
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

        # 特征增强
        self.enhance = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        # 边界检测
        edge = self.edge_detect[0](x) - self.edge_detect[1](x)
        edge = torch.abs(edge)

        # 注意力权重
        weight = self.attention(edge)

        # 特征增强
        enhanced = self.enhance(x * weight)

        return enhanced + x


# ASPP模块 - 用于Stage0
class ASPPModule(BaseModule):
    """Atrous Spatial Pyramid Pooling Module."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 dilations=(1, 6, 12, 18),
                 init_cfg=None):
        super(ASPPModule, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilations = dilations

        self.aspp_modules = nn.ModuleList()

        # 1x1卷积
        self.aspp_modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))

        # 多个扩张卷积
        for dilation in dilations[1:]:
            self.aspp_modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        padding=dilation,
                        dilation=dilation,
                        bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)))

        # 全局平均池化
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

        # 特征融合
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                out_channels * (len(dilations) + 1),
                out_channels,
                3,
                padding=1,
                bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.size()[2:]

        aspp_outs = []
        for aspp_module in self.aspp_modules:
            aspp_outs.append(aspp_module(x))

        global_avg_pool = self.global_avg_pool(x)
        global_avg_pool = F.interpolate(
            global_avg_pool, size=size, mode='bilinear', align_corners=False)
        aspp_outs.append(global_avg_pool)

        aspp_outs = torch.cat(aspp_outs, dim=1)
        out = self.bottleneck(aspp_outs)

        return out


# DropPath for ConvNeXt
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ConvNeXt Block - 用于Stages 1-3
class EnhancedConvNeXtBlock(nn.Module):
    """Enhanced ConvNeXt Block with multi-scale processing."""

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super(EnhancedConvNeXtBlock, self).__init__()

        # 多尺度处理分支
        self.branch1 = nn.Conv2d(dim, dim // 4, kernel_size=3, padding=1, groups=dim // 4)
        self.branch2 = nn.Conv2d(dim, dim // 4, kernel_size=5, padding=2, groups=dim // 4)
        self.branch3 = nn.Conv2d(dim, dim // 4, kernel_size=7, padding=3, groups=dim // 4)
        self.branch4 = nn.Conv2d(dim, dim // 4, kernel_size=9, padding=4, groups=dim // 4)

        # 分支融合
        self.branch_fusion = nn.Conv2d(dim, dim, kernel_size=1)
        self.branch_norm = nn.BatchNorm2d(dim)

        # 原始ConvNeXt部分
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # 注意力加权
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 16, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        input_x = x

        # 多尺度处理
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        # 多尺度特征融合
        branches = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        branches = self.branch_fusion(branches)
        branches = self.branch_norm(branches)

        # 原始ConvNeXt处理流程
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        # 组合原始ConvNeXt和多尺度特征
        combined = x + branches

        # 应用注意力机制
        att_weight = self.attention(combined)
        combined = combined * att_weight

        # 残差连接
        return input_x + self.drop_path(combined)


# ConvNeXt Module - 用于Stages 1-3 (统一版本，都包含边界感知模块)
class EnhancedConvNeXtModule(BaseModule):
    """Enhanced ConvNeXt Module with multi-scale processing and boundary awareness."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 depths=3,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 init_cfg=None):
        super(EnhancedConvNeXtModule, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 输入映射
        self.input_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # ConvNeXt块
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]
        self.blocks = nn.Sequential(*[
            EnhancedConvNeXtBlock(
                dim=out_channels,
                drop_path=dpr[i],
                layer_scale_init_value=layer_scale_init_value)
            for i in range(depths)
        ])

        # 输出映射与规范化
        self.norm = nn.LayerNorm(out_channels, eps=1e-6)
        self.output_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # 边界感知模块 - 现在所有stages都有
        self.boundary_aware = BoundaryAwareModule(out_channels)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)

        # 应用边界感知
        x = self.boundary_aware(x)

        # 应用规范化
        x_norm = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x_norm = self.norm(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = self.output_proj(x_norm)

        return x


@MODELS.register_module()
class SwinEnd(BaseModule):
    """Swin Transformer without hierarchical feature fusion and global context blocks - for ablation study."""

    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained=None,
                 frozen_stages=-1,
                 init_cfg=None,
                 aspp_dilations=(1, 6, 12, 18),
                 aspp_out_channels=256,
                 mask2former_in_channels=256,
                 convnext_depths=3,  # 为ConvNeXt添加的参数
                 layer_scale_init_value=1e-6,  # 为ConvNeXt添加的参数
                 spatial_attn_kernel=7,  # 空间注意力核大小
                 **kwargs):
        super(SwinEnd, self).__init__(init_cfg)

        # 创建Swin Transformer编码器
        # 从kwargs中明确移除SwinTransformer不接受的参数
        swin_valid_keys = [
            'pretrain_img_size', 'in_channels', 'embed_dims', 'patch_size',
            'window_size', 'mlp_ratio', 'depths', 'num_heads', 'strides',
            'out_indices', 'qkv_bias', 'qk_scale', 'patch_norm', 'drop_rate',
            'attn_drop_rate', 'drop_path_rate', 'use_abs_pos_embed', 'act_cfg',
            'norm_cfg', 'with_cp', 'pretrained', 'frozen_stages', 'init_cfg'
        ]

        # 创建一个只包含SwinTransformer接受的参数的字典
        swin_kwargs = {k: v for k, v in kwargs.items() if k in swin_valid_keys}

        # 添加明确传递的参数（覆盖kwargs中的同名参数）
        explicit_args = {
            'pretrain_img_size': pretrain_img_size,
            'in_channels': in_channels,
            'embed_dims': embed_dims,
            'patch_size': patch_size,
            'window_size': window_size,
            'mlp_ratio': mlp_ratio,
            'depths': depths,
            'num_heads': num_heads,
            'strides': strides,
            'out_indices': out_indices,
            'qkv_bias': qkv_bias,
            'qk_scale': qk_scale,
            'patch_norm': patch_norm,
            'drop_rate': drop_rate,
            'attn_drop_rate': attn_drop_rate,
            'drop_path_rate': drop_path_rate,
            'use_abs_pos_embed': use_abs_pos_embed,
            'act_cfg': act_cfg,
            'norm_cfg': norm_cfg,
            'with_cp': with_cp,
            'pretrained': pretrained,
            'frozen_stages': frozen_stages,
            'init_cfg': init_cfg
        }

        # 只添加不为None的参数
        swin_kwargs.update({k: v for k, v in explicit_args.items() if v is not None})

        # 创建Swin Transformer编码器
        self.backbone = SwinTransformer(**swin_kwargs)

        # 获取Swin Transformer每个阶段的输出通道数
        self.out_channels = [embed_dims * 2 ** i for i in range(len(depths))]

        # 为stage0添加ASPP模块，为stages 1-3添加EnhancedConvNeXt模块
        self.feature_modules = nn.ModuleList()
        for i, channels in enumerate(self.out_channels):
            if i == 0:  # stage0 使用 ASPP
                self.feature_modules.append(
                    ASPPModule(
                        in_channels=channels,
                        out_channels=aspp_out_channels,
                        dilations=aspp_dilations))
            else:  # stages 1-3 都使用带边界感知的 EnhancedConvNeXtModule
                self.feature_modules.append(
                    EnhancedConvNeXtModule(
                        in_channels=channels,
                        out_channels=aspp_out_channels,
                        depths=convnext_depths,
                        drop_path_rate=drop_path_rate * 0.5,
                        layer_scale_init_value=layer_scale_init_value))


    def forward(self, x):
        # Swin Transformer 提取特征
        backbone_outs = self.backbone(x)

        # 直接应用特征模块（stage0用ASPP，stages 1-3用EnhancedConvNeXt+BA）
        feature_outs = []
        for i, feat in enumerate(backbone_outs):
            feature_outs.append(self.feature_modules[i](feat))
        return feature_outs