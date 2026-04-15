import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import digamma

class UncertaintyEstimator(nn.Module):
    """计算特征的不确定性（基于证据熵）"""
    def __init__(self, in_dim, num_evidences=3):
        super().__init__()
        self.evidence_head = nn.Sequential(
            nn.Conv2d(in_dim, num_evidences * in_dim, kernel_size=1),
            nn.Softplus()  # 确保α > 0
        )
        self.num_evidences = num_evidences
        self.in_dim = in_dim

    def forward(self, x):
        B, C, H, W = x.shape
        alpha = self.evidence_head(x)  # [B, num_evidences*C, H, W]
        alpha = alpha.view(B, self.num_evidences, C, H, W) + 1.0  # α ≥ 1
        S = torch.sum(alpha, dim=1, keepdim=True)  # [B, 1, C, H, W]
        digamma_S = digamma(S)
        digamma_alpha = digamma(alpha)
        sum_digamma_alpha = torch.sum(digamma_alpha, dim=1, keepdim=True)
        uncertainty = digamma_S - (sum_digamma_alpha / self.num_evidences)
        uncertainty = torch.sigmoid(uncertainty).squeeze(1)  # [B, C, H, W]
        return uncertainty


class ChannelModule(nn.Module):
    """通道注意力模块 + 不确定性估计"""
    def __init__(self, in_dim, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_att = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_dim // reduction, in_dim, 1, bias=False),
            nn.Sigmoid()
        )
        self.uncertainty_estimator = UncertaintyEstimator(in_dim)

    def forward(self, x):
        att = self.channel_att(self.avg_pool(x))  # [B, C, 1, 1]
        channel_feat = x * att  # 通道加权
        uncertainty = self.uncertainty_estimator(channel_feat)  # 不确定性估计
        return channel_feat, uncertainty


class SpatialModule(nn.Module):
    """空间注意力模块 + 不确定性估计"""
    def __init__(self, in_dim):
        super().__init__()
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_dim, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.uncertainty_estimator = UncertaintyEstimator(in_dim)

    def forward(self, x):
        att = self.spatial_att(x)  # [B, 1, H, W]
        spatial_feat = x * att  # 空间加权
        uncertainty = self.uncertainty_estimator(spatial_feat)  # 不确定性估计
        return spatial_feat, uncertainty


class CrossScaleModule(nn.Module):
    """跨尺度融合模块 + 不确定性估计"""
    def __init__(self, in_dim, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.convs = nn.ModuleList([
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1)
            for _ in scales
        ])
        self.uncertainty_estimator = UncertaintyEstimator(in_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        scale_feats = []
        for s, conv in zip(self.scales, self.convs):
            scaled = F.interpolate(x, scale_factor=1/s, mode='bilinear', align_corners=False)
            scaled = conv(scaled)
            scaled = F.interpolate(scaled, size=(H, W), mode='bilinear', align_corners=False)
            scale_feats.append(scaled)
        cross_feat = torch.mean(torch.stack(scale_feats), dim=0)  # 跨尺度融合
        uncertainty = self.uncertainty_estimator(cross_feat)  # 不确定性估计
        return cross_feat, uncertainty


class UncertaintyGating(nn.Module):
    """基于不确定性的门控机制（内部完成加权，不输出权重）"""
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, modules_outputs):
        feats = [x[0] for x in modules_outputs]
        uncertainties = [x[1] for x in modules_outputs]
        # 计算权重并融合（完全在内部完成）
        weights = [torch.exp(-self.beta * u) for u in uncertainties]
        weights_sum = torch.sum(torch.stack(weights), dim=0)
        att_weights = [w / (weights_sum + 1e-8) for w in weights]
        fused_feat = torch.sum(torch.stack([a * f for a, f in zip(att_weights, feats)]), dim=0)
        return fused_feat  # 只返回融合特征，不输出权重列表


class UncertaintyAwareAttention(nn.Module):
    """不确定性注意力模块（单输入、单输出，内部完成所有加权）"""
    def __init__(self, in_dim):
        super().__init__()
        self.channel_module = ChannelModule(in_dim)
        self.spatial_module = SpatialModule(in_dim)
        self.crossscale_module = CrossScaleModule(in_dim)
        self.gating = UncertaintyGating(beta=2.0)  # 门控只输出融合特征
        self.residual_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)  # 残差连接

    def forward(self, x):
        # 输入：[B, C, H, W]
        residual = self.residual_conv(x)  # 残差特征
        # 多模块并行处理（特征+不确定性）
        channel_feat, channel_uncert = self.channel_module(x)
        spatial_feat, spatial_uncert = self.spatial_module(x)
        cross_feat, cross_uncert = self.crossscale_module(x)
        # 门控融合（内部完成不确定性加权）
        fused_feat = self.gating([
            (channel_feat, channel_uncert),
            (spatial_feat, spatial_uncert),
            (cross_feat, cross_uncert)
        ])
        # 残差相加，输出最终特征
        out = fused_feat + residual  # 输出：[B, C, H, W]（与输入维度完全一致）
        return out  # 仅返回一个输出