"""
Jaguar Re-ID 模型模块

提供 ReID 模型：Backbone + GeM Pooling + BNNeck + Embedding Head
支持复用 gastrovision 的 model_factory 中所有 backbone。

架构:
    Image → Backbone(去掉分类头) → GeM Pool → BNNeck → Embedding(512-d)
                                                    ↘ ArcFace Head (训练时)

参考:
- Bag of Tricks for Re-ID (CVPR 2019 Workshop)
- BNNeck: 在 embedding 和分类头之间加 BatchNorm，分离度量学习和分类目标
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gastrovision.models.model_factory import MODEL_CONFIGS


class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling (GeM)

    比 AvgPool 更关注高激活区域，p=1 退化为 AvgPool，p→∞ 退化为 MaxPool。
    Re-ID 场景下通常 p=3.0 效果优于 AvgPool。

    参考: Fine-tuning CNN Image Retrieval with No Human Annotation (TPAMI 2019)
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            output_size=1
        ).pow(1.0 / self.p).flatten(1)


class ReIDModel(nn.Module):
    """
    Re-ID 模型

    结构: Backbone → GeM → FC(embedding_dim) → BNNeck → (ArcFace分类头)

    Args:
        backbone_name: backbone 名称（需在 MODEL_CONFIGS 中）
        embedding_dim: 嵌入维度
        pretrained: 是否使用 ImageNet 预训练
        use_gem: 是否使用 GeM Pooling（否则用 AvgPool）
        gem_p: GeM 的初始 p 值
        dropout: embedding FC 前的 dropout
    """

    def __init__(
        self,
        backbone_name: str = 'convnext_base',
        embedding_dim: int = 512,
        pretrained: bool = True,
        use_gem: bool = True,
        gem_p: float = 3.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # ---- Backbone ----
        if backbone_name not in MODEL_CONFIGS:
            raise ValueError(f"不支持的 backbone: {backbone_name}. 可选: {list(MODEL_CONFIGS.keys())}")

        model_fn, weights_enum, head_type = MODEL_CONFIGS[backbone_name]
        weights = weights_enum if pretrained else None
        backbone = model_fn(weights=weights)

        # 获取 backbone 输出维度并移除分类头
        self.backbone_dim = self._remove_head(backbone, head_type)
        self.backbone = backbone

        # ---- Pooling ----
        if use_gem:
            self.pool = GeMPooling(p=gem_p)
        else:
            self.pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1)
            )
        # 对于某些 backbone（ConvNeXt, EfficientNet, Swin）已经内置了 pooling，
        # 需要在 forward 中判断是否需要额外 pool
        self._head_type = head_type
        self._needs_pool = head_type == 'fc'  # ResNet 系列需要手动 pool

        # ---- Embedding Head ----
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.embedding = nn.Linear(self.backbone_dim, embedding_dim, bias=False)
        # BNNeck: 用于分离度量损失和分类损失的特征空间
        self.bn_neck = nn.BatchNorm1d(embedding_dim)
        self.bn_neck.bias.requires_grad_(False)  # BNNeck 标准做法

        self._init_weights()

    def _remove_head(self, backbone: nn.Module, head_type: str) -> int:
        """移除 backbone 的分类头，返回 backbone 输出维度"""
        if head_type == 'fc':
            # ResNet 系列
            dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            return dim
        elif head_type == 'classifier':
            # ConvNeXt / EfficientNet
            if hasattr(backbone.classifier, '__getitem__'):
                dim = backbone.classifier[-1].in_features
                backbone.classifier[-1] = nn.Identity()
            else:
                dim = backbone.classifier.in_features
                backbone.classifier = nn.Identity()
            return dim
        elif head_type == 'head':
            # Swin Transformer
            dim = backbone.head.in_features
            backbone.head = nn.Identity()
            return dim
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def _init_weights(self):
        """初始化 embedding 层权重"""
        nn.init.kaiming_normal_(self.embedding.weight, mode='fan_out')
        nn.init.constant_(self.bn_neck.weight, 1.0)
        nn.init.constant_(self.bn_neck.bias, 0.0)

    def forward(self, x: torch.Tensor, return_both: bool = False):
        """
        Args:
            x: (B, 3, H, W) 输入图像
            return_both: 如果 True，返回 (bn_feat, raw_feat)
                         bn_feat 用于 ArcFace 分类头
                         raw_feat 用于推理时的余弦相似度

        Returns:
            默认返回 bn_feat (B, embedding_dim)
            如果 return_both=True，返回 (bn_feat, raw_feat)
        """
        feat = self.backbone(x)

        # ResNet 系列输出 (B, C, H, W)，需要 pooling
        if self._needs_pool and feat.dim() == 4:
            feat = self.pool(feat)

        # 已经是 (B, backbone_dim) 了
        feat = feat.view(feat.size(0), -1)

        feat = self.dropout(feat)
        raw_feat = self.embedding(feat)  # (B, embedding_dim)
        bn_feat = self.bn_neck(raw_feat)

        if return_both:
            return bn_feat, raw_feat
        return bn_feat

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取 L2 归一化后的 embedding（用于推理）

        Args:
            x: (B, 3, H, W)

        Returns:
            (B, embedding_dim) L2 归一化 embedding
        """
        _, raw_feat = self.forward(x, return_both=True)
        return F.normalize(raw_feat, p=2, dim=1)


def build_model(cfg) -> ReIDModel:
    """从配置创建 ReID 模型"""
    model = ReIDModel(
        backbone_name=cfg.backbone,
        embedding_dim=cfg.embedding_dim,
        pretrained=cfg.pretrained,
        use_gem=cfg.use_gem,
        gem_p=cfg.gem_p,
        dropout=cfg.dropout,
    )
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  模型参数: {total:.2f}M, 可训练: {trainable:.2f}M")
    return model
