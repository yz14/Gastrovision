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
- MLP Head: 两层非线性结构比单层 Linear 更强，业界最佳实践
- GeM Pooling: 通过 hook 机制对所有 backbone 生效（含 ConvNeXt/Swin）
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gastrovision.models.model_factory import MODEL_CONFIGS
from collections import OrderedDict


def _extract_state_dict(raw) -> dict:
    """
    从各种 checkpoint 格式中提取 state_dict。

    支持的格式:
      - 纯 state_dict (torchvision 缓存格式)
      - {'model_state_dict': ...}  (ReIDTrainer checkpoint)
      - {'state_dict': ...}        (通用 / timm / Lightning)
      - {'model': ...}             (Facebook ConvNeXt / Microsoft Swin 官方)
    """
    if not isinstance(raw, dict):
        return raw

    # 按优先级尝试已知键
    for key in ('model_state_dict', 'state_dict', 'model'):
        if key in raw:
            candidate = raw[key]
            if isinstance(candidate, (dict, OrderedDict)):
                return candidate

    # 没有已知键 → 检查是否本身就是 state_dict（值为 Tensor）
    # 排除只含 metadata 键（如 'epoch', 'optimizer'）的情况
    tensor_keys = [k for k, v in raw.items() if isinstance(v, torch.Tensor)]
    if tensor_keys:
        return raw

    # 无法识别的格式
    raise ValueError(
        f"无法识别的 checkpoint 格式，包含的键: {list(raw.keys())[:10]}。"
        f"预期包含 'model_state_dict' / 'state_dict' / 'model' 之一，"
        f"或直接为 state_dict (键为参数名，值为 Tensor)。"
    )


def _load_pretrained_backbone(
    backbone: nn.Module,
    state_dict: dict,
    pretrained_path: str,
) -> None:
    """
    将预训练 state_dict 健壮地加载到 backbone 中。

    处理逻辑:
      1. 自动检测并剥离常见键前缀 (backbone., module., model.)
      2. 逐键检查形状，过滤掉尺寸不匹配的键（避免 RuntimeError）
      3. 使用 strict=False 加载，仅加载形状兼容的键
      4. 打印详细的加载报告

    Args:
        backbone: 目标 backbone 模型（torchvision 实例）
        state_dict: 从文件中提取的 state_dict
        pretrained_path: 文件路径（仅用于日志）
    """
    model_dict = backbone.state_dict()

    # ---- 第一步: 尝试不同的前缀映射 ----
    # 计算每种前缀方案下能匹配的键数量，选最优的
    prefixes_to_try = ['', 'backbone.', 'module.', 'model.', 'module.backbone.']
    best_prefix = ''
    best_match_count = 0

    for prefix in prefixes_to_try:
        match_count = 0
        for k in state_dict.keys():
            if prefix and k.startswith(prefix):
                stripped = k[len(prefix):]
            elif not prefix:
                stripped = k
            else:
                continue
            if stripped in model_dict:
                match_count += 1
        if match_count > best_match_count:
            best_match_count = match_count
            best_prefix = prefix

    # ---- 第二步: 用最优前缀构建新 state_dict，过滤不兼容键 ----
    filtered_dict = OrderedDict()
    skipped_keys = []   # 形状不匹配
    matched_keys = []   # 成功匹配

    for k, v in state_dict.items():
        # 去前缀
        if best_prefix and k.startswith(best_prefix):
            new_k = k[len(best_prefix):]
        elif not best_prefix:
            new_k = k
        else:
            continue  # 这个键不以最优前缀开头，跳过

        if new_k not in model_dict:
            continue  # 模型中没有这个参数，跳过（strict=False 会处理）

        # 检查形状是否兼容
        if v.shape != model_dict[new_k].shape:
            skipped_keys.append(
                f"{new_k}: 预训练={list(v.shape)} vs 模型={list(model_dict[new_k].shape)}"
            )
            continue

        filtered_dict[new_k] = v
        matched_keys.append(new_k)

    # ---- 第三步: 加载并报告 ----
    missing, unexpected = backbone.load_state_dict(filtered_dict, strict=False)

    print(f"  加载本地预训练权重: {pretrained_path}")
    if best_prefix:
        print(f"  [提示] 自动去除键前缀: '{best_prefix}'")
    print(f"  [统计] 成功加载: {len(matched_keys)}/{len(model_dict)} 个参数")

    if skipped_keys:
        print(f"  [跳过] 尺寸不匹配的键 ({len(skipped_keys)} 个):")
        for s in skipped_keys[:5]:
            print(f"         {s}")
        if len(skipped_keys) > 5:
            print(f"         ... 及另外 {len(skipped_keys) - 5} 个")

    if len(matched_keys) == 0:
        print(f"  [警告] 没有任何参数被加载！请检查预训练权重是否与 backbone 匹配。")
        print(f"         预训练文件键示例: {list(state_dict.keys())[:5]}")
        print(f"         模型期望键示例: {list(model_dict.keys())[:5]}")


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


class MLPEmbeddingHead(nn.Module):
    """
    MLP Embedding Head: Linear → BN → ReLU → Linear

    相比单层 Linear 的优势：
    - 中间 BN 稳定梯度，减少对 backbone 特征分布的敏感性
    - ReLU 引入非线性，增强表达能力
    - 在 Re-ID 小数据集上经验上比单层 Linear 高 1~2% mAP

    参考: OSNet (ICCV 2019), FastReID (arXiv 2020)
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        hidden_dim = max(out_dim, in_dim // 2)
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out')
        nn.init.constant_(self.bn1.weight, 1.0)
        nn.init.constant_(self.bn1.bias, 0.0)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


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
        pretrained_path: str = '',
        use_gem: bool = True,
        gem_p: float = 3.0,
        dropout: float = 0.0,
        use_mlp_head: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self._use_gem = use_gem

        # ---- Backbone ----
        if backbone_name not in MODEL_CONFIGS:
            raise ValueError(f"不支持的 backbone: {backbone_name}. 可选: {list(MODEL_CONFIGS.keys())}")

        model_fn, weights_enum, head_type = MODEL_CONFIGS[backbone_name]

        # 加载预训练权重
        # 优先级: pretrained_path(本地文件) > torchvision 默认(网络/缓存) > 随机初始化
        if pretrained_path and os.path.isfile(pretrained_path):
            backbone = model_fn(weights=None)
            raw = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            state_dict = _extract_state_dict(raw)
            _load_pretrained_backbone(backbone, state_dict, pretrained_path)
        elif pretrained:
            weights = weights_enum
            backbone = model_fn(weights=weights)
            print(f"  加载 ImageNet 预训练权重: {weights}")
        else:
            backbone = model_fn(weights=None)
            print(f"  使用随机初始化权重")

        # ---- 获取 backbone 输出维度并移除分类头 ----
        self.backbone_dim = self._remove_head(backbone, head_type)
        self.backbone = backbone
        self._head_type = head_type

        # ---- Pooling ----
        # 统一用 hook 在内置 avgpool/flatten 之前截取 4D 特征图，
        # 这样 GeM 对所有 backbone（ResNet/ConvNeXt/EfficientNet）均生效。
        # Swin 因输出为序列格式不适合 GeM，自动退化为 AvgPool。
        self._gem_feat = None   # hook 捕获的特征图（每次 forward 刷新）
        self._gem_hook = None
        self._use_gem = use_gem
        self._needs_pool = head_type == 'fc'  # ResNet 系列需要手动 pool；其他 backbone 内置 pool

        if use_gem:
            self.pool = GeMPooling(p=gem_p)
            gem_registered = self._register_gem_hook(backbone, head_type)
            if not gem_registered:
                # Swin 等不支持 GeM 的 backbone 退化为 AvgPool
                self._use_gem = False
                self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        else:
            self.pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1)
            )

        # ---- Embedding Head ----
        # MLP Head (Linear→BN→ReLU→Linear) 比单层 Linear 更强
        if use_mlp_head:
            self.embedding = MLPEmbeddingHead(
                in_dim=self.backbone_dim,
                out_dim=embedding_dim,
                dropout=dropout,
            )
            print(f"  Embedding Head: MLP ({self.backbone_dim}→{max(embedding_dim, self.backbone_dim//2)}→{embedding_dim})")
        else:
            self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
            self.embedding = nn.Linear(self.backbone_dim, embedding_dim, bias=False)
            print(f"  Embedding Head: Linear ({self.backbone_dim}→{embedding_dim})")
        self._use_mlp_head = use_mlp_head

        # BNNeck: 用于分离度量损失和分类损失的特征空间
        self.bn_neck = nn.BatchNorm1d(embedding_dim)
        self.bn_neck.bias.requires_grad_(False)  # BNNeck 标准做法

        self._init_weights()

    def _remove_head(self, backbone: nn.Module, head_type: str) -> int:
        """移除 backbone 的分类头，返回 backbone 输出维度"""
        if head_type == 'fc':
            # ResNet 系列：移除 fc，保留 avgpool
            # avgpool 是否替换为 Identity 取决于 use_gem，在 _register_gem_hook 后决定
            dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            return dim
        elif head_type == 'classifier':
            # ConvNeXt / EfficientNet
            if hasattr(backbone.classifier, '__getitem__'):
                dim = backbone.classifier[-1].in_features
                backbone.classifier[-1] = nn.Identity()
                # 移除 EfficientNet 内置的 Dropout，避免与 MLPHead 重复
                for i, layer in enumerate(backbone.classifier):
                    if isinstance(layer, nn.Dropout):
                        backbone.classifier[i] = nn.Identity()
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

    def _register_gem_hook(self, backbone: nn.Module, head_type: str) -> bool:
        """
        注册 forward hook，截取最后一层卷积/特征块的 4D 输出 (B, C, H, W)，
        存入 self._gem_feat，供 forward() 中的 GeM Pooling 使用。

        - ResNet: hook layer4 输出 (B, C, H, W)，同时 avgpool 已替换为 Identity
        - ConvNeXt/EfficientNet: hook backbone.features 输出 (B, C, H, W)
        - Swin: 特征为序列格式，不适合 GeM → 返回 False，退化为 AvgPool

        Returns:
            True  = hook 注册成功，GeM 可用
            False = 不支持 GeM（Swin），需退化为 AvgPool
        """
        def _hook_fn(module, input, output):
            self._gem_feat = output

        if head_type == 'fc':
            # ResNet: hook layer4（最后卷积块），输出 (B, C, H, W)
            # 同时把 avgpool 替换为 Identity，避免 layer4 输出经过 avgpool 压缩
            if hasattr(backbone, 'layer4'):
                self._gem_hook = backbone.layer4.register_forward_hook(_hook_fn)
                backbone.avgpool = nn.Identity()  # GeM 替代 avgpool
                return True
            return False

        elif head_type == 'classifier':
            # ConvNeXt/EfficientNet: hook features 模块整体输出
            if hasattr(backbone, 'features'):
                self._gem_hook = backbone.features.register_forward_hook(_hook_fn)
                return True
            # Fallback: hook avgpool 输入
            if hasattr(backbone, 'avgpool'):
                def _pre_pool_hook(module, inp, output):
                    if inp[0].dim() == 4:
                        self._gem_feat = inp[0]
                self._gem_hook = backbone.avgpool.register_forward_hook(_pre_pool_hook)
                return True
            return False

        elif head_type == 'head':
            # Swin Transformer 特征为 (B, H*W, C) 序列，GeM 无意义
            print(f"  [提示] Swin Transformer 特征为序列格式，不支持 GeM，已自动切换为 AvgPool")
            return False

        return False

    def _init_weights(self):
        """初始化 BNNeck 权重（MLPHead 内部已自初始化）"""
        if not self._use_mlp_head:
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
        # 每次 forward 前清空 hook 捕获的特征
        self._gem_feat = None

        # backbone forward（同时触发 hook 捕获特征图）
        feat = self.backbone(x)

        if self._use_gem and self._gem_feat is not None:
            # hook 捕获了 4D 特征图，用 GeM pooling
            gem_input = self._gem_feat
            if gem_input.dim() == 4:
                feat = self.pool(gem_input)          # (B, C)
            else:
                # 非 4D 退化为直接使用 backbone 输出
                feat = feat.view(feat.size(0), -1)
        else:
            # 无 GeM hook 或不使用 GeM：backbone 输出可能是 4D 或 2D
            if feat.dim() == 4:
                # 备用 pool（use_gem=False 时的 AvgPool）
                feat = self.pool(feat)
            # feat.dim() == 2: 已经是 (B, C)，直接用

        # 统一形状为 (B, backbone_dim)
        feat = feat.view(feat.size(0), -1)

        # Embedding Head (MLP 或 Linear)
        if self._use_mlp_head:
            raw_feat = self.embedding(feat)  # MLPEmbeddingHead 内含 dropout
        else:
            feat = self.dropout(feat)
            raw_feat = self.embedding(feat)

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
        pretrained_path=getattr(cfg, 'pretrained_path', ''),
        use_gem=cfg.use_gem,
        gem_p=cfg.gem_p,
        dropout=cfg.dropout,
        use_mlp_head=getattr(cfg, 'use_mlp_head', True),
    )
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  模型参数: {total:.2f}M, 可训练: {trainable:.2f}M")
    return model
