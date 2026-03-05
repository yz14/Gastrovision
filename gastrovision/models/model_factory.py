"""
模型工厂

提供:
- get_model: 统一的模型创建接口，支持 ResNet/ConvNeXt/EfficientNet/Swin/GastroNet
"""

import os

import torch
import torch.nn as nn
import torchvision.models as models


# ================================================================
# 模型配置表: (工厂函数, 权重枚举, 分类头类型)
# ================================================================
MODEL_CONFIGS = {
    # ResNet 系列
    'resnet18':      (models.resnet18,          models.ResNet18_Weights.IMAGENET1K_V1,          'fc'),
    'resnet34':      (models.resnet34,          models.ResNet34_Weights.IMAGENET1K_V1,          'fc'),
    'resnet50':      (models.resnet50,          models.ResNet50_Weights.IMAGENET1K_V2,          'fc'),
    'resnet101':     (models.resnet101,         models.ResNet101_Weights.IMAGENET1K_V2,         'fc'),
    'resnet152':     (models.resnet152,         models.ResNet152_Weights.IMAGENET1K_V2,         'fc'),
    'resnext50':     (models.resnext50_32x4d,   models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2,   'fc'),
    'resnext101':    (models.resnext101_32x8d,  models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2,  'fc'),
    'wide_resnet50': (models.wide_resnet50_2,   models.Wide_ResNet50_2_Weights.IMAGENET1K_V2,   'fc'),
    'wide_resnet101':(models.wide_resnet101_2,  models.Wide_ResNet101_2_Weights.IMAGENET1K_V2,  'fc'),
    # ConvNeXt 系列
    'convnext_tiny':  (models.convnext_tiny,  models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,  'classifier'),
    'convnext_small': (models.convnext_small, models.ConvNeXt_Small_Weights.IMAGENET1K_V1, 'classifier'),
    'convnext_base':  (models.convnext_base,  models.ConvNeXt_Base_Weights.IMAGENET1K_V1,  'classifier'),
    'convnext_large': (models.convnext_large, models.ConvNeXt_Large_Weights.IMAGENET1K_V1, 'classifier'),
    # EfficientNet V2 系列
    'efficientnet_v2_s': (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.IMAGENET1K_V1, 'classifier'),
    'efficientnet_v2_m': (models.efficientnet_v2_m, models.EfficientNet_V2_M_Weights.IMAGENET1K_V1, 'classifier'),
    'efficientnet_v2_l': (models.efficientnet_v2_l, models.EfficientNet_V2_L_Weights.IMAGENET1K_V1, 'classifier'),
    # Swin Transformer 系列
    'swin_t': (models.swin_t, models.Swin_T_Weights.IMAGENET1K_V1, 'head'),
    'swin_s': (models.swin_s, models.Swin_S_Weights.IMAGENET1K_V1, 'head'),
    'swin_b': (models.swin_b, models.Swin_B_Weights.IMAGENET1K_V1, 'head'),
}


def get_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = False,
    weights_path: str = None,
    freeze_backbone: bool = False,
    classifier_head: str = 'linear',
    classifier_dropout: float = 0.2
) -> nn.Module:
    """
    获取模型

    Args:
        model_name: 模型名称 (见 MODEL_CONFIGS)
        num_classes: 分类类别数
        pretrained: 是否使用 ImageNet 预训练权重
        weights_path: 预训练权重路径
        freeze_backbone: 是否冻结backbone只训练分类头
        classifier_head: 分类头类型
            - "linear": 标准单层 Linear
            - "mlp":    两层 MLP (Linear → ReLU → Dropout → Linear)
        classifier_dropout: MLP head 的 dropout 概率

    Returns:
        PyTorch 模型
    """
    # GastroNet 预训练模型 (特殊处理)
    if model_name.startswith('gastronet_'):
        from GastroNet_5m import get_gastronet_model  # 可选依赖，延迟导入
        model = get_gastronet_model(
            model_name,
            num_classes=num_classes,
            weights_dir=weights_path,
            freeze_backbone=freeze_backbone)
        return model

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"不支持的模型: {model_name}. 支持: {list(MODEL_CONFIGS.keys())}")

    model_fn, weights_enum, head_type = MODEL_CONFIGS[model_name]

    # 加载模型权重
    model = _load_model_weights(model_fn, weights_enum, model_name, pretrained, weights_path)

    # 替换分类头
    in_features = _get_classifier_in_features(model, head_type)
    _replace_classifier(model, head_type, in_features, num_classes,
                        classifier_head, classifier_dropout)

    # 冻结 backbone
    if freeze_backbone:
        _freeze_backbone(model, head_type)

    _print_model_params(model)
    return model


def _load_model_weights(model_fn, weights_enum, model_name, pretrained, weights_path):
    """加载模型权重，优先级: weights_path > torch cache > 网络下载"""
    if weights_path and os.path.isfile(weights_path):
        model = model_fn(weights=None)
        print(f"  加载本地权重: {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict, strict=False)
    elif pretrained:
        model = model_fn(weights=weights_enum)
        print(f"  加载 ImageNet 预训练权重: {weights_enum}")
    else:
        model = model_fn(weights=None)
        print(f"  使用随机初始化权重")
    return model


def _get_classifier_in_features(model: nn.Module, head_type: str) -> int:
    """获取分类头的输入特征维度"""
    if head_type == 'fc':
        return model.fc.in_features
    elif head_type == 'classifier':
        if hasattr(model.classifier, '__getitem__'):
            return model.classifier[-1].in_features
        return model.classifier.in_features
    elif head_type == 'head':
        return model.head.in_features
    else:
        raise ValueError(f"Unknown head type: {head_type}")


def _replace_classifier(
    model: nn.Module,
    head_type: str,
    in_features: int,
    num_classes: int,
    classifier_head: str = 'linear',
    classifier_dropout: float = 0.2
):
    """
    替换分类头

    Args:
        classifier_head: "linear" 或 "mlp"
        classifier_dropout: MLP head 中的 dropout 概率
    """
    if classifier_head == 'mlp':
        # 两层 MLP head: 增强非线性拟合能力
        # Linear(in, in//2) → ReLU → Dropout → Linear(in//2, num_classes)
        hidden_dim = in_features // 2
        new_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        print(f"  分类头: MLP ({in_features} → {hidden_dim} → {num_classes}), dropout={classifier_dropout}")
    else:
        new_head = nn.Linear(in_features, num_classes)
        print(f"  分类头: Linear ({in_features} → {num_classes})")

    if head_type == 'fc':
        model.fc = new_head
    elif head_type == 'classifier':
        if hasattr(model.classifier, '__getitem__'):
            model.classifier[-1] = new_head
        else:
            model.classifier = new_head
    elif head_type == 'head':
        model.head = new_head


def _freeze_backbone(model: nn.Module, head_type: str):
    """冻结 backbone，只训练分类头"""
    for name, param in model.named_parameters():
        if head_type == 'fc' and 'fc' not in name:
            param.requires_grad = False
        elif head_type == 'classifier' and 'classifier' not in name:
            param.requires_grad = False
        elif head_type == 'head' and 'head' not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  冻结 backbone: 可训练参数 {trainable/1e6:.2f}M / 总参数 {total/1e6:.2f}M")


def _print_model_params(model: nn.Module):
    """打印模型参数量"""
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  模型参数: {total:.2f}M, 可训练: {trainable:.2f}M")
