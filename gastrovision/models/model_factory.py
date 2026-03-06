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
    classifier_dropout: float = 0.2,
    replace_head: bool = True
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
        replace_head: 是否替换分类头
            - True  (默认): 使用 ImageNet 预训练权重或 backbone-only 权重时，
                            必须替换分类头（ImageNet 是 1000 类）
            - False: resume/推理场景，weights_path 是本项目训练的完整 checkpoint，
                     分类头权重已包含在 checkpoint 中，不应替换

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
    # replace_head=False 时先初始化网络结构（含正确类别数的分类头），再严格加载 checkpoint
    model = _load_model_weights(
        model_fn, weights_enum, model_name, pretrained, weights_path,
        num_classes=num_classes, head_type=head_type,
        replace_head=replace_head)

    # 替换分类头（仅在 replace_head=True 时执行）
    if replace_head:
        in_features = _get_classifier_in_features(model, head_type)
        _replace_classifier(model, head_type, in_features, num_classes,
                            classifier_head, classifier_dropout)

    # 冻结 backbone
    if freeze_backbone:
        _freeze_backbone(model, head_type)

    _print_model_params(model)
    return model


def _load_model_weights(
    model_fn, weights_enum, model_name, pretrained, weights_path,
    num_classes: int = None, head_type: str = None, replace_head: bool = True
):
    """
    加载模型权重，优先级: weights_path > torch cache > 网络下载

    Args:
        replace_head: 是否要替换分类头（调用方在此函数后还会调用 _replace_classifier）
            - True : weights_path 是 backbone-only 或 ImageNet 权重，用 strict=False
            - False: weights_path 是本项目训练的完整 checkpoint，
                     先构建包含正确 num_classes 分类头的模型，再 strict=True 加载
    """
    if weights_path and os.path.isfile(weights_path):
        print(f"  加载本地权重: {weights_path}")
        raw = torch.load(weights_path, map_location='cpu', weights_only=False)

        # 检测是否为 Trainer.save_checkpoint 格式（含 model_state_dict 键）
        if isinstance(raw, dict) and 'model_state_dict' in raw:
            state_dict = raw['model_state_dict']
        else:
            state_dict = raw

        if replace_head:
            # backbone-only 或格式未知的权重：随机初始化网络，宽松加载（允许缺失/多余键）
            model = model_fn(weights=None)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  [提示] 以下键在 checkpoint 中缺失（将使用随机初始化）: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        else:
            # 完整 checkpoint（含分类头）：先构建正确 num_classes 的模型，再严格加载
            # 注意：此处需要先替换头使类别数正确，才能 strict=True 加载
            model = model_fn(weights=None)
            # 先临时替换为正确类别数（让 state_dict 的形状与模型一致）
            _replace_classifier_for_load(model, head_type, state_dict, num_classes)
            missing, unexpected = model.load_state_dict(state_dict, strict=True)
            print(f"  ✓ 完整 checkpoint 已加载（含分类头），类别数={num_classes}")
    elif pretrained:
        model = model_fn(weights=weights_enum)
        print(f"  加载 ImageNet 预训练权重: {weights_enum}")
    else:
        model = model_fn(weights=None)
        print(f"  使用随机初始化权重")
    return model


def _replace_classifier_for_load(model: nn.Module, head_type: str,
                                 state_dict: dict, num_classes: int):
    """
    在 strict 加载前，将模型分类头调整为与 checkpoint 中保存的 num_classes 一致。
    仅用于 replace_head=False 的路径，确保 load_state_dict(strict=True) 不报形状不匹配。
    """
    # 从 state_dict 读取分类头输出维度
    ckpt_num_classes = num_classes  # 默认使用传入值
    if head_type == 'fc':
        key = 'fc.weight'
        if key in state_dict:
            ckpt_num_classes = state_dict[key].shape[0]
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, ckpt_num_classes)
    elif head_type == 'classifier':
        if hasattr(model.classifier, '__getitem__'):
            key = f'classifier.{len(model.classifier) - 1}.weight'
            # 尝试查找最后一个 Linear 层的 key
            for k in state_dict:
                if k.startswith('classifier.') and k.endswith('.weight'):
                    key = k
            if key in state_dict:
                ckpt_num_classes = state_dict[key].shape[0]
            in_feats = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_feats, ckpt_num_classes)
        else:
            key = 'classifier.weight'
            if key in state_dict:
                ckpt_num_classes = state_dict[key].shape[0]
            in_feats = model.classifier.in_features
            model.classifier = nn.Linear(in_feats, ckpt_num_classes)
    elif head_type == 'head':
        key = 'head.weight'
        if key in state_dict:
            ckpt_num_classes = state_dict[key].shape[0]
        in_feats = model.head.in_features
        model.head = nn.Linear(in_feats, ckpt_num_classes)
    
    if ckpt_num_classes != num_classes:
        print(f"  [提示] checkpoint 分类头类别数={ckpt_num_classes}，与配置 num_classes={num_classes} 不同")
        print(f"         已按 checkpoint 类别数加载，若需微调请使用 replace_head=True")


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
