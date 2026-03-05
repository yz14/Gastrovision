"""
Gastrovision 损失函数工厂

提供:
- create_loss_function: 创建单标签分类损失函数
- create_metric_loss_function: 创建度量学习损失函数（同时包装模型）
- get_samples_per_class: 从训练数据获取每类样本数
"""

from pathlib import Path

import torch
import torch.nn as nn

from .classification import LabelSmoothingCrossEntropy, FocalLoss, ClassBalancedLoss
from .metric_learning import create_metric_loss
from ..models.wrapper import MetricLearningWrapper


def get_samples_per_class(data_dir: str) -> list:
    """获取每个类别的样本数

    Args:
        data_dir: 包含 train.txt 的目录

    Returns:
        每个类别样本数的列表，如果 train.txt 不存在返回 None
    """
    train_file = Path(data_dir) / 'train.txt'
    if not train_file.exists():
        return None

    class_counts = {}
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                label = int(parts[-1])
                class_counts[label] = class_counts.get(label, 0) + 1

    num_classes = max(class_counts.keys()) + 1
    samples_per_class = [class_counts.get(i, 0) for i in range(num_classes)]
    return samples_per_class


def create_loss_function(args, device):
    """
    创建单标签分类损失函数

    Args:
        args: 需要包含 loss_type, label_smoothing, focal_gamma, data_dir 属性
        device: 设备

    Returns:
        损失函数实例
    """
    if args.loss_type == 'ce':
        if args.label_smoothing > 0:
            return LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        return nn.CrossEntropyLoss()

    elif args.loss_type == 'focal':
        return FocalLoss(gamma=args.focal_gamma)

    elif args.loss_type in ['cb_focal', 'cb_softmax']:
        samples_per_class = get_samples_per_class(args.data_dir)
        if samples_per_class is None:
            print("警告: 无法获取类别样本数，使用普通 Focal Loss")
            return FocalLoss(gamma=args.focal_gamma)

        loss_type = 'focal' if args.loss_type == 'cb_focal' else 'softmax'
        return ClassBalancedLoss(
            samples_per_class=samples_per_class,
            loss_type=loss_type,
            beta=0.9999,
            gamma=args.focal_gamma)

    else:
        raise ValueError(f"不支持的损失函数: {args.loss_type}")


def create_metric_loss_function(args, num_classes: int, device, model: nn.Module = None):
    """
    创建度量学习损失函数，并在需要时用 MetricLearningWrapper 包装模型。

    Args:
        args: 需要包含 metric_loss, embedding_dim, metric_loss_margin,
              metric_loss_scale, metric_loss_weight 属性
        num_classes: 类别数量
        device: 设备
        model: 原始模型（将被包装为 MetricLearningWrapper 以输出 features）

    Returns:
        (metric_criterion, wrapped_model)
        - metric_criterion: 度量学习损失函数实例，或 None
        - wrapped_model: 包装后的模型（如果启用度量学习），否则为原始模型
    """
    if args.metric_loss == 'none':
        return None, model

    # 用 MetricLearningWrapper 包装模型以提取 backbone features
    if not isinstance(model, MetricLearningWrapper):
        model = MetricLearningWrapper(model)
        print(f"  已包装模型为 MetricLearningWrapper")

    # 自动检测 backbone 特征维度
    feature_dim = model.feature_dim
    print(f"  Backbone 特征维度: {feature_dim}")

    # 如果用户未指定 embedding_dim 或使用默认值，自动适配为 feature_dim
    # 对需要 embedding_dim 的损失（ProxyNCA, ArcFace, CosFace, SphereFace, CircleLoss_cls），
    # 必须与 backbone 特征维度匹配
    needs_embedding = args.metric_loss in ['proxy_nca', 'arcface', 'cosface', 'sphereface', 'circle_cls']
    if needs_embedding:
        effective_dim = feature_dim
        if args.embedding_dim != 512 and args.embedding_dim != feature_dim:
            # 用户显式指定了非默认值且不等于 feature_dim，给出警告
            print(f"  警告: --embedding_dim={args.embedding_dim} 与 backbone 特征维度 {feature_dim} 不匹配")
            print(f"         自动使用 backbone 特征维度 {feature_dim}")
        embedding_dim = effective_dim
    else:
        embedding_dim = feature_dim

    # 构造额外参数
    kwargs = {}
    if args.metric_loss_margin > 0:
        kwargs['margin'] = args.metric_loss_margin
    if args.metric_loss_scale > 0:
        kwargs['scale'] = args.metric_loss_scale

    metric_criterion = create_metric_loss(
        loss_type=args.metric_loss,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        **kwargs
    )
    metric_criterion = metric_criterion.to(device)

    print(f"度量学习损失: {args.metric_loss.upper()}")
    print(f"  - 权重: {args.metric_loss_weight}")
    print(f"  - embedding_dim: {embedding_dim} (backbone 特征维度)")
    if args.metric_loss_margin > 0:
        print(f"  - margin: {args.metric_loss_margin}")
    if args.metric_loss_scale > 0:
        print(f"  - scale: {args.metric_loss_scale}")
    if needs_embedding:
        print(f"  - 注意: 此损失含可学习参数，已加入优化器")

    return metric_criterion, model
