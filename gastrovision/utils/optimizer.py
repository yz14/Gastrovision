"""
Gastrovision 优化器工厂

提供:
- get_optimizer: 从模型获取优化器
- get_optimizer_from_params: 从参数列表获取优化器（支持合并多个模块的参数）
"""

import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW


def get_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
    weight_decay: float
) -> torch.optim.Optimizer:
    """获取优化器"""
    return get_optimizer_from_params(
        model.parameters(), optimizer_name, lr, weight_decay
    )


def get_optimizer_from_params(
    params,
    optimizer_name: str,
    lr: float,
    weight_decay: float
) -> torch.optim.Optimizer:
    """从参数列表获取优化器（支持合并模型参数和度量学习损失参数）"""
    if optimizer_name == 'adam':
        return Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")
