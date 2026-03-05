"""
Gastrovision 学习率调度器工厂

提供:
- get_scheduler: 获取学习率调度器
"""

import torch
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
)
from ..data.augmentation import WarmupCosineScheduler


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    steps_per_epoch: int = None,
    warmup_epochs: int = 5
):
    """
    获取学习率调度器

    Args:
        optimizer: 优化器
        scheduler_name: 调度器名称
            - 'step': StepLR (每 30 epoch lr *= 0.1)
            - 'cosine': CosineAnnealingLR
            - 'plateau': ReduceLROnPlateau
            - 'onecycle': OneCycleLR (需要 steps_per_epoch)
            - 'warmup_cosine': WarmupCosineScheduler (带 warmup 的余弦退火)
            - 'none': 不使用调度器
        epochs: 总训练轮数
        steps_per_epoch: 每 epoch 的步数（OneCycleLR 必需）
        warmup_epochs: 学习率预热轮数

    Returns:
        调度器实例或 None
    """
    if scheduler_name == 'step':
        return StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    elif scheduler_name == 'onecycle':
        return OneCycleLR(optimizer, max_lr=optimizer.param_groups[0]['lr'],
                         epochs=epochs, steps_per_epoch=steps_per_epoch)
    elif scheduler_name == 'warmup_cosine':
        return WarmupCosineScheduler(optimizer, warmup_epochs=warmup_epochs,
                                     total_epochs=epochs)
    elif scheduler_name == 'none':
        return None
    else:
        raise ValueError(f"不支持的调度器: {scheduler_name}")
