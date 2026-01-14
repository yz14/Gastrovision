"""
Gastrovision 数据增强与训练策略模块

提供高级数据增强和训练技巧：
- Mixup: 图像混合增强
- CutMix: 图像裁剪混合增强
- TTA: 测试时数据增强
- ProgressiveResizing: 渐进式图像增大
- WarmupCosineScheduler: 学习率预热调度器

注意：损失函数已移至 losses.py
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


# =============================================================================
# Mixup 数据增强
# =============================================================================

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Mixup 数据增强
    
    将两个样本线性混合: x' = λx_i + (1-λ)x_j
    
    Args:
        x: 输入图像 (B, C, H, W)
        y: 标签 (B,)
        alpha: Beta 分布参数（越大混合程度越高）
        device: 设备
        
    Returns:
        mixed_x: 混合后的图像
        y_a: 原始标签
        y_b: 配对标签
        lam: 混合系数
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    if device is None:
        device = x.device
    
    # 随机打乱索引
    index = torch.randperm(batch_size).to(device)
    
    # 混合图像
    mixed_x = lam * x + (1 - lam) * x[index, :]
    
    # 保留原始标签和配对标签
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """
    Mixup 损失计算
    
    Args:
        criterion: 损失函数
        pred: 模型预测
        y_a: 原始标签
        y_b: 配对标签
        lam: 混合系数
        
    Returns:
        混合损失
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============================================================================
# CutMix 数据增强
# =============================================================================

def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    CutMix 数据增强
    
    从一张图像裁剪一块区域粘贴到另一张图像
    
    Args:
        x: 输入图像 (B, C, H, W)
        y: 标签 (B,)
        alpha: Beta 分布参数
        device: 设备
        
    Returns:
        mixed_x: 混合后的图像
        y_a: 原始标签
        y_b: 配对标签
        lam: 混合系数（基于面积比例）
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    if device is None:
        device = x.device
    
    # 随机打乱索引
    index = torch.randperm(batch_size).to(device)
    
    # 获取随机裁剪框
    _, _, H, W = x.shape
    bbx1, bby1, bbx2, bby2 = rand_bbox(H, W, lam)
    
    # 裁剪并粘贴
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # 根据面积重新计算 lambda
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def rand_bbox(H: int, W: int, lam: float) -> Tuple[int, int, int, int]:
    """
    生成随机裁剪框
    
    Args:
        H: 图像高度
        W: 图像宽度
        lam: 混合系数
        
    Returns:
        (x1, y1, x2, y2) 裁剪框坐标
    """
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    
    # 随机选择中心点
    cx = np.random.randint(H)
    cy = np.random.randint(W)
    
    # 计算裁剪框坐标
    bbx1 = np.clip(cx - cut_h // 2, 0, H)
    bby1 = np.clip(cy - cut_w // 2, 0, W)
    bbx2 = np.clip(cx + cut_h // 2, 0, H)
    bby2 = np.clip(cy + cut_w // 2, 0, W)
    
    return bbx1, bby1, bbx2, bby2


# =============================================================================
# 学习率预热调度器
# =============================================================================

class WarmupCosineScheduler:
    """
    带预热的余弦退火学习率调度器
    
    分为两个阶段：
    1. Warmup: 学习率从 0 线性增长到 base_lr
    2. Cosine Annealing: 学习率按余弦曲线衰减
    
    Args:
        optimizer: 优化器
        warmup_epochs: 预热轮数
        total_epochs: 总训练轮数
        min_lr: 最小学习率
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        
        # 保存初始学习率
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
    
    def step(self, epoch: int = None):
        """更新学习率"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = self._get_lr(base_lr)
    
    def _get_lr(self, base_lr: float) -> float:
        """计算当前学习率"""
        if self.current_epoch < self.warmup_epochs:
            # Warmup 阶段：线性增长
            return base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine Annealing 阶段
            progress = (self.current_epoch - self.warmup_epochs) / \
                      (self.total_epochs - self.warmup_epochs)
            return self.min_lr + 0.5 * (base_lr - self.min_lr) * \
                   (1 + np.cos(np.pi * progress))
    
    def state_dict(self):
        return {'current_epoch': self.current_epoch}
    
    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict['current_epoch']


# =============================================================================
# 渐进式图像增大
# =============================================================================

class ProgressiveResizing:
    """
    渐进式图像增大
    
    训练初期使用小图像，逐渐增大到目标尺寸
    可以加速训练并提高最终精度
    
    Args:
        start_size: 起始尺寸
        end_size: 目标尺寸
        epochs_to_full: 达到目标尺寸所需的 epoch 数
    """
    
    def __init__(
        self,
        start_size: int = 128,
        end_size: int = 224,
        epochs_to_full: int = 10
    ):
        self.start_size = start_size
        self.end_size = end_size
        self.epochs_to_full = epochs_to_full
    
    def get_size(self, epoch: int) -> int:
        """获取当前 epoch 的图像尺寸"""
        if epoch >= self.epochs_to_full:
            return self.end_size
        
        progress = epoch / self.epochs_to_full
        size = int(self.start_size + (self.end_size - self.start_size) * progress)
        # 确保尺寸是 32 的倍数
        return (size // 32) * 32


# =============================================================================
# Test Time Augmentation (TTA)
# =============================================================================

def tta_predict(
    model: nn.Module,
    images: torch.Tensor,
    num_augments: int = 5
) -> torch.Tensor:
    """
    测试时数据增强 (TTA)
    
    对测试图像应用多种增强，取平均预测结果
    
    Args:
        model: 训练好的模型
        images: 输入图像 (B, C, H, W)
        num_augments: 增强次数
        
    Returns:
        平均预测结果
    """
    model.eval()
    device = next(model.parameters()).device
    images = images.to(device)
    
    predictions = []
    
    with torch.no_grad():
        # 原始图像
        pred = model(images)
        predictions.append(pred)
        
        # 水平翻转
        if num_augments >= 2:
            flipped = torch.flip(images, dims=[3])
            pred = model(flipped)
            predictions.append(pred)
        
        # 垂直翻转
        if num_augments >= 3:
            flipped = torch.flip(images, dims=[2])
            pred = model(flipped)
            predictions.append(pred)
        
        # 旋转 90 度
        if num_augments >= 4:
            rotated = torch.rot90(images, k=1, dims=[2, 3])
            pred = model(rotated)
            predictions.append(pred)
        
        # 旋转 270 度
        if num_augments >= 5:
            rotated = torch.rot90(images, k=3, dims=[2, 3])
            pred = model(rotated)
            predictions.append(pred)
    
    # 取平均
    avg_pred = torch.stack(predictions).mean(dim=0)
    return avg_pred


# =============================================================================
# 向后兼容：从 losses.py 导入（将在未来版本移除）
# =============================================================================

# 保持向后兼容，但建议直接从 losses.py 导入
try:
    from losses import LabelSmoothingCrossEntropy, FocalLoss, ClassBalancedLoss
except ImportError:
    pass


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("数据增强与训练策略模块")
    print("=" * 50)
    
    # 测试 Mixup
    print("\n1. Mixup 示例:")
    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0, 1, 2, 3])
    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.4)
    print(f"   Lambda: {lam:.4f}")
    print(f"   y_a: {y_a.tolist()}, y_b: {y_b.tolist()}")
    
    # 测试 CutMix
    print("\n2. CutMix 示例:")
    mixed_x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0)
    print(f"   Lambda (area ratio): {lam:.4f}")
    
    # 测试 Warmup 调度器
    print("\n3. Warmup Scheduler 示例:")
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, total_epochs=50)
    lrs = []
    for epoch in range(50):
        scheduler.step(epoch)
        lrs.append(optimizer.param_groups[0]['lr'])
    print(f"   Epoch 0: {lrs[0]:.6f}")
    print(f"   Epoch 5: {lrs[5]:.6f}")
    print(f"   Epoch 25: {lrs[25]:.6f}")
    print(f"   Epoch 49: {lrs[49]:.6f}")
    
    print("\n✓ 模块测试完成!")
    print("\n注意: 损失函数已移至 losses.py")
