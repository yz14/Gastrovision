"""
Gastrovision 单标签分类损失函数

提供:
- LabelSmoothingCrossEntropy: 标签平滑交叉熵
- FocalLoss: 聚焦难样本的损失
- ClassBalancedLoss: 基于有效样本数的类别平衡损失

参考文献:
- Focal Loss: Lin et al., 2017
- Class-Balanced Loss: Cui et al., 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class LabelSmoothingCrossEntropy(nn.Module):
    """
    带标签平滑的交叉熵损失
    
    Args:
        smoothing: 平滑系数 (0.0 = 无平滑, 0.1 = 常用值)
        weight: 类别权重（可选）
    """
    
    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = pred.size(1)
        smooth_target = torch.zeros_like(pred)
        smooth_target.fill_(self.smoothing / (num_classes - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        log_prob = F.log_softmax(pred, dim=1)
        
        if self.weight is not None:
            weight = self.weight.unsqueeze(0).expand_as(log_prob)
            loss = -(smooth_target * log_prob * weight).sum(dim=1)
        else:
            loss = -(smooth_target * log_prob).sum(dim=1)
        
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss - 处理类别不平衡问题 (单标签版本)
    
    论文: Focal Loss for Dense Object Detection (Lin et al., 2017)
    
    Args:
        alpha: 类别权重（Tensor）
        gamma: 聚焦参数（推荐值: 2.0）
        reduction: 'mean', 'sum', 或 'none'
    """
    
    def __init__(
        self, 
        alpha: Optional[torch.Tensor] = None, 
        gamma: float = 2.0, 
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            if self.alpha.device != pred.device:
                self.alpha = self.alpha.to(pred.device)
            alpha_t = self.alpha[target]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss - 基于有效样本数的类别平衡损失
    
    论文: Class-Balanced Loss Based on Effective Number of Samples (Cui et al., 2019)
    
    Args:
        samples_per_class: 每个类别的样本数列表
        beta: 平衡因子（推荐值: 0.9999）
        gamma: Focal Loss 的 gamma 参数
        loss_type: 'focal' 或 'softmax'
    """
    
    def __init__(
        self,
        samples_per_class: list,
        beta: float = 0.9999,
        gamma: float = 2.0,
        loss_type: str = 'focal'
    ):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.num_classes = len(samples_per_class)
        
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * self.num_classes
        self.weights = torch.tensor(weights, dtype=torch.float32)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.weights.device != pred.device:
            self.weights = self.weights.to(pred.device)
        
        if self.loss_type == 'focal':
            ce_loss = F.cross_entropy(pred, target, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_weight = (1 - pt) ** self.gamma
            cb_weights = self.weights[target]
            loss = cb_weights * focal_weight * ce_loss
        else:
            cb_weights = self.weights[target]
            ce_loss = F.cross_entropy(pred, target, reduction='none')
            loss = cb_weights * ce_loss
        
        return loss.mean()
