"""
Gastrovision 损失函数模块

提供用于处理类别不平衡和提升泛化性能的损失函数：

单标签分类:
- LabelSmoothingCrossEntropy: 标签平滑交叉熵
- FocalLoss: 聚焦难样本的损失
- ClassBalancedLoss: 基于有效样本数的类别平衡损失

多标签分类:
- FocalLossMultilabel: 多标签 Focal Loss + OHEM
- FocalOHEMLoss: 完整 Focal + OHEM 组合损失 (WhaleSSL)
- TripletLoss: 三元组损失 (度量学习)
- AsymmetricLoss: 非对称损失 (正负样本不同权重)
- LabelSmoothingBCE: 标签平滑 BCE
- WeightedBCELoss: 加权 BCE (支持正样本/类别权重)
- PolyLoss: 多项式损失
- DiceLoss: Dice 损失
- SoftmaxLossMultilabel: 多标签 Softmax 损失 (WhaleSSL)
- CombinedMultilabelLoss: 统一接口 (支持上述所有损失)

参考文献:
- Focal Loss: Lin et al., 2017
- Class-Balanced Loss: Cui et al., 2019
- WhaleSSL: Kaggle Humpback Whale Identification 2nd place solution
- Asymmetric Loss: Ridnik et al., 2021
- PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# =============================================================================
# 单标签分类损失函数
# =============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """
    带标签平滑的交叉熵损失
    
    标签平滑可以防止模型过度自信，提升泛化性能
    
    Args:
        smoothing: 平滑系数 (0.0 = 无平滑, 0.1 = 常用值)
        weight: 类别权重（可选）
    """
    
    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算标签平滑交叉熵损失
        
        Args:
            pred: 模型预测 (B, C)
            target: 真实标签 (B,)
            
        Returns:
            损失值
        """
        num_classes = pred.size(1)
        
        # 创建平滑后的标签分布
        smooth_target = torch.zeros_like(pred)
        smooth_target.fill_(self.smoothing / (num_classes - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        
        # 计算损失
        log_prob = F.log_softmax(pred, dim=1)
        
        if self.weight is not None:
            weight = self.weight.unsqueeze(0).expand_as(log_prob)
            loss = -(smooth_target * log_prob * weight).sum(dim=1)
        else:
            loss = -(smooth_target * log_prob).sum(dim=1)
        
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss - 专门处理类别不平衡问题 (单标签版本)
    
    通过降低易分类样本的权重，让模型更关注难分类样本
    
    论文: Focal Loss for Dense Object Detection (Lin et al., 2017)
    
    Args:
        alpha: 类别权重（可以是 float 或 Tensor）
        gamma: 聚焦参数，越大越关注难样本（推荐值: 2.0）
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
        """
        计算 Focal Loss
        
        Args:
            pred: 模型预测 (B, C)
            target: 真实标签 (B,)
            
        Returns:
            损失值
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)  # 预测正确的概率
        
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
    
    有效样本数 E_n = (1 - β^n) / (1 - β)，其中 n 是样本数
    
    Args:
        samples_per_class: 每个类别的样本数列表
        beta: 平衡因子（推荐值: 0.9999）
        gamma: Focal Loss 的 gamma 参数（可选）
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
        
        # 计算有效样本数和权重
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * self.num_classes
        
        self.weights = torch.tensor(weights, dtype=torch.float32)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算 Class-Balanced Loss
        
        Args:
            pred: 模型预测 (B, C)
            target: 真实标签 (B,)
            
        Returns:
            损失值
        """
        if self.weights.device != pred.device:
            self.weights = self.weights.to(pred.device)
        
        if self.loss_type == 'focal':
            # 使用 Focal Loss
            ce_loss = F.cross_entropy(pred, target, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_weight = (1 - pt) ** self.gamma
            cb_weights = self.weights[target]
            loss = cb_weights * focal_weight * ce_loss
        else:
            # 使用普通 Softmax Cross Entropy
            cb_weights = self.weights[target]
            ce_loss = F.cross_entropy(pred, target, reduction='none')
            loss = cb_weights * ce_loss
        
        return loss.mean()


# =============================================================================
# 多标签分类损失函数
# =============================================================================

def l2_norm(x: torch.Tensor, axis: int = 1) -> torch.Tensor:
    """L2 归一化"""
    norm = torch.norm(x, 2, axis, True)
    return torch.div(x, norm.clamp(min=1e-12))


def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """计算欧氏距离矩阵"""
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


class FocalLossMultilabel(nn.Module):
    """
    多标签 Focal Loss + OHEM (Online Hard Example Mining)
    
    基于 WhaleSSL 方案，适用于多标签分类任务
    
    公式: FL(p) = -(1-p)^γ × log(p)
    
    当 γ=2 时:
    - 简单样本 (p=0.9): 权重 = 0.01
    - 困难样本 (p=0.1): 权重 = 0.81
    
    Args:
        gamma: 聚焦参数 (默认 2.0)
        alpha: 正样本权重 (可选，用于处理正负样本不平衡)
        ohem_ratio: OHEM 比例，只保留最难的 k 个样本 (None = 不使用 OHEM)
        reduction: 'mean', 'sum', 或 'none'
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        ohem_ratio: Optional[float] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算多标签 Focal Loss
        
        Args:
            logits: 模型输出 logits (B, C)，未经 sigmoid
            targets: 多热编码标签 (B, C)，取值 0 或 1
            
        Returns:
            损失值
        """
        # BCE Loss 的稳定计算
        max_val = (-logits).clamp(min=0)
        bce_loss = logits - logits * targets + max_val + \
                   ((-max_val).exp() + (-logits - max_val).exp()).log()
        
        # Focal 权重: (1-p)^γ
        # 对于正样本 (target=1): p = sigmoid(logit)
        # 对于负样本 (target=0): p = 1 - sigmoid(logit)
        invprobs = F.logsigmoid(-logits * (targets * 2 - 1))
        focal_weight = (invprobs * self.gamma).exp()
        
        # 应用 Focal 权重
        loss = focal_weight * bce_loss
        
        # 应用 alpha (正样本权重)
        if self.alpha is not None:
            alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            loss = alpha_weight * loss
        
        # OHEM: 只保留最难的样本
        if self.ohem_ratio is not None and self.ohem_ratio < 1.0:
            num_classes = logits.shape[1]
            k = max(1, int(num_classes * self.ohem_ratio))
            loss, _ = loss.topk(k=k, dim=1, largest=True, sorted=True)
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalOHEMLoss(nn.Module):
    """
    完整的 Focal + OHEM 组合损失 (WhaleSSL 原版)
    
    组合三个损失:
    1. BCE Loss (带 OHEM)
    2. Focal Loss (带 OHEM)
    3. 正确类别的 Focal Loss
    
    Args:
        gamma: Focal Loss 的 gamma 参数 (默认 2.0)
        ohem_ratio: OHEM 比例 (默认 0.01 = 1%)
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        ohem_ratio: float = 0.01
    ):
        super().__init__()
        self.gamma = gamma
        self.ohem_ratio = ohem_ratio
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算组合损失
        
        Args:
            logits: 模型输出 logits (B, C)
            targets: 多热编码标签 (B, C)
            
        Returns:
            损失值
        """
        num_classes = logits.shape[1]
        k = max(1, int(num_classes * self.ohem_ratio))
        
        # 1. BCE Loss with OHEM
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        bce_ohem, _ = bce_loss.topk(k=k, dim=1, largest=True, sorted=True)
        loss0 = bce_ohem.mean()
        
        # 2. Focal Loss with OHEM
        max_val = (-logits).clamp(min=0)
        focal_bce = logits - logits * targets + max_val + \
                    ((-max_val).exp() + (-logits - max_val).exp()).log()
        invprobs = F.logsigmoid(-logits * (targets * 2 - 1))
        focal_loss = (invprobs * self.gamma).exp() * focal_bce
        focal_ohem, _ = focal_loss.topk(k=k, dim=1, largest=True, sorted=True)
        loss1 = focal_ohem.mean()
        
        # 3. 正样本 Focal Loss (只对有正标签的位置)
        positive_mask = targets > 0.5
        if positive_mask.any():
            positive_logits = logits[positive_mask]
            positive_focal = F.binary_cross_entropy_with_logits(
                positive_logits,
                torch.ones_like(positive_logits),
                reduction='none'
            )
            pt = torch.sigmoid(positive_logits)
            focal_weight = (1 - pt) ** self.gamma
            loss2 = (focal_weight * positive_focal).mean()
        else:
            loss2 = torch.tensor(0.0, device=logits.device)
        
        return loss0 + loss1 + loss2


class TripletLoss(nn.Module):
    """
    三元组损失 - 用于度量学习
    
    目标: d(anchor, positive) + margin < d(anchor, negative)
    损失: max(0, d(a,p) - d(a,n) + margin)
    
    使用硬样本挖掘 (Hard Example Mining):
    - 对于每个 anchor，找到最远的正样本 (hardest positive)
    - 对于每个 anchor，找到最近的负样本 (hardest negative)
    
    Args:
        margin: 边距值 (默认 0.3)
        normalize: 是否对特征进行 L2 归一化 (默认 True)
    """
    
    def __init__(
        self,
        margin: float = 0.3,
        normalize: bool = True
    ):
        super().__init__()
        self.margin = margin
        self.normalize = normalize
        
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()
    
    def _hard_example_mining(
        self,
        dist_mat: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        硬样本挖掘
        
        Args:
            dist_mat: 距离矩阵 (N, N)
            labels: 标签 (N,) - 对于多标签，需要传入主标签
            
        Returns:
            dist_ap: anchor-positive 距离
            dist_an: anchor-negative 距离
        """
        N = dist_mat.size(0)
        
        # 构建正负样本 mask
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
        
        # 对角线设为 False (排除自身)
        is_pos = is_pos & ~torch.eye(N, dtype=torch.bool, device=labels.device)
        
        # 检查是否有足够的正负样本对
        if not is_pos.any() or not is_neg.any():
            # 返回零损失
            return (
                torch.zeros(N, device=dist_mat.device),
                torch.ones(N, device=dist_mat.device)
            )
        
        # 最远正样本 (hardest positive)
        dist_ap = []
        for i in range(N):
            pos_mask = is_pos[i]
            if pos_mask.any():
                dist_ap.append(dist_mat[i][pos_mask].max())
            else:
                dist_ap.append(torch.tensor(0.0, device=dist_mat.device))
        dist_ap = torch.stack(dist_ap)
        
        # 最近负样本 (hardest negative)
        dist_an = []
        for i in range(N):
            neg_mask = is_neg[i]
            if neg_mask.any():
                dist_an.append(dist_mat[i][neg_mask].min())
            else:
                dist_an.append(torch.tensor(1.0, device=dist_mat.device))
        dist_an = torch.stack(dist_an)
        
        return dist_ap, dist_an
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Triplet Loss
        
        Args:
            features: 特征向量 (B, D)
            labels: 标签 (B,) - 对于多标签，传入主标签 (argmax)
            
        Returns:
            损失值
        """
        # L2 归一化
        if self.normalize:
            features = l2_norm(features)
        
        # 计算距离矩阵
        dist_mat = euclidean_dist(features, features)
        
        # 硬样本挖掘
        dist_ap, dist_an = self._hard_example_mining(dist_mat, labels)
        
        # 计算损失
        y = torch.ones_like(dist_an)
        
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        
        return loss


class AsymmetricLoss(nn.Module):
    """
    非对称损失 (Asymmetric Loss) - 专为多标签分类设计
    
    论文: Asymmetric Loss For Multi-Label Classification (Ridnik et al., 2021)
    
    核心思想:
    - 正样本: 使用较小的 gamma (γ+)，保持梯度
    - 负样本: 使用较大的 gamma (γ-)，抑制简单负样本
    - 概率偏移: 对负样本概率进行裁剪 (margin clipping)
    
    Args:
        gamma_neg: 负样本的 gamma (默认 4.0)
        gamma_pos: 正样本的 gamma (默认 1.0)
        clip: 概率裁剪值 (默认 0.05)
        disable_torch_grad_focal_loss: 是否禁用 focal 部分的梯度
    """
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        disable_torch_grad_focal_loss: bool = False
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算非对称损失
        
        Args:
            logits: 模型输出 logits (B, C)
            targets: 多热编码标签 (B, C)
            
        Returns:
            损失值
        """
        # 计算概率
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1 - probs
        
        # 概率裁剪 (Probability Margin Clipping)
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1)
        
        # 分别计算正负样本的损失
        los_pos = targets * torch.log(probs_pos.clamp(min=1e-8))
        los_neg = (1 - targets) * torch.log(probs_neg.clamp(min=1e-8))
        loss = los_pos + los_neg
        
        # Asymmetric Focusing (matches official ASL implementation)
        # Key formula: weight = (1 - pt)^γ, where pt is the probability of
        # the ground-truth class:
        #   - For positives (target=1): pt = p,       weight = (1-p)^γ+
        #   - For negatives (target=0): pt = 1-p_m,   weight = p_m^γ-
        #     where p_m = max(p - clip, 0) (after probability margin clipping)
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            
            # pt = ground-truth class probability for each position
            pt = probs_pos * targets + probs_neg * (1 - targets)
            # Per-position gamma: γ+ for positives, γ- for negatives
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            focal_weight = torch.pow(1 - pt, one_sided_gamma)
            
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            
            loss = loss * focal_weight
        
        return -loss.sum() / logits.shape[0]  # 按 batch 平均


class LabelSmoothingBCE(nn.Module):
    """
    多标签标签平滑 BCE Loss
    
    对正负标签都进行平滑处理，防止过拟合
    
    Args:
        smoothing: 平滑系数 (默认 0.1)
        reduction: 'mean', 'sum', 或 'none'
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算标签平滑 BCE Loss
        
        Args:
            logits: 模型输出 logits (B, C)
            targets: 多热编码标签 (B, C)
            
        Returns:
            损失值
        """
        # 标签平滑: 将 0 -> smoothing/2, 1 -> 1 - smoothing/2
        smooth_targets = targets * (1 - self.smoothing) + self.smoothing / 2
        
        loss = F.binary_cross_entropy_with_logits(
            logits, smooth_targets, reduction=self.reduction
        )
        
        return loss


class WeightedBCELoss(nn.Module):
    """
    加权 BCE Loss - 支持正负样本权重和类别权重
    
    Args:
        pos_weight: 正样本权重 (可以是标量或每类权重向量)
        class_weight: 每个类别的权重 (可选)
        reduction: 'mean', 'sum', 或 'none'
    """
    
    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        class_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.class_weight = class_weight
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算加权 BCE Loss
        """
        # 处理 pos_weight
        if self.pos_weight is not None:
            if self.pos_weight.device != logits.device:
                self.pos_weight = self.pos_weight.to(logits.device)
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight, reduction='none'
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction='none'
            )
        
        # 应用类别权重
        if self.class_weight is not None:
            if self.class_weight.device != logits.device:
                self.class_weight = self.class_weight.to(logits.device)
            loss = loss * self.class_weight.unsqueeze(0)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class PolyLoss(nn.Module):
    """
    Poly Loss - 多项式损失函数
    
    论文: PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
    
    Poly-1 Loss: L = CE + ε₁(1-pₜ)
    
    比 CE 和 Focal Loss 更平滑，对困难样本更友好
    
    Args:
        epsilon: 多项式系数 (默认 1.0)
        reduction: 'mean', 'sum', 或 'none'
    """
    
    def __init__(
        self,
        epsilon: float = 1.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Poly-1 BCE Loss
        
        Args:
            logits: 模型输出 logits (B, C)
            targets: 多热编码标签 (B, C)
            
        Returns:
            损失值
        """
        # 基础 BCE
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # 计算 p_t (目标类的概率)
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        
        # Poly Loss = BCE + ε(1-pt)
        poly_loss = bce + self.epsilon * (1 - pt)
        
        if self.reduction == 'mean':
            return poly_loss.mean()
        elif self.reduction == 'sum':
            return poly_loss.sum()
        else:
            return poly_loss


class DiceLoss(nn.Module):
    """
    Dice Loss - 基于 Dice 系数的损失
    
    常用于分割任务，对类别不平衡鲁棒
    Dice = 2|A∩B| / (|A| + |B|)
    
    Args:
        smooth: 平滑项，防止除零 (默认 1.0)
        reduction: 'mean', 'sum', 或 'none'
    """
    
    def __init__(
        self,
        smooth: float = 1.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Dice Loss
        
        Args:
            logits: 模型输出 logits (B, C)
            targets: 多热编码标签 (B, C)
            
        Returns:
            损失值
        """
        probs = torch.sigmoid(logits)
        
        # 计算每个类别的 Dice
        intersection = (probs * targets).sum(dim=0)
        cardinality = (probs + targets).sum(dim=0)
        
        dice_scores = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1 - dice_scores
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class SoftmaxLossMultilabel(nn.Module):
    """
    多标签 Softmax Loss (WhaleSSL 风格)
    
    将多标签问题视为多个二分类问题的组合
    使用 logsumexp 技巧稳定计算
    
    Args:
        reduction: 'mean', 'sum', 或 'none'
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Softmax Loss
        
        对每个样本，将正样本 logit 与所有 logit 的 logsumexp 比较
        """
        # 对于每个样本，计算 log(sum(exp(logits)))
        logsumexp = torch.logsumexp(logits, dim=1, keepdim=True)
        
        # 正样本的 log softmax
        log_softmax = logits - logsumexp
        
        # 只对正样本计算损失
        loss = -log_softmax * targets
        
        # 归一化: 除以每个样本的正标签数
        num_pos = targets.sum(dim=1, keepdim=True).clamp(min=1)
        loss = loss.sum(dim=1) / num_pos.squeeze()
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedMultilabelLoss(nn.Module):

    """
    组合多标签损失
    
    支持多种损失的加权组合:
    - bce: 标准 BCE Loss
    - focal: Focal Loss (多标签版)
    - focal_ohem: Focal + OHEM (WhaleSSL)
    - asymmetric/asl: Asymmetric Loss
    - label_smoothing: 标签平滑 BCE
    - poly: Poly Loss
    - dice: Dice Loss
    - softmax: Softmax Loss (WhaleSSL)
    
    Args:
        loss_type: 损失类型
        focal_gamma: Focal Loss 的 gamma
        ohem_ratio: OHEM 比例
        asl_gamma_neg: ASL 负样本 gamma
        asl_gamma_pos: ASL 正样本 gamma
        asl_clip: ASL 概率裁剪
        label_smoothing: 标签平滑系数
        poly_epsilon: Poly Loss 的 epsilon
    """
    
    def __init__(
        self,
        loss_type: str = 'bce',
        focal_gamma: float = 2.0,
        ohem_ratio: float = None,
        asl_gamma_neg: float = 4.0,
        asl_gamma_pos: float = 1.0,
        asl_clip: float = 0.05,
        label_smoothing: float = 0.1,
        poly_epsilon: float = 1.0
    ):
        super().__init__()
        self.loss_type = loss_type.lower()
        
        if self.loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.loss_type == 'focal':
            self.criterion = FocalLossMultilabel(
                gamma=focal_gamma,
                ohem_ratio=ohem_ratio
            )
        elif self.loss_type == 'focal_ohem':
            self.criterion = FocalOHEMLoss(
                gamma=focal_gamma,
                ohem_ratio=ohem_ratio if ohem_ratio else 0.01
            )
        elif self.loss_type == 'asymmetric' or self.loss_type == 'asl':
            self.criterion = AsymmetricLoss(
                gamma_neg=asl_gamma_neg,
                gamma_pos=asl_gamma_pos,
                clip=asl_clip
            )
        elif self.loss_type == 'label_smoothing':
            self.criterion = LabelSmoothingBCE(smoothing=label_smoothing)
        elif self.loss_type == 'poly':
            self.criterion = PolyLoss(epsilon=poly_epsilon)
        elif self.loss_type == 'dice':
            self.criterion = DiceLoss()
        elif self.loss_type == 'softmax':
            self.criterion = SoftmaxLossMultilabel()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. "
                           f"Supported: bce, focal, focal_ohem, asl, label_smoothing, poly, dice, softmax")
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        return self.criterion(logits, targets)


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("损失函数模块测试")
    print("=" * 60)
    
    # ========== 单标签测试 ==========
    print("\n【单标签分类损失】")
    pred_single = torch.randn(4, 10)
    target_single = torch.tensor([0, 1, 2, 3])
    
    print("\n1. LabelSmoothingCrossEntropy:")
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    loss = criterion(pred_single, target_single)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n2. FocalLoss (单标签):")
    criterion = FocalLoss(gamma=2.0)
    loss = criterion(pred_single, target_single)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n3. ClassBalancedLoss:")
    samples_per_class = [100, 50, 30, 20, 15, 10, 8, 5, 3, 2]
    criterion = ClassBalancedLoss(samples_per_class, beta=0.9999, gamma=2.0)
    loss = criterion(pred_single, target_single)
    print(f"   Loss: {loss.item():.4f}")
    
    # ========== 多标签测试 ==========
    print("\n" + "=" * 60)
    print("【多标签分类损失】")
    pred_multi = torch.randn(4, 16)  # 4 samples, 16 classes
    target_multi = torch.zeros(4, 16)
    target_multi[0, [0, 5, 10]] = 1
    target_multi[1, [1, 3]] = 1
    target_multi[2, [2, 7, 8, 12]] = 1
    target_multi[3, [4, 6]] = 1
    
    print("\n4. FocalLossMultilabel:")
    criterion = FocalLossMultilabel(gamma=2.0)
    loss = criterion(pred_multi, target_multi)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n5. FocalLossMultilabel + OHEM (10%):")
    criterion = FocalLossMultilabel(gamma=2.0, ohem_ratio=0.1)
    loss = criterion(pred_multi, target_multi)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n6. FocalOHEMLoss (WhaleSSL 风格):")
    criterion = FocalOHEMLoss(gamma=2.0, ohem_ratio=0.01)
    loss = criterion(pred_multi, target_multi)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n7. AsymmetricLoss:")
    criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
    loss = criterion(pred_multi, target_multi)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n8. TripletLoss:")
    features = torch.randn(8, 512)  # 8 samples, 512-dim features
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])  # 4 identities, 2 each
    criterion = TripletLoss(margin=0.3)
    loss = criterion(features, labels)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n9. LabelSmoothingBCE:")
    criterion = LabelSmoothingBCE(smoothing=0.1)
    loss = criterion(pred_multi, target_multi)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n10. PolyLoss:")
    criterion = PolyLoss(epsilon=1.0)
    loss = criterion(pred_multi, target_multi)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n11. DiceLoss:")
    criterion = DiceLoss()
    loss = criterion(pred_multi, target_multi)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n12. SoftmaxLossMultilabel:")
    criterion = SoftmaxLossMultilabel()
    loss = criterion(pred_multi, target_multi)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n13. CombinedMultilabelLoss (各类型):")
    for loss_type in ['bce', 'focal', 'focal_ohem', 'asl', 'label_smoothing', 'poly', 'dice', 'softmax']:
        criterion = CombinedMultilabelLoss(loss_type=loss_type)
        loss = criterion(pred_multi, target_multi)
        print(f"   {loss_type:16s}: {loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过!")


