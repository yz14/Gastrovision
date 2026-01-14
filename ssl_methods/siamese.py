"""
Siamese Network + Contrastive Loss / Triplet Loss

早期度量学习方法，奠定了对比学习的基础。

Siamese Network:
- 论文: "Signature Verification using a Siamese Time Delay Neural Network" (1993)
- 使用对比损失学习相似性
- L = y * d² + (1-y) * max(0, margin - d)²

Triplet Loss (FaceNet):
- 论文: https://arxiv.org/abs/1503.03832
- 使用三元组 (anchor, positive, negative)
- L = max(0, d(a,p) - d(a,n) + margin)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .base import SSLMethod, get_backbone_output_dim


class SiameseNetwork(SSLMethod):
    """
    Siamese Network with Contrastive Loss
    
    架构:
        x1, x2 → Encoder (共享权重) → z1, z2
        Loss = y * ||z1-z2||² + (1-y) * max(0, margin - ||z1-z2||)²
    
    Args:
        backbone: 特征提取网络
        proj_dim: 投影维度 (默认 128)
        margin: 对比损失的 margin (默认 1.0)
        
    Example:
        >>> model = SiameseNetwork(backbone, proj_dim=128, margin=1.0)
        >>> # 需要提供标签: 1=正样本对, 0=负样本对
        >>> output = model(x1, x2, labels=pair_labels)
        >>> loss = output['loss']
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        proj_dim: int = 128,
        margin: float = 1.0
    ):
        feature_dim = get_backbone_output_dim(backbone)
        super().__init__(backbone, feature_dim)
        
        self.proj_dim = proj_dim
        self.margin = margin
        
        # 移除 backbone 的分类头
        self._remove_fc()
        
        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, proj_dim)
        )
    
    def _remove_fc(self):
        """移除 backbone 的分类头"""
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()
    
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        labels: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x1: 第一个图像 [B, C, H, W]
            x2: 第二个图像 [B, C, H, W]
            labels: 配对标签 [B], 1=正样本对 (同类), 0=负样本对 (异类)
            
        Returns:
            包含 'loss', 'distance', 'z1', 'z2' 的字典
        """
        # 编码
        h1 = self.backbone(x1)
        h2 = self.backbone(x2)
        
        # 投影
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        
        # 计算欧氏距离
        distance = F.pairwise_distance(z1, z2, p=2)
        
        # 如果没有提供标签，假设都是正样本对 (用于预训练场景)
        if labels is None:
            # 使用 SSL 场景：x1 和 x2 是同一图像的不同增强，应该相似
            labels = torch.ones(x1.shape[0], device=x1.device)
        
        # 对比损失
        loss = self._contrastive_loss(distance, labels)
        
        return {
            'loss': loss,
            'distance': distance.detach(),
            'z1': z1.detach(),
            'z2': z2.detach()
        }
    
    def _contrastive_loss(
        self,
        distance: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        对比损失
        
        L = y * d² + (1-y) * max(0, margin - d)²
        """
        labels = labels.float()
        
        # 正样本对损失：距离越小越好
        pos_loss = labels * distance.pow(2)
        
        # 负样本对损失：距离小于 margin 时有惩罚
        neg_loss = (1 - labels) * F.relu(self.margin - distance).pow(2)
        
        loss = (pos_loss + neg_loss).mean()
        return loss
    
    def get_encoder(self) -> nn.Module:
        """返回用于下游任务的编码器"""
        return self.backbone


class TripletNetwork(SSLMethod):
    """
    Triplet Network with Triplet Loss (FaceNet style)
    
    架构:
        anchor, positive, negative → Encoder (共享权重) → embeddings
        Loss = max(0, d(a,p) - d(a,n) + margin)
    
    Args:
        backbone: 特征提取网络
        proj_dim: 投影维度 (默认 128)
        margin: 三元组损失的 margin (默认 0.2)
        
    Example:
        >>> model = TripletNetwork(backbone)
        >>> output = model(anchor, positive, negative)
        >>> loss = output['loss']
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        proj_dim: int = 128,
        margin: float = 0.2
    ):
        feature_dim = get_backbone_output_dim(backbone)
        super().__init__(backbone, feature_dim)
        
        self.proj_dim = proj_dim
        self.margin = margin
        
        # 移除 backbone 的分类头
        self._remove_fc()
        
        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, proj_dim)
        )
    
    def _remove_fc(self):
        """移除 backbone 的分类头"""
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()
    
    def forward(
        self,
        x1: torch.Tensor,  # anchor 或 view1
        x2: torch.Tensor,  # positive 或 view2
        x3: torch.Tensor = None  # negative (可选)
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        两种模式:
        1. 三元组模式: (anchor, positive, negative) 都提供
        2. SSL 模式: 只提供 (x1, x2)，使用批内负样本
        
        Args:
            x1: anchor 图像 [B, C, H, W]
            x2: positive 图像 [B, C, H, W]
            x3: negative 图像 [B, C, H, W] (可选)
            
        Returns:
            包含 'loss', 'pos_dist', 'neg_dist' 等的字典
        """
        # 编码
        z_anchor = self.projector(self.backbone(x1))
        z_positive = self.projector(self.backbone(x2))
        
        if x3 is not None:
            # 模式 1: 使用提供的负样本
            z_negative = self.projector(self.backbone(x3))
            return self._triplet_loss_explicit(z_anchor, z_positive, z_negative)
        else:
            # 模式 2: 使用批内负样本 (滚动)
            return self._triplet_loss_batch(z_anchor, z_positive)
    
    def _triplet_loss_explicit(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """使用显式提供的三元组"""
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_dist - neg_dist + self.margin).mean()
        
        return {
            'loss': loss,
            'pos_dist': pos_dist.mean().item(),
            'neg_dist': neg_dist.mean().item(),
            'z_anchor': anchor.detach(),
            'z_positive': positive.detach(),
            'z_negative': negative.detach()
        }
    
    def _triplet_loss_batch(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        使用批内负样本
        
        对于 anchor[i]，negative 是 positive[(i+1) % B]
        """
        batch_size = anchor.shape[0]
        
        # 滚动创建负样本
        negative = torch.roll(positive, shifts=1, dims=0)
        
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_dist - neg_dist + self.margin).mean()
        
        return {
            'loss': loss,
            'pos_dist': pos_dist.mean().item(),
            'neg_dist': neg_dist.mean().item(),
            'z1': anchor.detach(),
            'z2': positive.detach()
        }
    
    def get_encoder(self) -> nn.Module:
        """返回用于下游任务的编码器"""
        return self.backbone


# ============ 测试代码 ============
if __name__ == '__main__':
    print("测试 Siamese Network 和 Triplet Network...")
    
    from torchvision.models import resnet18
    
    # 测试 Siamese Network
    print("\n=== Siamese Network ===")
    backbone = resnet18(weights=None)
    siamese = SiameseNetwork(backbone, proj_dim=128, margin=1.0)
    
    x1 = torch.randn(8, 3, 224, 224)
    x2 = torch.randn(8, 3, 224, 224)
    labels = torch.randint(0, 2, (8,)).float()  # 0 或 1
    
    output = siamese(x1, x2, labels)
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"Distance mean: {output['distance'].mean().item():.4f}")
    print(f"z1 shape: {output['z1'].shape}")
    
    # 测试 SSL 模式 (无标签)
    output_ssl = siamese(x1, x2)
    print(f"SSL mode loss: {output_ssl['loss'].item():.4f}")
    
    # 测试 Triplet Network
    print("\n=== Triplet Network ===")
    backbone = resnet18(weights=None)
    triplet = TripletNetwork(backbone, proj_dim=128, margin=0.2)
    
    anchor = torch.randn(8, 3, 224, 224)
    positive = torch.randn(8, 3, 224, 224)
    negative = torch.randn(8, 3, 224, 224)
    
    # 显式三元组模式
    output = triplet(anchor, positive, negative)
    print(f"Explicit mode - Loss: {output['loss'].item():.4f}")
    print(f"  Pos dist: {output['pos_dist']:.4f}, Neg dist: {output['neg_dist']:.4f}")
    
    # 批内负样本模式
    output_batch = triplet(anchor, positive)
    print(f"Batch mode - Loss: {output_batch['loss'].item():.4f}")
    print(f"  Pos dist: {output_batch['pos_dist']:.4f}, Neg dist: {output_batch['neg_dist']:.4f}")
    
    # 参数量
    print(f"\nSiamese params: {sum(p.numel() for p in siamese.parameters()) / 1e6:.2f}M")
    print(f"Triplet params: {sum(p.numel() for p in triplet.parameters()) / 1e6:.2f}M")
    
    print("\nSiamese & Triplet 测试通过！")
