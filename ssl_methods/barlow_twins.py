"""
Barlow Twins: Self-Supervised Learning via Redundancy Reduction

论文: https://arxiv.org/abs/2103.03230
官方实现: https://github.com/facebookresearch/barlowtwins

核心思想:
- 通过减少特征冗余来学习表示
- 两个视图特征的互相关矩阵应趋近于单位阵
- 无需负样本、动量编码器、大批量

损失函数:
L = Σ_i (1 - C_ii)² + λ * Σ_i Σ_{j≠i} C_ij²

官方实现关键点:
- 使用 BatchNorm1d (affine=False) 进行归一化
- off_diagonal 函数提取非对角元素
- 投影器: Linear-BN-ReLU-Linear-BN-ReLU-Linear (无最后的 BN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .base import SSLMethod, get_backbone_output_dim


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """
    提取方阵的非对角元素 (官方实现)
    
    Returns a flattened view of the off-diagonal elements of a square matrix
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(SSLMethod):
    """
    Barlow Twins: Self-Supervised Learning via Redundancy Reduction
    
    官方实现参考: https://github.com/facebookresearch/barlowtwins
    
    架构:
        x1, x2 → Backbone → Projector → z1, z2
        z1_norm = BN(z1), z2_norm = BN(z2)
        C = z1_norm.T @ z2_norm / batch_size  # 互相关矩阵
        Loss = Σ(1 - diag(C))² + λ * Σ(off_diag(C))²
    
    Args:
        backbone: 特征提取网络
        projector_sizes: 投影头的维度列表 (默认 [8192, 8192, 8192])
        lambda_coeff: 非对角项权重 (默认 0.0051)
        
    Example:
        >>> backbone = resnet50
        >>> model = BarlowTwins(backbone, projector_sizes=[8192, 8192, 8192])
        >>> output = model(x1, x2)
        >>> loss = output['loss']
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        projector_sizes: list = None,  # 默认 [8192, 8192, 8192]
        lambda_coeff: float = 0.0051
    ):
        feature_dim = get_backbone_output_dim(backbone)
        super().__init__(backbone, feature_dim)
        
        if projector_sizes is None:
            projector_sizes = [8192, 8192, 8192]
        
        self.lambda_coeff = lambda_coeff
        self.proj_dim = projector_sizes[-1]
        
        # 移除 backbone 的分类头 (官方: self.backbone.fc = nn.Identity())
        self._remove_fc()
        
        # 构建投影器 (官方实现风格)
        # sizes = [2048] + [8192, 8192, 8192]
        sizes = [feature_dim] + projector_sizes
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        # 最后一层没有 BN 和 ReLU
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        
        # 归一化层 (官方: affine=False)
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
    
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
        x2: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播 (严格按照官方实现)
        
        Args:
            x1: 第一个视图 [B, C, H, W]
            x2: 第二个视图 [B, C, H, W]
            
        Returns:
            包含 'loss', 'on_diag', 'off_diag', 'cross_correlation' 的字典
        """
        batch_size = x1.shape[0]
        
        # 编码并投影
        z1 = self.projector(self.backbone(x1))  # [B, proj_dim]
        z2 = self.projector(self.backbone(x2))
        
        # 使用 BatchNorm 归一化 (官方实现)
        z1_norm = self.bn(z1)
        z2_norm = self.bn(z2)
        
        # 计算互相关矩阵 [proj_dim, proj_dim]
        # 官方: c = self.bn(z1).T @ self.bn(z2)
        # 官方: c.div_(batch_size)
        c = z1_norm.T @ z2_norm
        c.div_(batch_size)
        
        # 对角项损失: (1 - C_ii)²
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        
        # 非对角项损失: C_ij² (使用官方的 off_diagonal 函数)
        off_diag_loss = off_diagonal(c).pow_(2).sum()
        
        # 总损失
        loss = on_diag + self.lambda_coeff * off_diag_loss
        
        return {
            'loss': loss,
            'z1': z1.detach(),
            'z2': z2.detach(),
            'cross_correlation': c.detach(),
            'on_diag': on_diag.item(),
            'off_diag': off_diag_loss.item()
        }
    
    def get_encoder(self) -> nn.Module:
        """返回用于下游任务的编码器"""
        return self.backbone
    
    def get_representations(self, x: torch.Tensor) -> torch.Tensor:
        """获取特征表示"""
        return self.backbone(x)


# ============ 测试代码 ============
if __name__ == '__main__':
    print("测试 Barlow Twins (基于官方实现)...")
    
    from torchvision.models import resnet18
    
    # 使用较小的 proj_dim 用于测试
    backbone = resnet18(weights=None)
    model = BarlowTwins(
        backbone, 
        projector_sizes=[2048, 2048, 2048], 
        lambda_coeff=0.0051
    )
    
    # 测试前向传播
    x1 = torch.randn(32, 3, 224, 224)
    x2 = torch.randn(32, 3, 224, 224)
    
    output = model(x1, x2)
    
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"On-diag loss: {output['on_diag']:.4f}")
    print(f"Off-diag loss: {output['off_diag']:.4f}")
    print(f"z1 shape: {output['z1'].shape}")
    print(f"Cross-correlation shape: {output['cross_correlation'].shape}")
    
    # 检查互相关矩阵
    c = output['cross_correlation']
    print(f"Cross-correlation diagonal mean: {torch.diagonal(c).mean().item():.4f}")
    
    # 计算模型参数量
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total params: {num_params:.2f}M")
    
    print("\nBarlow Twins 测试通过！")
