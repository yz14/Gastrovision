"""
SwAV: Swapping Assignments between Views

论文: https://arxiv.org/abs/2006.09882

核心思想:
- 在线聚类 + 交换预测
- 使用原型 (prototypes) 进行聚类
- Sinkhorn-Knopp 算法确保聚类均匀分布
- 无需负样本或 memory bank

关键组件:
- 多尺度裁剪 (multi-crop)
- 可学习的原型向量
- Sinkhorn-Knopp 正则化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

from .base import SSLMethod, get_backbone_output_dim


class SwAV(SSLMethod):
    """
    SwAV: Swapping Assignments between Views
    
    架构:
        x1, x2 → Encoder → Projector → z1, z2
        q1 = softmax(z1 @ prototypes / T)  # 聚类分配
        q2 = softmax(z2 @ prototypes / T)
        
        Loss = -(q1 * log(p2) + q2 * log(p1))  # 交换预测
    
    Args:
        backbone: 特征提取网络
        proj_dim: 投影维度 (默认 128)
        hidden_dim: 隐藏层维度 (默认 2048)
        num_prototypes: 原型数量 (默认 3000)
        temperature: 温度参数 (默认 0.1)
        sinkhorn_iters: Sinkhorn 迭代次数 (默认 3)
        epsilon: Sinkhorn epsilon (默认 0.05)
        
    Example:
        >>> model = SwAV(backbone, num_prototypes=3000)
        >>> output = model(x1, x2)
        >>> loss = output['loss']
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        proj_dim: int = 128,
        hidden_dim: int = 2048,
        num_prototypes: int = 3000,
        temperature: float = 0.1,
        sinkhorn_iters: int = 3,
        epsilon: float = 0.05
    ):
        feature_dim = get_backbone_output_dim(backbone)
        super().__init__(backbone, feature_dim)
        
        self.proj_dim = proj_dim
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.sinkhorn_iters = sinkhorn_iters
        self.epsilon = epsilon
        
        # 移除 backbone 的分类头
        self._remove_fc()
        
        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim)
        )
        
        # 原型向量 (可学习)
        self.prototypes = nn.Linear(proj_dim, num_prototypes, bias=False)
        
        # 初始化原型
        nn.init.normal_(self.prototypes.weight, std=0.01)
    
    def _remove_fc(self):
        """移除 backbone 的分类头"""
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()
    
    @torch.no_grad()
    def _sinkhorn(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Sinkhorn-Knopp 算法
        
        确保聚类分配均匀分布
        """
        Q = torch.exp(scores / self.epsilon).t()  # [K, B]
        Q /= Q.sum()
        
        K, B = Q.shape
        
        for _ in range(self.sinkhorn_iters):
            # 行归一化
            Q /= Q.sum(dim=1, keepdim=True)
            Q /= K
            
            # 列归一化
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= B
        
        return (Q / Q.sum(dim=0, keepdim=True)).t()  # [B, K]
    
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x1: 第一个视图 [B, C, H, W]
            x2: 第二个视图 [B, C, H, W]
            
        Returns:
            包含 'loss', 'z1', 'z2' 等的字典
        """
        # 编码并归一化
        z1 = F.normalize(self.projector(self.backbone(x1)), dim=1)
        z2 = F.normalize(self.projector(self.backbone(x2)), dim=1)
        
        # 归一化原型
        with torch.no_grad():
            w = self.prototypes.weight.data
            w = F.normalize(w, dim=1)
            self.prototypes.weight.data = w
        
        # 计算与原型的相似度
        scores1 = self.prototypes(z1)  # [B, K]
        scores2 = self.prototypes(z2)
        
        # Sinkhorn 得到软聚类分配
        with torch.no_grad():
            q1 = self._sinkhorn(scores1)  # [B, K]
            q2 = self._sinkhorn(scores2)
        
        # 交换预测损失
        p1 = F.log_softmax(scores1 / self.temperature, dim=1)
        p2 = F.log_softmax(scores2 / self.temperature, dim=1)
        
        loss = -0.5 * (
            (q1 * p2).sum(dim=1).mean() + 
            (q2 * p1).sum(dim=1).mean()
        )
        
        return {
            'loss': loss,
            'z1': z1.detach(),
            'z2': z2.detach(),
            'q1': q1.detach(),
            'q2': q2.detach()
        }
    
    def get_encoder(self) -> nn.Module:
        """返回用于下游任务的编码器"""
        return self.backbone


# ============ 测试代码 ============
if __name__ == '__main__':
    print("测试 SwAV...")
    
    from torchvision.models import resnet18
    
    backbone = resnet18(weights=None)
    model = SwAV(backbone, proj_dim=128, num_prototypes=1000)
    
    # 测试前向传播
    x1 = torch.randn(32, 3, 224, 224)
    x2 = torch.randn(32, 3, 224, 224)
    
    output = model(x1, x2)
    
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"z1 shape: {output['z1'].shape}")
    print(f"q1 shape: {output['q1'].shape}")  # [B, K]
    print(f"q1 sum per sample: {output['q1'].sum(dim=1)[:5]}")  # 应该接近 1
    
    # 参数量
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total params: {num_params:.2f}M")
    
    print("\nSwAV 测试通过！")
