"""
Instance Discrimination (InstDisc)

论文: "Unsupervised Feature Learning via Non-Parametric Instance Discrimination"
链接: https://arxiv.org/abs/1805.01978

核心思想:
- 将每个图像视为独立的类别
- 使用 Memory Bank 存储所有样本的特征
- NCE Loss 作为损失函数

这是 MoCo 的前身，引入了非参数化实例判别的概念。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .base import SSLMethod, get_backbone_output_dim


class MemoryBank(nn.Module):
    """
    Memory Bank: 存储所有样本的特征
    
    用于提供大量负样本，而不需要大批量
    """
    
    def __init__(self, num_samples: int, feature_dim: int):
        """
        Args:
            num_samples: 数据集样本数
            feature_dim: 特征维度
        """
        super().__init__()
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        
        # 初始化为随机向量并归一化
        self.register_buffer(
            'bank',
            F.normalize(torch.randn(num_samples, feature_dim), dim=1)
        )
    
    def update(self, features: torch.Tensor, indices: torch.Tensor):
        """
        更新 memory bank 中的特征
        
        Args:
            features: 新特征 [B, D]
            indices: 样本索引 [B]
        """
        with torch.no_grad():
            features = F.normalize(features, dim=1)
            self.bank[indices] = features
    
    def get_negatives(
        self,
        indices: torch.Tensor,
        num_negatives: int = 4096
    ) -> torch.Tensor:
        """
        获取负样本
        
        Args:
            indices: 当前样本的索引 (需要排除)
            num_negatives: 负样本数量
            
        Returns:
            负样本特征 [num_negatives, D]
        """
        # 随机采样负样本索引
        all_indices = torch.arange(self.num_samples, device=self.bank.device)
        
        # 创建掩码排除当前样本
        mask = torch.ones(self.num_samples, dtype=torch.bool, device=self.bank.device)
        mask[indices] = False
        
        # 从可用索引中采样
        available_indices = all_indices[mask]
        num_negatives = min(num_negatives, len(available_indices))
        
        perm = torch.randperm(len(available_indices))[:num_negatives]
        neg_indices = available_indices[perm]
        
        return self.bank[neg_indices]


class InstDisc(SSLMethod):
    """
    Instance Discrimination with Memory Bank
    
    架构:
        x → Encoder → z (normalized)
        Score = z · [z_pos, memory_bank] / T
        Loss = NCE
    
    Args:
        backbone: 特征提取网络
        num_samples: 数据集样本数 (用于初始化 memory bank)
        feature_dim: 特征维度 (默认 128)
        num_negatives: 每次采样的负样本数 (默认 4096)
        temperature: 温度参数 (默认 0.07)
        momentum: memory bank 更新动量 (默认 0.5)
        
    Example:
        >>> model = InstDisc(backbone, num_samples=50000, feature_dim=128)
        >>> output = model(images, indices)  # indices 是批次中每个样本的数据集索引
        >>> loss = output['loss']
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        num_samples: int,
        feature_dim: int = 128,
        num_negatives: int = 4096,
        temperature: float = 0.07,
        momentum: float = 0.5
    ):
        backbone_dim = get_backbone_output_dim(backbone)
        super().__init__(backbone, backbone_dim)
        
        self.feature_dim = feature_dim
        self.num_negatives = num_negatives
        self.temperature = temperature
        self.momentum = momentum
        self.num_samples = num_samples
        
        # 移除 backbone 的分类头
        self._remove_fc()
        
        # 投影头 (简单的线性层)
        self.projector = nn.Linear(backbone_dim, feature_dim)
        
        # Memory Bank
        self.memory_bank = MemoryBank(num_samples, feature_dim)
    
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
        x2: torch.Tensor = None,
        indices: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x1: 图像 [B, C, H, W]
            x2: 第二个视图 (可选，用于标准 SSL 模式)
            indices: 样本在数据集中的索引 [B]
            
        Returns:
            包含 'loss', 'features' 等的字典
        """
        batch_size = x1.shape[0]
        device = x1.device
        
        # 如果没有提供索引，生成假索引
        if indices is None:
            indices = torch.arange(batch_size, device=device)
        
        # 编码当前批次
        h = self.backbone(x1)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        
        # 从 memory bank 获取负样本
        negatives = self.memory_bank.get_negatives(indices, self.num_negatives)
        
        # 计算 NCE loss
        # 正样本: memory bank 中自己的旧特征
        positives = self.memory_bank.bank[indices]
        
        # 计算相似度
        pos_sim = (z * positives).sum(dim=1, keepdim=True)  # [B, 1]
        neg_sim = torch.mm(z, negatives.t())  # [B, K]
        
        logits = torch.cat([pos_sim, neg_sim], dim=1) / self.temperature  # [B, 1+K]
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        loss = F.cross_entropy(logits, labels)
        
        # 更新 memory bank (使用动量)
        with torch.no_grad():
            new_features = self.momentum * self.memory_bank.bank[indices] + (1 - self.momentum) * z
            self.memory_bank.update(new_features, indices)
        
        return {
            'loss': loss,
            'features': z.detach(),
            'logits': logits.detach()
        }
    
    def get_encoder(self) -> nn.Module:
        """返回用于下游任务的编码器"""
        return self.backbone


# ============ 测试代码 ============
if __name__ == '__main__':
    print("测试 InstDisc...")
    
    from torchvision.models import resnet18
    
    # 假设数据集有 1000 个样本
    num_samples = 1000
    backbone = resnet18(weights=None)
    model = InstDisc(backbone, num_samples=num_samples, feature_dim=128, num_negatives=256)
    
    # 测试前向传播
    batch_size = 16
    x = torch.randn(batch_size, 3, 224, 224)
    indices = torch.randint(0, num_samples, (batch_size,))
    
    output = model(x, indices=indices)
    
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"Features shape: {output['features'].shape}")
    print(f"Logits shape: {output['logits'].shape}")  # [16, 257] = [B, 1+K]
    
    # 测试多次更新 memory bank
    for i in range(10):
        indices = torch.randint(0, num_samples, (batch_size,))
        output = model(x, indices=indices)
    print(f"After 10 updates, loss: {output['loss'].item():.4f}")
    
    # 参数量
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total params: {num_params:.2f}M")
    
    print("\nInstDisc 测试通过！")
