"""
SimCLR: A Simple Framework for Contrastive Learning of Visual Representations

论文: https://arxiv.org/abs/2002.05709

核心思想:
- 使用强数据增强生成正样本对
- 批内对比：每个样本的正样本是其增强版本，负样本是批内其他样本
- NT-Xent 损失函数
- 需要大批量 (4096-8192)

关键组件:
- 强数据增强
- 投影头 (MLP)
- NT-Xent 损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .base import SSLMethod, ProjectionHead, get_backbone_output_dim


class SimCLR(SSLMethod):
    """
    SimCLR: Simple Contrastive Learning of Representations
    
    架构:
        x1, x2 → Encoder → h → Projector → z1, z2
        Loss = NT-Xent(z1, z2)  (批内对比)
    
    Args:
        backbone: 特征提取网络
        proj_dim: 投影头输出维度 (默认 128)
        hidden_dim: 投影头隐藏层维度 (默认 2048)
        temperature: 温度参数 (默认 0.5)
        
    Example:
        >>> backbone = resnet50
        >>> model = SimCLR(backbone, proj_dim=128, temperature=0.5)
        >>> output = model(x1, x2)
        >>> loss = output['loss']
    
    Note:
        SimCLR 需要大批量 (推荐 256-4096)，小批量效果会显著下降
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        proj_dim: int = 128,
        hidden_dim: int = 2048,
        temperature: float = 0.5,
        proj_num_layers: int = 2
    ):
        feature_dim = get_backbone_output_dim(backbone)
        super().__init__(backbone, feature_dim)
        
        self.proj_dim = proj_dim
        self.temperature = temperature
        
        # 移除 backbone 的分类头
        self._remove_fc()
        
        # 投影头 (MLP)
        # SimCLR 使用 2 层 MLP: fc → ReLU → fc
        if proj_num_layers == 2:
            self.projector = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, proj_dim)
            )
        else:
            # 3 层投影头
            self.projector = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, proj_dim)
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
        x2: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x1: 第一个视图 [B, C, H, W]
            x2: 第二个视图 [B, C, H, W]
            
        Returns:
            包含 'loss', 'z1', 'z2', 'logits', 'labels' 的字典
        """
        batch_size = x1.shape[0]
        
        # 编码
        h1 = self.backbone(x1)  # [B, feature_dim]
        h2 = self.backbone(x2)
        
        # 投影
        z1 = self.projector(h1)  # [B, proj_dim]
        z2 = self.projector(h2)
        
        # 计算 NT-Xent 损失
        loss, logits, labels = self._nt_xent_loss(z1, z2)
        
        return {
            'loss': loss,
            'z1': z1.detach(),
            'z2': z2.detach(),
            'h1': h1.detach(),
            'h2': h2.detach(),
            'logits': logits.detach(),
            'labels': labels
        }
    
    def _nt_xent_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> tuple:
        """
        NT-Xent (Normalized Temperature-scaled Cross Entropy) 损失
        
        对于样本 i:
        - 正样本: i 的另一个视图
        - 负样本: 批内所有其他样本 (包括它们的两个视图)
        """
        batch_size = z1.shape[0]
        device = z1.device
        
        # L2 归一化
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # 拼接: [z1_0, z1_1, ..., z1_B, z2_0, z2_1, ..., z2_B]
        z = torch.cat([z1, z2], dim=0)  # [2B, proj_dim]
        
        # 相似度矩阵 [2B, 2B]
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        
        # 创建掩码，排除自己与自己的相似度
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        
        # 正样本标签
        # 对于 z1[i]，正样本是 z2[i]，位置是 batch_size + i
        # 对于 z2[i]，正样本是 z1[i]，位置是 i
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),  # z1 的正样本位置
            torch.arange(batch_size, device=device)  # z2 的正样本位置
        ])
        
        # 交叉熵损失
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss, sim_matrix, labels
    
    def get_encoder(self) -> nn.Module:
        """返回用于下游任务的编码器"""
        return self.backbone
    
    def get_representations(self, x: torch.Tensor) -> torch.Tensor:
        """获取特征表示"""
        return self.backbone(x)


# ============ 测试代码 ============
if __name__ == '__main__':
    print("测试 SimCLR...")
    
    from torchvision.models import resnet18
    
    backbone = resnet18(weights=None)
    model = SimCLR(backbone, proj_dim=128, temperature=0.5)
    
    # 测试前向传播
    batch_size = 32  # SimCLR 需要较大批量
    x1 = torch.randn(batch_size, 3, 224, 224)
    x2 = torch.randn(batch_size, 3, 224, 224)
    
    output = model(x1, x2)
    
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"z1 shape: {output['z1'].shape}")  # [32, 128]
    print(f"h1 shape: {output['h1'].shape}")  # [32, 512]
    print(f"Logits shape: {output['logits'].shape}")  # [64, 64]
    print(f"Labels shape: {output['labels'].shape}")  # [64]
    
    # 计算模型参数量
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total params: {num_params:.2f}M")
    
    # 测试不同批量大小的效果
    print("\n不同批量大小的 loss:")
    for bs in [8, 16, 32, 64]:
        x1 = torch.randn(bs, 3, 224, 224)
        x2 = torch.randn(bs, 3, 224, 224)
        output = model(x1, x2)
        print(f"  batch_size={bs}: loss={output['loss'].item():.4f}")
    
    print("\nSimCLR 测试通过！")
