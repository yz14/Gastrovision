"""
SimSiam: Exploring Simple Siamese Representation Learning

论文: https://arxiv.org/abs/2011.10566
官方实现: https://github.com/facebookresearch/simsiam

核心思想:
- 无需负样本、动量编码器、大批量
- 使用 stop-gradient 防止模式坍塌
- 结构: Encoder → Projector → Predictor

损失函数:
L = -0.5 * (cos(p1, z2.detach()) + cos(p2, z1.detach()))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .base import SSLMethod, ProjectionHead, PredictionHead, get_backbone_output_dim


class SimSiam(SSLMethod):
    """
    SimSiam: Simple Siamese Representation Learning
    
    架构:
        x1, x2 → Encoder → Projector → z1, z2
                                  ↓
                            Predictor → p1, p2
        
        Loss = -cos(p1, z2.detach()) - cos(p2, z1.detach())
    
    Args:
        backbone: 特征提取网络 (如 ResNet)
        proj_dim: 投影头输出维度 (默认 2048)
        pred_dim: 预测头隐藏层维度 (默认 512)
        
    Example:
        >>> backbone = resnet50(num_classes=1000)
        >>> model = SimSiam(backbone, proj_dim=2048, pred_dim=512)
        >>> x1, x2 = augment(images)  # 两个增强视图
        >>> output = model(x1, x2)
        >>> loss = output['loss']
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        proj_dim: int = 2048,
        pred_dim: int = 512,
        proj_num_layers: int = 3
    ):
        # 获取 backbone 输出维度
        feature_dim = get_backbone_output_dim(backbone)
        super().__init__(backbone, feature_dim)
        
        self.proj_dim = proj_dim
        self.pred_dim = pred_dim
        
        # 移除 backbone 的分类头
        self._remove_fc()
        
        # 3 层投影头 (与官方实现一致)
        # fc → BN → ReLU → fc → BN → ReLU → fc → BN
        self.projector = self._build_projector(
            in_dim=feature_dim,
            hidden_dim=feature_dim,  # 官方使用 prev_dim
            out_dim=proj_dim,
            num_layers=proj_num_layers)
        
        # 2 层预测头
        # fc → BN → ReLU → fc
        self.predictor = PredictionHead(
            in_dim=proj_dim,
            hidden_dim=pred_dim,
            out_dim=proj_dim,
            use_bn=True)
    
    def _remove_fc(self):
        """移除 backbone 的分类头"""
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()
    
    def _build_projector(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3
    ) -> nn.Module:
        """
        构建投影头 (与官方实现一致)
        
        官方实现:
            fc(prev_dim, prev_dim) → BN → ReLU
            fc(prev_dim, prev_dim) → BN → ReLU
            fc(prev_dim, dim) → BN (无 ReLU, 无 affine)
        """
        layers = []
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)])
        
        # 最后一层
        layers.extend([
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False)  # 无 affine，与官方一致
        ])
        
        return nn.Sequential(*layers)
    
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
            包含 'loss', 'z1', 'z2', 'p1', 'p2' 的字典
        """
        # 编码
        f1 = self.backbone(x1)  # [B, feature_dim]
        f2 = self.backbone(x2)
        
        # 投影
        z1 = self.projector(f1)  # [B, proj_dim]
        z2 = self.projector(f2)
        
        # 预测
        p1 = self.predictor(z1)  # [B, proj_dim]
        p2 = self.predictor(z2)
        
        # 计算损失 (负余弦相似度)
        loss = self._loss(p1, p2, z1, z2)
        
        return {
            'loss': loss,
            'z1': z1.detach(),
            'z2': z2.detach(),
            'p1': p1.detach(),
            'p2': p2.detach()
        }
    
    def _loss(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 SimSiam 损失
        
        L = -0.5 * (cos(p1, z2.detach()) + cos(p2, z1.detach()))
        """
        # L2 归一化
        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # 负余弦相似度 (z 需要 stop-gradient)
        loss1 = -(p1 * z2.detach()).sum(dim=1).mean()
        loss2 = -(p2 * z1.detach()).sum(dim=1).mean()
        
        loss = 0.5 * (loss1 + loss2)
        return loss
    
    def get_encoder(self) -> nn.Module:
        """
        返回用于下游任务的编码器 (backbone)
        """
        return self.backbone
    
    def get_representations(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取图像的特征表示 (用于评估)
        
        Args:
            x: 输入图像 [B, C, H, W]
            
        Returns:
            特征向量 [B, feature_dim]
        """
        return self.backbone(x)


# ============ 测试代码 ============
if __name__ == '__main__':
    print("测试 SimSiam...")
    
    # 创建一个简单的 backbone
    import sys
    sys.path.append('..')
    
    from torchvision.models import resnet18
    
    backbone = resnet18(weights=None)
    model = SimSiam(backbone, proj_dim=2048, pred_dim=512)
    
    # 测试前向传播
    x1 = torch.randn(4, 3, 224, 224)
    x2 = torch.randn(4, 3, 224, 224)
    
    output = model(x1, x2)
    
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"z1 shape: {output['z1'].shape}")
    print(f"p1 shape: {output['p1'].shape}")
    
    # 测试 get_encoder
    encoder = model.get_encoder()
    features = encoder(x1)
    print(f"Encoder output shape: {features.shape}")
    
    # 计算模型参数量
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total params: {num_params:.2f}M")
    
    print("\nSimSiam 测试通过！")
