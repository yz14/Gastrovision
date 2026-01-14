"""
BYOL: Bootstrap Your Own Latent

论文: https://arxiv.org/abs/2006.07733

核心思想:
- 不需要负样本
- Online network 预测 Target network 的输出
- Target network 通过动量更新
- Stop-gradient 防止坍塌

架构:
    Online:  x → encoder → projector → predictor → prediction
    Target:  x → encoder → projector → target (stop-grad)
    Loss = -cosine_similarity(prediction, target.detach())
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import copy

from .base import SSLMethod, ProjectionHead, PredictionHead, get_backbone_output_dim


class BYOL(SSLMethod):
    """
    BYOL: Bootstrap Your Own Latent
    
    架构:
        Online:  x1 → encoder_o → projector_o → predictor → p1
        Target:  x2 → encoder_t → projector_t → z2 (stop-grad)
        
        Loss = 2 - 2 * cos(p1, z2.detach())
    
    Args:
        backbone: 特征提取网络
        proj_dim: 投影头输出维度 (默认 256)
        hidden_dim: 隐藏层维度 (默认 4096)
        pred_dim: 预测头隐藏层维度 (默认 256)
        momentum: EMA 动量系数 (默认 0.996)
        
    Example:
        >>> backbone = resnet50
        >>> model = BYOL(backbone, proj_dim=256, hidden_dim=4096)
        >>> output = model(x1, x2)
        >>> loss = output['loss']
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        proj_dim: int = 256,
        hidden_dim: int = 4096,
        pred_dim: int = 256,
        momentum: float = 0.996
    ):
        feature_dim = get_backbone_output_dim(backbone)
        super().__init__(backbone, feature_dim)
        
        self.proj_dim = proj_dim
        self.momentum = momentum
        
        # Online network
        self.online_encoder = backbone
        self._remove_fc(self.online_encoder)
        
        self.online_projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_dim),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, proj_dim)
        )
        
        # Target network (EMA of online network)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        
        # 冻结 target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
    
    def _remove_fc(self, model: nn.Module):
        """移除分类头"""
        if hasattr(model, 'fc'):
            model.fc = nn.Identity()
        elif hasattr(model, 'classifier'):
            model.classifier = nn.Identity()
        elif hasattr(model, 'head'):
            model.head = nn.Identity()
    
    @torch.no_grad()
    def _update_target_network(self):
        """动量更新 target network"""
        for param_o, param_t in zip(
            self.online_encoder.parameters(), 
            self.target_encoder.parameters()
        ):
            param_t.data = self.momentum * param_t.data + (1 - self.momentum) * param_o.data
        
        for param_o, param_t in zip(
            self.online_projector.parameters(), 
            self.target_projector.parameters()
        ):
            param_t.data = self.momentum * param_t.data + (1 - self.momentum) * param_o.data
    
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
            包含 'loss' 等的字典
        """
        # Online network
        h1 = self.online_encoder(x1)
        h2 = self.online_encoder(x2)
        
        z1 = self.online_projector(h1)
        z2 = self.online_projector(h2)
        
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        
        # Target network (no gradient)
        with torch.no_grad():
            self._update_target_network()
            
            t1 = self.target_projector(self.target_encoder(x1))
            t2 = self.target_projector(self.target_encoder(x2))
        
        # 计算损失 (对称)
        loss = self._loss(p1, t2) + self._loss(p2, t1)
        loss = loss / 2
        
        return {
            'loss': loss,
            'z1': z1.detach(),
            'z2': z2.detach(),
            'p1': p1.detach(),
            'p2': p2.detach()
        }
    
    def _loss(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        BYOL 损失: 2 - 2 * cosine_similarity
        
        等价于 MSE(normalize(p), normalize(t.detach()))
        """
        p = F.normalize(p, dim=1)
        t = F.normalize(t, dim=1)
        
        # 负余弦相似度
        loss = 2 - 2 * (p * t.detach()).sum(dim=1).mean()
        return loss
    
    def get_encoder(self) -> nn.Module:
        """返回 online encoder (用于下游任务)"""
        return self.online_encoder
    
    def get_representations(self, x: torch.Tensor) -> torch.Tensor:
        """获取特征表示"""
        return self.online_encoder(x)


# ============ 测试代码 ============
if __name__ == '__main__':
    print("测试 BYOL...")
    
    from torchvision.models import resnet18
    
    backbone = resnet18(weights=None)
    model = BYOL(backbone, proj_dim=256, hidden_dim=4096, momentum=0.996)
    
    # 测试前向传播
    x1 = torch.randn(8, 3, 224, 224)
    x2 = torch.randn(8, 3, 224, 224)
    
    output = model(x1, x2)
    
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"z1 shape: {output['z1'].shape}")
    print(f"p1 shape: {output['p1'].shape}")
    
    # 测试动量更新
    old_param = next(model.target_encoder.parameters()).clone()
    for _ in range(10):
        output = model(x1, x2)
    new_param = next(model.target_encoder.parameters())
    print(f"Target encoder updated: {not torch.equal(old_param, new_param)}")
    
    # 计算模型参数量
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total params: {num_params:.2f}M, Trainable: {trainable:.2f}M")
    
    print("\nBYOL 测试通过！")
