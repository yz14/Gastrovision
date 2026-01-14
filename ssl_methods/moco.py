"""
MoCo: Momentum Contrast for Unsupervised Visual Representation Learning

论文:
- MoCo v1: https://arxiv.org/abs/1911.05722
- MoCo v2: https://arxiv.org/abs/2003.04297

官方实现: https://github.com/facebookresearch/moco

核心思想:
- 使用动量更新的 key encoder 保持特征一致性
- 使用队列存储大量负样本，解耦批大小和负样本数量
- InfoNCE 损失

MoCo v2 改进:
- 使用 MLP 投影头 (SimCLR 启发)
- 更强的数据增强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import copy

from .base import SSLMethod, ProjectionHead, get_backbone_output_dim


class MoCo(SSLMethod):
    """
    MoCo: Momentum Contrast
    
    架构:
        Query: x_q → encoder_q → q
        Key:   x_k → encoder_k (momentum) → k
        
        Loss = CrossEntropy(q·[k, queue] / T, labels=0)
    
    Args:
        backbone: 特征提取网络
        dim: 输出特征维度 (默认 128)
        K: 队列大小 (默认 65536)
        m: 动量系数 (默认 0.999)
        T: 温度参数 (默认 0.07)
        mlp: 是否使用 MLP 投影头 (MoCo v2)
        
    Example:
        >>> backbone = resnet50
        >>> model = MoCo(backbone, dim=128, K=65536, mlp=True)  # MoCo v2
        >>> output = model(x_q, x_k)
        >>> loss = output['loss']
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        dim: int = 128,
        K: int = 65536,
        m: float = 0.999,
        T: float = 0.07,
        mlp: bool = True,  # MoCo v2 默认使用 MLP
        hidden_dim: int = 2048
    ):
        feature_dim = get_backbone_output_dim(backbone)
        super().__init__(backbone, feature_dim)
        
        self.K = K
        self.m = m
        self.T = T
        self.dim = dim
        self.use_mlp = mlp
        
        # Query encoder
        self.encoder_q = backbone
        self._remove_fc(self.encoder_q)
        
        # Key encoder (动量更新)
        self.encoder_k = copy.deepcopy(backbone)
        self._remove_fc(self.encoder_k)
        
        # 投影头
        if mlp:
            # MoCo v2: 2-layer MLP
            self.projector_q = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, dim)
            )
            self.projector_k = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, dim)
            )
        else:
            # MoCo v1: linear projection
            self.projector_q = nn.Linear(feature_dim, dim)
            self.projector_k = nn.Linear(feature_dim, dim)
        
        # 初始化 key encoder 和 projector（复制 query 的权重）
        self._copy_params(self.encoder_q, self.encoder_k)
        self._copy_params(self.projector_q, self.projector_k)
        
        # 冻结 key encoder
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        for param in self.projector_k.parameters():
            param.requires_grad = False
        
        # 队列
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def _remove_fc(self, model: nn.Module):
        """移除分类头"""
        if hasattr(model, 'fc'):
            model.fc = nn.Identity()
        elif hasattr(model, 'classifier'):
            model.classifier = nn.Identity()
        elif hasattr(model, 'head'):
            model.head = nn.Identity()
    
    def _copy_params(self, src: nn.Module, dst: nn.Module):
        """复制参数"""
        for param_src, param_dst in zip(src.parameters(), dst.parameters()):
            param_dst.data.copy_(param_src.data)
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """动量更新 key encoder"""
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        
        for param_q, param_k in zip(
            self.projector_q.parameters(), self.projector_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """更新队列"""
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # 如果队列不够大，循环填充
        if ptr + batch_size > self.K:
            # 分两部分填充
            remaining = self.K - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
            ptr = batch_size - remaining
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K
        
        self.queue_ptr[0] = ptr
    
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x1: query 图像 [B, C, H, W]
            x2: key 图像 [B, C, H, W]
            
        Returns:
            包含 'loss', 'logits', 'labels' 的字典
        """
        # Query forward
        q = self.projector_q(self.encoder_q(x1))
        q = F.normalize(q, dim=1)
        
        # Key forward (no gradient)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            
            k = self.projector_k(self.encoder_k(x2))
            k = F.normalize(k, dim=1)
        
        # 计算 logits
        # 正样本: N x 1
        l_pos = torch.einsum('nc,nc->n', q, k).unsqueeze(-1)
        # 负样本 (从队列): N x K
        l_neg = torch.einsum('nc,ck->nk', q, self.queue.clone().detach())
        
        # logits: N x (1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits = logits / self.T
        
        # labels: 正样本在第 0 位
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # 计算损失
        loss = F.cross_entropy(logits, labels)
        
        # 更新队列
        self._dequeue_and_enqueue(k)
        
        return {
            'loss': loss,
            'logits': logits.detach(),
            'labels': labels,
            'q': q.detach(),
            'k': k.detach()
        }
    
    def get_encoder(self) -> nn.Module:
        """返回 query encoder (用于下游任务)"""
        return self.encoder_q
    
    def get_representations(self, x: torch.Tensor) -> torch.Tensor:
        """获取特征表示"""
        return self.encoder_q(x)


# ============ 测试代码 ============
if __name__ == '__main__':
    print("测试 MoCo...")
    
    from torchvision.models import resnet18
    
    backbone = resnet18(weights=None)
    model = MoCo(backbone, dim=128, K=4096, mlp=True)  # 小队列用于测试
    
    # 测试前向传播
    x1 = torch.randn(8, 3, 224, 224)
    x2 = torch.randn(8, 3, 224, 224)
    
    output = model(x1, x2)
    
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"Logits shape: {output['logits'].shape}")  # [8, 1+4096]
    print(f"q shape: {output['q'].shape}")  # [8, 128]
    
    # 多次前向传播测试队列更新
    for i in range(10):
        output = model(x1, x2)
    print(f"After 10 updates, queue_ptr: {model.queue_ptr.item()}")
    
    # 计算模型参数量
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total params: {num_params:.2f}M, Trainable: {trainable_params:.2f}M")
    
    print("\nMoCo 测试通过！")
