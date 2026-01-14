"""
DINO: Self-Distillation with No Labels

论文: https://arxiv.org/abs/2104.14294
官方实现: https://github.com/facebookresearch/dino

核心思想:
- 自蒸馏：Student 学习 Teacher 的输出
- Teacher 通过 EMA 更新
- Centering 防止模式坍塌
- Multi-crop 策略

官方实现关键点:
- DINOLoss: 分离的损失类
- center 通过 EMA 更新
- teacher_temp 有 warmup
- 跳过 student 和 teacher 处理相同 view 的情况
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import copy

from .base import SSLMethod, get_backbone_output_dim


class DINOHead(nn.Module):
    """
    DINO 头部 (官方实现风格)
    
    MLP with last layer optionally normalized
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        use_bn: bool = False,
        norm_last_layer: bool = True
    ):
        super().__init__()
        nlayers = 3
        
        layers = [nn.Linear(in_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        
        # Last layer
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOLoss(nn.Module):
    """
    DINO 损失函数 (官方实现)
    
    Cross-entropy between softmax outputs of teacher and student.
    """
    
    def __init__(
        self,
        out_dim: int,
        ncrops: int = 2,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 0,
        nepochs: int = 100,
        student_temp: float = 0.1,
        center_momentum: float = 0.9
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        
        # Teacher temperature schedule
        import numpy as np
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, max(1, warmup_teacher_temp_epochs)),
            np.ones(max(1, nepochs - warmup_teacher_temp_epochs)) * teacher_temp
        ))
    
    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        epoch: int = 0
    ) -> torch.Tensor:
        """
        Args:
            student_output: [B*ncrops, out_dim]
            teacher_output: [B*2, out_dim] (only global views)
            epoch: current epoch
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
        
        # Teacher centering and sharpening
        epoch = min(epoch, len(self.teacher_temp_schedule) - 1)
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)
        
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # Skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss
    
    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        """Update center using EMA"""
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINO(SSLMethod):
    """
    DINO: Self-Distillation with No Labels
    
    官方实现参考: https://github.com/facebookresearch/dino
    
    架构:
        Student: crops → backbone → head → student_output
        Teacher: global_crops → backbone → head → teacher_output (EMA)
        
        Loss = DINOLoss(student_output, teacher_output)
    
    Args:
        backbone: 特征提取网络
        out_dim: 输出维度 (默认 65536)
        hidden_dim: 隐藏层维度 (默认 2048)
        bottleneck_dim: bottleneck 维度 (默认 256)
        momentum: Teacher 动量 (默认 0.996)
        teacher_temp: Teacher 温度 (默认 0.04)
        student_temp: Student 温度 (默认 0.1)
        center_momentum: 中心动量 (默认 0.9)
        use_bn: 是否在 head 中使用 BN (默认 False)
        norm_last_layer: 是否归一化最后一层 (默认 True)
        
    Example:
        >>> model = DINO(backbone, out_dim=65536)
        >>> output = model(global_crop1, global_crop2)
        >>> loss = output['loss']
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        out_dim: int = 65536,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        momentum: float = 0.996,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        use_bn: bool = False,
        norm_last_layer: bool = True
    ):
        feature_dim = get_backbone_output_dim(backbone)
        super().__init__(backbone, feature_dim)
        
        self.momentum = momentum
        self.out_dim = out_dim
        
        # Student network
        self.student_backbone = backbone
        self._remove_fc(self.student_backbone)
        
        self.student_head = DINOHead(
            in_dim=feature_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            use_bn=use_bn,
            norm_last_layer=norm_last_layer
        )
        
        # Teacher network (EMA)
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_head = DINOHead(
            in_dim=feature_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            use_bn=use_bn,
            norm_last_layer=False  # Teacher 不归一化最后一层
        )
        
        # 复制 student head 的权重到 teacher head
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        
        # 冻结 teacher
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
        for param in self.teacher_head.parameters():
            param.requires_grad = False
        
        # Loss
        self.dino_loss = DINOLoss(
            out_dim=out_dim,
            ncrops=2,  # 简化版本只使用 2 个 global crops
            teacher_temp=teacher_temp,
            student_temp=student_temp,
            center_momentum=center_momentum
        )
    
    def _remove_fc(self, model: nn.Module):
        """移除分类头"""
        if hasattr(model, 'fc'):
            model.fc = nn.Identity()
        elif hasattr(model, 'classifier'):
            model.classifier = nn.Identity()
        elif hasattr(model, 'head'):
            model.head = nn.Identity()
    
    @torch.no_grad()
    def _update_teacher(self):
        """动量更新 Teacher (官方实现)"""
        for param_s, param_t in zip(
            self.student_backbone.parameters(),
            self.teacher_backbone.parameters()
        ):
            param_t.data.mul_(self.momentum).add_((1 - self.momentum) * param_s.data)
        
        for param_s, param_t in zip(
            self.student_head.parameters(),
            self.teacher_head.parameters()
        ):
            param_t.data.mul_(self.momentum).add_((1 - self.momentum) * param_s.data)
    
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        epoch: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播 (简化版本，只使用 2 个 global crops)
        
        Args:
            x1: 第一个 global crop [B, C, H, W]
            x2: 第二个 global crop [B, C, H, W]
            epoch: 当前 epoch (用于温度调度)
            
        Returns:
            包含 'loss' 等的字典
        """
        # Student forward
        s1 = self.student_head(self.student_backbone(x1))
        s2 = self.student_head(self.student_backbone(x2))
        student_output = torch.cat([s1, s2], dim=0)  # [2B, out_dim]
        
        # Teacher forward (no gradient)
        with torch.no_grad():
            self._update_teacher()
            t1 = self.teacher_head(self.teacher_backbone(x1))
            t2 = self.teacher_head(self.teacher_backbone(x2))
            teacher_output = torch.cat([t1, t2], dim=0)  # [2B, out_dim]
        
        # 计算损失
        loss = self.dino_loss(student_output, teacher_output, epoch)
        
        return {
            'loss': loss,
            'student_output': student_output.detach(),
            'teacher_output': teacher_output.detach(),
            'center': self.dino_loss.center.clone()
        }
    
    def get_encoder(self) -> nn.Module:
        """返回 Student backbone (用于下游任务)"""
        return self.student_backbone


# ============ 测试代码 ============
if __name__ == '__main__':
    print("测试 DINO (基于官方实现)...")
    
    from torchvision.models import resnet18
    
    # 使用较小的 out_dim 用于测试
    backbone = resnet18(weights=None)
    model = DINO(backbone, out_dim=4096)
    
    # 测试前向传播
    x1 = torch.randn(8, 3, 224, 224)
    x2 = torch.randn(8, 3, 224, 224)
    
    output = model(x1, x2, epoch=0)
    
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"Student output shape: {output['student_output'].shape}")
    print(f"Teacher output shape: {output['teacher_output'].shape}")
    print(f"Center norm: {output['center'].norm().item():.4f}")
    
    # 测试动量更新
    old_param = next(model.teacher_backbone.parameters()).clone()
    for _ in range(5):
        output = model(x1, x2)
    new_param = next(model.teacher_backbone.parameters())
    print(f"Teacher updated: {not torch.equal(old_param, new_param)}")
    
    # 参数量
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total params: {num_params:.2f}M, Trainable: {trainable:.2f}M")
    
    print("\nDINO 测试通过！")
