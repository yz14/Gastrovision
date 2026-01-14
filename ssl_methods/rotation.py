"""
Rotation Prediction: Unsupervised Representation Learning by Predicting Image Rotations

论文: https://arxiv.org/abs/1803.07728

核心思想:
- 将图像随机旋转 0°, 90°, 180°, 270°
- 训练网络预测旋转角度
- 简单有效的预文本任务

这是最简单的自监督方法之一，不需要负样本对。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .base import SSLMethod, get_backbone_output_dim


class RotationPrediction(SSLMethod):
    """
    Rotation Prediction: 预测图像旋转角度
    
    架构:
        x (随机旋转) → Encoder → 4 分类头 → rotation_class
        Loss = CrossEntropy
    
    Args:
        backbone: 特征提取网络
        
    Example:
        >>> backbone = resnet50
        >>> model = RotationPrediction(backbone)
        >>> x = images  # 原始图像
        >>> output = model(x)  # 内部自动旋转并预测
        >>> loss = output['loss']
    
    Note:
        与其他 SSL 方法不同，Rotation 只需要单个视图，
        会在内部生成 4 个旋转版本
    """
    
    ROTATIONS = [0, 90, 180, 270]
    
    def __init__(self, backbone: nn.Module):
        feature_dim = get_backbone_output_dim(backbone)
        super().__init__(backbone, feature_dim)
        
        # 移除 backbone 的分类头
        self._remove_fc()
        
        # 4 分类头 (预测 0°, 90°, 180°, 270°)
        self.classifier = nn.Linear(feature_dim, 4)
    
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
        x2: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        有两种使用方式:
        1. 传入 (images, None): 内部自动生成所有旋转
        2. 传入 (rotated_images, labels): 直接使用提供的旋转图像和标签
        
        Args:
            x1: 图像 [B, C, H, W] 或 预旋转图像
            x2: None 或 旋转标签 [B]
            
        Returns:
            包含 'loss', 'logits', 'labels', 'accuracy' 的字典
        """
        if x2 is None:
            # 模式 1: 自动生成所有旋转
            return self._forward_auto_rotate(x1)
        else:
            # 模式 2: 使用提供的标签
            return self._forward_with_labels(x1, x2)
    
    def _forward_auto_rotate(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """自动生成所有旋转版本并预测"""
        batch_size = x.shape[0]
        device = x.device
        
        # 生成 4 个旋转版本
        rotated_images = []
        labels = []
        
        for i, angle in enumerate(self.ROTATIONS):
            rotated = self._rotate(x, angle)
            rotated_images.append(rotated)
            labels.append(torch.full((batch_size,), i, dtype=torch.long, device=device))
        
        # 拼接 [4B, C, H, W]
        rotated_images = torch.cat(rotated_images, dim=0)
        labels = torch.cat(labels, dim=0)
        
        return self._forward_with_labels(rotated_images, labels)
    
    def _forward_with_labels(
        self,
        x: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """使用提供的标签计算损失"""
        # 编码
        features = self.backbone(x)
        
        # 分类
        logits = self.classifier(features)
        
        # 损失
        loss = F.cross_entropy(logits, labels)
        
        # 准确率
        preds = logits.argmax(dim=1)
        accuracy = (preds == labels).float().mean()
        
        return {
            'loss': loss,
            'logits': logits.detach(),
            'labels': labels,
            'accuracy': accuracy.item(),
            'features': features.detach()
        }
    
    def _rotate(self, x: torch.Tensor, angle: int) -> torch.Tensor:
        """旋转图像"""
        if angle == 0:
            return x
        elif angle == 90:
            return x.flip(3).transpose(2, 3)
        elif angle == 180:
            return x.flip(2).flip(3)
        elif angle == 270:
            return x.flip(2).transpose(2, 3)
        else:
            raise ValueError(f"不支持的角度: {angle}")
    
    def get_encoder(self) -> nn.Module:
        """返回用于下游任务的编码器"""
        return self.backbone
    
    def get_representations(self, x: torch.Tensor) -> torch.Tensor:
        """获取特征表示"""
        return self.backbone(x)


# ============ 测试代码 ============
if __name__ == '__main__':
    print("测试 Rotation Prediction...")
    
    from torchvision.models import resnet18
    
    backbone = resnet18(weights=None)
    model = RotationPrediction(backbone)
    
    # 测试自动旋转模式
    x = torch.randn(8, 3, 224, 224)
    output = model(x)
    
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"Accuracy: {output['accuracy']:.4f}")
    print(f"Logits shape: {output['logits'].shape}")  # [32, 4] (4 rotations * 8 batch)
    print(f"Labels shape: {output['labels'].shape}")  # [32]
    
    # 测试旋转函数
    print("\n测试旋转函数:")
    for angle in [0, 90, 180, 270]:
        rotated = model._rotate(x[:1], angle)
        print(f"  {angle}°: shape = {rotated.shape}")
    
    # 计算模型参数量
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nTotal params: {num_params:.2f}M")
    
    print("\nRotation Prediction 测试通过！")
