"""
SSL 基类模块

定义所有自监督学习方法的抽象基类和通用组件。
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional


class SSLMethod(ABC, nn.Module):
    """
    自监督学习方法抽象基类
    
    所有 SSL 方法都应继承此类并实现必要的抽象方法。
    
    Attributes:
        backbone: 特征提取网络 (如 ResNet)
        feature_dim: backbone 输出特征维度
    """
    
    def __init__(self, backbone: nn.Module, feature_dim: int = 2048):
        """
        Args:
            backbone: 特征提取网络
            feature_dim: backbone 输出特征维度
        """
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
    
    @abstractmethod
    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x1: 第一个视图的图像批次 [B, C, H, W]
            x2: 第二个视图的图像批次 [B, C, H, W] (可选，某些方法如 Rotation 不需要)
            
        Returns:
            包含以下键的字典:
            - 'loss': 训练损失
            - 'features': 学习到的特征 (可选)
            - 其他方法特定的输出
        """
        pass
    
    @abstractmethod
    def get_encoder(self) -> nn.Module:
        """
        返回用于下游任务微调的编码器
        
        Returns:
            可用于分类任务的编码器 (通常是 backbone 部分)
        """
        pass
    
    def get_feature_dim(self) -> int:
        """返回特征维度"""
        return self.feature_dim
    
    def load_pretrained(self, checkpoint_path: str, strict: bool = False):
        """
        加载预训练权重
        
        Args:
            checkpoint_path: checkpoint 文件路径
            strict: 是否严格匹配所有键
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 移除 'module.' 前缀 (DDP 训练产生的)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        if missing:
            print(f"  缺失的键: {missing[:5]}..." if len(missing) > 5 else f"  缺失的键: {missing}")
        if unexpected:
            print(f"  多余的键: {unexpected[:5]}..." if len(unexpected) > 5 else f"  多余的键: {unexpected}")
        
        return missing, unexpected


class ProjectionHead(nn.Module):
    """
    MLP 投影头 (SimCLR, MoCo, BYOL, SimSiam 等通用)
    
    将 backbone 特征投影到对比学习空间。
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 2048,
        out_dim: int = 2048,
        num_layers: int = 2,
        use_bn: bool = True,
        use_bias: bool = False,
        last_bn: bool = True,
        last_bn_affine: bool = False
    ):
        """
        Args:
            in_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            out_dim: 输出维度
            num_layers: MLP 层数 (2 或 3)
            use_bn: 是否使用 BatchNorm
            use_bias: 线性层是否使用 bias
            last_bn: 最后是否加 BN
            last_bn_affine: 最后的 BN 是否有可学习参数
        """
        super().__init__()
        
        layers = []
        
        # 第一层
        layers.append(nn.Linear(in_dim, hidden_dim, bias=use_bias))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        
        # 中间层 (如果 num_layers > 2)
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=use_bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        
        # 最后一层
        layers.append(nn.Linear(hidden_dim, out_dim, bias=use_bias))
        if last_bn:
            layers.append(nn.BatchNorm1d(out_dim, affine=last_bn_affine))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class PredictionHead(nn.Module):
    """
    预测头 (BYOL, SimSiam 专用)
    
    用于预测 target network 的输出。
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        out_dim: int = 2048,
        use_bn: bool = True
    ):
        """
        Args:
            in_dim: 输入维度 (projector 输出维度)
            hidden_dim: 隐藏层维度
            out_dim: 输出维度 (通常与 in_dim 相同)
        """
        super().__init__()
        
        layers = [
            nn.Linear(in_dim, hidden_dim, bias=False),
        ]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.extend([
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)  # 最后一层有 bias
        ])
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def get_backbone_output_dim(backbone: nn.Module) -> int:
    """
    获取 backbone 的输出特征维度
    
    Args:
        backbone: 特征提取网络
        
    Returns:
        特征维度
    """
    # 尝试常见的属性名
    if hasattr(backbone, 'fc'):
        if hasattr(backbone.fc, 'in_features'):
            return backbone.fc.in_features
        elif hasattr(backbone.fc, 'weight'):
            return backbone.fc.weight.shape[1]
    
    if hasattr(backbone, 'classifier'):
        if isinstance(backbone.classifier, nn.Sequential):
            for layer in reversed(list(backbone.classifier)):
                if hasattr(layer, 'in_features'):
                    return layer.in_features
        elif hasattr(backbone.classifier, 'in_features'):
            return backbone.classifier.in_features
    
    if hasattr(backbone, 'head'):
        if hasattr(backbone.head, 'in_features'):
            return backbone.head.in_features
    
    if hasattr(backbone, 'num_features'):
        return backbone.num_features
    
    raise ValueError("无法自动检测 backbone 输出维度，请手动指定 feature_dim")


def remove_fc(backbone: nn.Module) -> Tuple[nn.Module, int]:
    """
    移除 backbone 的分类头，返回纯特征提取器
    
    Args:
        backbone: 带分类头的网络
        
    Returns:
        (无分类头的 backbone, 特征维度)
    """
    feature_dim = get_backbone_output_dim(backbone)
    
    if hasattr(backbone, 'fc'):
        backbone.fc = nn.Identity()
    elif hasattr(backbone, 'classifier'):
        backbone.classifier = nn.Identity()
    elif hasattr(backbone, 'head'):
        backbone.head = nn.Identity()
    
    return backbone, feature_dim
