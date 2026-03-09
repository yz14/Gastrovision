"""
MetricLearningWrapper - 为任意 backbone 添加 (logits, features) 双输出

核心问题：
  torchvision 模型的 forward() 只返回 logits（分类结果），
  而度量学习损失需要 backbone 的中间特征（如 ResNet50 的 2048-dim features）。

解决方案：
  通过 PyTorch forward hook 捕获 backbone 最后一层池化输出，
  使 forward() 返回 (logits, features) 元组。
  
  Trainer 已经正确处理 tuple 输出：
    if isinstance(outputs, tuple):
        logits, features = outputs[0], outputs[-1]

支持的模型架构：
  - ResNet / ResNeXt / Wide ResNet (hook avgpool, feature_dim=2048)
  - ConvNeXt (hook avgpool, feature_dim varies)
  - EfficientNet V2 (hook avgpool, feature_dim varies)
  - Swin Transformer (hook flatten/norm, feature_dim varies)
  - GastroNet (hook avgpool, feature_dim=2048)
"""

import torch
import torch.nn as nn
from typing import Optional


class MetricLearningWrapper(nn.Module):
    """
    包装任意分类模型，使其同时输出 logits 和 features。
    
    原理:
      在 backbone 的全局池化层注册 forward hook，
      捕获池化后的特征向量。forward() 返回 (logits, features)。
    
    用法:
      model = torchvision.models.resnet50(num_classes=23)
      wrapped = MetricLearningWrapper(model)
      logits, features = wrapped(images)  # features.shape = (B, 2048)
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._features: Optional[torch.Tensor] = None
        self._hook_handle = None
        self._feature_dim: Optional[int] = None
        
        # 自动检测并注册 hook
        self._register_feature_hook()
    
    def _register_feature_hook(self):
        """自动检测 backbone 的池化层并注册 hook"""
        target_layer = None
        
        # 策略 1: ResNet / ResNeXt / Wide ResNet / GastroNet — 有 avgpool 属性
        if hasattr(self.model, 'avgpool'):
            target_layer = self.model.avgpool
        
        # 策略 2: ConvNeXt — classifier 是 Sequential(LayerNorm, Flatten, Linear)
        # 需要 hook Flatten 之前的 avgpool
        elif hasattr(self.model, 'classifier') and hasattr(self.model, 'avgpool'):
            target_layer = self.model.avgpool
        
        # 策略 3: EfficientNet V2 — 有 avgpool
        elif hasattr(self.model, 'features') and hasattr(self.model, 'avgpool'):
            target_layer = self.model.avgpool
        
        # 策略 4: Swin Transformer — 有 norm + head
        elif hasattr(self.model, 'norm') and hasattr(self.model, 'head'):
            target_layer = self.model.norm
        
        # 策略 5: 通用 — 尝试找最后一个 AdaptiveAvgPool2d
        if target_layer is None:
            for name, module in reversed(list(self.model.named_modules())):
                if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d)):
                    target_layer = module
                    break
        
        if target_layer is None:
            raise RuntimeError(
                "无法自动检测 backbone 的池化层。"
                "请确保模型包含 avgpool 或 AdaptiveAvgPool2d 层。"
            )
        
        def hook_fn(module, input, output):
            # 展平池化输出 → (batch_size, feature_dim)
            self._features = output.flatten(1)
        
        self._hook_handle = target_layer.register_forward_hook(hook_fn)
    
    @property
    def feature_dim(self) -> int:
        """获取 backbone 特征维度（首次前向传播后可用）

        对于 Linear head:    feature_dim = head.in_features
        对于 MLP head (Sequential): feature_dim = 第一个 Linear 层的 in_features
                                    （即 backbone avgpool 的输出维度）
        """
        if self._feature_dim is not None:
            return self._feature_dim

        def _first_linear_in_features(module: nn.Module) -> int:
            """从模块或 Sequential 中提取第一个 Linear 层的 in_features。"""
            if hasattr(module, 'in_features'):
                # 单层 Linear
                return module.in_features
            if hasattr(module, '__iter__'):
                # Sequential / ModuleList：取第一个 Linear 层
                for layer in module:
                    if hasattr(layer, 'in_features'):
                        return layer.in_features
            return None

        # ResNet / ResNeXt / Wide ResNet: fc
        if hasattr(self.model, 'fc'):
            dim = _first_linear_in_features(self.model.fc)
            if dim is not None:
                self._feature_dim = dim
                return self._feature_dim

        # Swin Transformer: head
        if hasattr(self.model, 'head'):
            dim = _first_linear_in_features(self.model.head)
            if dim is not None:
                self._feature_dim = dim
                return self._feature_dim

        # ConvNeXt / EfficientNet: classifier
        if hasattr(self.model, 'classifier'):
            clf = self.model.classifier
            if hasattr(clf, '__getitem__'):
                # Sequential classifier（ConvNeXt 等）
                # ConvNeXt 的 classifier = Sequential(LayerNorm, Flatten, Linear)
                # EfficientNet 的 classifier = Sequential(Dropout, Linear)
                # 我们要的是 backbone 特征维度 = 最后一个 Linear 的 in_features
                for layer in reversed(list(clf)):
                    if hasattr(layer, 'in_features'):
                        self._feature_dim = layer.in_features
                        return self._feature_dim
            else:
                dim = _first_linear_in_features(clf)
                if dim is not None:
                    self._feature_dim = dim
                    return self._feature_dim

        raise RuntimeError(
            "无法推断 feature_dim，请先执行一次前向传播。\n"
            "这通常发生在自定义分类头不包含 Linear 层时。"
        )

    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        前向传播，返回 (logits, features)
        
        Args:
            x: 输入图像 (B, C, H, W)
            
        Returns:
            (logits, features) 元组
              - logits: (B, num_classes) 分类输出
              - features: (B, feature_dim) backbone 特征
        """
        logits = self.model(x)
        features = self._features
        
        # 缓存 feature_dim
        if self._feature_dim is None and features is not None:
            self._feature_dim = features.shape[1]
        
        return logits, features
    
    def remove_hook(self):
        """移除 hook（如果不再需要 features 输出）"""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
    
    def __del__(self):
        self.remove_hook()
