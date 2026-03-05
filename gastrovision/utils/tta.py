"""
Test-Time Augmentation (TTA) 工具

原理:
  推理时对每个输入图像生成多个增强视图，将预测结果聚合以获得更稳定的预测。
  因为不同增强展现了图像的不同方面，聚合后的预测更鲁棒。

使用:
  from gastrovision.utils.tta import TTAPredictor

  tta_predictor = TTAPredictor(model, device, num_classes=23)
  predictions, targets = tta_predictor.predict(test_loader)

  # 或者单张图像
  probs = tta_predictor.predict_single(image_tensor)  # (num_classes,) 概率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from typing import List, Callable, Optional, Tuple
from tqdm import tqdm


# ===== 预定义的 TTA 变换 =====

def identity(x: torch.Tensor) -> torch.Tensor:
    """恒等变换（原图）"""
    return x


def hflip(x: torch.Tensor) -> torch.Tensor:
    """水平翻转"""
    return torch.flip(x, dims=[-1])


def vflip(x: torch.Tensor) -> torch.Tensor:
    """垂直翻转"""
    return torch.flip(x, dims=[-2])


def rotate90(x: torch.Tensor) -> torch.Tensor:
    """旋转 90°"""
    return torch.rot90(x, k=1, dims=[-2, -1])


def rotate180(x: torch.Tensor) -> torch.Tensor:
    """旋转 180°"""
    return torch.rot90(x, k=2, dims=[-2, -1])


def rotate270(x: torch.Tensor) -> torch.Tensor:
    """旋转 270°"""
    return torch.rot90(x, k=3, dims=[-2, -1])


# 预定义变换集：从轻量到全面
TTA_LIGHT = [identity, hflip]
TTA_MEDIUM = [identity, hflip, vflip, rotate180]
TTA_HEAVY = [identity, hflip, vflip, rotate90, rotate180, rotate270]


class TTAPredictor:
    """
    Test-Time Augmentation 预测器
    
    Args:
        model: 训练好的模型
        device: 设备
        num_classes: 类别数量
        transforms: TTA 变换列表 (默认 TTA_MEDIUM)
        aggregate: 聚合方式 ('mean' 或 'max')
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int,
        transforms: Optional[List[Callable]] = None,
        aggregate: str = 'mean'
    ):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.transforms = transforms or TTA_MEDIUM
        self.aggregate = aggregate
    
    @torch.no_grad()
    def predict_single(self, image: torch.Tensor) -> torch.Tensor:
        """
        对单张图像进行 TTA 预测
        
        Args:
            image: (C, H, W) 或 (1, C, H, W) 输入图像
            
        Returns:
            (num_classes,) 聚合后的概率向量
        """
        self.model.eval()
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        
        all_probs = []
        for transform in self.transforms:
            augmented = transform(image)
            outputs = self.model(augmented)
            
            # 处理 (logits, features) 元组输出
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs)
        
        # 聚合
        stacked = torch.stack(all_probs, dim=0)  # (T, 1, C)
        if self.aggregate == 'max':
            aggregated = stacked.max(dim=0)[0]
        else:
            aggregated = stacked.mean(dim=0)
        
        return aggregated.squeeze(0)  # (C,)
    
    @torch.no_grad()
    def predict(
        self,
        data_loader: DataLoader,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对整个数据集进行 TTA 预测
        
        Args:
            data_loader: 数据加载器
            return_features: 是否返回特征（用于可视化）
            
        Returns:
            (predictions, targets)
            - predictions: (N, num_classes) 每个样本的聚合概率
            - targets: (N,) ground truth 标签
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        for images, targets in tqdm(data_loader, desc=f'TTA ({len(self.transforms)} views)'):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            batch_probs = []
            for transform in self.transforms:
                augmented = transform(images)
                outputs = self.model(augmented)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                probs = F.softmax(outputs, dim=1)
                batch_probs.append(probs)
            
            stacked = torch.stack(batch_probs, dim=0)  # (T, B, C)
            if self.aggregate == 'max':
                aggregated = stacked.max(dim=0)[0]
            else:
                aggregated = stacked.mean(dim=0)
            
            all_predictions.append(aggregated.cpu())
            all_targets.append(targets.cpu())
        
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        return predictions, targets
    
    def evaluate(
        self,
        data_loader: DataLoader,
        class_names: Optional[List[str]] = None
    ) -> dict:
        """
        使用 TTA 评估模型
        
        Args:
            data_loader: 测试数据加载器
            class_names: 类别名称（用于报告）
            
        Returns:
            包含各种评估指标的字典
        """
        predictions, targets = self.predict(data_loader)
        
        # Top-1 准确率
        _, pred_classes = predictions.max(dim=1)
        correct = pred_classes.eq(targets).sum().item()
        total = targets.size(0)
        accuracy = correct / total
        
        # 每个类别的准确率
        per_class_acc = {}
        for cls in range(self.num_classes):
            mask = targets == cls
            if mask.sum() > 0:
                cls_acc = pred_classes[mask].eq(targets[mask]).float().mean().item()
                name = class_names[cls] if class_names else str(cls)
                per_class_acc[name] = cls_acc
        
        results = {
            'accuracy': accuracy,
            'total_samples': total,
            'correct': correct,
            'tta_views': len(self.transforms),
            'aggregate': self.aggregate,
            'per_class_accuracy': per_class_acc,
        }
        
        # 打印报告
        print(f"\n{'='*50}")
        print(f"TTA 评估结果 ({len(self.transforms)} views, {self.aggregate})")
        print(f"{'='*50}")
        print(f"Top-1 Accuracy: {accuracy:.4f} ({correct}/{total})")
        if class_names:
            print(f"\n每类准确率:")
            for name, acc in sorted(per_class_acc.items(), key=lambda x: x[1]):
                print(f"  {name:30s}: {acc:.4f}")
        print(f"{'='*50}")
        
        return results
