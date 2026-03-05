"""
Gastrovision 多标签训练器模块

提供：
- MultilabelTrainer: 多标签分类训练器
- 评估指标: per-class AUC, Top-1/2/3 accuracy, F1, mAP
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score
)

from ..data.augmentation import mixup_data, mixup_criterion, cutmix_data
from ..utils.metrics import AverageMeter





class MultilabelTrainer:
    """
    多标签分类训练器
    
    Args:
        model: PyTorch 模型
        criterion: 损失函数 (BCEWithLogitsLoss)
        optimizer: 优化器
        device: 训练设备
        scheduler: 学习率调度器（可选）
        output_dir: 输出目录
        class_names: 类别名称列表
        threshold: 分类阈值
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        scheduler: Optional[_LRScheduler] = None,
        output_dir: str = "./output",
        class_names: Optional[List[str]] = None,
        threshold: float = 0.5,
        triplet_loss: Optional[nn.Module] = None,
        triplet_weight: float = 1.0,
        metric_loss: Optional[nn.Module] = None,
        metric_loss_weight: float = 0.5,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.output_dir = Path(output_dir)
        self.class_names = class_names or []
        self.threshold = threshold
        
        # Triplet Loss (WhaleSSL 风格)
        self.triplet_loss = triplet_loss
        self.triplet_weight = triplet_weight
        
        # 通用度量学习损失
        self.metric_loss = metric_loss
        self.metric_loss_weight = metric_loss_weight
        
        # Mixup / CutMix
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史
        self.history: Dict[str, List] = {
            'train_loss': [],
            'valid_loss': [],
            'valid_auc_macro': [],
            'valid_map': [],
            'learning_rate': [],
        }
        
        # 最佳验证指标
        self.best_valid_auc = 0.0
        self.best_epoch = 0

    def _compute_metric_loss(
        self,
        logits: torch.Tensor,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """计算度量学习损失并做维度安全检查。"""
        metric_inputs = features

        expected_dim = None
        if hasattr(self.metric_loss, 'weight') and getattr(self.metric_loss, 'weight') is not None:
            expected_dim = self.metric_loss.weight.shape[1]
        elif hasattr(self.metric_loss, 'proxies') and getattr(self.metric_loss, 'proxies') is not None:
            expected_dim = self.metric_loss.proxies.shape[1]

        if expected_dim is not None and metric_inputs.dim() == 2 and metric_inputs.shape[1] != expected_dim:
            if logits.dim() == 2 and logits.shape[1] == expected_dim:
                metric_inputs = logits
            else:
                raise ValueError(
                    f"Metric loss embedding dim mismatch: got {metric_inputs.shape[1]}, expected {expected_dim}. "
                    f"Please set --embedding_dim to model feature dim or use a model that outputs (logits, features)."
                )

        return self.metric_loss(metric_inputs, labels)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        total_epochs: int
    ) -> Dict[str, float]:
        """
        训练一个 epoch
        """
        self.model.train()
        
        loss_meter = AverageMeter()
        triplet_loss_meter = AverageMeter()
        metric_loss_meter = AverageMeter()
        start_time = time.time()
        num_batches = len(train_loader)
        
        use_mixup = self.mixup_alpha > 0 or self.cutmix_alpha > 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Mixup / CutMix 数据增强
            mixed = False
            if use_mixup:
                if self.mixup_alpha > 0 and self.cutmix_alpha > 0:
                    if np.random.random() > 0.5:
                        images, y_a, y_b, lam = mixup_data(images, targets, self.mixup_alpha)
                    else:
                        images, y_a, y_b, lam = cutmix_data(images, targets, self.cutmix_alpha)
                elif self.mixup_alpha > 0:
                    images, y_a, y_b, lam = mixup_data(images, targets, self.mixup_alpha)
                else:
                    images, y_a, y_b, lam = cutmix_data(images, targets, self.cutmix_alpha)
                mixed = True
            
            # 前向传播
            outputs = self.model(images)
            
            # 处理模型输出 (可能是元组: logits, features)
            if isinstance(outputs, tuple):
                logits, features = outputs[0], outputs[-1]
            else:
                logits = outputs
                features = outputs  # 使用 logits 作为 embeddings
            
            # 主损失 (BCE/Focal/ASL 等)
            if mixed:
                loss = mixup_criterion(self.criterion, logits, y_a, y_b, lam)
            else:
                loss = self.criterion(logits, targets)
            
            # Triplet Loss (WhaleSSL 风格) — 混合样本时跳过
            triplet_loss_val = 0.0
            if self.triplet_loss is not None and not mixed:
                # 从多标签中提取主标签作为身份
                primary_labels = targets.argmax(dim=1)  # 使用最大激活标签
                triplet_loss_val = self.triplet_loss(features, primary_labels)
                loss = loss + self.triplet_weight * triplet_loss_val
                triplet_loss_meter.update(triplet_loss_val.item(), targets.size(0))
            
            # 通用度量学习损失 — 混合样本时跳过
            if self.metric_loss is not None and not mixed:
                primary_labels = targets.argmax(dim=1)
                ml_loss = self._compute_metric_loss(logits, features, primary_labels)
                loss = loss + self.metric_loss_weight * ml_loss
                metric_loss_meter.update(ml_loss.item(), targets.size(0))
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # OneCycleLR 需要每 step 调用（而非每 epoch）
            if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()
            
            # 更新统计
            loss_meter.update(loss.item(), targets.size(0))
            
            # 打印进度（每 20% 打印一次）
            if (batch_idx + 1) % max(1, num_batches // 5) == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)
                triplet_info = f" | Triplet: {triplet_loss_meter.avg:.4f}" if self.triplet_loss else ""
                metric_info = f" | Metric: {metric_loss_meter.avg:.4f}" if self.metric_loss else ""
                print(f"  [{batch_idx + 1}/{num_batches}] "
                      f"Loss: {loss_meter.avg:.4f}{triplet_info}{metric_info} | "
                      f"ETA: {eta:.0f}s")
        
        elapsed = time.time() - start_time
        
        result = {
            'loss': loss_meter.avg,
            'time': elapsed
        }
        if self.triplet_loss:
            result['triplet_loss'] = triplet_loss_meter.avg
        if self.metric_loss:
            result['metric_loss'] = metric_loss_meter.avg
        
        return result
    
    @torch.no_grad()
    def validate(
        self,
        valid_loader: DataLoader,
        desc: str = "Validation"
    ) -> Dict[str, float]:
        """
        验证模型
        """
        self.model.eval()
        
        loss_meter = AverageMeter()
        all_probs = []
        all_targets = []
        
        for images, targets in valid_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            outputs = self.model(images)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = self.criterion(logits, targets)
            
            probs = torch.sigmoid(logits)
            
            loss_meter.update(loss.item(), targets.size(0))
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
        
        all_probs = np.concatenate(all_probs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # 计算 AUC
        try:
            auc_macro = roc_auc_score(all_targets, all_probs, average='macro')
        except ValueError:
            auc_macro = 0.0
        
        # 计算 mAP
        try:
            map_score = average_precision_score(all_targets, all_probs, average='macro')
        except ValueError:
            map_score = 0.0
        
        return {
            'loss': loss_meter.avg,
            'auc_macro': auc_macro,
            'map': map_score
        }
    
    @torch.no_grad()
    def optimize_thresholds(
        self,
        valid_loader: DataLoader,
        metric: str = 'f1'
    ) -> Dict[str, float]:
        """
        在验证集上为每个类别寻找最优阈值
        
        Args:
            valid_loader: 验证数据加载器
            metric: 优化目标，'f1' 或 'precision' 或 'recall'
            
        Returns:
            包含每个类别最优阈值的字典
        """
        self.model.eval()
        
        all_probs = []
        all_targets = []
        
        for images, targets in valid_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            outputs = self.model(images)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probs = torch.sigmoid(logits)
            
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
        
        all_probs = np.concatenate(all_probs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        num_classes = all_probs.shape[1]
        optimal_thresholds = {}
        
        print("=" * 50)
        print("优化每个类别的阈值 (基于验证集)")
        print("=" * 50)
        
        # 候选阈值
        candidate_thresholds = np.arange(0.1, 0.9, 0.05)
        
        for i in range(num_classes):
            class_name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
            y_true = all_targets[:, i]
            y_prob = all_probs[:, i]
            
            best_threshold = 0.5
            best_score = 0.0
            
            for thresh in candidate_thresholds:
                y_pred = (y_prob >= thresh).astype(int)
                
                if metric == 'f1':
                    score = f1_score(y_true, y_pred, zero_division=0)
                elif metric == 'precision':
                    score = precision_score(y_true, y_pred, zero_division=0)
                elif metric == 'recall':
                    score = recall_score(y_true, y_pred, zero_division=0)
                else:
                    score = f1_score(y_true, y_pred, zero_division=0)
                
                if score > best_score:
                    best_score = score
                    best_threshold = thresh
            
            optimal_thresholds[class_name] = best_threshold
            print(f"  {i:2d}. {class_name}: {best_threshold:.2f} ({metric}={best_score:.3f})")
        
        # 保存到实例变量
        self.thresholds_per_class = optimal_thresholds
        
        # 保存到文件
        thresh_path = self.output_dir / 'optimal_thresholds.json'
        with open(thresh_path, 'w', encoding='utf-8') as f:
            json.dump(optimal_thresholds, f, indent=2, ensure_ascii=False)
        print(f"\n阈值已保存到: {thresh_path}")
        
        return optimal_thresholds
    
    @torch.no_grad()
    def test(
        self,
        test_loader: DataLoader,
        save_roc_curves: bool = True
    ) -> Dict[str, Any]:
        """
        测试模型，输出详细指标
        
        包含：
        - Per-class AUC
        - Top-1/2/3 准确率
        - macro/micro F1, precision, recall
        - mAP
        """
        self.model.eval()
        
        all_probs = []
        all_targets = []
        
        for images, targets in test_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            outputs = self.model(images)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probs = torch.sigmoid(logits)
            
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
        
        all_probs = np.concatenate(all_probs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        num_classes = all_probs.shape[1]
        num_samples = len(all_targets)
        
        # ========== Per-class AUC ==========
        per_class_auc = {}
        for i in range(num_classes):
            class_name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
            try:
                auc = roc_auc_score(all_targets[:, i], all_probs[:, i])
            except ValueError:
                auc = 0.0
            per_class_auc[class_name] = auc
        
        # Macro/Micro AUC
        try:
            auc_macro = roc_auc_score(all_targets, all_probs, average='macro')
            auc_micro = roc_auc_score(all_targets, all_probs, average='micro')
            auc_weighted = roc_auc_score(all_targets, all_probs, average='weighted')
        except ValueError:
            auc_macro = auc_micro = auc_weighted = 0.0
        
        # ========== Top-K 准确率 ==========
        # 两种计算方式:
        # 1. top_k_any: 前 k 个预测中至少有一个在真实标签中（宽松）
        # 2. top_k_strict: 如果真实标签有 N 个，Top-1 检查前 N 个是否全中，Top-2 检查前 N+1（严格）
        
        # 宽松版本 (any)
        top1_any_correct = 0
        top2_any_correct = 0
        top3_any_correct = 0
        
        # 严格版本 (strict)
        top1_strict_correct = 0
        top2_strict_correct = 0
        top3_strict_correct = 0
        
        valid_samples = 0  # 有标签的样本数
        
        for i in range(num_samples):
            true_labels = set(np.where(all_targets[i] > 0.5)[0])
            n_true = len(true_labels)
            if n_true == 0:
                continue
            valid_samples += 1
            
            # 按概率排序（降序）
            sorted_indices = np.argsort(all_probs[i])[::-1]
            
            # ===== 宽松版本: 前 k 个预测中至少有一个在真实标签中 =====
            if sorted_indices[0] in true_labels:
                top1_any_correct += 1
            if any(idx in true_labels for idx in sorted_indices[:2]):
                top2_any_correct += 1
            if any(idx in true_labels for idx in sorted_indices[:3]):
                top3_any_correct += 1
            
            # ===== 严格版本: 基于真实标签数 N =====
            # Top-1 strict: 前 N 个预测是否包含所有真实标签
            # Top-2 strict: 前 N+1 个预测是否包含所有真实标签
            # Top-3 strict: 前 N+2 个预测是否包含所有真实标签
            top_n = set(sorted_indices[:n_true])
            top_n_plus_1 = set(sorted_indices[:n_true + 1])
            top_n_plus_2 = set(sorted_indices[:n_true + 2])
            
            if true_labels <= top_n:  # 子集关系判断
                top1_strict_correct += 1
            if true_labels <= top_n_plus_1:
                top2_strict_correct += 1
            if true_labels <= top_n_plus_2:
                top3_strict_correct += 1
        
        # 宽松版本准确率
        top1_any_acc = top1_any_correct / valid_samples if valid_samples > 0 else 0
        top2_any_acc = top2_any_correct / valid_samples if valid_samples > 0 else 0
        top3_any_acc = top3_any_correct / valid_samples if valid_samples > 0 else 0
        
        # 严格版本准确率
        top1_strict_acc = top1_strict_correct / valid_samples if valid_samples > 0 else 0
        top2_strict_acc = top2_strict_correct / valid_samples if valid_samples > 0 else 0
        top3_strict_acc = top3_strict_correct / valid_samples if valid_samples > 0 else 0
        
        # ========== 二值化预测后的指标 ==========
        # 如果有 per-class 阈值则使用，否则使用全局阈值
        if hasattr(self, 'thresholds_per_class') and self.thresholds_per_class:
            all_preds = np.zeros_like(all_probs, dtype=int)
            for i in range(num_classes):
                class_name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
                thresh = self.thresholds_per_class.get(class_name, self.threshold)
                all_preds[:, i] = (all_probs[:, i] >= thresh).astype(int)
        else:
            all_preds = (all_probs >= self.threshold).astype(int)
        
        # F1 Score
        f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        f1_micro = f1_score(all_targets, all_preds, average='micro', zero_division=0)
        f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        # Precision, Recall
        precision_macro = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        recall_macro = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        precision_micro = precision_score(all_targets, all_preds, average='micro', zero_division=0)
        recall_micro = recall_score(all_targets, all_preds, average='micro', zero_division=0)
        
        # mAP
        try:
            map_score = average_precision_score(all_targets, all_probs, average='macro')
        except ValueError:
            map_score = 0.0
        
        # Subset accuracy (完全匹配)
        subset_acc = (all_preds == all_targets).all(axis=1).mean()
        
        # Per-class Precision, Recall, F1
        per_class_metrics = {}
        for i in range(num_classes):
            class_name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
            try:
                p = precision_score(all_targets[:, i], all_preds[:, i], zero_division=0)
                r = recall_score(all_targets[:, i], all_preds[:, i], zero_division=0)
                f1 = f1_score(all_targets[:, i], all_preds[:, i], zero_division=0)
            except:
                p = r = f1 = 0.0
            per_class_metrics[class_name] = {
                'precision': p,
                'recall': r,
                'f1': f1,
                'auc': per_class_auc.get(class_name, 0.0)
            }
        
        results = {
            'num_samples': num_samples,
            'num_classes': num_classes,
            'threshold': self.threshold,
            'thresholds_per_class': getattr(self, 'thresholds_per_class', None),
            # Top-K 准确率 (宽松版本: 前 k 个预测中至少有一个正确)
            'top1_any_accuracy': top1_any_acc,
            'top2_any_accuracy': top2_any_acc,
            'top3_any_accuracy': top3_any_acc,
            # Top-K 准确率 (严格版本: 若有 N 个真实标签，前 N+k-1 个预测需包含全部)
            'top1_strict_accuracy': top1_strict_acc,
            'top2_strict_accuracy': top2_strict_acc,
            'top3_strict_accuracy': top3_strict_acc,
            # AUC
            'auc_macro': auc_macro,
            'auc_micro': auc_micro,
            'auc_weighted': auc_weighted,
            'per_class_auc': per_class_auc,
            # mAP
            'map': map_score,
            # F1
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_weighted': f1_weighted,
            # Precision, Recall
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            # Subset accuracy
            'subset_accuracy': subset_acc,
            # Per-class metrics
            'per_class_metrics': per_class_metrics,
        }
        
        # 保存 ROC 曲线（4x4 合并图）
        if save_roc_curves:
            self._save_roc_curves_grid(all_targets, all_probs)
        
        # 保存结果
        self._save_test_results(results)
        
        return results
    
    def _save_roc_curves_grid(
        self,
        all_targets: np.ndarray,
        all_probs: np.ndarray
    ) -> None:
        """保存 ROC 曲线为 4x4 合并图"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            num_classes = all_probs.shape[1]
            rows, cols = 4, 4
            
            fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
            fig.suptitle('ROC Curves for All Classes', fontsize=16, fontweight='bold')
            
            for i in range(num_classes):
                row, col = i // cols, i % cols
                ax = axes[row, col]
                
                class_name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
                
                try:
                    fpr, tpr, _ = roc_curve(all_targets[:, i], all_probs[:, i])
                    auc = roc_auc_score(all_targets[:, i], all_probs[:, i])
                    
                    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.3f}')
                    ax.plot([0, 1], [0, 1], 'r--', linewidth=1)
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('FPR', fontsize=8)
                    ax.set_ylabel('TPR', fontsize=8)
                    ax.set_title(f'{i}: {class_name[:15]}', fontsize=9)
                    ax.legend(loc='lower right', fontsize=7)
                    ax.grid(True, alpha=0.3)
                except ValueError:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
                    ax.set_title(f'{i}: {class_name[:15]}', fontsize=9)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            save_path = self.output_dir / 'roc_curves_grid.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ROC 曲线已保存到: {save_path}")
            
        except ImportError:
            print("  警告: matplotlib 未安装，跳过 ROC 曲线生成")
    
    def _save_test_results(self, results: Dict) -> None:
        """保存测试结果"""
        # 保存 JSON
        report_path = self.output_dir / 'multilabel_test_results.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  测试结果已保存到: {report_path}")
    
    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        epochs: int,
        early_stopping: int = 0
    ) -> Dict[str, List]:
        """
        完整训练循环
        """
        print("=" * 60)
        print("开始多标签训练")
        print("=" * 60)
        print(f"设备: {self.device}")
        print(f"训练样本: {len(train_loader.dataset)}")
        print(f"验证样本: {len(valid_loader.dataset)}")
        print(f"总 Epochs: {epochs}")
        print(f"批次大小: {train_loader.batch_size}")
        print(f"分类阈值: {self.threshold}")
        print()
        
        no_improve_count = 0
        total_start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            print("-" * 40)
            
            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"学习率: {current_lr:.6f}")
            
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch, epochs)
            print(f"训练 - Loss: {train_metrics['loss']:.4f}, "
                  f"Time: {train_metrics['time']:.1f}s")
            
            # 验证
            valid_metrics = self.validate(valid_loader)
            print(f"验证 - Loss: {valid_metrics['loss']:.4f}, "
                  f"AUC: {valid_metrics['auc_macro']:.4f}, "
                  f"mAP: {valid_metrics['map']:.4f}")
            
            # 更新学习率（OneCycleLR 已在 train_epoch 中按 step 更新）
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    pass  # 已在 train_epoch 中每 batch 调用
                elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(valid_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['valid_loss'].append(valid_metrics['loss'])
            self.history['valid_auc_macro'].append(valid_metrics['auc_macro'])
            self.history['valid_map'].append(valid_metrics['map'])
            self.history['learning_rate'].append(current_lr)
            
            # 保存最佳模型（基于 AUC）
            if valid_metrics['auc_macro'] > self.best_valid_auc:
                self.best_valid_auc = valid_metrics['auc_macro']
                self.best_epoch = epoch
                self.save_checkpoint('best_model.pth', epoch, valid_metrics)
                print(f"  ★ 新最佳模型! AUC: {self.best_valid_auc:.4f}")
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            print()
            
            # 早停检查
            if early_stopping > 0 and no_improve_count >= early_stopping:
                print(f"早停! {early_stopping} 个 epoch 无改进")
                break
        
        total_time = time.time() - total_start_time
        
        print("=" * 60)
        print("训练完成")
        print("=" * 60)
        print(f"总耗时: {total_time / 60:.1f} 分钟")
        print(f"最佳验证 AUC: {self.best_valid_auc:.4f} (Epoch {self.best_epoch})")
        
        # 保存训练日志
        self.save_training_log()
        
        return self.history
    
    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """保存模型 checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_valid_auc': self.best_valid_auc,
            'best_epoch': self.best_epoch,
            'threshold': self.threshold,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.output_dir / filename)
    
    def load_checkpoint(self, filename: str) -> Dict:
        """加载模型 checkpoint"""
        from pathlib import Path
        
        filepath = Path(filename)
        if not filepath.is_absolute():
            filepath = self.output_dir / filename
        
        checkpoint = torch.load(
            filepath, 
            map_location=self.device,
            weights_only=False
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_valid_auc = checkpoint.get('best_valid_auc', 0.0)
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.threshold = checkpoint.get('threshold', 0.5)
        
        return checkpoint
    
    def save_training_log(self) -> None:
        """保存训练日志"""
        log = {
            'history': self.history,
            'best_valid_auc': self.best_valid_auc,
            'best_epoch': self.best_epoch,
            'timestamp': datetime.now().isoformat(),
        }
        
        log_path = self.output_dir / 'multilabel_training_log.json'
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2)
        
        print(f"训练日志已保存到: {log_path}")


def print_multilabel_test_results(results: Dict[str, Any]) -> None:
    """打印多标签测试结果"""
    print("\n" + "=" * 70)
    print("多标签测试结果")
    print("=" * 70)
    print(f"样本数: {results['num_samples']}")
    print(f"类别数: {results['num_classes']}")
    print(f"默认阈值: {results['threshold']}")
    if results.get('thresholds_per_class'):
        print("  (使用 per-class 优化阈值)")
    print()
    
    print("Top-K 准确率 (宽松: 前k个至少1个正确):")
    print(f"  Top-1 Any: {results['top1_any_accuracy']:.4f}")
    print(f"  Top-2 Any: {results['top2_any_accuracy']:.4f}")
    print(f"  Top-3 Any: {results['top3_any_accuracy']:.4f}")
    print()
    
    print("Top-K 准确率 (严格: 若有N个真实标签，前N+k-1个需全覆盖):")
    print(f"  Top-1 Strict: {results['top1_strict_accuracy']:.4f}")
    print(f"  Top-2 Strict: {results['top2_strict_accuracy']:.4f}")
    print(f"  Top-3 Strict: {results['top3_strict_accuracy']:.4f}")
    print()
    
    print("AUC 指标:")
    print(f"  Macro AUC:    {results['auc_macro']:.4f}")
    print(f"  Micro AUC:    {results['auc_micro']:.4f}")
    print(f"  Weighted AUC: {results['auc_weighted']:.4f}")
    print()
    
    print("F1 Score:")
    print(f"  Macro F1:    {results['f1_macro']:.4f}")
    print(f"  Micro F1:    {results['f1_micro']:.4f}")
    print(f"  Weighted F1: {results['f1_weighted']:.4f}")
    print()
    
    print(f"mAP: {results['map']:.4f}")
    print(f"Subset Accuracy: {results['subset_accuracy']:.4f}")
    print()
    
    print("每类别 AUC:")
    print("-" * 50)
    for class_name, auc in results['per_class_auc'].items():
        print(f"  {class_name}: {auc:.4f}")
    print("=" * 70)
