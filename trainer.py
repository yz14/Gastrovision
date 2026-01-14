"""
Gastrovision 训练器模块

提供：
- Trainer: 模型训练器类
- 丰富的评估指标
- 训练日志和 checkpoint 保存
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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class AverageMeter:
    """计算和存储平均值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer:
    """
    模型训练器
    
    Args:
        model: PyTorch 模型
        criterion: 损失函数
        optimizer: 优化器
        device: 训练设备
        scheduler: 学习率调度器（可选）
        output_dir: 输出目录（保存 checkpoint 和日志）
        class_names: 类别名称列表（用于生成报告）
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        scheduler: Optional[_LRScheduler] = None,
        output_dir: str = "./output",
        class_names: Optional[List[str]] = None
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.output_dir = Path(output_dir)
        self.class_names = class_names
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史
        self.history: Dict[str, List] = {
            'train_loss': [],
            'train_acc': [],
            'valid_loss': [],
            'valid_acc': [],
            'learning_rate': [],
        }
        
        # 最佳验证准确率
        self.best_valid_acc = 0.0
        self.best_epoch = 0
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        total_epochs: int
    ) -> Dict[str, float]:
        """
        训练一个 epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前 epoch（从1开始）
            total_epochs: 总 epoch 数
            
        Returns:
            包含 loss 和 accuracy 的字典
        """
        self.model.train()
        
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        start_time = time.time()
        num_batches = len(train_loader)
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 计算准确率
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum().item()
            batch_size = targets.size(0)
            acc = correct / batch_size
            
            # 更新统计
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc, batch_size)
            
            # 打印进度（每 20% 打印一次）
            if (batch_idx + 1) % max(1, num_batches // 5) == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)
                print(f"  [{batch_idx + 1}/{num_batches}] "
                      f"Loss: {loss_meter.avg:.4f} | "
                      f"Acc: {acc_meter.avg:.4f} | "
                      f"ETA: {eta:.0f}s")
        
        elapsed = time.time() - start_time
        
        return {
            'loss': loss_meter.avg,
            'accuracy': acc_meter.avg,
            'time': elapsed
        }
    
    @torch.no_grad()
    def validate(
        self,
        valid_loader: DataLoader,
        desc: str = "Validation"
    ) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            valid_loader: 验证数据加载器
            desc: 描述字符串
            
        Returns:
            包含 loss、accuracy 等指标的字典
        """
        self.model.eval()
        
        loss_meter = AverageMeter()
        all_predictions = []
        all_targets = []
        
        for images, targets in valid_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            _, predicted = outputs.max(1)
            
            loss_meter.update(loss.item(), targets.size(0))
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        # 计算指标
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        accuracy = accuracy_score(all_targets, all_predictions)
        precision_macro = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
        recall_macro = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
        f1_macro = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
        
        return {
            'loss': loss_meter.avg,
            'accuracy': accuracy,
            'precision': precision_macro,
            'recall': recall_macro,
            'f1': f1_macro
        }
    
    @torch.no_grad()
    def test(
        self,
        test_loader: DataLoader,
        save_confusion_matrix: bool = True
    ) -> Dict[str, Any]:
        """
        测试模型，输出详细指标
        
        Args:
            test_loader: 测试数据加载器
            save_confusion_matrix: 是否保存混淆矩阵
            
        Returns:
            包含详细指标的字典
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probs = []
        
        for images, targets in test_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # 计算整体指标
        accuracy = accuracy_score(all_targets, all_predictions)
        
        # Top-5 准确率
        num_classes = all_probs.shape[1]
        if num_classes >= 5:
            top5_predictions = np.argsort(all_probs, axis=1)[:, -5:]
            top5_correct = np.array([t in p for t, p in zip(all_targets, top5_predictions)])
            top5_accuracy = top5_correct.mean()
        else:
            top5_accuracy = accuracy
        
        # Macro 和 Weighted 指标
        precision_macro = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
        recall_macro = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
        f1_macro = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
        
        precision_weighted = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall_weighted = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1_weighted = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        # 混淆矩阵
        cm = confusion_matrix(all_targets, all_predictions)
        
        # 分类报告
        target_names = self.class_names if self.class_names else None
        report = classification_report(
            all_targets,
            all_predictions,
            target_names=target_names,
            zero_division=0,
            output_dict=True
        )
        
        results = {
            'accuracy': accuracy,
            'top5_accuracy': top5_accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'num_samples': len(all_targets),
            'num_classes': num_classes
        }
        
        # 保存混淆矩阵图
        if save_confusion_matrix:
            self._save_confusion_matrix(cm, target_names)
        
        # 保存分类报告
        self._save_classification_report(results)
        
        return results
    
    def _save_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> None:
        """保存混淆矩阵为图片（使用类别索引作为标签，并添加legend）"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            num_classes = len(cm)
            # 增加宽度以容纳右侧 legend
            fig_size = max(12, num_classes * 0.5)
            fig, ax = plt.subplots(figsize=(fig_size + 5, fig_size))
            
            # 使用类别索引作为标签
            class_indices = [str(i) for i in range(num_classes)]
            
            sns.heatmap(
                cm,
                annot=num_classes <= 20,  # 类别太多时不显示数字
                fmt='d',
                cmap='Blues',
                xticklabels=class_indices if num_classes <= 30 else False,
                yticklabels=class_indices if num_classes <= 30 else False,
                ax=ax
            )
            
            ax.set_xlabel('Predicted Label (Index)', fontsize=12)
            ax.set_ylabel('True Label (Index)', fontsize=12)
            ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
            
            # 添加类别索引到名称的 legend 表格
            if class_names and num_classes <= 30:
                legend_text = "Class Legend:\n" + "-" * 32 + "\n"
                for idx, name in enumerate(class_names):
                    display_name = name[:25] + "..." if len(name) > 25 else name
                    legend_text += f"{idx:2d}: {display_name}\n"
                
                plt.gcf().text(0.82, 0.5, legend_text, fontsize=7, family='monospace',
                               verticalalignment='center', 
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                plt.subplots_adjust(right=0.78)
            else:
                plt.tight_layout()
            
            plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  混淆矩阵已保存到: {self.output_dir / 'confusion_matrix.png'}")
            
        except ImportError:
            print("  警告: 未安装 matplotlib/seaborn，跳过混淆矩阵图生成")
    
    def _save_classification_report(self, results: Dict) -> None:
        """保存分类报告"""
        report_path = self.output_dir / 'test_results.json'
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
        
        Args:
            train_loader: 训练数据加载器
            valid_loader: 验证数据加载器
            epochs: 训练轮数
            early_stopping: 早停耐心值（0表示不使用早停）
            
        Returns:
            训练历史
        """
        print("=" * 60)
        print("开始训练")
        print("=" * 60)
        print(f"设备: {self.device}")
        print(f"训练样本: {len(train_loader.dataset)}")
        print(f"验证样本: {len(valid_loader.dataset)}")
        print(f"总 Epochs: {epochs}")
        print(f"批次大小: {train_loader.batch_size}")
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
                  f"Acc: {train_metrics['accuracy']:.4f}, "
                  f"Time: {train_metrics['time']:.1f}s")
            
            # 验证
            valid_metrics = self.validate(valid_loader)
            print(f"验证 - Loss: {valid_metrics['loss']:.4f}, "
                  f"Acc: {valid_metrics['accuracy']:.4f}, "
                  f"F1: {valid_metrics['f1']:.4f}")
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(valid_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['valid_loss'].append(valid_metrics['loss'])
            self.history['valid_acc'].append(valid_metrics['accuracy'])
            self.history['learning_rate'].append(current_lr)
            
            # 保存最佳模型
            if valid_metrics['accuracy'] > self.best_valid_acc:
                self.best_valid_acc = valid_metrics['accuracy']
                self.best_epoch = epoch
                self.save_checkpoint('best_model.pth', epoch, valid_metrics)
                print(f"  ★ 新最佳模型! Acc: {self.best_valid_acc:.4f}")
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # 保存每个 epoch 的 checkpoint
            self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', epoch, valid_metrics)
            
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
        print(f"最佳验证准确率: {self.best_valid_acc:.4f} (Epoch {self.best_epoch})")
        
        # 保存训练日志
        self.save_training_log()
        
        # 清理非最佳 checkpoint
        self._cleanup_checkpoints()
        
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
            'best_valid_acc': self.best_valid_acc,
            'best_epoch': self.best_epoch,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.output_dir / filename)
    
    def load_checkpoint(self, filename: str) -> Dict:
        """加载模型 checkpoint
        
        支持预训练+微调场景：当分类头尺寸不匹配时，自动跳过分类头权重
        """
        from pathlib import Path
        
        # 支持绝对路径和相对路径
        filepath = Path(filename)
        if not filepath.is_absolute():
            filepath = self.output_dir / filename
        
        # PyTorch 2.6+ 兼容性：添加 weights_only=False
        checkpoint = torch.load(
            filepath, 
            map_location=self.device,
            weights_only=False
        )
        
        # 尝试加载模型权重
        model_state_dict = checkpoint['model_state_dict']
        current_state_dict = self.model.state_dict()
        
        # 检查分类头尺寸是否匹配
        fc_keys = [k for k in model_state_dict.keys() if 'fc' in k or 'classifier' in k or 'head' in k]
        size_mismatch = False
        for key in fc_keys:
            if key in current_state_dict:
                if model_state_dict[key].shape != current_state_dict[key].shape:
                    size_mismatch = True
                    print(f"⚠ 跳过分类头权重 '{key}': checkpoint {model_state_dict[key].shape} != 当前模型 {current_state_dict[key].shape}")
        
        if size_mismatch:
            # 过滤掉分类头权重，只加载 backbone
            filtered_state_dict = {}
            for key, value in model_state_dict.items():
                if any(fc_name in key for fc_name in ['fc.', 'classifier.', 'head.']):
                    continue  # 跳过分类头
                if key in current_state_dict and value.shape == current_state_dict[key].shape:
                    filtered_state_dict[key] = value
            
            # 加载过滤后的权重
            missing_keys, unexpected_keys = self.model.load_state_dict(filtered_state_dict, strict=False)
            print(f"✓ 已加载 backbone 权重 ({len(filtered_state_dict)} 个参数)")
            print(f"  分类头将使用随机初始化（用于微调场景）")
            
            # 微调场景：重置最佳准确率（因为分类头已重新初始化）
            self.best_valid_acc = 0.0
            self.best_epoch = 0
            print(f"  已重置 best_valid_acc 为 0.0（微调模式）")
        else:
            # 正常加载所有权重
            self.model.load_state_dict(model_state_dict)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # 恢复训练场景：保留最佳准确率
            self.best_valid_acc = checkpoint.get('best_valid_acc', 0.0)
            self.best_epoch = checkpoint.get('best_epoch', 0)
        
        return checkpoint
    
    def _cleanup_checkpoints(self) -> None:
        """清理非最佳 checkpoint，只保留 best_model.pth"""
        import glob
        
        # 查找所有 checkpoint_epoch_*.pth 文件
        pattern = str(self.output_dir / 'checkpoint_epoch_*.pth')
        checkpoint_files = glob.glob(pattern)
        
        if checkpoint_files:
            deleted_count = 0
            for ckpt_file in checkpoint_files:
                try:
                    os.remove(ckpt_file)
                    deleted_count += 1
                except OSError as e:
                    print(f"⚠ 无法删除 {ckpt_file}: {e}")
            
            if deleted_count > 0:
                print(f"✓ 已清理 {deleted_count} 个临时 checkpoint 文件")
        
        # 确认 best_model.pth 存在
        best_model_path = self.output_dir / 'best_model.pth'
        if best_model_path.exists():
            print(f"✓ 最佳模型已保存: {best_model_path}")
        else:
            print(f"⚠ 警告: best_model.pth 未找到!")
    
    def save_training_log(self) -> None:
        """保存训练日志"""
        log = {
            'history': self.history,
            'best_valid_acc': self.best_valid_acc,
            'best_epoch': self.best_epoch,
            'timestamp': datetime.now().isoformat(),
        }
        
        log_path = self.output_dir / 'training_log.json'
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2)
        
        print(f"训练日志已保存到: {log_path}")


def print_test_results(results: Dict[str, Any]) -> None:
    """打印测试结果"""
    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    print(f"样本数: {results['num_samples']}")
    print(f"类别数: {results['num_classes']}")
    print()
    print("整体指标:")
    print(f"  Top-1 Accuracy: {results['accuracy']:.4f}")
    print(f"  Top-5 Accuracy: {results['top5_accuracy']:.4f}")
    print()
    print("Macro 平均:")
    print(f"  Precision: {results['precision_macro']:.4f}")
    print(f"  Recall:    {results['recall_macro']:.4f}")
    print(f"  F1-Score:  {results['f1_macro']:.4f}")
    print()
    print("Weighted 平均:")
    print(f"  Precision: {results['precision_weighted']:.4f}")
    print(f"  Recall:    {results['recall_weighted']:.4f}")
    print(f"  F1-Score:  {results['f1_weighted']:.4f}")
    print("=" * 60)
