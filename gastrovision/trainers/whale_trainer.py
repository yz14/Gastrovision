"""
鲸鱼分类训练器

基于原始 colonnav_ssl/train_whale.py 的 run_train + do_valid 逻辑重构,
适配 Gastrovision 风格的模块化训练流程。

关键特性:
- 支持双头模型 (BinaryHead + MarginHead)
- 支持三损失组合 (focal_OHEM + softmax + triplet) + YAML 权重开关
- 支持 Gastrovision 度量学习损失替换 triplet
- 动态 hard_ratio 和学习率调度
- 双验证 (正常 + 翻转) + 阈值搜索 + MAP 评估
"""

import os
import time
import json
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import numpy as np

from ..losses.whale_losses import metric


class WhaleTrainer:
    """
    鲸鱼分类训练器

    Args:
        model: WhaleNet 双头模型
        composite_loss: WhaleCompositeLoss 组合损失
        optimizer: 优化器
        device: 训练设备
        scheduler: 学习率调度器 (可选)
        output_dir: 输出目录
        whale_id_num: 鲸鱼个体数 (默认 5004)
    """

    def __init__(
        self,
        model: nn.Module,
        composite_loss,
        optimizer: Optimizer,
        device: torch.device,
        scheduler: Optional[_LRScheduler] = None,
        output_dir: str = "./output",
        whale_id_num: int = 5004
    ):
        self.model = model.to(device)
        self.composite_loss = composite_loss
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.output_dir = Path(output_dir)
        self.whale_id_num = whale_id_num
        self.num_class = whale_id_num * 2  # 含翻转

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 训练历史
        self.history: Dict[str, List] = {
            'train_loss': [],
            'valid_map': [],
            'valid_threshold': [],
            'learning_rate': [],
        }

        # 最佳验证 MAP
        self.best_valid_map = 0.0
        self.best_epoch = 0

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        total_epochs: int,
        hard_ratio: float = 0.5
    ) -> Dict[str, float]:
        """
        训练一个 epoch

        Args:
            train_loader: 训练数据加载器
            epoch: 当前 epoch (从 1 开始)
            total_epochs: 总 epoch 数
            hard_ratio: OHEM 难例比例

        Returns:
            包含 loss 分量的字典
        """
        self.model.train()

        running_losses = {}
        batch_count = 0
        start_time = time.time()
        num_batches = len(train_loader)

        for batch_idx, (input_, truth_, truth_NW_binary) in enumerate(train_loader):
            input_ = input_.to(self.device, non_blocking=True)
            truth_ = truth_.to(self.device, non_blocking=True)
            truth_NW_binary = truth_NW_binary.to(self.device, non_blocking=True)

            # 非 new_whale 的索引 (用于 softmax 和 triplet)
            indexs_NoNew = (truth_NW_binary == 0).nonzero(as_tuple=True)[0]

            # one-hot 编码 (用于 focal_OHEM)
            labels_onehot = torch.FloatTensor(
                len(truth_), self.num_class).to(self.device)
            labels_onehot.zero_()
            # new_whale 标签 = num_class, 超出范围, scatter 会忽略
            labels_onehot.scatter_(
                1, truth_.view(-1, 1).clamp(0, self.num_class - 1), 1)

            # 前向传播
            # 注意: 原始代码中 is_infer=True, ArcFace margin 仅在 MarginHead 内部
            # 通过 softmax_loss 作用。BinaryHead 不受 is_infer 影响。
            logit_binary, logit_margin, features = self.model(
                input_, label=truth_, is_infer=True)

            # 计算组合损失
            loss, loss_dict = self.composite_loss(
                logit_binary=logit_binary,
                logit_softmax=logit_margin,
                features=features,
                labels=truth_,
                labels_onehot=labels_onehot,
                hard_ratio=hard_ratio,
                indexs_NoNew=indexs_NoNew
            )

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            # 累计损失
            for k, v in loss_dict.items():
                if k not in running_losses:
                    running_losses[k] = 0.0
                running_losses[k] += v
            batch_count += 1

            # 打印进度 (每 20% 打印一次)
            if (batch_idx + 1) % max(1, num_batches // 5) == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)
                loss_str = ' | '.join(
                    f'{k}: {v/batch_count:.4f}' for k, v in running_losses.items())
                print(f"  [{batch_idx + 1}/{num_batches}] "
                      f"{loss_str} | ETA: {eta:.0f}s")

        elapsed = time.time() - start_time

        # 平均损失
        avg_losses = {k: v / batch_count for k, v in running_losses.items()}
        avg_losses['time'] = elapsed

        return avg_losses

    @torch.no_grad()
    def validate(
        self,
        dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        desc: str = "Validation"
    ) -> Dict[str, float]:
        """
        验证模型 (正常 + 翻转 TTA)

        Args:
            dataset: WhaleDataset (会自动设置 val 模式)
            batch_size: 批次大小
            num_workers: 工作进程数
            desc: 描述字符串

        Returns:
            包含 MAP、阈值等信息的字典
        """
        self.model.eval()

        def _do_valid_pass(dataset_obj, is_flip):
            """
            单次验证 (与原始 do_valid 逻辑一致)

            使用 sigmoid 而非 softmax (原始代码使用 BCE-style focal_OHEM)
            正常方向: 取 prob[:, :whale_id_num]
            翻转方向: 取 prob[:, whale_id_num:], label -= whale_id_num
            """
            dataset_obj.set_mode('val', dataset_obj.fold_index)
            dataset_obj.is_flip = is_flip

            loader = DataLoader(
                dataset_obj, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True, drop_last=False)

            all_probs = []
            all_labels = []

            for input_batch, truth_batch, _ in loader:
                input_batch = input_batch.to(self.device)

                logit_binary, _, _ = self.model(
                    input_batch, label=None, is_infer=True)
                # 关键: 使用 sigmoid, 与原始代码一致
                prob = torch.sigmoid(logit_binary)
                prob = prob.cpu().numpy()
                label = truth_batch.numpy()

                # 原始 do_valid 逻辑:
                # 翻转方向: 取后半部分 prob, label -= whale_id_num
                # 正常方向: 取前半部分 prob, new_whale label 设为 whale_id_num
                if is_flip:
                    prob = prob[:, self.whale_id_num:]
                    label = label - self.whale_id_num
                else:
                    prob = prob[:, :self.whale_id_num]
                    # new_whale: label = num_class (10008) → whale_id_num (5004)
                    label[label == self.num_class] = self.whale_id_num

                all_probs.append(prob)
                all_labels.append(label)

            dataset_obj.is_flip = False

            prob_all = np.concatenate(all_probs)
            label_all = np.concatenate(all_labels)
            return prob_all, label_all

        # ---- 正常方向 + 翻转方向 ----
        prob_normal, label_normal = _do_valid_pass(dataset, is_flip=False)
        prob_flip, label_flip = _do_valid_pass(dataset, is_flip=True)

        # ---- 合并: 概率平均 ----
        prob_merged = (prob_normal + prob_flip) / 2.0
        # 使用正常方向的 label (翻转方向的 label 只是偏移版本)
        labels = label_normal

        # ---- 搜索最佳 new_whale 阈值 ----
        best_threshold = 0.5
        best_map = 0.0

        for threshold in np.arange(0.02, 0.98, 0.02):
            map_score, top5 = metric(prob_merged, labels, threshold)
            if map_score > best_map:
                best_map = map_score
                best_threshold = threshold

        # 用最佳阈值计算 top5
        _, best_top5 = metric(prob_merged, labels, best_threshold)

        results = {
            'map': best_map,
            'threshold': best_threshold,
            'top1': best_top5[0] if len(best_top5) > 0 else 0,
            'top5': sum(best_top5[:5]) if best_top5 else 0,
        }

        return results

    def fit(
        self,
        train_loader: DataLoader,
        val_dataset,
        epochs: int,
        val_batch_size: int = 32,
        val_num_workers: int = 4,
        initial_hard_ratio: float = 1.0,
        min_hard_ratio: float = 0.2,
        hard_ratio_step: float = 0.1,
        early_stopping: int = 0
    ) -> Dict[str, List]:
        """
        完整训练循环

        Args:
            train_loader: 训练数据加载器
            val_dataset: 验证用 WhaleDataset
            epochs: 训练轮数
            val_batch_size: 验证批次大小
            val_num_workers: 验证数据工作进程数
            initial_hard_ratio: 初始 OHEM 比例 (逐步衰减)
            min_hard_ratio: 最小 OHEM 比例
            hard_ratio_step: 每 epoch 衰减步长
            early_stopping: 早停耐心值

        Returns:
            训练历史
        """
        print("=" * 60)
        print("开始鲸鱼分类训练")
        print("=" * 60)
        print(f"设备: {self.device}")
        print(f"训练样本: {len(train_loader.dataset)}")
        print(f"总 Epochs: {epochs}")
        print(f"OHEM 比例: {initial_hard_ratio} → {min_hard_ratio}")
        print()

        no_improve_count = 0
        total_start_time = time.time()
        hard_ratio = initial_hard_ratio

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            print("-" * 40)

            # 动态 hard_ratio
            hard_ratio = max(min_hard_ratio,
                             initial_hard_ratio - hard_ratio_step * (epoch - 1))
            print(f"OHEM hard_ratio: {hard_ratio:.2f}")

            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"学习率: {current_lr:.6f}")

            # 训练
            train_metrics = self.train_epoch(
                train_loader, epoch, epochs, hard_ratio)
            loss_str = ' | '.join(
                f'{k}: {v:.4f}' for k, v in train_metrics.items()
                if k != 'time')
            print(f"训练 - {loss_str} | Time: {train_metrics['time']:.1f}s")

            # 验证
            valid_metrics = self.validate(
                val_dataset,
                batch_size=val_batch_size,
                num_workers=val_num_workers
            )
            print(f"验证 - MAP: {valid_metrics['map']:.4f}, "
                  f"Threshold: {valid_metrics['threshold']:.2f}, "
                  f"Top1: {valid_metrics['top1']:.4f}")

            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(valid_metrics['map'])
                elif isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    pass  # 已在 train_epoch 中更新
                else:
                    self.scheduler.step()

            # 记录历史
            self.history['train_loss'].append(train_metrics.get('total', 0))
            self.history['valid_map'].append(valid_metrics['map'])
            self.history['valid_threshold'].append(valid_metrics['threshold'])
            self.history['learning_rate'].append(current_lr)

            # 保存最佳模型
            if valid_metrics['map'] > self.best_valid_map:
                self.best_valid_map = valid_metrics['map']
                self.best_epoch = epoch
                self.save_checkpoint('best_model.pth', epoch, valid_metrics)
                print(f"  ★ 新最佳模型! MAP: {self.best_valid_map:.4f}")
                no_improve_count = 0
            else:
                no_improve_count += 1

            # 保存每个 epoch 的 checkpoint
            self.save_checkpoint(
                f'checkpoint_epoch_{epoch}.pth', epoch, valid_metrics)

            print()

            # 早停
            if early_stopping > 0 and no_improve_count >= early_stopping:
                print(f"早停! {early_stopping} 个 epoch 无改进")
                break

        total_time = time.time() - total_start_time

        print("=" * 60)
        print("训练完成")
        print("=" * 60)
        print(f"总耗时: {total_time / 60:.1f} 分钟")
        print(f"最佳验证 MAP: {self.best_valid_map:.4f} (Epoch {self.best_epoch})")

        # 保存训练日志
        self.save_training_log()
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
            'best_valid_map': self.best_valid_map,
            'best_epoch': self.best_epoch,
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, self.output_dir / filename)

    def load_checkpoint(self, filename: str) -> Dict:
        """加载模型 checkpoint"""
        filepath = Path(filename)
        if not filepath.is_absolute():
            filepath = self.output_dir / filename

        checkpoint = torch.load(
            filepath, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_valid_map = checkpoint.get('best_valid_map', 0.0)
        self.best_epoch = checkpoint.get('best_epoch', 0)

        print(f"已加载 checkpoint: {filepath}")
        print(f"  Epoch: {checkpoint['epoch']}, "
              f"Best MAP: {self.best_valid_map:.4f}")

        return checkpoint

    def _cleanup_checkpoints(self) -> None:
        """清理非最佳 checkpoint"""
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

    def save_training_log(self) -> None:
        """保存训练日志"""
        log = {
            'history': self.history,
            'best_valid_map': self.best_valid_map,
            'best_epoch': self.best_epoch,
            'timestamp': datetime.now().isoformat(),
        }

        log_path = self.output_dir / 'training_log.json'
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2)

        print(f"训练日志已保存到: {log_path}")
