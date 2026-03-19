"""
Jaguar Re-ID 训练器

训练循环：ArcFace 分类损失 + 可选的度量学习辅助损失。
验证指标：Rank-1 准确率 + mAP（基于 embedding 余弦相似度）。
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gastrovision.utils.metrics import AverageMeter
from gastrovision.utils.ema import ModelEMA


class ReIDTrainer:
    """
    Re-ID 训练器

    Args:
        model: ReIDModel
        arcface_head: ArcFace 损失（含可学习权重）
        optimizer: 优化器（需包含 model + arcface_head 的参数）
        device: 训练设备
        scheduler: 学习率调度器
        output_dir: 输出目录
        use_ema: 是否使用 EMA
        ema_decay: EMA 衰减率
        label_smooth: 标签平滑（应用于 ArcFace 的 CE 损失）
    """

    def __init__(
        self,
        model: nn.Module,
        arcface_head: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        scheduler: Optional[_LRScheduler] = None,
        output_dir: str = './output',
        use_ema: bool = False,
        ema_decay: float = 0.9999,
        label_smooth: float = 0.0,
    ):
        self.model = model.to(device)
        self.arcface_head = arcface_head.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.label_smooth = label_smooth

        # EMA
        self.ema = None
        if use_ema:
            self.ema = ModelEMA(model, decay=ema_decay)
            print(f"  EMA 已启用 (decay={ema_decay})")

        # 训练状态
        self.best_metric = 0.0
        self.best_epoch = 0
        self.train_history = []
        self.start_epoch = 0

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping: int = 15,
        val_label_map: dict = None,
    ):
        """
        训练主循环

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 总训练轮数
            early_stopping: 早停耐心值 (0=禁用)
            val_label_map: 验证集的 label_map（用于 Re-ID 评估）
        """
        patience_counter = 0
        total_start = time.time()

        log_path = self.output_dir / 'training_log.txt'

        for epoch in range(self.start_epoch, epochs):
            epoch_start = time.time()

            # ---- 训练 ----
            train_metrics = self._train_epoch(train_loader, epoch, epochs)

            # ---- 验证 ----
            val_metrics = self._validate(val_loader)

            # ---- 学习率调度 ----
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if hasattr(self.scheduler, 'step'):
                    # ReduceLROnPlateau 需要传入 metric
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['rank1'])
                    else:
                        self.scheduler.step()

            epoch_time = time.time() - epoch_start

            # ---- 日志 ----
            log_line = (
                f"Epoch [{epoch+1}/{epochs}] "
                f"lr={current_lr:.6f} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['acc']:.4f} "
                f"val_rank1={val_metrics['rank1']:.4f} "
                f"val_mAP={val_metrics['mAP']:.4f} "
                f"time={epoch_time:.1f}s"
            )
            print(log_line)

            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(log_line + '\n')
                f.flush()

            # ---- 保存最佳模型 ----
            metric = val_metrics['rank1']
            if metric > self.best_metric:
                self.best_metric = metric
                self.best_epoch = epoch + 1
                patience_counter = 0
                self._save_checkpoint(epoch, 'best_model.pth', val_metrics)
                print(f"  ★ 新最佳! Rank-1={metric:.4f}")
            else:
                patience_counter += 1

            # 每 10 epoch 保存一次
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, f'checkpoint_epoch{epoch+1}.pth', val_metrics)

            # ---- 早停 ----
            if early_stopping > 0 and patience_counter >= early_stopping:
                print(f"\n早停触发 (连续 {early_stopping} 轮无改善)")
                break

            self.train_history.append({
                'epoch': epoch + 1,
                **train_metrics,
                **{f'val_{k}': v for k, v in val_metrics.items()},
                'lr': current_lr,
            })

        total_time = time.time() - total_start
        print(f"\n训练完成! 总时间: {total_time/60:.1f} 分钟")
        print(f"最佳 Rank-1: {self.best_metric:.4f} (Epoch {self.best_epoch})")

        # 保存训练历史
        with open(self.output_dir / 'train_history.json', 'w') as f:
            json.dump(self.train_history, f, indent=2)

    def _train_epoch(self, loader: DataLoader, epoch: int, total_epochs: int) -> dict:
        """单个 epoch 训练"""
        self.model.train()
        self.arcface_head.train()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward: 获取 BN 后的特征用于 ArcFace
            bn_feat = self.model(images)

            # ArcFace loss
            loss = self.arcface_head(bn_feat, labels)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.arcface_head.parameters()),
                max_norm=5.0
            )
            self.optimizer.step()

            # EMA 更新
            if self.ema is not None:
                self.ema.update(self.model)

            # 统计
            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)

            # 计算分类准确率（用 ArcFace 的 logit）
            with torch.no_grad():
                feat_norm = F.normalize(bn_feat, p=2, dim=1)
                weight_norm = F.normalize(self.arcface_head.weight, p=2, dim=1)
                cosine = torch.mm(feat_norm, weight_norm.t())
                preds = cosine.argmax(dim=1)
                acc = (preds == labels).float().mean().item()
                acc_meter.update(acc, batch_size)

        return {'loss': loss_meter.avg, 'acc': acc_meter.avg}

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict:
        """
        验证：基于 embedding 的 Rank-1 准确率和 mAP

        策略：对验证集所有样本提取 embedding，
        每个样本作为 query，其他同类样本作为 gallery，计算 Rank-1 和 mAP。
        """
        model = self.model
        if self.ema is not None:
            self.ema.apply_shadow(model)

        model.eval()

        all_embeddings = []
        all_labels = []

        for images, labels in loader:
            images = images.to(self.device)
            embeddings = model.extract_embedding(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

        all_embeddings = torch.cat(all_embeddings, dim=0)  # (N, D)
        all_labels = torch.cat(all_labels, dim=0)  # (N,)

        # 计算余弦相似度矩阵
        sim_matrix = torch.mm(all_embeddings, all_embeddings.t())  # (N, N)

        # 对角线设为 -inf（排除自身）
        N = sim_matrix.size(0)
        sim_matrix.fill_diagonal_(-float('inf'))

        # 计算 Rank-1 和 mAP
        rank1_correct = 0
        ap_sum = 0.0
        valid_queries = 0

        for i in range(N):
            query_label = all_labels[i].item()

            # 其他样本中是否有同类
            mask_same = (all_labels == query_label)
            mask_same[i] = False  # 排除自身
            n_same = mask_same.sum().item()

            if n_same == 0:
                continue  # 这个类在验证集中只有 1 个样本

            valid_queries += 1

            # 按相似度降序排列
            sorted_indices = sim_matrix[i].argsort(descending=True)
            sorted_match = mask_same[sorted_indices].float()

            # Rank-1
            if sorted_match[0] == 1.0:
                rank1_correct += 1

            # AP (Average Precision)
            cum_correct = sorted_match.cumsum(0)
            precision_at_k = cum_correct / torch.arange(1, N + 1, dtype=torch.float32)
            ap = (precision_at_k * sorted_match).sum().item() / n_same
            ap_sum += ap

        rank1 = rank1_correct / max(valid_queries, 1)
        mAP = ap_sum / max(valid_queries, 1)

        if self.ema is not None:
            self.ema.restore(model)

        return {'rank1': rank1, 'mAP': mAP}

    def _save_checkpoint(self, epoch: int, filename: str, metrics: dict = None):
        """保存 checkpoint"""
        path = self.output_dir / filename
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'arcface_state_dict': self.arcface_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
        }
        if self.scheduler is not None:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.ema is not None:
            state['ema_state_dict'] = self.ema.state_dict()
        if metrics is not None:
            state['metrics'] = metrics
        torch.save(state, path)

    def load_checkpoint(self, checkpoint_path: str):
        """加载 checkpoint"""
        path = Path(checkpoint_path)
        if not path.exists():
            # 尝试在 output_dir 中查找
            path = self.output_dir / checkpoint_path
        if not path.exists():
            print(f"  [警告] Checkpoint 不存在: {path}")
            return

        print(f"  加载 checkpoint: {path}")
        state = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(state['model_state_dict'])
        if 'arcface_state_dict' in state:
            self.arcface_head.load_state_dict(state['arcface_state_dict'])
        if 'optimizer_state_dict' in state:
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
        if self.scheduler is not None and 'scheduler_state_dict' in state:
            self.scheduler.load_state_dict(state['scheduler_state_dict'])
        if self.ema is not None and 'ema_state_dict' in state:
            self.ema.load_state_dict(state['ema_state_dict'])

        self.best_metric = state.get('best_metric', 0.0)
        self.best_epoch = state.get('best_epoch', 0)
        self.start_epoch = state.get('epoch', 0) + 1

        print(f"  恢复自 Epoch {self.start_epoch}, 最佳 Rank-1={self.best_metric:.4f}")
