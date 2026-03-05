"""
度量学习诊断工具

提供训练过程中对度量学习的深度监控:
- 特征范数分布
- 梯度大小监控
- 类内/类间距离在训练中的变化
- 损失权重平衡分析
- 收敛性指标
- 批次中正/负样本对数量

设计原则:
  - 低开销: 默认不启用，通过 --metric_debug 或配置开启
  - 无侵入: 通过回调方式接入 trainer，不修改核心训练逻辑
  - 结构清晰: 所有诊断数据存为 JSON，方便后续分析
"""

import json
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import numpy as np


class MetricLearningDebugger:
    """
    度量学习诊断器

    通过 hook 方式接入训练循环，监控度量学习的健康状况。

    用法:
        debugger = MetricLearningDebugger(output_dir, log_interval=50)
        
        # 在 train_epoch 里调用
        debugger.on_batch_end(
            epoch=epoch, batch_idx=batch_idx,
            features=features, labels=labels,
            cls_loss=cls_loss, metric_loss=ml_loss,
            metric_module=self.metric_loss)
        
        # epoch 结束时调用
        debugger.on_epoch_end(epoch, model)
        
        # 训练结束后保存
        debugger.save_report()
    """

    def __init__(
        self,
        output_dir: str,
        log_interval: int = 50,
        enabled: bool = True
    ):
        """
        Args:
            output_dir: 诊断报告输出目录
            log_interval: 每隔多少 batch 打印一次诊断信息
            enabled: 是否启用诊断 (False 时所有方法变为 no-op)
        """
        self.output_dir = Path(output_dir)
        self.log_interval = log_interval
        self.enabled = enabled

        if not self.enabled:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 诊断数据存储
        self.history: Dict[str, List] = defaultdict(list)

        # 每 epoch 的汇总统计
        self._epoch_stats: Dict[str, List[float]] = defaultdict(list)

        self._print_header = True

    def on_batch_end(
        self,
        epoch: int,
        batch_idx: int,
        features: torch.Tensor,
        labels: torch.Tensor,
        cls_loss: float,
        metric_loss: Optional[float],
        metric_module: Optional[nn.Module] = None
    ):
        """每个 batch 结束时调用"""
        if not self.enabled:
            return

        with torch.no_grad():
            stats = self._compute_batch_stats(features, labels, metric_module)

        stats['cls_loss'] = cls_loss
        stats['metric_loss'] = metric_loss if metric_loss is not None else 0.0

        # 累积到 epoch 统计
        for key, val in stats.items():
            if isinstance(val, (int, float)):
                self._epoch_stats[key].append(val)

        # 定时打印
        if batch_idx > 0 and batch_idx % self.log_interval == 0:
            self._print_batch_diagnostics(epoch, batch_idx, stats)

    def on_epoch_end(
        self,
        epoch: int,
        model: nn.Module,
        metric_module: Optional[nn.Module] = None
    ):
        """每个 epoch 结束时调用"""
        if not self.enabled:
            return

        # 汇总 epoch 统计
        epoch_summary = {}
        for key, values in self._epoch_stats.items():
            if values:
                epoch_summary[f'{key}_mean'] = float(np.mean(values))
                epoch_summary[f'{key}_std'] = float(np.std(values))
        epoch_summary['epoch'] = epoch

        # 度量学习模块参数分析
        if metric_module is not None:
            param_stats = self._analyze_metric_params(metric_module)
            epoch_summary.update(param_stats)

        # 梯度统计
        grad_stats = self._analyze_gradients(model, metric_module)
        epoch_summary.update(grad_stats)

        self.history['epoch_summaries'].append(epoch_summary)

        # 打印 epoch 诊断报告
        self._print_epoch_report(epoch, epoch_summary)

        # 清空 epoch 累积
        self._epoch_stats.clear()

    def save_report(self, filename: str = 'metric_learning_diagnostics.json'):
        """保存完整诊断报告"""
        if not self.enabled:
            return

        report_path = self.output_dir / filename
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(dict(self.history), f, indent=2, ensure_ascii=False)
        print(f"\n  📊 度量学习诊断报告已保存: {report_path}")

    # ---- 内部分析方法 ----

    def _compute_batch_stats(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        metric_module: Optional[nn.Module]
    ) -> Dict[str, Any]:
        """计算单个 batch 的诊断统计"""
        B = features.size(0)
        stats = {}

        # 1. 特征范数统计
        feat_norms = features.norm(dim=1)
        stats['feat_norm_mean'] = feat_norms.mean().item()
        stats['feat_norm_std'] = feat_norms.std().item()
        stats['feat_norm_min'] = feat_norms.min().item()
        stats['feat_norm_max'] = feat_norms.max().item()

        # 2. 类内/类间距离
        feat_normed = nn.functional.normalize(features, dim=1)
        sim_mat = torch.mm(feat_normed, feat_normed.t())  # (B, B)

        labels_eq = (labels.unsqueeze(0) == labels.unsqueeze(1))  # (B, B)
        eye = torch.eye(B, dtype=torch.bool, device=features.device)

        pos_mask = labels_eq & ~eye  # 同类但非自身
        neg_mask = ~labels_eq

        if pos_mask.any():
            intra_sim = sim_mat[pos_mask].float()
            stats['intra_class_sim_mean'] = intra_sim.mean().item()
            stats['intra_class_sim_std'] = intra_sim.std().item()
        else:
            stats['intra_class_sim_mean'] = 0.0
            stats['intra_class_sim_std'] = 0.0

        if neg_mask.any():
            inter_sim = sim_mat[neg_mask].float()
            stats['inter_class_sim_mean'] = inter_sim.mean().item()
            stats['inter_class_sim_std'] = inter_sim.std().item()
        else:
            stats['inter_class_sim_mean'] = 0.0
            stats['inter_class_sim_std'] = 0.0

        # 3. 类内/类间距离差值（越大越好，表示分类边界更清晰）
        stats['sim_gap'] = stats['intra_class_sim_mean'] - stats['inter_class_sim_mean']

        # 4. 批次中唯一类别数和正样本对数
        unique_classes = labels.unique().numel()
        num_pos_pairs = pos_mask.sum().item() // 2  # 对称矩阵除以 2
        stats['unique_classes_in_batch'] = unique_classes
        stats['num_pos_pairs'] = num_pos_pairs
        stats['samples_per_class_avg'] = B / max(unique_classes, 1)

        # 5. ArcFace/ProxyNCA 专用: 权重（proxy）与特征的余弦距离
        if metric_module is not None:
            if hasattr(metric_module, 'weight'):
                weight = metric_module.weight.data
                weight_normed = nn.functional.normalize(weight, dim=1)
                # 权重之间的余弦相似度（类间距）
                w_sim = torch.mm(weight_normed, weight_normed.t())
                w_eye = torch.eye(w_sim.size(0), dtype=torch.bool, device=w_sim.device)
                stats['proxy_inter_sim_mean'] = w_sim[~w_eye].mean().item()
                # 权重范数
                stats['proxy_norm_mean'] = weight.norm(dim=1).mean().item()
            elif hasattr(metric_module, 'proxies'):
                proxies = metric_module.proxies.data
                proxies_normed = nn.functional.normalize(proxies, dim=1)
                p_sim = torch.mm(proxies_normed, proxies_normed.t())
                p_eye = torch.eye(p_sim.size(0), dtype=torch.bool, device=p_sim.device)
                stats['proxy_inter_sim_mean'] = p_sim[~p_eye].mean().item()
                stats['proxy_norm_mean'] = proxies.norm(dim=1).mean().item()

        return stats

    def _analyze_metric_params(self, metric_module: nn.Module) -> Dict[str, float]:
        """分析度量学习模块的参数状态"""
        stats = {}
        total_params = 0
        total_grad_norm = 0.0

        for name, param in metric_module.named_parameters():
            if param.requires_grad:
                total_params += param.numel()
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
                    stats[f'metric_param_{name}_grad_norm'] = grad_norm
                    stats[f'metric_param_{name}_value_norm'] = param.data.norm().item()

        stats['metric_total_params'] = total_params
        stats['metric_total_grad_norm'] = total_grad_norm
        return stats

    def _analyze_gradients(self, model: nn.Module, metric_module: Optional[nn.Module]) -> Dict[str, float]:
        """分析模型和度量学习模块的梯度"""
        stats = {}

        # backbone 梯度统计
        backbone_grad_norms = []
        classifier_grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if any(kw in name for kw in ['fc.', 'classifier.', 'head.']):
                    classifier_grad_norms.append(grad_norm)
                else:
                    backbone_grad_norms.append(grad_norm)

        if backbone_grad_norms:
            stats['backbone_grad_norm_mean'] = float(np.mean(backbone_grad_norms))
            stats['backbone_grad_norm_max'] = float(np.max(backbone_grad_norms))
        if classifier_grad_norms:
            stats['classifier_grad_norm_mean'] = float(np.mean(classifier_grad_norms))

        return stats

    def _print_batch_diagnostics(self, epoch: int, batch_idx: int, stats: Dict):
        """打印 batch 级诊断"""
        if self._print_header:
            print("\n  [MetricDebug] 度量学习诊断信息:")
            print("  " + "-" * 70)
            self._print_header = False

        sim_gap = stats.get('sim_gap', 0)
        gap_indicator = '✓' if sim_gap > 0.1 else ('△' if sim_gap > 0 else '✗')

        parts = [
            f"  [E{epoch} B{batch_idx}]",
            f"SimGap={sim_gap:.3f}{gap_indicator}",
            f"Intra={stats.get('intra_class_sim_mean', 0):.3f}",
            f"Inter={stats.get('inter_class_sim_mean', 0):.3f}",
            f"FeatNorm={stats.get('feat_norm_mean', 0):.1f}±{stats.get('feat_norm_std', 0):.1f}",
            f"PosPairs={stats.get('num_pos_pairs', 0)}",
            f"Classes={stats.get('unique_classes_in_batch', 0)}",
        ]
        if 'proxy_inter_sim_mean' in stats:
            parts.append(f"ProxySim={stats['proxy_inter_sim_mean']:.3f}")
        if stats.get('metric_loss', 0) > 0:
            ratio = stats['cls_loss'] / max(stats['metric_loss'], 1e-8)
            parts.append(f"Cls/Metric={ratio:.2f}")

        print(" | ".join(parts))

    def _print_epoch_report(self, epoch: int, summary: Dict):
        """打印 epoch 级诊断报告"""
        print(f"\n  {'='*60}")
        print(f"  📊 度量学习诊断报告 — Epoch {epoch}")
        print(f"  {'='*60}")

        # 类内/类间距离趋势
        sim_gap = summary.get('sim_gap_mean', 0)
        intra = summary.get('intra_class_sim_mean_mean', 0)
        inter = summary.get('inter_class_sim_mean_mean', 0)
        print(f"  类内余弦相似度:      {intra:.4f}")
        print(f"  类间余弦相似度:      {inter:.4f}")
        print(f"  Similarity Gap:      {sim_gap:.4f} {'✓ 正常' if sim_gap > 0.1 else '⚠ 偏低'}")

        # 特征范数
        fn_mean = summary.get('feat_norm_mean_mean', 0)
        fn_std = summary.get('feat_norm_std_mean', 0)
        print(f"  特征范数:            {fn_mean:.2f} ± {fn_std:.2f}")

        # 正样本对数量
        pos_pairs = summary.get('num_pos_pairs_mean', 0)
        classes = summary.get('unique_classes_in_batch_mean', 0)
        spc = summary.get('samples_per_class_avg_mean', 0)
        print(f"  每批次正样本对:      {pos_pairs:.0f}")
        print(f"  每批次唯一类别数:    {classes:.0f}")
        print(f"  每批次类均样本数:    {spc:.1f}")

        # 对 pair-level 损失的警告
        if pos_pairs < 5:
            print(f"  ⚠ 警告: 正样本对过少 ({pos_pairs:.0f})!")
            print(f"    → pair-level 损失(contrastive/triplet/circle)需要更大 batch_size")
            print(f"    → 建议: 使用 proxy-based 损失(arcface/proxy_nca/circle_cls)")

        # 损失平衡
        cls_loss = summary.get('cls_loss_mean', 0)
        ml_loss = summary.get('metric_loss_mean', 0)
        if ml_loss > 0:
            ratio = cls_loss / max(ml_loss, 1e-8)
            print(f"  分类损失/度量损失:   {ratio:.2f}")
            if ratio > 10:
                print(f"  ⚠ 度量损失相对过小，可能无效。建议增大 --metric_loss_weight")
            elif ratio < 0.1:
                print(f"  ⚠ 度量损失主导训练，可能影响分类精度。建议减小 --metric_loss_weight")

        # Proxy/Weight 分析
        if 'proxy_inter_sim_mean_mean' in summary:
            proxy_sim = summary['proxy_inter_sim_mean_mean']
            print(f"  Proxy 类间余弦:      {proxy_sim:.4f}")
            if proxy_sim > 0.8:
                print(f"  ⚠ Proxy 向量过于相似，区分性差。考虑增大 margin 或降低学习率")

        # 梯度健康
        bb_grad = summary.get('backbone_grad_norm_mean', 0)
        clf_grad = summary.get('classifier_grad_norm_mean', 0)
        ml_grad = summary.get('metric_total_grad_norm', 0)
        if bb_grad > 0:
            print(f"  梯度范数:            backbone={bb_grad:.4f} | classifier={clf_grad:.4f} | metric={ml_grad:.4f}")
            if bb_grad < 1e-6:
                print(f"  ⚠ Backbone 梯度接近零，可能存在梯度消失")

        print(f"  {'='*60}\n")

        self._print_header = True
