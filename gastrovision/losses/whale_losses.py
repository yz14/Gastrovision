"""
鲸鱼分类损失函数模块

保留原始三损失组合, 每个损失通过 YAML 权重控制开关:
- focal_OHEM: BCE + Focal + 目标类 Focal (权重 focal_w)
- softmax: CrossEntropy (权重 softmax_w)
- TripletLoss: 带 hard mining 的三元组损失 (权重 triplet_w)

所有损失通过 WhaleCompositeLoss 统一管理。

原始来源: colonnav_ssl/loss/loss.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# 基础损失函数
# ============================================================

def l2_norm(x, axis=1):
    """L2 归一化"""
    norm = torch.norm(x, 2, axis, True)
    return torch.div(x, norm)


def focal_loss(input, target, OHEM_percent=None):
    """
    Focal Loss (二分类形式)

    Args:
        input: 模型输出 logits
        target: 目标 (与 input 同尺寸)
        OHEM_percent: 在线难例挖掘比例, None 表示使用所有样本
    """
    gamma = 2

    assert target.size() == input.size(), \
        f"target size {target.size()} != input size {input.size()}"

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + \
        ((-max_val).exp() + (-input - max_val).exp()).log()

    invprobs = F.logsigmoid(-input * (target * 2 - 1))
    loss = (invprobs * gamma).exp() * loss

    if OHEM_percent is None:
        return loss.mean()
    else:
        # 动态获取类别数, 避免硬编码
        class_num = input.shape[1] if input.dim() > 1 else input.shape[0]
        k = max(1, int(class_num * OHEM_percent))
        if input.dim() > 1:
            OHEM, _ = loss.topk(k=k, dim=1, largest=True, sorted=True)
        else:
            OHEM, _ = loss.topk(k=k, largest=True, sorted=True)
        return OHEM.mean()


def bce_loss(input, target, OHEM_percent=None):
    """
    Binary Cross Entropy Loss

    Args:
        input: 模型输出 logits
        target: 目标 (与 input 同尺寸)
        OHEM_percent: 在线难例挖掘比例
    """
    if OHEM_percent is None:
        return F.binary_cross_entropy_with_logits(input, target, reduction='mean')
    else:
        loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        class_num = input.shape[1] if input.dim() > 1 else input.shape[0]
        k = max(1, int(class_num * OHEM_percent))
        if input.dim() > 1:
            value, _ = loss.topk(k, dim=1, largest=True, sorted=True)
        else:
            value, _ = loss.topk(k, largest=True, sorted=True)
        return value.mean()


def focal_OHEM(results, labels, labels_onehot, OHEM_percent=100):
    """
    Focal + OHEM 组合损失

    用于 BinaryHead 的输出。
    组合: BCE loss + Focal loss + 目标类的 Focal loss。

    Args:
        results: 模型输出 logits [B, C]
        labels: 类别标签 [B] (整数)
        labels_onehot: one-hot 编码 [B, C]
        OHEM_percent: OHEM 比例
    """
    batch_size, class_num = results.shape
    labels = labels.view(-1)

    loss0 = bce_loss(results, labels_onehot, OHEM_percent)
    loss1 = focal_loss(results, labels_onehot, OHEM_percent)

    # 目标类的额外 focal loss (排除 new_whale 等超出类别范围的标签)
    indexs_ = (labels != class_num).nonzero().view(-1)
    if len(indexs_) == 0:
        return loss0 + loss1

    # 取出每个样本在其目标类上的 logit
    results_ = results[
        torch.arange(0, len(results), device=results.device)[indexs_],
        labels[indexs_]
    ].contiguous()
    loss2 = focal_loss(results_, torch.ones_like(results_).float())

    return loss0 + loss1 + loss2


def softmax_loss(results, labels):
    """
    标准 Softmax (CrossEntropy) 损失

    用于 MarginHead 的输出。

    Args:
        results: logits [B, C]
        labels: 类别标签 [B]
    """
    labels = labels.view(-1)
    return F.cross_entropy(results, labels)


# ============================================================
# Triplet Loss
# ============================================================

def euclidean_dist(x, y):
    """计算欧氏距离矩阵"""
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def hard_example_mining(dist_mat, labels):
    """
    批内 Hard Mining

    对每个 anchor 找到:
    - 最远的正样本 (hardest positive)
    - 最近的负样本 (hardest negative)

    Args:
        dist_mat: 距离矩阵 [N, N]
        labels: 标签 [N]

    Returns:
        dist_ap: anchor-positive 距离 [N]
        dist_an: anchor-negative 距离 [N]
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # 最远正样本
    dist_ap, _ = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # 最近负样本
    dist_an, _ = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)

    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    return dist_ap, dist_an


class TripletLoss:
    """
    Triplet Loss (带 Hard Mining)

    参考: 'In Defense of the Triplet Loss for Person Re-Identification'

    Args:
        margin: 间隔值, None 则使用 SoftMarginLoss (无需调参)
    """

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels):
        """
        Args:
            global_feat: 特征向量 [B, D]
            labels: 标签 [B]
        """
        global_feat = l2_norm(global_feat)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        y = torch.ones_like(dist_an)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)

        return loss


# ============================================================
# 评估指标
# ============================================================

def metric(prob, label, thres=0.5):
    """
    计算 MAP@5 指标

    Args:
        prob: 预测概率 [N, C]
        label: 真实标签 [N]
        thres: new_whale 阈值

    Returns:
        (precision, top5): MAP 和 top-5 准确率列表
    """
    shape = prob.shape
    prob_tmp = np.ones([shape[0], shape[1] + 1]) * thres
    prob_tmp[:, :shape[1]] = prob
    precision, top5 = _top_n_np(prob_tmp, label)
    return precision, top5


def _top_n_np(preds, labels, n=5):
    """计算 top-n 准确率和 MAP"""
    predicted = np.fliplr(preds.argsort(axis=1)[:, -n:])
    top5 = []

    re = 0
    for i in range(len(preds)):
        predicted_tmp = predicted[i]
        labels_tmp = labels[i]
        for n_ in range(5):
            re += np.sum(labels_tmp == predicted_tmp[n_]) / (n_ + 1.0)

    re = re / len(preds)
    for i in range(n):
        top5.append(np.sum(labels == predicted[:, i]) / (1.0 * len(labels)))
    return re, top5


# ============================================================
# 组合损失管理器
# ============================================================

class WhaleCompositeLoss:
    """
    鲸鱼分类组合损失管理器

    通过 YAML 配置控制各损失的权重 (权重=0 即关闭该损失)。

    支持的损失组合:
    1. focal_OHEM (BinaryHead) + softmax (MarginHead) + triplet (features)
    2. 任意子集 (通过权重控制)
    3. triplet 可替换为 Gastrovision 的 metric_loss

    Args:
        focal_w: focal_OHEM 损失权重
        softmax_w: softmax 损失权重
        triplet_w: triplet 损失权重
        triplet_margin: triplet 间隔 (None=SoftMargin, 0.3=MarginRanking)
        metric_loss: Gastrovision 度量学习损失实例 (替代 triplet, 可选)
        metric_loss_weight: 度量学习损失权重
    """

    def __init__(
        self,
        focal_w=1.0,
        softmax_w=0.1,
        triplet_w=1.0,
        triplet_margin=0.3,
        metric_loss=None,
        metric_loss_weight=0.5
    ):
        self.focal_w = focal_w
        self.softmax_w = softmax_w
        self.triplet_w = triplet_w
        self.metric_loss = metric_loss
        self.metric_loss_weight = metric_loss_weight

        # 仅在需要时创建 TripletLoss
        self.triplet_loss = TripletLoss(margin=triplet_margin) if triplet_w > 0 else None

        # 打印损失配置
        print("鲸鱼组合损失配置:")
        if self.focal_w > 0:
            print(f"  ✓ focal_OHEM: 权重={self.focal_w}")
        else:
            print(f"  ✗ focal_OHEM: 已关闭")
        if self.softmax_w > 0:
            print(f"  ✓ softmax: 权重={self.softmax_w}")
        else:
            print(f"  ✗ softmax: 已关闭")
        if self.triplet_w > 0:
            print(f"  ✓ triplet: 权重={self.triplet_w}, margin={triplet_margin}")
        else:
            print(f"  ✗ triplet: 已关闭")
        if self.metric_loss is not None:
            print(f"  ✓ metric_loss (替代/叠加 triplet): 权重={self.metric_loss_weight}")

    def __call__(
        self,
        logit_binary,
        logit_softmax,
        features,
        labels,
        labels_onehot,
        hard_ratio,
        indexs_NoNew
    ):
        """
        计算组合损失

        Args:
            logit_binary: BinaryHead 输出 [B, C]
            logit_softmax: MarginHead 输出 [B, C]
            features: Backbone 特征 [B, D]
            labels: 整数标签 [B]
            labels_onehot: one-hot 标签 [B, C]
            hard_ratio: OHEM 难例比例
            indexs_NoNew: 非 new_whale 样本的索引

        Returns:
            (total_loss, loss_dict): 总损失和各分量字典
        """
        total_loss = 0.0
        loss_dict = {}

        # 1. focal_OHEM 损失 (BinaryHead)
        if self.focal_w > 0:
            loss_focal = focal_OHEM(logit_binary, labels, labels_onehot, hard_ratio)
            total_loss = total_loss + loss_focal * self.focal_w
            loss_dict['focal'] = loss_focal.item()

        # 2. softmax 损失 (MarginHead, 仅非 new_whale 样本)
        if self.softmax_w > 0 and len(indexs_NoNew) > 0:
            loss_softmax = softmax_loss(logit_softmax[indexs_NoNew], labels[indexs_NoNew])
            total_loss = total_loss + loss_softmax * self.softmax_w
            loss_dict['softmax'] = loss_softmax.item()

        # 3. Triplet 损失 (原生)
        if self.triplet_w > 0 and self.triplet_loss is not None:
            loss_triplet = self.triplet_loss(features, labels)
            total_loss = total_loss + loss_triplet * self.triplet_w
            loss_dict['triplet'] = loss_triplet.item()

        # 4. Gastrovision 度量学习损失 (替代/叠加 triplet)
        if self.metric_loss is not None:
            ml_loss = self.metric_loss(features, labels)
            total_loss = total_loss + ml_loss * self.metric_loss_weight
            loss_dict['metric'] = ml_loss.item()

        loss_dict['total'] = total_loss.item() if torch.is_tensor(total_loss) else total_loss

        return total_loss, loss_dict

    def parameters(self):
        """返回可学习损失函数的参数 (用于加入优化器)"""
        params = []
        if self.metric_loss is not None and hasattr(self.metric_loss, 'parameters'):
            params.extend(list(self.metric_loss.parameters()))
        return params
