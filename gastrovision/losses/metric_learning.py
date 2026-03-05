"""
Gastrovision 度量学习损失函数模块

实现以下度量学习损失：
1. ContrastiveLoss - 对比损失 (Siamese Network, DrLIM CVPR 2006)
2. TripletMarginLoss - 三元组损失 (FaceNet CVPR 2015, 带在线硬样本挖掘)
3. LiftedStructureLoss - 提升结构损失 (CVPR 2016)
4. ProxyNCALoss - 代理 NCA 损失 (NeurIPS 2017)
5. NPairLoss - N-pair 损失 (NeurIPS 2016)
6. ArcFaceLoss / CosFaceLoss / SphereFaceLoss - 角度间隔系列 (CVPR 2017/2018/2019)
7. CircleLoss - 圆损失 (CVPR 2020)

参考文献：
- Contrastive: Hadsell et al., "Dimensionality Reduction by Learning an Invariant Mapping", CVPR 2006
- Triplet: Schroff et al., "FaceNet: A Unified Embedding for Face Recognition and Clustering", CVPR 2015
- Lifted Structure: Oh Song et al., "Deep Metric Learning via Lifted Structured Feature Embedding", CVPR 2016
- Proxy NCA: Movshovitz-Attias et al., "No Fuss Distance Metric Learning Using Proxies", NeurIPS 2017
- N-pair: Sohn, "Improved Deep Metric Learning with Multi-class N-pair Loss Objective", NeurIPS 2016
- SphereFace: Liu et al., "SphereFace: Deep Hypersphere Embedding for Face Recognition", CVPR 2017
- CosFace: Wang et al., "CosFace: Large Margin Cosine Loss for Deep Face Recognition", CVPR 2018
- ArcFace: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", CVPR 2019
- Circle Loss: Sun et al., "Circle Loss: A Unified Perspective of Pair Similarity Optimization", CVPR 2020
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# =============================================================================
# 工具函数
# =============================================================================

def l2_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    """L2 归一化"""
    return F.normalize(x, p=2, dim=dim, eps=eps)


def pairwise_euclidean_distance(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    """
    计算成对欧氏距离矩阵
    
    Args:
        x: (N, D) 特征矩阵
        y: (M, D) 特征矩阵（默认与 x 相同）
    
    Returns:
        (N, M) 距离矩阵
    """
    if y is None:
        y = x
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x @ y^T
    xx = (x * x).sum(dim=1, keepdim=True)  # (N, 1)
    yy = (y * y).sum(dim=1, keepdim=True)  # (M, 1)
    dist_sq = xx + yy.t() - 2.0 * torch.mm(x, y.t())
    dist_sq = dist_sq.clamp(min=0.0)  # 数值稳定
    # 避免 sqrt(0) 在反向传播时出现非有限梯度
    return (dist_sq + 1e-12).sqrt()


def pairwise_cosine_similarity(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    """
    计算成对余弦相似度矩阵
    
    Args:
        x: (N, D) 特征矩阵
        y: (M, D) 特征矩阵（默认与 x 相同）
    
    Returns:
        (N, M) 余弦相似度矩阵
    """
    if y is None:
        y = x
    x_norm = l2_normalize(x, dim=1)
    y_norm = l2_normalize(y, dim=1)
    return torch.mm(x_norm, y_norm.t())


def get_pairwise_labels(labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从标签生成正负样本对掩码
    
    Args:
        labels: (N,) 标签向量
        
    Returns:
        pos_mask: (N, N) 正样本对掩码（同类别且非自身）
        neg_mask: (N, N) 负样本对掩码（不同类别）
    """
    labels = labels.unsqueeze(0)  # (1, N)
    match = (labels == labels.t()).float()  # (N, N)
    eye = torch.eye(labels.size(1), device=labels.device)
    pos_mask = match - eye  # 排除自身
    neg_mask = 1.0 - match
    return pos_mask, neg_mask


# =============================================================================
# 1. Contrastive Loss (对比损失)
# =============================================================================

class ContrastiveLoss(nn.Module):
    """
    对比损失 (DrLIM, CVPR 2006)
    
    基于 batch 的实现：从 batch 中自动构造正负样本对。
    
    L = (1/|P|) * Σ_{(i,j)∈P} d(i,j)²
      + (1/|N|) * Σ_{(i,j)∈N} max(0, margin - d(i,j))²
    
    其中 P 为同类别对, N 为不同类别对, d 为欧氏距离。
    
    Args:
        margin: 负样本对的最小距离间隔 (默认 1.0)
        normalize: 是否对特征做 L2 归一化 (默认 True)
    """
    
    def __init__(self, margin: float = 1.0, normalize: bool = True):
        super().__init__()
        self.margin = margin
        self.normalize = normalize
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) 嵌入特征
            labels: (B,) 类别标签
            
        Returns:
            标量损失
        """
        if self.normalize:
            features = l2_normalize(features)
        
        dist_mat = pairwise_euclidean_distance(features)
        pos_mask, neg_mask = get_pairwise_labels(labels)
        
        # 正样本对损失: 拉近同类
        pos_loss = (dist_mat.pow(2) * pos_mask)
        num_pos = pos_mask.sum().clamp(min=1.0)
        pos_loss = pos_loss.sum() / num_pos
        
        # 负样本对损失: 推远异类
        neg_dist = F.relu(self.margin - dist_mat)
        neg_loss = (neg_dist.pow(2) * neg_mask)
        num_neg = neg_mask.sum().clamp(min=1.0)
        neg_loss = neg_loss.sum() / num_neg
        
        return pos_loss + neg_loss


# =============================================================================
# 2. Triplet Margin Loss (三元组损失, 带在线硬样本挖掘)
# =============================================================================

class TripletMarginLoss(nn.Module):
    """
    三元组损失 (FaceNet, CVPR 2015)
    
    带在线 Batch Hard Mining：对每个 anchor 选择最难的正样本和最难的负样本。
    
    L = (1/B) * Σ_a max(0, d(a, p_hard) - d(a, n_hard) + margin)
    
    其中 p_hard = argmax_{p: y_p=y_a} d(a,p), n_hard = argmin_{n: y_n≠y_a} d(a,n)
    
    Args:
        margin: 三元组间隔 (默认 0.3)
        normalize: 是否对特征做 L2 归一化 (默认 True)
        soft: 使用 soft margin (log(1+exp(d_ap - d_an))) 替代 hard margin
    """
    
    def __init__(self, margin: float = 0.3, normalize: bool = True, soft: bool = False):
        super().__init__()
        self.margin = margin
        self.normalize = normalize
        self.soft = soft
        
        if self.soft:
            self.ranking_loss = nn.SoftMarginLoss()
        else:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    
    def _hard_example_mining(
        self, 
        dist_mat: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch Hard Mining: 对每个 anchor 选择最远正样本和最近负样本
        """
        pos_mask, neg_mask = get_pairwise_labels(labels)
        
        # 最远正样本 (hardest positive)
        # 将非正样本对的距离设为 -inf，取 max
        dist_ap = dist_mat * pos_mask + (-1e12) * (1.0 - pos_mask)
        dist_ap, _ = dist_ap.max(dim=1)
        
        # 最近负样本 (hardest negative)  
        # 将非负样本对的距离设为 +inf，取 min
        dist_an = dist_mat * neg_mask + 1e12 * (1.0 - neg_mask)
        dist_an, _ = dist_an.min(dim=1)
        
        return dist_ap, dist_an
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) 嵌入特征
            labels: (B,) 类别标签
            
        Returns:
            标量损失
        """
        if self.normalize:
            features = l2_normalize(features)
        
        dist_mat = pairwise_euclidean_distance(features)
        dist_ap, dist_an = self._hard_example_mining(dist_mat, labels)
        
        y = torch.ones_like(dist_an)
        
        if self.soft:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        
        return loss


# =============================================================================
# 3. Lifted Structure Loss (提升结构损失)
# =============================================================================

class LiftedStructureLoss(nn.Module):
    """
    提升结构损失 (Oh Song et al., CVPR 2016)
    
    改进版本 (论文中 Eq. 4 的 smooth upper bound):
    
    L = (1/2|P|) * Σ_{(i,j)∈P} max(0, J_{ij})²
    
    J_{ij} = max(max_k{α - D_{ik}}, 0) + max(max_l{α - D_{jl}}, 0) + D_{ij}
    
    实现使用 logsumexp 近似 max 以获得更好的梯度：
    J_{ij} = log(Σ_k exp(α - D_{ik})) + log(Σ_l exp(α - D_{jl})) + D_{ij}
    
    Args:
        margin: 距离间隔 α (默认 1.0)
        normalize: 是否对特征做 L2 归一化 (默认 True)
    """
    
    def __init__(self, margin: float = 1.0, normalize: bool = True):
        super().__init__()
        self.margin = margin
        self.normalize = normalize
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) 嵌入特征
            labels: (B,) 类别标签
            
        Returns:
            标量损失
        """
        if self.normalize:
            features = l2_normalize(features)
        
        B = features.size(0)
        dist_mat = pairwise_euclidean_distance(features)
        pos_mask, neg_mask = get_pairwise_labels(labels)
        
        # 对每个样本 i，计算 logsumexp(α - D_{ik}) 对所有负样本 k
        # 先构造 margin - dist，再 mask 掉非负样本并做 logsumexp
        margin_dist = self.margin - dist_mat  # (B, B)
        
        # 对负样本取 logsumexp
        # 将非负样本的位置设为 -inf，这样 exp(-inf) = 0
        neg_margin_dist = margin_dist.clone()
        neg_margin_dist[neg_mask == 0] = -float('inf')
        lse_neg = torch.logsumexp(neg_margin_dist, dim=1)  # (B,)
        
        # 计算每对正样本 (i,j) 的损失
        # J_{ij} = lse_neg[i] + lse_neg[j] + D_{ij}
        pos_pairs = pos_mask.nonzero(as_tuple=False)  # (|P|, 2)
        
        if pos_pairs.size(0) == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        i_idx = pos_pairs[:, 0]
        j_idx = pos_pairs[:, 1]
        
        J = lse_neg[i_idx] + lse_neg[j_idx] + dist_mat[i_idx, j_idx]
        loss = F.relu(J).pow(2)
        
        return loss.mean() * 0.5


# =============================================================================
# 4. Proxy NCA Loss (代理 NCA 损失)
# =============================================================================

class ProxyNCALoss(nn.Module):
    """
    代理 NCA 损失 (Movshovitz-Attias et al., NeurIPS 2017)
    
    为每个类别维护一个可学习的 proxy 向量，用 NCA softmax 损失训练。
    避免了在线挖掘样本对的复杂性。
    
    L = -log( exp(-d(x, p_y)) / Σ_{z≠y} exp(-d(x, p_z)) )
    
    其中 p_y 是类别 y 的 proxy，d 是平方欧氏距离。
    
    Args:
        num_classes: 类别数量
        embedding_dim: 嵌入维度
        scale: 缩放因子 (默认 8.0，参考论文)
    """
    
    def __init__(self, num_classes: int, embedding_dim: int, scale: float = 8.0):
        super().__init__()
        self.num_classes = num_classes
        self.scale = scale
        
        # 可学习的 proxy 向量，每个类别一个
        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.kaiming_uniform_(self.proxies, a=math.sqrt(5))
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) 嵌入特征（L2 归一化后）
            labels: (B,) 类别标签
            
        Returns:
            标量损失
        """
        # 归一化特征和代理
        features = l2_normalize(features)
        proxies = l2_normalize(self.proxies)
        
        # 计算特征到所有 proxy 的距离
        # 使用负余弦相似度 (等效于归一化后的欧氏距离)
        # similarity: (B, C) where C = num_classes
        similarity = torch.mm(features, proxies.t()) * self.scale  # (B, C)
        
        # NCA softmax loss (排除正类 proxy 之外)
        # 等价于 CrossEntropyLoss
        loss = F.cross_entropy(similarity, labels)
        
        return loss


# =============================================================================
# 5. N-pair Loss (N-pair 损失)
# =============================================================================

class NPairLoss(nn.Module):
    """
    N-pair 损失 (Sohn, NeurIPS 2016)
    
    多类别推广的三元组损失，每个 anchor 同时考虑多个负样本。
    
    L = (1/B) * Σ_i log(1 + Σ_{j: y_j≠y_i} exp(f_i · f_j - f_i · f_i⁺))
    
    其中 f_i⁺ 是与 anchor i 同类的正样本特征。
    
    Args:
        normalize: 是否对特征做 L2 归一化 (默认 True)
        l2_reg: L2 正则化权重 (默认 0.02)
    """
    
    def __init__(self, normalize: bool = True, l2_reg: float = 0.02):
        super().__init__()
        self.normalize = normalize
        self.l2_reg = l2_reg
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) 嵌入特征
            labels: (B,) 类别标签
            
        Returns:
            标量损失
        """
        if self.normalize:
            features = l2_normalize(features)
        
        B = features.size(0)
        pos_mask, neg_mask = get_pairwise_labels(labels)
        
        # 相似度矩阵: S_{ij} = f_i · f_j
        similarity = torch.mm(features, features.t())  # (B, B)
        
        # 对每个 anchor i，找一个正样本 (取第一个同类样本)
        # pos_sim[i] = f_i · f_{p_i} 其中 p_i 是 i 的正样本
        pos_sim = (similarity * pos_mask)
        # 取每行最大的正样本相似度（最难正样本实际上应该是最小，但这里我们用平均）
        # 标准 N-pair loss: 使用 anchor 的 "正伙伴"
        has_pos = pos_mask.sum(dim=1) > 0  # (B,)
        
        if has_pos.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # 对每个有正样本的 anchor，取平均正相似度
        pos_sim_mean = pos_sim.sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)  # (B,)
        
        # N-pair loss: log(1 + Σ_{neg j} exp(s_{i,j} - s_{i,p}))
        # = logsumexp([0, s_{i,j1} - s_{i,p}, s_{i,j2} - s_{i,p}, ...])
        neg_exp = (similarity - pos_sim_mean.unsqueeze(1)) * neg_mask  # (B, B)
        neg_exp[neg_mask == 0] = -float('inf')
        
        # 加入 0 项: log(1 + Σ exp(...)) = logsumexp([0, ...])
        zero_col = torch.zeros(B, 1, device=features.device)
        neg_with_zero = torch.cat([zero_col, neg_exp], dim=1)  # (B, B+1)
        
        loss = torch.logsumexp(neg_with_zero, dim=1)  # (B,)
        loss = loss[has_pos].mean()
        
        # L2 正则化
        if self.l2_reg > 0:
            l2_loss = self.l2_reg * (features.pow(2).sum(dim=1).mean())
            loss = loss + l2_loss
        
        return loss


# =============================================================================
# 6. Angular Margin Loss 系列
# =============================================================================

class ArcFaceLoss(nn.Module):
    """
    ArcFace 损失 (Deng et al., CVPR 2019)
    
    同时包含 SphereFace (m1) 和 CosFace (m3) 变体:
    
    ArcFace:   cos(m1 * θ + m2) - m3   (m1=1, m2=arcface_m, m3=0)
    CosFace:   cos(θ) - m3              (m1=1, m2=0, m3=cosface_m)
    SphereFace: cos(m1 * θ)             (m1=sphereface_m, m2=0, m3=0)
    Combined:  cos(m1 * θ + m2) - m3    (全部启用)
    
    L = -log( exp(s * (cos(m1·θ_{y} + m2) - m3)) / 
              (exp(s * (cos(m1·θ_{y} + m2) - m3)) + Σ_{j≠y} exp(s * cos(θ_j))) )
    
    此类同时充当 **分类头** (替代最后的 nn.Linear)，
    内部维护一个归一化的权重矩阵。
    
    Args:
        num_classes: 类别数量
        embedding_dim: 嵌入维度
        scale: 缩放因子 s (默认 30.0)
        margin: ArcFace 角度间隔 m2 (默认 0.5, 对应 ~28.6°)
        easy_margin: 使用简化的 margin 计算 (默认 False)
        margin_type: 'arcface' (默认), 'cosface', 'sphereface', 'combined'
        m1: SphereFace 的角度倍数 (默认 1.0)
        m3: CosFace 的余弦间隔 (默认 0.0)
    """
    
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        scale: float = 30.0,
        margin: float = 0.5,
        easy_margin: bool = False,
        margin_type: str = 'arcface',
        m1: float = 1.0,
        m3: float = 0.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.margin = margin  # m2 in combined formulation
        self.easy_margin = easy_margin
        self.margin_type = margin_type
        self.m1 = m1
        self.m3 = m3
        
        # 可学习的类别权重（相当于分类头的权重）
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # 预计算 margin 相关常量
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # cos(π - m) = -cos(m)
        self.th = math.cos(math.pi - margin)
        # sin(π - m) * m
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) 嵌入特征（不需要预先归一化）
            labels: (B,) 类别标签
            
        Returns:
            标量损失 (CrossEntropy with angular margin)
        """
        # 归一化特征和权重
        features = l2_normalize(features)
        weight = l2_normalize(self.weight)
        
        # 计算余弦相似度: cos(θ) = normalized_features @ normalized_weight^T
        cosine = torch.mm(features, weight.t())  # (B, C)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)  # 数值稳定
        
        if self.margin_type == 'cosface':
            # CosFace: cos(θ) - m3
            phi = cosine - self.m3
        elif self.margin_type == 'sphereface':
            # SphereFace: cos(m1 * θ)
            theta = torch.acos(cosine)
            phi = torch.cos(self.m1 * theta)
        elif self.margin_type == 'combined':
            # Combined: cos(m1 * θ + m2) - m3
            theta = torch.acos(cosine)
            phi = torch.cos(self.m1 * theta + self.margin) - self.m3
        else:
            # ArcFace (默认): cos(θ + m)
            # 使用三角恒等式避免 arccos: cos(θ+m) = cos(θ)cos(m) - sin(θ)sin(m)
            sine = torch.sqrt(1.0 - cosine.pow(2))
            phi = cosine * self.cos_m - sine * self.sin_m
            
            if self.easy_margin:
                # easy_margin: 当 cos(θ) > 0 时使用 margin, 否则不使用
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                # 标准 ArcFace: 当 cos(θ) > cos(π-m) 时使用 margin
                # 否则使用线性回退 cos(θ) - mm 以保证单调性
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # 构造 one-hot 标签
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        
        # 对正确类别使用 phi, 其他类别使用 cosine
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits = logits * self.scale
        
        loss = F.cross_entropy(logits, labels)
        return loss


# =============================================================================
# 7. Circle Loss (圆损失)
# =============================================================================

class CircleLoss(nn.Module):
    """
    Circle Loss (Sun et al., CVPR 2020)
    
    统一对比损失和分类损失的视角，使用自适应权重。
    
    L = log(1 + Σ_{(i,j)∈N} exp(γ·α_n·(s_n - Δ_n)) · Σ_{(i,j)∈P} exp(-γ·α_p·(s_p - Δ_p)))
    
    其中:
    - s_p, s_n 为正/负样本对的余弦相似度
    - Δ_p = 1 - margin, Δ_n = margin (最优相似度)
    - α_p = relu(O_p - s_p), α_n = relu(s_n - O_n) (自适应权重)
    - O_p = 1 + margin, O_n = -margin
    
    Args:
        margin: 间隔 m (默认 0.25)
        scale: 缩放因子 γ (默认 256)
        normalize: 是否对特征做 L2 归一化 (默认 True)
    """
    
    def __init__(self, margin: float = 0.25, scale: float = 256.0, normalize: bool = True):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.normalize = normalize
        
        # 最优相似度点
        self.O_p = 1 + margin
        self.O_n = -margin
        # 收敛目标
        self.delta_p = 1 - margin
        self.delta_n = margin
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) 嵌入特征
            labels: (B,) 类别标签
            
        Returns:
            标量损失
        """
        if self.normalize:
            features = l2_normalize(features)
        
        # 余弦相似度矩阵
        sim_mat = pairwise_cosine_similarity(features)  # (B, B)
        pos_mask, neg_mask = get_pairwise_labels(labels)
        
        # 自适应权重
        # α_p = max(O_p - s_p, 0), 正样本对距离越远权重越大
        # α_n = max(s_n - O_n, 0), 负样本对距离越近权重越大
        alpha_p = F.relu(self.O_p - sim_mat.detach())
        alpha_n = F.relu(sim_mat.detach() - self.O_n)
        
        # 带权重的 logit
        # 正样本: -γ * α_p * (s_p - Δ_p)
        # 负样本: γ * α_n * (s_n - Δ_n)
        logit_p = -self.scale * alpha_p * (sim_mat - self.delta_p)  # 越大越好 → 负值
        logit_n = self.scale * alpha_n * (sim_mat - self.delta_n)   # 越小越好 → 正值
        
        # 对每个 anchor，聚合正负样本的 logit
        # 将非正/非负样本位置设为 -inf
        logit_p[pos_mask == 0] = -float('inf')
        logit_n[neg_mask == 0] = -float('inf')
        
        # logsumexp 聚合
        lse_p = torch.logsumexp(logit_p, dim=1)  # (B,)
        lse_n = torch.logsumexp(logit_n, dim=1)  # (B,)
        
        # 最终损失: log(1 + exp(lse_n + lse_p)) = softplus(lse_n + lse_p)
        loss = F.softplus(lse_n + lse_p)
        
        # 只对有正样本的 anchor 计算损失
        has_pos = pos_mask.sum(dim=1) > 0
        if has_pos.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        return loss[has_pos].mean()


# =============================================================================
# 8. Circle Loss - 分类版本 (Class-level)
# =============================================================================

class CircleLossClassLevel(nn.Module):
    """
    Circle Loss 的分类版本 (论文 Section 3.3)
    
    使用可学习的类别中心替代 batch 内样本对，类似 ProxyNCA。
    
    L = log(1 + exp(s_n) · exp(-s_p))
    
    其中 s_p = γ·α_p·(cos(θ_y) - Δ_p), s_n = γ·α_n·(cos(θ_j) - Δ_n)
    
    Args:
        num_classes: 类别数量
        embedding_dim: 嵌入维度
        margin: 间隔 (默认 0.25)
        scale: 缩放因子 (默认 256)
    """
    
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        margin: float = 0.25,
        scale: float = 256.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.O_p = 1 + margin
        self.O_n = -margin
        self.delta_p = 1 - margin
        self.delta_n = margin
        
        # 可学习的类别权重
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) 嵌入特征
            labels: (B,) 类别标签
            
        Returns:
            标量损失
        """
        features = l2_normalize(features)
        weight = l2_normalize(self.weight)
        
        # 余弦相似度: (B, C)
        sim = torch.mm(features, weight.t())
        
        # one-hot
        one_hot = torch.zeros_like(sim)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        
        # 自适应权重
        alpha_p = F.relu(self.O_p - sim.detach())
        alpha_n = F.relu(sim.detach() - self.O_n)
        
        # 正类 logit: -γ * α_p * (s - Δ_p)
        logit_p = -self.scale * alpha_p * (sim - self.delta_p)
        # 负类 logit: γ * α_n * (s - Δ_n)
        logit_n = self.scale * alpha_n * (sim - self.delta_n)
        
        # 正类位置用 logit_p, 负类位置用 logit_n
        logit_p[one_hot == 0] = -float('inf')
        logit_n[one_hot == 1] = -float('inf')
        
        # logsumexp 聚合
        lse_p = torch.logsumexp(logit_p, dim=1)
        lse_n = torch.logsumexp(logit_n, dim=1)
        
        loss = F.softplus(lse_n + lse_p).mean()
        return loss


# =============================================================================
# 工厂函数
# =============================================================================

def create_metric_loss(
    loss_type: str,
    num_classes: int = 0,
    embedding_dim: int = 0,
    **kwargs
) -> nn.Module:
    """
    创建度量学习损失函数的工厂函数
    
    Args:
        loss_type: 损失类型名称
            - 'contrastive': ContrastiveLoss
            - 'triplet': TripletMarginLoss
            - 'lifted': LiftedStructureLoss
            - 'proxy_nca': ProxyNCALoss (需要 num_classes, embedding_dim)
            - 'npair': NPairLoss
            - 'arcface': ArcFaceLoss (需要 num_classes, embedding_dim)
            - 'cosface': CosFace 变体
            - 'sphereface': SphereFace 变体
            - 'circle': CircleLoss (pair-level)
            - 'circle_cls': CircleLossClassLevel (class-level, 需要 num_classes, embedding_dim)
        num_classes: 类别数量 (proxy_nca, arcface, circle_cls 需要)
        embedding_dim: 嵌入维度 (proxy_nca, arcface, circle_cls 需要)
        **kwargs: 传递给具体损失函数的额外参数
    
    Returns:
        nn.Module 损失函数实例
    """
    loss_type = loss_type.lower()
    
    if loss_type == 'contrastive':
        return ContrastiveLoss(
            margin=kwargs.get('margin', 1.0),
            normalize=kwargs.get('normalize', True)
        )
    
    elif loss_type == 'triplet':
        return TripletMarginLoss(
            margin=kwargs.get('margin', 0.3),
            normalize=kwargs.get('normalize', True),
            soft=kwargs.get('soft', False)
        )
    
    elif loss_type == 'lifted':
        return LiftedStructureLoss(
            margin=kwargs.get('margin', 1.0),
            normalize=kwargs.get('normalize', True)
        )
    
    elif loss_type == 'proxy_nca':
        assert num_classes > 0 and embedding_dim > 0, \
            "ProxyNCALoss 需要 num_classes 和 embedding_dim"
        return ProxyNCALoss(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            scale=kwargs.get('scale', 8.0)
        )
    
    elif loss_type == 'npair':
        return NPairLoss(
            normalize=kwargs.get('normalize', True),
            l2_reg=kwargs.get('l2_reg', 0.02)
        )
    
    elif loss_type == 'arcface':
        assert num_classes > 0 and embedding_dim > 0, \
            "ArcFaceLoss 需要 num_classes 和 embedding_dim"
        return ArcFaceLoss(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            scale=kwargs.get('scale', 30.0),
            margin=kwargs.get('margin', 0.5),
            easy_margin=kwargs.get('easy_margin', False),
            margin_type='arcface'
        )
    
    elif loss_type == 'cosface':
        assert num_classes > 0 and embedding_dim > 0, \
            "CosFaceLoss 需要 num_classes 和 embedding_dim"
        # CosFace 仅使用 m3 (cosine margin)，用户的 --metric_loss_margin 应映射到 m3
        m3_val = kwargs.get('margin', 0.35)
        return ArcFaceLoss(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            scale=kwargs.get('scale', 30.0),
            margin=0.0,  # m2 (ArcFace angular margin) 不用于 CosFace
            margin_type='cosface',
            m3=m3_val
        )
    
    elif loss_type == 'sphereface':
        assert num_classes > 0 and embedding_dim > 0, \
            "SphereFaceLoss 需要 num_classes 和 embedding_dim"
        # SphereFace 仅使用 m1 (角度乘数)，用户的 --metric_loss_margin 应映射到 m1
        m1_val = kwargs.get('margin', 4.0)
        return ArcFaceLoss(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            scale=kwargs.get('scale', 30.0),
            margin=0.0,  # m2 (ArcFace angular margin) 不用于 SphereFace
            margin_type='sphereface',
            m1=m1_val
        )
    
    elif loss_type == 'circle':
        return CircleLoss(
            margin=kwargs.get('margin', 0.25),
            scale=kwargs.get('scale', 256.0),
            normalize=kwargs.get('normalize', True)
        )
    
    elif loss_type == 'circle_cls':
        assert num_classes > 0 and embedding_dim > 0, \
            "CircleLossClassLevel 需要 num_classes 和 embedding_dim"
        return CircleLossClassLevel(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            margin=kwargs.get('margin', 0.25),
            scale=kwargs.get('scale', 256.0)
        )
    
    else:
        raise ValueError(
            f"不支持的度量学习损失: {loss_type}. "
            f"可选: contrastive, triplet, lifted, proxy_nca, npair, "
            f"arcface, cosface, sphereface, circle, circle_cls"
        )
