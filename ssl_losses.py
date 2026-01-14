"""
SSL 损失函数模块

包含各种自监督学习方法使用的损失函数。

时间线:
- ContrastiveLoss: Siamese Network (1993)
- TripletLoss: FaceNet (2015)
- InfoNCELoss: CPC (2018), InstDisc, MoCo, SimCLR
- NegativeCosineSimilarity: SimSiam, BYOL (2020-2021)
- BarlowTwinsLoss: Barlow Twins (2021)
- MAELoss: MAE (2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ContrastiveLoss(nn.Module):
    """
    对比损失 (Contrastive Loss) - Siamese Network
    
    L = y * d² + (1-y) * max(0, margin - d)²
    
    其中:
    - y=1 表示正样本对 (同类)
    - y=0 表示负样本对 (异类)
    - d 是两个特征之间的欧氏距离
    
    References:
        - "Dimensionality Reduction by Learning an Invariant Mapping" (2005)
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin: 负样本对的最小距离阈值
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z1: 第一个样本的特征 [B, D]
            z2: 第二个样本的特征 [B, D]
            labels: 标签，1=正样本对，0=负样本对 [B]
            
        Returns:
            对比损失
        """
        # 计算欧氏距离
        distance = F.pairwise_distance(z1, z2, p=2)
        
        # 正样本对损失：距离越小越好
        pos_loss = labels * distance.pow(2)
        
        # 负样本对损失：距离小于 margin 时有惩罚
        neg_loss = (1 - labels) * F.relu(self.margin - distance).pow(2)
        
        loss = (pos_loss + neg_loss).mean()
        return loss


class TripletLoss(nn.Module):
    """
    三元组损失 (Triplet Loss) - FaceNet
    
    L = max(0, d(a,p) - d(a,n) + margin)
    
    目标：使 anchor 与 positive 的距离 + margin < anchor 与 negative 的距离
    
    References:
        - FaceNet: "A Unified Embedding for Face Recognition and Clustering" (2015)
    """
    
    def __init__(self, margin: float = 0.2, distance: str = 'euclidean'):
        """
        Args:
            margin: 正负样本对之间的最小距离差
            distance: 距离度量 ('euclidean' 或 'cosine')
        """
        super().__init__()
        self.margin = margin
        self.distance = distance
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            anchor: 锚点特征 [B, D]
            positive: 正样本特征 [B, D]
            negative: 负样本特征 [B, D]
            
        Returns:
            三元组损失
        """
        if self.distance == 'euclidean':
            pos_dist = F.pairwise_distance(anchor, positive, p=2)
            neg_dist = F.pairwise_distance(anchor, negative, p=2)
        else:  # cosine
            # 余弦距离 = 1 - 余弦相似度
            pos_dist = 1 - F.cosine_similarity(anchor, positive)
            neg_dist = 1 - F.cosine_similarity(anchor, negative)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class InfoNCELoss(nn.Module):
    """
    InfoNCE 损失 - CPC, MoCo, SimCLR, InstDisc
    
    L = -log( exp(sim(q, k+) / τ) / Σ exp(sim(q, k) / τ) )
    
    本质是一个 softmax 分类问题，目标是从所有样本中识别正样本。
    
    References:
        - CPC: "Representation Learning with Contrastive Predictive Coding" (2018)
        - MoCo: "Momentum Contrast for Unsupervised Visual Representation Learning" (2020)
        - SimCLR: "A Simple Framework for Contrastive Learning" (2020)
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: 温度参数，控制分布的平滑程度
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        query: torch.Tensor,
        key_pos: torch.Tensor,
        key_neg: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: 查询特征 [B, D]
            key_pos: 正样本键特征 [B, D]
            key_neg: 负样本键特征 [K, D] (可选，如果为 None 则使用批内负样本)
            
        Returns:
            InfoNCE 损失
        """
        # L2 归一化
        query = F.normalize(query, dim=1)
        key_pos = F.normalize(key_pos, dim=1)
        
        # 正样本相似度 [B, 1]
        pos_sim = torch.einsum('nc,nc->n', query, key_pos).unsqueeze(-1)
        
        if key_neg is not None:
            # 使用外部负样本 (如 MoCo 的队列)
            key_neg = F.normalize(key_neg, dim=1)
            neg_sim = torch.einsum('nc,kc->nk', query, key_neg)  # [B, K]
            logits = torch.cat([pos_sim, neg_sim], dim=1)  # [B, 1+K]
        else:
            # 使用批内负样本 (SimCLR 方式)
            # 所有样本与所有样本的相似度矩阵
            all_keys = key_pos  # [B, D]
            sim_matrix = torch.einsum('nc,mc->nm', query, all_keys)  # [B, B]
            
            # 对角线是正样本，非对角线是负样本
            batch_size = query.shape[0]
            labels = torch.arange(batch_size, device=query.device)
            
            logits = sim_matrix
        
        # 温度缩放
        logits = logits / self.temperature
        
        # 交叉熵损失，标签是 0 (正样本在第一个位置)
        if key_neg is not None:
            labels = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) 损失 - SimCLR
    
    SimCLR 专用的对比损失，使用批内正负样本对。
    对于每个样本，其增强版本是正样本，其他样本是负样本。
    
    References:
        - SimCLR: "A Simple Framework for Contrastive Learning" (2020)
    """
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: 第一个视图的特征 [B, D]
            z2: 第二个视图的特征 [B, D]
            
        Returns:
            NT-Xent 损失
        """
        batch_size = z1.shape[0]
        device = z1.device
        
        # L2 归一化
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # 拼接两个视图
        z = torch.cat([z1, z2], dim=0)  # [2B, D]
        
        # 计算相似度矩阵
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # [2B, 2B]
        
        # 创建掩码，排除自己与自己的相似度
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        
        # 创建正样本标签
        # 对于 z1[i]，正样本是 z2[i]，即位置 i+B
        # 对于 z2[i]，正样本是 z1[i]，即位置 i
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(batch_size, device=device)
        ])
        
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


class NegativeCosineSimilarity(nn.Module):
    """
    负余弦相似度损失 - SimSiam, BYOL
    
    L = -cos(p, z) = -(p·z) / (||p|| ||z||)
    
    注意：z 需要 stop gradient (detach)
    
    References:
        - SimSiam: "Exploring Simple Siamese Representation Learning" (2021)
        - BYOL: "Bootstrap Your Own Latent" (2020)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        p: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            p: predictor 输出 [B, D]
            z: encoder 输出 (已 detach) [B, D]
            
        Returns:
            负余弦相似度损失
        """
        # L2 归一化
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        
        # 负余弦相似度
        loss = -(p * z).sum(dim=1).mean()
        return loss


class BarlowTwinsLoss(nn.Module):
    """
    Barlow Twins 损失
    
    目标：使两个视图特征的互相关矩阵趋近于单位阵
    
    L = Σ_i (1 - C_ii)² + λ Σ_i Σ_{j≠i} C_ij²
    
    其中 C 是归一化后特征的互相关矩阵
    
    References:
        - "Barlow Twins: Self-Supervised Learning via Redundancy Reduction" (2021)
    """
    
    def __init__(self, lambda_coeff: float = 0.005):
        """
        Args:
            lambda_coeff: 非对角项的权重系数
        """
        super().__init__()
        self.lambda_coeff = lambda_coeff
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: 第一个视图的特征 [B, D]
            z2: 第二个视图的特征 [B, D]
            
        Returns:
            Barlow Twins 损失
        """
        batch_size = z1.shape[0]
        feature_dim = z1.shape[1]
        
        # 沿批次维度标准化
        z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-5)
        z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-5)
        
        # 计算互相关矩阵 [D, D]
        c = torch.mm(z1_norm.T, z2_norm) / batch_size
        
        # 对角项损失：希望对角线元素接近 1
        on_diag = (torch.diagonal(c) - 1).pow(2).sum()
        
        # 非对角项损失：希望非对角线元素接近 0
        off_diag = (c.flatten()[:-1].view(feature_dim - 1, feature_dim + 1)[:, 1:]).pow(2).sum()
        
        loss = on_diag + self.lambda_coeff * off_diag
        return loss


class MAELoss(nn.Module):
    """
    MAE 重建损失 - Masked Autoencoder
    
    仅在被掩码的 patch 上计算 MSE 损失
    
    References:
        - "Masked Autoencoders Are Scalable Vision Learners" (2022)
    """
    
    def __init__(self, norm_pix_loss: bool = True):
        """
        Args:
            norm_pix_loss: 是否使用像素归一化的损失
        """
        super().__init__()
        self.norm_pix_loss = norm_pix_loss
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测的像素值 [B, L, patch_size²*C]
            target: 目标像素值 [B, L, patch_size²*C]
            mask: 掩码，1 表示被掩码的位置 [B, L]
            
        Returns:
            重建损失
        """
        if self.norm_pix_loss:
            # 对每个 patch 进行归一化
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()
        
        # MSE 损失
        loss = (pred - target).pow(2).mean(dim=-1)  # [B, L]
        
        # 仅计算被掩码位置的损失
        loss = (loss * mask).sum() / mask.sum()
        
        return loss


# ============ 测试代码 ============
if __name__ == '__main__':
    print("测试 SSL 损失函数...")
    
    batch_size = 32
    feature_dim = 128
    
    z1 = torch.randn(batch_size, feature_dim)
    z2 = torch.randn(batch_size, feature_dim)
    
    # 测试 ContrastiveLoss
    labels = torch.randint(0, 2, (batch_size,)).float()
    contrastive = ContrastiveLoss(margin=1.0)
    loss = contrastive(z1, z2, labels)
    print(f"ContrastiveLoss: {loss.item():.4f}")
    
    # 测试 TripletLoss
    anchor = torch.randn(batch_size, feature_dim)
    positive = torch.randn(batch_size, feature_dim)
    negative = torch.randn(batch_size, feature_dim)
    triplet = TripletLoss(margin=0.2)
    loss = triplet(anchor, positive, negative)
    print(f"TripletLoss: {loss.item():.4f}")
    
    # 测试 NTXentLoss
    ntxent = NTXentLoss(temperature=0.5)
    loss = ntxent(z1, z2)
    print(f"NTXentLoss: {loss.item():.4f}")
    
    # 测试 NegativeCosineSimilarity
    neg_cos = NegativeCosineSimilarity()
    loss = neg_cos(z1, z2.detach())
    print(f"NegativeCosineSimilarity: {loss.item():.4f}")
    
    # 测试 BarlowTwinsLoss
    barlow = BarlowTwinsLoss(lambda_coeff=0.005)
    loss = barlow(z1, z2)
    print(f"BarlowTwinsLoss: {loss.item():.4f}")
    
    print("\n所有损失函数测试通过！")
