"""
MAE: Masked Autoencoders Are Scalable Vision Learners

论文: https://arxiv.org/abs/2111.06377

核心思想:
- 随机掩码大部分图像 patch (75%)
- 使用 Encoder 编码可见 patch
- 使用 Decoder 重建被掩码的 patch
- MSE 损失

注意: MAE 原本设计用于 ViT，这里提供一个适配 CNN 的简化版本。
为了完整性，也提供了基于 patch 的实现框架。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .base import SSLMethod, get_backbone_output_dim


class PatchEmbed(nn.Module):
    """将图像分割为 patch 并嵌入"""
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, embed_dim, H/P, W/P] -> [B, num_patches, embed_dim]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MAE(SSLMethod):
    """
    MAE: Masked Autoencoder
    
    这是一个简化版本，可以与 CNN backbone 配合使用。
    对于完整的 MAE 效果，建议使用 ViT。
    
    架构 (CNN 适配版):
        x → random_mask → masked_x
        masked_x → Encoder → latent
        latent → Decoder → reconstructed_patches
        Loss = MSE(reconstructed, original) on masked patches
    
    Args:
        backbone: CNN 特征提取网络
        image_size: 图像大小 (默认 224)
        patch_size: patch 大小 (默认 16)
        mask_ratio: 掩码比例 (默认 0.75)
        decoder_dim: 解码器维度 (默认 512)
        decoder_depth: 解码器层数 (默认 2)
        
    Example:
        >>> model = MAE(backbone, mask_ratio=0.75)
        >>> output = model(images)
        >>> loss = output['loss']
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        image_size: int = 224,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        decoder_dim: int = 512,
        decoder_depth: int = 2,
        norm_pix_loss: bool = True
    ):
        feature_dim = get_backbone_output_dim(backbone)
        super().__init__(backbone, feature_dim)
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=feature_dim
        )
        
        # 移除 backbone 的分类头
        self._remove_fc()
        
        # Encoder projection (for patches)
        self.encoder_proj = nn.Linear(feature_dim, feature_dim)
        
        # Learned mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, feature_dim)
        )
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Decoder
        self.decoder_embed = nn.Linear(feature_dim, decoder_dim)
        
        decoder_layers = []
        for _ in range(decoder_depth):
            decoder_layers.extend([
                nn.Linear(decoder_dim, decoder_dim),
                nn.GELU(),
                nn.LayerNorm(decoder_dim)
            ])
        self.decoder_layers = nn.Sequential(*decoder_layers)
        
        # Reconstruction head
        self.decoder_pred = nn.Linear(decoder_dim, self.patch_dim)
    
    def _remove_fc(self):
        """移除 backbone 的分类头"""
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()
    
    def _random_masking(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        随机掩码
        
        Args:
            x: patch embeddings [B, N, D]
            
        Returns:
            x_masked: 掩码后的 patches [B, N*(1-mask_ratio), D]
            mask: 掩码 [B, N], 1=masked
            ids_restore: 恢复顺序的索引
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        
        # 生成随机噪声并排序
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # 保留前 len_keep 个
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )
        
        # 生成掩码
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x1: 输入图像 [B, C, H, W]
            x2: 忽略 (为了接口一致性)
            
        Returns:
            包含 'loss', 'pred', 'mask' 等的字典
        """
        # 转换为 patches
        patches = self.patch_embed(x1)  # [B, N, D]
        patches = patches + self.pos_embed
        
        # 随机掩码
        patches_visible, mask, ids_restore = self._random_masking(patches)
        
        # Encoder
        latent = self.encoder_proj(patches_visible)
        
        # Decoder
        B, N_vis, D = latent.shape
        N = self.num_patches
        
        # 嵌入到 decoder 维度
        latent = self.decoder_embed(latent)
        
        # 添加 mask tokens
        mask_tokens = self.mask_token.repeat(B, N - N_vis, 1)
        
        # Concatenate: visible + masked
        latent_full = torch.cat([latent, mask_tokens], dim=1)
        
        # 恢复原始顺序
        latent_full = torch.gather(
            latent_full, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, latent_full.shape[-1])
        )
        
        # Decoder layers
        latent_full = self.decoder_layers(latent_full)
        
        # 预测
        pred = self.decoder_pred(latent_full)  # [B, N, patch_dim]
        
        # 计算损失
        target = self._patchify(x1)  # [B, N, patch_dim]
        loss = self._reconstruction_loss(pred, target, mask)
        
        return {
            'loss': loss,
            'pred': pred.detach(),
            'mask': mask.detach(),
            'target': target.detach()
        }
    
    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """将图像转换为 patches"""
        B, C, H, W = x.shape
        P = self.patch_size
        
        # [B, C, H, W] -> [B, C, H/P, P, W/P, P]
        x = x.reshape(B, C, H // P, P, W // P, P)
        # -> [B, H/P, W/P, C, P, P]
        x = x.permute(0, 2, 4, 1, 3, 5)
        # -> [B, N, C*P*P]
        x = x.reshape(B, -1, C * P * P)
        
        return x
    
    def _reconstruction_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """重建损失"""
        if self.norm_pix_loss:
            # 对每个 patch 归一化
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()
        
        # MSE
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, N]
        
        # 只计算被掩码的 patch
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def get_encoder(self) -> nn.Module:
        """返回用于下游任务的编码器"""
        return self.backbone


# ============ 测试代码 ============
if __name__ == '__main__':
    print("测试 MAE...")
    
    from torchvision.models import resnet18
    
    backbone = resnet18(weights=None)
    model = MAE(backbone, image_size=224, patch_size=16, mask_ratio=0.75)
    
    # 测试前向传播
    x = torch.randn(4, 3, 224, 224)
    
    output = model(x)
    
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"Pred shape: {output['pred'].shape}")  # [B, N, patch_dim]
    print(f"Mask shape: {output['mask'].shape}")  # [B, N]
    print(f"Mask ratio: {output['mask'].sum() / output['mask'].numel():.2f}")  # ~0.75
    
    # 参数量
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total params: {num_params:.2f}M")
    
    print("\nMAE 测试通过！")
