"""
SSL 数据增强模块

自监督学习的核心在于数据增强，不同方法有不同的增强策略。

增强级别:
- SimCLR/SimSiam/BYOL: 强增强 (RandomResizedCrop + ColorJitter + GaussianBlur)
- MoCo: 中等增强
- Rotation: 仅旋转变换

参考:
- SimCLR: https://github.com/google-research/simclr
- MoCo: https://github.com/facebookresearch/moco
- SimSiam: https://github.com/facebookresearch/simsiam
"""

import torch
from torch import Tensor
from torchvision import transforms
from typing import Tuple, List, Optional
import random
from PIL import ImageFilter, ImageOps


class GaussianBlur:
    """
    高斯模糊变换
    
    SimCLR 和 BYOL 使用的变换之一
    """
    
    def __init__(self, sigma: Tuple[float, float] = (0.1, 2.0)):
        """
        Args:
            sigma: 高斯模糊的 sigma 范围
        """
        self.sigma = sigma
    
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarization:
    """
    Solarization 变换
    
    BYOL 使用的额外变换
    """
    
    def __init__(self, threshold: int = 128):
        self.threshold = threshold
    
    def __call__(self, x):
        return ImageOps.solarize(x, self.threshold)


class TwoCropsTransform:
    """
    双视图变换
    
    对每张图像应用两次随机变换，生成两个不同的视图。
    用于 SimCLR、SimSiam、BYOL、MoCo 等方法。
    """
    
    def __init__(self, base_transform: transforms.Compose):
        """
        Args:
            base_transform: 基础变换
        """
        self.base_transform = base_transform
    
    def __call__(self, x) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: PIL Image
            
        Returns:
            (view1, view2): 两个增强视图
        """
        return self.base_transform(x), self.base_transform(x)


class MultiCropsTransform:
    """
    多视图变换 (SwAV 使用)
    
    生成多个不同大小的视图
    """
    
    def __init__(
        self,
        global_transform: transforms.Compose,
        local_transform: transforms.Compose,
        n_global_crops: int = 2,
        n_local_crops: int = 6
    ):
        self.global_transform = global_transform
        self.local_transform = local_transform
        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops
    
    def __call__(self, x) -> List[Tensor]:
        crops = []
        for _ in range(self.n_global_crops):
            crops.append(self.global_transform(x))
        for _ in range(self.n_local_crops):
            crops.append(self.local_transform(x))
        return crops


def get_ssl_augmentation(
    method: str = 'simsiam',
    image_size: int = 224,
    normalize: Optional[transforms.Normalize] = None
) -> transforms.Compose:
    """
    获取 SSL 方法对应的数据增强
    
    Args:
        method: SSL 方法名称
        image_size: 图像大小
        normalize: 归一化变换 (默认使用 ImageNet 标准)
        
    Returns:
        数据增强 Compose
    """
    if normalize is None:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    if method in ['simclr', 'simsiam', 'byol', 'barlow_twins']:
        # 强增强
        return get_strong_augmentation(image_size, normalize)
    elif method in ['moco', 'instdisc']:
        # MoCo 增强
        return get_moco_augmentation(image_size, normalize)
    elif method == 'rotation':
        # 旋转任务使用基础增强
        return get_basic_augmentation(image_size, normalize)
    elif method in ['siamese', 'triplet']:
        # 度量学习使用中等增强
        return get_moderate_augmentation(image_size, normalize)
    else:
        # 默认使用强增强
        return get_strong_augmentation(image_size, normalize)


def get_strong_augmentation(
    image_size: int = 224,
    normalize: transforms.Normalize = None
) -> transforms.Compose:
    """
    SimCLR/SimSiam/BYOL 强增强
    
    包含:
    - RandomResizedCrop
    - RandomHorizontalFlip
    - ColorJitter (0.4, 0.4, 0.4, 0.1)
    - RandomGrayscale (p=0.2)
    - GaussianBlur (p=0.5)
    """
    if normalize is None:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            )
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur(sigma=(0.1, 2.0))], p=0.5),
        transforms.ToTensor(),
        normalize
    ])


def get_moco_augmentation(
    image_size: int = 224,
    normalize: transforms.Normalize = None
) -> transforms.Compose:
    """
    MoCo v2 增强 (与 SimCLR 类似但略有不同)
    """
    if normalize is None:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur()], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def get_moderate_augmentation(
    image_size: int = 224,
    normalize: transforms.Normalize = None
) -> transforms.Compose:
    """
    中等强度增强 (用于度量学习)
    """
    if normalize is None:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        ], p=0.5),
        transforms.ToTensor(),
        normalize
    ])


def get_basic_augmentation(
    image_size: int = 224,
    normalize: transforms.Normalize = None
) -> transforms.Compose:
    """
    基础增强 (用于 Rotation 等)
    """
    if normalize is None:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize
    ])


def get_eval_augmentation(
    image_size: int = 224,
    normalize: transforms.Normalize = None
) -> transforms.Compose:
    """
    评估/测试时的增强 (只有 resize 和 normalize)
    """
    if normalize is None:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize
    ])


class RotationTransform:
    """
    旋转变换 (用于 Rotation Prediction 预文本任务)
    
    随机将图像旋转 0°, 90°, 180°, 270° 中的一个角度
    """
    
    def __init__(
        self,
        base_transform: transforms.Compose = None,
        image_size: int = 224
    ):
        """
        Args:
            base_transform: 可选的基础变换
            image_size: 图像大小
        """
        if base_transform is None:
            self.base_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.base_transform = base_transform
        
        self.angles = [0, 90, 180, 270]
    
    def __call__(self, x) -> Tuple[Tensor, int]:
        """
        Args:
            x: PIL Image
            
        Returns:
            (rotated_image, rotation_label): 旋转后的图像和旋转标签 (0-3)
        """
        # 随机选择旋转角度
        label = random.randint(0, 3)
        angle = self.angles[label]
        
        # 旋转图像
        if angle != 0:
            x = x.rotate(angle)
        
        # 应用基础变换
        x = self.base_transform(x)
        
        return x, label


# ============ 测试代码 ============
if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    
    print("测试 SSL 数据增强...")
    
    # 创建测试图像
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    
    # 测试双视图变换
    base_aug = get_ssl_augmentation('simsiam', image_size=224)
    two_crop = TwoCropsTransform(base_aug)
    v1, v2 = two_crop(img)
    print(f"TwoCropsTransform: v1={v1.shape}, v2={v2.shape}")
    
    # 测试旋转变换
    rot_transform = RotationTransform(image_size=224)
    rot_img, label = rot_transform(img)
    print(f"RotationTransform: img={rot_img.shape}, label={label}")
    
    # 测试各种增强
    for method in ['simclr', 'moco', 'rotation', 'siamese']:
        aug = get_ssl_augmentation(method, image_size=224)
        out = aug(img)
        print(f"{method} augmentation: {out.shape}")
    
    print("\n所有增强测试通过！")
