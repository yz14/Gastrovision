"""
Jaguar Re-ID 数据增强模块

针对动物 Re-ID 优化的增强策略：
- 训练：强几何 + 颜色增强 + Random Erasing（模拟遮挡）
- 验证/测试：仅 Resize + CenterCrop + Normalize
"""

import numpy as np
from typing import Tuple, Optional, Callable

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    A = None
    ToTensorV2 = None

# ImageNet 归一化参数
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(
    image_size: int = 224,
    augment_level: str = 'medium',
    mean: Tuple[float, ...] = IMAGENET_MEAN,
    std: Tuple[float, ...] = IMAGENET_STD,
) -> Callable:
    """
    获取训练阶段数据增强

    Args:
        image_size: 输出图像尺寸
        augment_level: 增强强度 ('light', 'medium', 'heavy')
        mean: 归一化均值
        std: 归一化标准差

    Returns:
        Albumentations Compose 变换
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("请安装 albumentations: pip install albumentations")

    size = (image_size, image_size)

    if augment_level == 'light':
        transform = A.Compose([
            A.RandomResizedCrop(size=size, scale=(0.85, 1.0), ratio=(0.8, 1.2)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    elif augment_level == 'heavy':
        transform = A.Compose([
            A.RandomResizedCrop(size=size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent=(-0.1, 0.1),
                scale=(0.8, 1.2),
                rotate=(-25, 25),
                p=0.6
            ),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                A.ElasticTransform(alpha=80, sigma=80 * 0.05, p=1.0),
            ], p=0.2),
            # 颜色增强 — 对斑纹识别重要
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=30, p=0.5
            ),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
            ], p=0.3),
            # Random Erasing — 模拟遮挡，对 Re-ID 至关重要
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(int(image_size * 0.1), int(image_size * 0.3)),
                hole_width_range=(int(image_size * 0.1), int(image_size * 0.3)),
                fill="random",
                p=0.5
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        # medium（推荐默认）
        transform = A.Compose([
            A.RandomResizedCrop(size=size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent=(-0.08, 0.08),
                scale=(0.85, 1.15),
                rotate=(-20, 20),
                p=0.5
            ),
            # 颜色增强
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=25, p=0.4
            ),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.2),
            # Random Erasing
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(int(image_size * 0.08), int(image_size * 0.25)),
                hole_width_range=(int(image_size * 0.08), int(image_size * 0.25)),
                fill="random",
                p=0.4
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    return transform


def get_val_transforms(
    image_size: int = 224,
    mean: Tuple[float, ...] = IMAGENET_MEAN,
    std: Tuple[float, ...] = IMAGENET_STD,
) -> Callable:
    """
    获取验证/测试阶段变换（只做 Resize + CenterCrop + Normalize）
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("请安装 albumentations: pip install albumentations")

    resize_size = int(image_size * 1.14)
    return A.Compose([
        A.Resize(height=resize_size, width=resize_size),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


class AlbumentationsWrapper:
    """将 Albumentations 变换包装为可调用对象，接受 numpy array 或 PIL Image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        if hasattr(image, 'convert'):
            image = np.array(image.convert('RGB'))
        return self.transform(image=image)['image']
