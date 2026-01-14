"""
Gastrovision Albumentations 数据增强模块

提供基于 Albumentations 的高级数据增强策略，专门针对医学内镜图像优化。

优势:
- 更丰富的增强方法（CLAHE、GridDistortion、ElasticTransform 等）
- 更高性能（基于 numpy/opencv）
- 更灵活的组合

使用方法:
    from transforms_album import get_albumentations_transforms
    
    train_transform = get_albumentations_transforms('train')
    valid_transform = get_albumentations_transforms('valid')

依赖:
    pip install albumentations

注意：
    Albumentations 2.0+ API: 使用 size=(height, width) 元组格式
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


def get_albumentations_transforms(
    mode: str = 'train',
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    augment_level: str = 'medium'
) -> Optional[Callable]:
    """
    获取 Albumentations 数据增强变换
    
    Args:
        mode: 'train', 'valid' 或 'test'
        image_size: 输出图像尺寸
        mean: 归一化均值（ImageNet 默认值）
        std: 归一化标准差
        augment_level: 增强强度 ('light', 'medium', 'heavy')
        
    Returns:
        Albumentations Compose 对象，或 None（如果未安装）
    """
    if not HAS_ALBUMENTATIONS:
        print("⚠ Albumentations 未安装，使用默认 torchvision 增强")
        print("  安装: pip install albumentations")
        return None
    
    size = (image_size, image_size)  # (height, width)
    
    if mode == 'train':
        if augment_level == 'light':
            # 轻度增强
            transform = A.Compose([
                A.RandomResizedCrop(size=size, scale=(0.85, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        elif augment_level == 'heavy':
            # 重度增强（适合小数据集或过拟合情况）
            transform = A.Compose([
                A.RandomResizedCrop(size=size, scale=(0.7, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.15, rotate_limit=20, 
                    border_mode=0, p=0.6
                ),
                # 医学影像常用增强
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
                A.OneOf([
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1.0),
                ], p=0.3),
                # 颜色增强
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=30, p=0.4
                ),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                ], p=0.2),
                # Cutout 效果
                A.CoarseDropout(
                    max_holes=12, max_height=20, max_width=20,
                    min_holes=4, min_height=8, min_width=8, p=0.4
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        else:
            # 中等增强（推荐默认值）
            transform = A.Compose([
                A.RandomResizedCrop(size=size, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=15, 
                    border_mode=0, p=0.5
                ),
                # 医学影像常用：CLAHE 对比度增强
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                # 颜色增强
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=20, p=0.3
                ),
                # Cutout 效果
                A.CoarseDropout(
                    max_holes=8, max_height=16, max_width=16,
                    min_holes=1, min_height=4, min_width=4, p=0.3
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
    else:
        # 验证/测试：只做基本预处理
        resize_size = int(image_size * 1.14)
        transform = A.Compose([
            A.Resize(height=resize_size, width=resize_size),
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    
    return transform


class AlbumentationsTransformWrapper:
    """
    Albumentations 变换包装器
    
    用于将 Albumentations 变换适配到 PyTorch Dataset 的 __getitem__ 中
    （因为 Albumentations 接受 numpy array，而 PIL Image 需要转换）
    """
    
    def __init__(self, transform: A.Compose):
        self.transform = transform
    
    def __call__(self, image):
        """
        应用变换
        
        Args:
            image: PIL Image 或 numpy array
            
        Returns:
            变换后的 tensor
        """
        # 如果是 PIL Image，转换为 numpy array
        if hasattr(image, 'convert'):
            image = np.array(image.convert('RGB'))
        
        # 应用 Albumentations 变换
        transformed = self.transform(image=image)
        return transformed['image']


def get_wrapped_transforms(
    mode: str = 'train',
    image_size: int = 224,
    augment_level: str = 'medium'
) -> Optional[Callable]:
    """
    获取包装后的 Albumentations 变换
    
    可直接用于 GastrovisionDataset 的 transform 参数
    
    Args:
        mode: 'train', 'valid' 或 'test'
        image_size: 输出图像尺寸
        augment_level: 增强强度
        
    Returns:
        包装后的变换函数
    """
    transform = get_albumentations_transforms(mode, image_size, augment_level=augment_level)
    if transform is None:
        return None
    return AlbumentationsTransformWrapper(transform)


# 测试代码
if __name__ == "__main__":
    print("Albumentations 数据增强模块")
    print("=" * 50)
    
    if not HAS_ALBUMENTATIONS:
        print("\n❌ Albumentations 未安装")
        print("安装命令: pip install albumentations")
        exit(1)
    
    print(f"\n✓ Albumentations 版本: {A.__version__}")
    
    # 测试不同增强级别
    for level in ['light', 'medium', 'heavy']:
        transform = get_albumentations_transforms('train', augment_level=level)
        print(f"\n{level.upper()} 级别增强:")
        print(f"  变换数量: {len(transform.transforms)}")
    
    # 测试包装器
    print("\n测试包装器:")
    wrapper = get_wrapped_transforms('train', image_size=224)
    
    # 创建模拟图像
    from PIL import Image
    dummy_image = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
    
    result = wrapper(dummy_image)
    print(f"  输入尺寸: (300, 300, 3)")
    print(f"  输出尺寸: {tuple(result.shape)}")
    print(f"  输出类型: {type(result)}")
    
    print("\n✓ 模块测试完成!")
