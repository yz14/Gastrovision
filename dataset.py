"""
Gastrovision 数据加载模块

提供：
- GastrovisionDataset: PyTorch Dataset 类
- get_transforms: 数据增强变换
- create_dataloaders: DataLoader 工厂函数
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional, Callable, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


class GastrovisionDataset(Dataset):
    """
    Gastrovision 数据集
    
    从 txt 文件读取图像路径和标签，支持自定义数据增强
    
    Args:
        txt_file: 包含 "图像路径 类别索引" 的文本文件
        transform: 图像变换（数据增强）
        class_names_file: 类别名称文件（可选，用于获取类别名称）
    """
    
    def __init__(
        self,
        txt_file: str,
        transform: Optional[Callable] = None,
        class_names_file: Optional[str] = None
    ):
        self.txt_file = txt_file
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.class_names: List[str] = []
        
        # 加载样本列表
        self._load_samples()
        
        # 加载类别名称（如果提供）
        if class_names_file and os.path.exists(class_names_file):
            self._load_class_names(class_names_file)
    
    def _load_samples(self) -> None:
        """从 txt 文件加载样本"""
        with open(self.txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 格式: 图像路径 类别索引
                # 注意：路径可能包含空格，所以从右边分割
                parts = line.rsplit(' ', 1)
                if len(parts) != 2:
                    print(f"警告: 无法解析行: {line}")
                    continue
                img_path, class_idx = parts
                self.samples.append((img_path, int(class_idx)))
    
    def _load_class_names(self, class_names_file: str) -> None:
        """从文件加载类别名称"""
        with open(class_names_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 格式: 类别索引 类别名称
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    self.class_names.append(parts[1])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"错误: 无法加载图像 {img_path}: {e}")
            # 返回一个全零张量作为占位符
            if self.transform:
                dummy = Image.new('RGB', (224, 224), (0, 0, 0))
                return self.transform(dummy), label
            return torch.zeros(3, 224, 224), label
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, idx: int) -> str:
        """获取类别名称"""
        if idx < len(self.class_names):
            return self.class_names[idx]
        return f"class_{idx}"
    
    @property
    def num_classes(self) -> int:
        """获取类别数量"""
        if self.class_names:
            return len(self.class_names)
        if self.samples:
            return max(label for _, label in self.samples) + 1
        return 0


def get_transforms(
    mode: str = 'train',
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> T.Compose:
    """
    获取数据增强变换
    
    Args:
        mode: 'train', 'valid' 或 'test'
        image_size: 输出图像尺寸
        mean: 归一化均值（ImageNet 默认值）
        std: 归一化标准差（ImageNet 默认值）
        
    Returns:
        torchvision.transforms.Compose 对象
    """
    if mode == 'train':
        # 训练时使用数据增强
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=15),
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05
            ),
            T.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05)
            ),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            # 随机擦除（Cutout 效果）
            T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
    else:
        # 验证和测试时只做基本预处理
        return T.Compose([
            T.Resize(int(image_size * 1.14)),  # 256 for 224
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    pin_memory: bool = True,
    use_albumentations: bool = False,
    augment_level: str = 'medium'
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    创建训练、验证、测试 DataLoader
    
    Args:
        data_dir: 包含 train.txt, valid.txt, test.txt, class_names.txt 的目录
        batch_size: 批次大小
        num_workers: DataLoader 工作进程数
        image_size: 图像尺寸
        pin_memory: 是否固定内存（GPU 训练时推荐开启）
        use_albumentations: 是否使用 Albumentations 增强
        augment_level: Albumentations 增强级别 ('light', 'medium', 'heavy')
        
    Returns:
        (train_loader, valid_loader, test_loader, num_classes)
    """
    data_dir = Path(data_dir)
    
    train_txt = data_dir / "train.txt"
    valid_txt = data_dir / "valid.txt"
    test_txt = data_dir / "test.txt"
    class_names_txt = data_dir / "class_names.txt"
    
    # 检查文件是否存在
    for f in [train_txt, valid_txt, test_txt]:
        if not f.exists():
            raise FileNotFoundError(f"找不到文件: {f}")
    
    class_names_file = str(class_names_txt) if class_names_txt.exists() else None
    
    # 选择数据增强方式
    if use_albumentations:
        try:
            from transforms_album import get_wrapped_transforms
            train_transform = get_wrapped_transforms('train', image_size, augment_level)
            valid_transform = get_wrapped_transforms('valid', image_size)
            test_transform = get_wrapped_transforms('test', image_size)
            
            if train_transform is None:
                # Albumentations 未安装，回退到 torchvision
                print("⚠ 回退到 torchvision 增强")
                train_transform = get_transforms('train', image_size)
                valid_transform = get_transforms('valid', image_size)
                test_transform = get_transforms('test', image_size)
            else:
                print(f"✓ 使用 Albumentations 增强 (级别: {augment_level})")
        except ImportError:
            print("⚠ transforms_album.py 未找到，使用 torchvision 增强")
            train_transform = get_transforms('train', image_size)
            valid_transform = get_transforms('valid', image_size)
            test_transform = get_transforms('test', image_size)
    else:
        train_transform = get_transforms('train', image_size)
        valid_transform = get_transforms('valid', image_size)
        test_transform = get_transforms('test', image_size)
    
    # 创建数据集
    train_dataset = GastrovisionDataset(
        txt_file=str(train_txt),
        transform=train_transform,
        class_names_file=class_names_file
    )
    
    valid_dataset = GastrovisionDataset(
        txt_file=str(valid_txt),
        transform=valid_transform,
        class_names_file=class_names_file
    )
    
    test_dataset = GastrovisionDataset(
        txt_file=str(test_txt),
        transform=test_transform,
        class_names_file=class_names_file
    )
    
    num_classes = train_dataset.num_classes
    
    print(f"数据集加载完成:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(valid_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    print(f"  类别数: {num_classes}")
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # 丢弃最后不完整的 batch
        persistent_workers=num_workers > 0
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    return train_loader, valid_loader, test_loader, num_classes


def get_class_weights(
    data_dir: str,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    计算类别权重（用于处理类别不平衡）
    
    使用逆频率加权：weight[c] = N / (num_classes * count[c])
    
    Args:
        data_dir: 包含 train.txt 的目录
        device: 权重张量的设备
        
    Returns:
        类别权重张量
    """
    from collections import Counter
    
    train_txt = Path(data_dir) / "train.txt"
    
    # 统计每个类别的样本数
    class_counts = Counter()
    with open(train_txt, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(' ', 1)
            if len(parts) == 2:
                class_idx = int(parts[1])
                class_counts[class_idx] += 1
    
    # 计算权重
    num_classes = max(class_counts.keys()) + 1
    total_samples = sum(class_counts.values())
    
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)  # 避免除零
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    
    # 归一化，使平均权重为 1
    weights_tensor = weights_tensor / weights_tensor.mean()
    
    return weights_tensor


# 用于测试模块
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试数据加载模块")
    parser.add_argument("--data_dir", type=str, 
                        default="D:/codes/work-projects/Gastrovision_model",
                        help="数据目录")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--num_workers", type=int, default=0, help="工作进程数")
    
    args = parser.parse_args()
    
    print("测试数据加载模块")
    print("=" * 50)
    
    try:
        train_loader, valid_loader, test_loader, num_classes = create_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=224
        )
        
        print(f"\n测试一个 batch:")
        for images, labels in train_loader:
            print(f"  图像形状: {images.shape}")
            print(f"  标签形状: {labels.shape}")
            print(f"  标签值: {labels.tolist()}")
            break
        
        print("\n✓ 数据加载模块测试通过!")
        
    except FileNotFoundError as e:
        print(f"\n⚠ 请先运行 split_dataset.py 生成划分文件")
        print(f"  错误信息: {e}")
