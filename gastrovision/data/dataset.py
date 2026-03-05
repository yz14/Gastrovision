"""
Gastrovision 数据加载模块

提供：
- GastrovisionDataset: PyTorch Dataset 类
- get_transforms: 数据增强变换
- create_dataloaders: DataLoader 工厂函数
"""

import os
import copy
import random
from pathlib import Path
from typing import Tuple, List, Optional, Callable, Dict
from collections import defaultdict, Counter

import numpy as np
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


class GastrovisionMultilabelDataset(Dataset):
    """
    Gastrovision 多标签数据集
    
    从 txt 文件读取图像路径和多个标签索引，返回 multi-hot 向量
    
    Args:
        txt_file: 包含 "图像路径 标签1 标签2 ..." 的文本文件
        num_classes: 类别总数（默认 16）
        transform: 图像变换（数据增强）
        class_names_file: 类别名称文件（可选）
    """
    
    def __init__(
        self,
        txt_file: str,
        num_classes: int = 16,
        transform: Optional[Callable] = None,
        class_names_file: Optional[str] = None
    ):
        self.txt_file = txt_file
        self.num_classes = num_classes
        self.transform = transform
        self.samples: List[Tuple[str, torch.Tensor]] = []
        self.class_names: List[str] = []
        
        # 加载样本列表
        self._load_samples()
        
        # 加载类别名称（如果提供）
        if class_names_file and os.path.exists(class_names_file):
            self._load_class_names(class_names_file)
    
    def _load_samples(self) -> None:
        """从 txt 文件加载多标签样本"""
        with open(self.txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 格式: 图像路径 标签索引1 标签索引2 ...
                # 路径可能包含空格，所以需要特殊处理
                # 假设标签索引都是数字，从右边开始解析
                parts = line.split()
                
                # 找到第一个非纯数字的部分作为路径的结尾
                label_start_idx = len(parts)
                for i in range(len(parts) - 1, -1, -1):
                    if not parts[i].isdigit():
                        label_start_idx = i + 1
                        break
                
                # 解析路径和标签
                img_path = ' '.join(parts[:label_start_idx])
                label_indices = [int(x) for x in parts[label_start_idx:]]
                
                # 创建 multi-hot 向量
                multi_hot = torch.zeros(self.num_classes, dtype=torch.float32)
                for idx in label_indices:
                    if 0 <= idx < self.num_classes:
                        multi_hot[idx] = 1.0
                
                self.samples.append((img_path, multi_hot))
    
    def _load_class_names(self, class_names_file: str) -> None:
        """从文件加载类别名称"""
        with open(class_names_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.class_names.append(line)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, labels = self.samples[idx]
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"错误: 无法加载图像 {img_path}: {e}")
            # 返回一个全零张量作为占位符
            if self.transform:
                dummy = Image.new('RGB', (224, 224), (0, 0, 0))
                return self.transform(dummy), labels
            return torch.zeros(3, 224, 224), labels
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, labels
    
    def get_class_name(self, idx: int) -> str:
        """获取类别名称"""
        if idx < len(self.class_names):
            return self.class_names[idx]
        return f"class_{idx}"
    
    def get_label_statistics(self) -> Dict[int, int]:
        """统计各标签的样本数"""
        stats = {i: 0 for i in range(self.num_classes)}
        for _, labels in self.samples:
            for i in range(self.num_classes):
                if labels[i] > 0:
                    stats[i] += 1
        return stats


class MultilabelIdentitySampler:
    """
    多标签身份采样器 - 基于 WhaleSSL 的 WhaleRandomIdentitySampler
    
    为 Triplet Loss 提供有效的采样策略：
    - 每个 batch 包含 N 个"身份" (主标签)
    - 每个身份采样 K 个实例
    - 保证每个 batch 有足够的正负样本对
    
    对于多标签数据，使用第一个正标签（或最稀有的正标签）作为"身份"
    
    Args:
        dataset: GastrovisionMultilabelDataset 实例
        batch_size: 批次大小 (应为 num_instances 的整数倍)
        num_instances: 每个身份的实例数 (默认 4)
        identity_strategy: 'first' (第一个正标签) 或 'rarest' (最稀有正标签)
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        num_instances: int = 4,
        identity_strategy: str = 'first'
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.identity_strategy = identity_strategy
        
        # 统计每个类别的频率 (用于 'rarest' 策略)
        self.label_counts = dataset.get_label_statistics() if hasattr(dataset, 'get_label_statistics') else None
        
        # 构建身份索引字典 {identity -> [sample_indices]}
        self.index_dic = defaultdict(list)
        
        for idx, (_, labels) in enumerate(dataset.samples):
            # 获取这个样本的身份 (identity)
            identity = self._get_identity(labels)
            if identity is not None:
                self.index_dic[identity].append(idx)
        
        self.pids = list(self.index_dic.keys())
        
        # 估计每个 epoch 的样本数
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances
        
        print(f"MultilabelIdentitySampler 初始化:")
        print(f"  - 身份数: {len(self.pids)}")
        print(f"  - 每 batch 身份数: {self.num_pids_per_batch}")
        print(f"  - 每身份实例数: {self.num_instances}")
        print(f"  - 预计每 epoch 样本数: {self.length}")
    
    def _get_identity(self, labels) -> Optional[int]:
        """从多热标签中获取身份 (主标签)"""
        
        # 获取所有正标签的索引
        if isinstance(labels, torch.Tensor):
            positive_indices = torch.where(labels > 0)[0].tolist()
        else:
            positive_indices = [i for i, v in enumerate(labels) if v > 0]
        
        if not positive_indices:
            return None
        
        if self.identity_strategy == 'first':
            # 使用第一个正标签
            return positive_indices[0]
        elif self.identity_strategy == 'rarest':
            # 使用最稀有的正标签
            if self.label_counts:
                rarest_idx = min(positive_indices, key=lambda x: self.label_counts.get(x, float('inf')))
                return rarest_idx
            return positive_indices[0]
        else:
            return positive_indices[0]
    
    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            
            # 如果样本数不足 num_instances，进行重采样
            if len(idxs) < self.num_instances:
                idxs = list(np.random.choice(idxs, size=self.num_instances, replace=True))
            
            random.shuffle(idxs)
            batch_idxs = []
            
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
        
        # 构建最终索引
        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
        
        return iter(final_idxs)
    
    def __len__(self):
        return self.length

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
            from .transforms import get_wrapped_transforms
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


def create_multilabel_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    pin_memory: bool = True,
    num_classes: int = 16,
    use_triplet_sampler: bool = False,
    triplet_num_instances: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader, int, List[str]]:
    """
    创建多标签训练、验证、测试 DataLoader
    
    Args:
        data_dir: 包含 train_mul.txt, valid_mul.txt, test_mul.txt, class_names_mul.txt 的目录
        batch_size: 批次大小
        num_workers: DataLoader 工作进程数
        image_size: 图像尺寸
        pin_memory: 是否固定内存
        num_classes: 类别数量
        use_triplet_sampler: 是否使用 Triplet 采样器
        triplet_num_instances: Triplet 采样器每个身份的实例数
        
    Returns:
        (train_loader, valid_loader, test_loader, num_classes, class_names)
    """
    data_dir = Path(data_dir)
    
    train_txt = data_dir / "train_mul.txt"
    valid_txt = data_dir / "valid_mul.txt"
    test_txt = data_dir / "test_mul.txt"
    class_names_txt = data_dir / "class_names_mul.txt"
    
    # 检查文件是否存在
    for f in [train_txt, valid_txt, test_txt]:
        if not f.exists():
            raise FileNotFoundError(f"找不到多标签数据文件: {f}")
    
    class_names_file = str(class_names_txt) if class_names_txt.exists() else None
    
    # 数据增强
    train_transform = get_transforms('train', image_size)
    valid_transform = get_transforms('valid', image_size)
    test_transform = get_transforms('test', image_size)
    
    # 创建数据集
    train_dataset = GastrovisionMultilabelDataset(
        txt_file=str(train_txt),
        num_classes=num_classes,
        transform=train_transform,
        class_names_file=class_names_file
    )
    
    valid_dataset = GastrovisionMultilabelDataset(
        txt_file=str(valid_txt),
        num_classes=num_classes,
        transform=valid_transform,
        class_names_file=class_names_file
    )
    
    test_dataset = GastrovisionMultilabelDataset(
        txt_file=str(test_txt),
        num_classes=num_classes,
        transform=test_transform,
        class_names_file=class_names_file
    )
    
    # 获取类别名称
    class_names = train_dataset.class_names if train_dataset.class_names else [f"class_{i}" for i in range(num_classes)]
    
    print(f"多标签数据集加载完成:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(valid_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    print(f"  类别数: {num_classes}")
    
    # 打印标签统计
    train_stats = train_dataset.get_label_statistics()
    print(f"  训练集标签分布:")
    for i, count in train_stats.items():
        name = class_names[i] if i < len(class_names) else f"class_{i}"
        print(f"    {i:2d}. {name}: {count}")
    
    # 创建 DataLoader
    if use_triplet_sampler:
        # 使用 Triplet 采样器 (WhaleSSL 风格)
        triplet_sampler = MultilabelIdentitySampler(
            dataset=train_dataset,
            batch_size=batch_size,
            num_instances=triplet_num_instances,
            identity_strategy='first'
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=triplet_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=num_workers > 0
        )
        print(f"  使用 Triplet 采样器: 每身份 {triplet_num_instances} 实例")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
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
    
    return train_loader, valid_loader, test_loader, num_classes, class_names


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
