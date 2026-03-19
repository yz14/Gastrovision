"""
Jaguar Re-ID 数据集模块

提供:
- JaguarTrainDataset: 训练/验证用数据集（返回 image, label）
- JaguarTestDataset:  推理用数据集（返回 image, filename）
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class JaguarTrainDataset(Dataset):
    """
    训练/验证数据集

    从 CSV 读取 (filename, ground_truth)，按 label_map 转为整数标签。

    Args:
        df: DataFrame（含 filename, ground_truth 列）
        image_dir: 图片所在目录
        label_map: {name: int} 标签映射
        transform: 图像变换（Albumentations Compose 或 wrapper）
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        label_map: dict,
        transform=None
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.label_map = label_map
        self.transform = transform

        # 预计算标签数组，避免每次查表
        self.filenames = self.df['filename'].tolist()
        self.labels = [label_map[gt] for gt in self.df['ground_truth']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_dir / self.filenames[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image=image)['image']

        label = self.labels[idx]
        return image, label

    @property
    def num_classes(self):
        return len(self.label_map)

    def get_labels(self):
        """返回所有标签列表（供 sampler 使用）"""
        return self.labels


class JaguarTestDataset(Dataset):
    """
    测试数据集（用于提取 embedding）

    从图片目录中加载所有唯一图片。

    Args:
        image_paths: 图片路径列表
        transform: 图像变换
    """

    def __init__(self, image_paths: list, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image=image)['image']

        filename = Path(img_path).name
        return image, filename
