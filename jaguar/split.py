"""
数据划分模块

将 train.csv 按分层采样划分为训练集和验证集。
保证验证集中每个类别至少有 1 个样本。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter


def stratified_split(
    csv_path: str,
    val_ratio: float = 0.15,
    min_val_per_class: int = 1,
    seed: int = 42,
    output_dir: str = None
) -> tuple:
    """
    分层划分训练集和验证集

    Args:
        csv_path: train.csv 路径（含 filename, ground_truth 列）
        val_ratio: 验证集比例
        min_val_per_class: 每个类别在验证集中的最少样本数
        seed: 随机种子
        output_dir: 如果指定，保存 train_split.csv 和 val_split.csv

    Returns:
        (train_df, val_df): 划分后的 DataFrame
    """
    rng = np.random.RandomState(seed)
    df = pd.read_csv(csv_path)

    # 统计每个类别的样本数
    class_counts = Counter(df['ground_truth'])
    classes = sorted(class_counts.keys())
    num_classes = len(classes)

    print(f"总样本数: {len(df)}")
    print(f"类别数:   {num_classes}")
    print(f"样本数范围: {min(class_counts.values())} ~ {max(class_counts.values())}")

    train_indices = []
    val_indices = []

    for cls in classes:
        cls_indices = df[df['ground_truth'] == cls].index.tolist()
        n = len(cls_indices)
        rng.shuffle(cls_indices)

        # 每个类至少 min_val_per_class 个验证样本
        n_val = max(min_val_per_class, int(n * val_ratio))
        # 但也要保证训练集至少有 1 个样本
        n_val = min(n_val, n - 1) if n > 1 else 0

        if n_val == 0 and n == 1:
            # 只有 1 个样本的类放入训练集（无法验证，但避免丢失）
            train_indices.extend(cls_indices)
            print(f"  [警告] 类别 '{cls}' 仅 1 个样本，全部放入训练集")
        else:
            val_indices.extend(cls_indices[:n_val])
            train_indices.extend(cls_indices[n_val:])

    train_df = df.loc[train_indices].reset_index(drop=True)
    val_df = df.loc[val_indices].reset_index(drop=True)

    print(f"\n划分结果:")
    print(f"  训练集: {len(train_df)} 样本")
    print(f"  验证集: {len(val_df)} 样本")

    # 验证每个类别在验证集中的覆盖情况
    val_classes = set(val_df['ground_truth'])
    train_classes = set(train_df['ground_truth'])
    missing = train_classes - val_classes
    if missing:
        print(f"  [提示] {len(missing)} 个类别仅出现在训练集中: {missing}")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(out / 'train_split.csv', index=False)
        val_df.to_csv(out / 'val_split.csv', index=False)
        print(f"  已保存到: {out}")

    return train_df, val_df


def build_label_map(df: pd.DataFrame) -> dict:
    """
    构建 ground_truth -> label_id 的映射

    Args:
        df: 包含 ground_truth 列的 DataFrame

    Returns:
        {name: int} 映射字典
    """
    classes = sorted(df['ground_truth'].unique())
    return {name: idx for idx, name in enumerate(classes)}


if __name__ == '__main__':
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'data/train.csv'
    train_df, val_df = stratified_split(csv_path, output_dir='data')
    label_map = build_label_map(pd.concat([train_df, val_df]))
    print(f"\n标签映射 ({len(label_map)} 类):")
    for name, idx in label_map.items():
        print(f"  {idx:3d}: {name}")
