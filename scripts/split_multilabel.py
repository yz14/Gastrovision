"""
Gastrovision 多标签数据集划分脚本

功能：
- 从 CSV 文件读取多标签数据
- 使用 Iterative Stratification 进行分层划分（7:1.5:1.5）
- 确保每个类别在验证集和测试集中至少有1个样本
- 输出 train_mul.txt, valid_mul.txt, test_mul.txt

使用方法：
    python split_multilabel.py --data_csv "D:/codes/data/Gastrovision/2026-01-26-v1.csv"

依赖：
    pip install iterative-stratification pandas
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

# 中文类别到英文文件夹名的映射
CATEGORY_TO_FOLDER = {
    "血管扩张": "Angiectasia",
    "腔内积血": "Blood in lumen",
    "盲肠": "Cecum",
    "结肠憩室": "Colon diverticula",
    "结肠息肉": "Colon polyps",
    "结直肠癌": "Colorectal cancer",
    "染色抬举后的息肉": "Dyed-lifted-polyps",
    "染色切除边缘": "Dyed-resection-margins",
    "红斑": "Erythema",
    "回盲瓣": "Ileocecal valve",
    "大肠黏膜炎症": "Mucosal inflammation large bowel",
    "大肠正常黏膜及血管形态": "Normal mucosa and vascular pattern in the large bowel",
    "切除的息肉": "Resected polyps",
    "切缘": "Resection margins",
    "直肠倒镜观察": "Retroflex rectum",
    "小肠末端回肠": "Small bowel_terminal ileum",
}

# 16 个多标签类别（按 CSV 列顺序）
LABEL_COLUMNS = [
    "血管扩张", "腔内积血", "盲肠", "结肠憩室", "结肠息肉", "结直肠癌",
    "染色抬举后的息肉", "染色切除边缘", "红斑", "回盲瓣", "大肠黏膜炎症",
    "大肠正常黏膜及血管形态", "切除的息肉", "切缘", "直肠倒镜观察", "小肠末端回肠"
]


def find_image_path(image_name: str, data_dir: str, original_category: str) -> Optional[str]:
    """
    根据图像名和原始类别查找图像路径
    
    优先在原始类别对应的文件夹中查找，找不到再全局搜索
    """
    # 首先尝试原始类别对应的文件夹
    if original_category in CATEGORY_TO_FOLDER:
        folder_name = CATEGORY_TO_FOLDER[original_category]
        candidate_path = Path(data_dir) / folder_name / image_name
        if candidate_path.exists():
            return str(candidate_path.absolute())
    
    # 如果找不到，在所有文件夹中搜索
    for folder in Path(data_dir).iterdir():
        if folder.is_dir():
            candidate_path = folder / image_name
            if candidate_path.exists():
                return str(candidate_path.absolute())
    
    return None


def load_csv_data(csv_path: str, data_dir: str) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    加载 CSV 数据
    
    Returns:
        image_paths: 图像路径列表
        labels: 多标签数组 (N, 16)
        missing_images: 找不到的图像名列表
    """
    df = pd.read_csv(csv_path)
    
    image_paths = []
    labels = []
    missing_images = []
    
    for _, row in df.iterrows():
        image_name = row["样本名字"]
        original_category = row["原始类别"]
        
        # 查找图像路径
        image_path = find_image_path(image_name, data_dir, original_category)
        
        if image_path is None:
            missing_images.append(image_name)
            continue
        
        # 提取16个标签
        label_values = [int(row[col]) for col in LABEL_COLUMNS]
        
        image_paths.append(image_path)
        labels.append(label_values)
    
    return image_paths, np.array(labels), missing_images


def iterative_stratification_split(
    labels: np.ndarray,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 Iterative Stratification 进行多标签数据划分
    
    Returns:
        train_indices, valid_indices, test_indices
    """
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError:
        print("错误: 请安装 iterative-stratification 库:")
        print("  pip install iterative-stratification")
        raise
    
    n_samples = len(labels)
    indices = np.arange(n_samples)
    
    np.random.seed(seed)
    
    # 第一步：分出 train 和 (valid + test)
    test_valid_ratio = valid_ratio + test_ratio
    
    splitter1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, 
        test_size=test_valid_ratio, 
        random_state=seed
    )
    
    train_idx, temp_idx = next(splitter1.split(indices, labels))
    train_indices = indices[train_idx]
    temp_indices = indices[temp_idx]
    
    # 第二步：将 (valid + test) 分成 valid 和 test
    # valid_ratio / (valid_ratio + test_ratio) = 0.15 / 0.30 = 0.5
    relative_valid_ratio = valid_ratio / test_valid_ratio
    
    splitter2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, 
        test_size=1 - relative_valid_ratio,  # test 在这里是最终的 test
        random_state=seed + 1
    )
    
    temp_labels = labels[temp_indices]
    valid_idx_relative, test_idx_relative = next(splitter2.split(np.arange(len(temp_indices)), temp_labels))
    
    valid_indices = temp_indices[valid_idx_relative]
    test_indices = temp_indices[test_idx_relative]
    
    return train_indices, valid_indices, test_indices


def ensure_min_samples(
    train_indices: np.ndarray,
    valid_indices: np.ndarray, 
    test_indices: np.ndarray,
    labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    确保每个类别在 valid 和 test 中至少有 1 个样本
    
    如果某个类别在 valid/test 中缺失，从 train 中移动一个样本过去
    """
    n_labels = labels.shape[1]
    
    train_set = set(train_indices.tolist())
    valid_set = set(valid_indices.tolist())
    test_set = set(test_indices.tolist())
    
    for label_idx in range(n_labels):
        # 检查 valid 集合
        valid_has_label = any(labels[i, label_idx] == 1 for i in valid_set)
        if not valid_has_label:
            # 从 train 中找一个有这个标签的样本移到 valid
            for i in list(train_set):
                if labels[i, label_idx] == 1:
                    train_set.remove(i)
                    valid_set.add(i)
                    print(f"  标签 {label_idx} 在 valid 中缺失，从 train 移动样本 {i}")
                    break
        
        # 检查 test 集合
        test_has_label = any(labels[i, label_idx] == 1 for i in test_set)
        if not test_has_label:
            # 从 train 中找一个有这个标签的样本移到 test
            for i in list(train_set):
                if labels[i, label_idx] == 1:
                    train_set.remove(i)
                    test_set.add(i)
                    print(f"  标签 {label_idx} 在 test 中缺失，从 train 移动样本 {i}")
                    break
    
    return (
        np.array(sorted(train_set)),
        np.array(sorted(valid_set)),
        np.array(sorted(test_set))
    )


def save_split_file(
    image_paths: List[str],
    labels: np.ndarray,
    indices: np.ndarray,
    output_path: str
):
    """
    保存划分结果到文件
    
    格式: 图像路径 标签索引1 标签索引2 ...
    例如: D:/path/to/image.jpg 0 4 10
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx in indices:
            path = image_paths[idx]
            label_indices = np.where(labels[idx] == 1)[0]
            label_str = " ".join(str(l) for l in label_indices)
            f.write(f"{path} {label_str}\n")


def save_class_names(output_path: str):
    """保存类别名称文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for name in LABEL_COLUMNS:
            f.write(f"{name}\n")


def print_statistics(
    labels: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    test_indices: np.ndarray
):
    """打印划分统计信息"""
    n_labels = labels.shape[1]
    
    print("\n" + "=" * 70)
    print("划分统计")
    print("=" * 70)
    
    print(f"\n样本数量:")
    print(f"  训练集: {len(train_indices)}")
    print(f"  验证集: {len(valid_indices)}")
    print(f"  测试集: {len(test_indices)}")
    print(f"  总计:   {len(train_indices) + len(valid_indices) + len(test_indices)}")
    
    print(f"\n各标签分布:")
    print("-" * 70)
    print(f"{'标签名称':<25} {'Train':>8} {'Valid':>8} {'Test':>8} {'Total':>8}")
    print("-" * 70)
    
    for i, name in enumerate(LABEL_COLUMNS):
        train_count = labels[train_indices, i].sum()
        valid_count = labels[valid_indices, i].sum()
        test_count = labels[test_indices, i].sum()
        total_count = train_count + valid_count + test_count
        
        # 如果 valid 或 test 中没有样本，用警告标记
        valid_mark = " ⚠" if valid_count == 0 else ""
        test_mark = " ⚠" if test_count == 0 else ""
        
        print(f"{name:<25} {train_count:>8} {valid_count:>7}{valid_mark:>1} {test_count:>7}{test_mark:>1} {total_count:>8}")
    
    print("-" * 70)
    
    # 多标签统计
    print("\n多标签样本统计:")
    for split_name, indices in [("训练集", train_indices), ("验证集", valid_indices), ("测试集", test_indices)]:
        label_counts = labels[indices].sum(axis=1)
        multi_label = (label_counts > 1).sum()
        single_label = (label_counts == 1).sum()
        print(f"  {split_name}: 单标签 {single_label}, 多标签 {multi_label} ({multi_label/len(indices)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Gastrovision 多标签数据集划分",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data_csv", 
        type=str, 
        default="D:/codes/data/Gastrovision/2026-01-26-v1.csv",
        help="CSV 数据文件路径"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="D:/codes/data/Gastrovision",
        help="图像数据目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="D:/codes/work-projects/Gastrovision_models",
        help="输出目录"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.7,
        help="训练集比例"
    )
    parser.add_argument(
        "--valid_ratio", type=float, default=0.15,
        help="验证集比例"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.15,
        help="测试集比例"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="输出详细信息"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Gastrovision 多标签数据集划分")
    print("=" * 70)
    print(f"CSV 文件: {args.data_csv}")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"划分比例: {args.train_ratio}:{args.valid_ratio}:{args.test_ratio}")
    print(f"随机种子: {args.seed}")
    
    # 加载数据
    print("\n加载数据...")
    image_paths, labels, missing_images = load_csv_data(args.data_csv, args.data_dir)
    
    print(f"  成功加载: {len(image_paths)} 个样本")
    if missing_images:
        print(f"  ⚠ 找不到 {len(missing_images)} 个图像文件:")
        if args.verbose:
            for img in missing_images[:10]:
                print(f"    - {img}")
            if len(missing_images) > 10:
                print(f"    ... 还有 {len(missing_images) - 10} 个")
    
    # 执行划分
    print("\n执行 Iterative Stratification 划分...")
    train_indices, valid_indices, test_indices = iterative_stratification_split(
        labels,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # 确保每个类别在 valid/test 中至少有 1 个样本
    print("\n检查并确保每个类别至少有 1 个样本...")
    train_indices, valid_indices, test_indices = ensure_min_samples(
        train_indices, valid_indices, test_indices, labels
    )
    
    # 打印统计信息
    print_statistics(labels, train_indices, valid_indices, test_indices)
    
    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n保存划分结果...")
    save_split_file(image_paths, labels, train_indices, output_dir / "train_mul.txt")
    save_split_file(image_paths, labels, valid_indices, output_dir / "valid_mul.txt")
    save_split_file(image_paths, labels, test_indices, output_dir / "test_mul.txt")
    save_class_names(output_dir / "class_names_mul.txt")
    
    print(f"\n✓ 已保存到:")
    print(f"  - {output_dir / 'train_mul.txt'}")
    print(f"  - {output_dir / 'valid_mul.txt'}")
    print(f"  - {output_dir / 'test_mul.txt'}")
    print(f"  - {output_dir / 'class_names_mul.txt'}")
    
    print("\n完成!")


if __name__ == "__main__":
    main()
