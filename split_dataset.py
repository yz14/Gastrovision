"""
Gastrovision 数据集划分脚本

功能：
- 扫描数据集目录下所有类别文件夹
- 按 7:1.5:1.5 的比例进行分层划分（训练:验证:测试）
- 保证每个类别在验证集和测试集中至少有1张图像
- 生成 train.txt, valid.txt, test.txt 以及 class_names.txt

用法：
    python split_dataset.py --data_dir D:/codes/data/Gastrovision --output_dir D:/codes/work-projects/Gastrovision_model
"""

import os
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict


def scan_dataset(data_dir: str) -> Dict[str, List[str]]:
    """
    扫描数据集目录，获取每个类别的所有图像路径
    
    Args:
        data_dir: 数据集根目录
        
    Returns:
        字典，键为类别名称，值为该类别下所有图像的绝对路径列表
    """
    data_dir = Path(data_dir)
    class_to_images = defaultdict(list)
    
    # 支持的图像扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    
    # 遍历所有子目录
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        
        # 收集该类别下的所有图像
        for img_path in class_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                class_to_images[class_name].append(str(img_path.absolute()))
    
    return dict(class_to_images)


def stratified_split(
    images: List[str],
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    对单个类别的图像进行划分
    
    对于小类别（样本数 < 7），优先保证 valid 和 test 各至少1张
    
    Args:
        images: 图像路径列表
        train_ratio: 训练集比例
        valid_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        
    Returns:
        (train_images, valid_images, test_images)
    """
    random.seed(seed)
    images = images.copy()
    random.shuffle(images)
    
    n = len(images)
    
    if n == 0:
        return [], [], []
    
    # 小类别特殊处理：确保 valid 和 test 各至少1张
    if n <= 2:
        # 只有1-2张：全部给训练集，valid/test 各分1张（如果有）
        if n == 1:
            return images, [], []
        else:  # n == 2
            return [], [images[0]], [images[1]]
    elif n <= 4:
        # 3-4张：valid/test 各1张，剩余给训练
        return images[2:], [images[0]], [images[1]]
    elif n <= 6:
        # 5-6张：valid/test 各1张，剩余给训练
        return images[2:], [images[0]], [images[1]]
    else:
        # 正常情况：按比例划分
        n_valid = max(1, int(n * valid_ratio))
        n_test = max(1, int(n * test_ratio))
        n_train = n - n_valid - n_test
        
        # 确保训练集至少有1张
        if n_train < 1:
            n_train = 1
            n_valid = max(1, (n - 1) // 2)
            n_test = n - 1 - n_valid
        
        train_images = images[:n_train]
        valid_images = images[n_train:n_train + n_valid]
        test_images = images[n_train + n_valid:]
        
        return train_images, valid_images, test_images


def save_split_file(
    data: List[Tuple[str, int]],
    output_path: str
) -> None:
    """
    保存划分结果到文件
    
    Args:
        data: [(图像路径, 类别索引), ...]
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for img_path, class_idx in data:
            f.write(f"{img_path} {class_idx}\n")


def save_class_names(
    class_names: List[str],
    output_path: str
) -> None:
    """
    保存类别名称到索引的映射
    
    Args:
        class_names: 类别名称列表（索引即为类别编号）
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, name in enumerate(class_names):
            f.write(f"{idx} {name}\n")


def split_dataset(
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, int]:
    """
    主函数：执行数据集划分
    
    Args:
        data_dir: 数据集根目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        valid_ratio: 验证集比例  
        test_ratio: 测试集比例
        seed: 随机种子
        
    Returns:
        统计信息字典
    """
    print("=" * 60)
    print("Gastrovision 数据集划分")
    print("=" * 60)
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"划分比例: {train_ratio}:{valid_ratio}:{test_ratio}")
    print(f"随机种子: {seed}")
    print()
    
    # 扫描数据集
    print("正在扫描数据集...")
    class_to_images = scan_dataset(data_dir)
    
    if not class_to_images:
        raise ValueError(f"在 {data_dir} 中未找到任何类别文件夹或图像")
    
    # 按数据量降序排列类别：标签0=数据最多，标签N=数据最少
    # 这样方便在图表中按索引0→N展示时直观看到数据量从多到少的分布
    class_names = sorted(class_to_images.keys(), 
                         key=lambda x: len(class_to_images[x]), 
                         reverse=True)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"发现 {len(class_names)} 个类别（按数据量降序排列）")
    print()
    
    # 对每个类别进行划分
    train_data = []
    valid_data = []
    test_data = []
    
    stats = {
        'num_classes': len(class_names),
        'train_total': 0,
        'valid_total': 0,
        'test_total': 0,
        'per_class': {}
    }
    
    print(f"{'类别名称':<50} {'总数':>6} {'训练':>6} {'验证':>6} {'测试':>6}")
    print("-" * 80)
    
    for class_name in class_names:
        images = class_to_images[class_name]
        class_idx = class_to_idx[class_name]
        
        train_imgs, valid_imgs, test_imgs = stratified_split(
            images, train_ratio, valid_ratio, test_ratio, seed
        )
        
        # 添加到总列表
        train_data.extend([(img, class_idx) for img in train_imgs])
        valid_data.extend([(img, class_idx) for img in valid_imgs])
        test_data.extend([(img, class_idx) for img in test_imgs])
        
        # 统计
        n_total = len(images)
        n_train = len(train_imgs)
        n_valid = len(valid_imgs)
        n_test = len(test_imgs)
        
        stats['train_total'] += n_train
        stats['valid_total'] += n_valid
        stats['test_total'] += n_test
        stats['per_class'][class_name] = {
            'total': n_total,
            'train': n_train,
            'valid': n_valid,
            'test': n_test
        }
        
        # 输出警告：如果 valid 或 test 为空
        warning = ""
        if n_valid == 0:
            warning += " [!valid=0]"
        if n_test == 0:
            warning += " [!test=0]"
            
        # 截断过长的类别名称
        display_name = class_name[:47] + "..." if len(class_name) > 50 else class_name
        print(f"{display_name:<50} {n_total:>6} {n_train:>6} {n_valid:>6} {n_test:>6}{warning}")
    
    print("-" * 80)
    print(f"{'总计':<50} {stats['train_total'] + stats['valid_total'] + stats['test_total']:>6} "
          f"{stats['train_total']:>6} {stats['valid_total']:>6} {stats['test_total']:>6}")
    print()
    
    # 按标签排序（便于查看类别不平衡）
    train_data.sort(key=lambda x: x[1])
    valid_data.sort(key=lambda x: x[1])
    test_data.sort(key=lambda x: x[1])
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存划分文件
    train_path = output_dir / "train.txt"
    valid_path = output_dir / "valid.txt"
    test_path = output_dir / "test.txt"
    class_names_path = output_dir / "class_names.txt"
    
    save_split_file(train_data, str(train_path))
    save_split_file(valid_data, str(valid_path))
    save_split_file(test_data, str(test_path))
    save_class_names(class_names, str(class_names_path))
    
    print("生成的文件:")
    print(f"  - {train_path} ({len(train_data)} 条记录)")
    print(f"  - {valid_path} ({len(valid_data)} 条记录)")
    print(f"  - {test_path} ({len(test_data)} 条记录)")
    print(f"  - {class_names_path} ({len(class_names)} 个类别)")
    print()
    print("✓ 数据集划分完成!")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Gastrovision 数据集划分工具",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="D:/codes/data/Gastrovision",
        help="数据集根目录 (默认: D:/codes/data/Gastrovision)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="D:/codes/work-projects/Gastrovision_model",
        help="输出目录 (默认: D:/codes/work-projects/Gastrovision_model)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="训练集比例 (默认: 0.7)"
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.15,
        help="验证集比例 (默认: 0.15)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="测试集比例 (默认: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )
    
    args = parser.parse_args()
    
    # 验证比例之和
    ratio_sum = args.train_ratio + args.valid_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"划分比例之和必须为1.0，当前为 {ratio_sum}")
    
    split_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
