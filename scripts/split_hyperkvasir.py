"""
Hyper-Kvasir 数据集划分脚本

根据 Hyper-Kvasir 的目录结构生成训练/验证/测试集划分文件：
- train.txt
- valid.txt
- test.txt
- class_names.txt

使用方法:
    python split_hyperkvasir.py --data_dir D:/codes/data/HH/hyper-kvasir-labeled-images/labeled-images
    
生成的文件将保存在 data_dir 目录下，之后可以直接在 main.py 中替换 --data_dir 参数进行训练。

作者: Gastrovision Team
"""

import os
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict


def scan_hyperkvasir_dataset(data_dir: str) -> Dict[str, List[str]]:
    """
    扫描 Hyper-Kvasir 数据集目录结构
    
    目录格式:
    labeled-images/
    ├── lower-gi-tract/
    │   ├── anatomical-landmarks/
    │   │   ├── cecum/
    │   │   │   ├── xxx.jpg
    │   │   │   └── ...
    │   │   └── ...
    │   ├── pathological-findings/
    │   │   ├── polyps/
    │   │   └── ...
    │   └── ...
    └── upper-gi-tract/
        ├── anatomical-landmarks/
        │   ├── pylorus/
        │   └── ...
        └── ...
    
    Args:
        data_dir: labeled-images 目录路径
        
    Returns:
        class_to_images: {类别名: [图像路径列表]}
    """
    class_to_images = defaultdict(list)
    data_dir = Path(data_dir)
    
    # 支持的图像扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    # 遍历 upper-gi-tract 和 lower-gi-tract
    for gi_tract in ['upper-gi-tract', 'lower-gi-tract']:
        gi_tract_dir = data_dir / gi_tract
        if not gi_tract_dir.exists():
            print(f"警告: 目录不存在 {gi_tract_dir}")
            continue
        
        # 遍历子分类 (anatomical-landmarks, pathological-findings, etc.)
        for category_dir in gi_tract_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            # 遍历具体类别 (cecum, polyps, etc.)
            for class_dir in category_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                
                class_name = class_dir.name
                
                # 收集该类别的所有图像
                for img_file in class_dir.iterdir():
                    if img_file.suffix.lower() in image_extensions:
                        class_to_images[class_name].append(str(img_file.absolute()))
    
    return dict(class_to_images)


def split_dataset(
    class_to_images: Dict[str, List[str]],
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    min_samples: int = 3
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]], List[str]]:
    """
    按比例划分数据集
    
    Args:
        class_to_images: {类别名: [图像路径列表]}
        train_ratio: 训练集比例
        valid_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        min_samples: 最少样本数（少于此数的类别会被跳过）
        
    Returns:
        train_samples, valid_samples, test_samples, class_names
    """
    random.seed(seed)
    
    train_samples = []
    valid_samples = []
    test_samples = []
    
    # 按样本数量降序排列类别
    sorted_classes = sorted(class_to_images.keys(), 
                           key=lambda x: len(class_to_images[x]), 
                           reverse=True)
    
    # 过滤掉样本数太少的类别
    valid_classes = []
    for class_name in sorted_classes:
        if len(class_to_images[class_name]) >= min_samples:
            valid_classes.append(class_name)
        else:
            print(f"跳过类别 '{class_name}' (样本数: {len(class_to_images[class_name])} < {min_samples})")
    
    # 为每个类别分配索引
    class_names = valid_classes  # 已按数量降序排列
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # 划分每个类别
    for class_name in class_names:
        images = class_to_images[class_name]
        random.shuffle(images)
        
        n = len(images)
        n_train = max(1, int(n * train_ratio))
        n_valid = max(1, int(n * valid_ratio))
        n_test = n - n_train - n_valid
        
        if n_test < 1:
            n_test = 1
            n_train = n - n_valid - n_test
        
        class_idx = class_to_idx[class_name]
        
        for img in images[:n_train]:
            train_samples.append((img, class_idx))
        for img in images[n_train:n_train + n_valid]:
            valid_samples.append((img, class_idx))
        for img in images[n_train + n_valid:]:
            test_samples.append((img, class_idx))
    
    # 打乱顺序
    random.shuffle(train_samples)
    random.shuffle(valid_samples)
    random.shuffle(test_samples)
    
    return train_samples, valid_samples, test_samples, class_names


def save_split_files(
    output_dir: str,
    train_samples: List[Tuple[str, int]],
    valid_samples: List[Tuple[str, int]],
    test_samples: List[Tuple[str, int]],
    class_names: List[str]
) -> None:
    """
    保存划分文件
    
    Args:
        output_dir: 输出目录
        train_samples: 训练集 [(图像路径, 类别索引), ...]
        valid_samples: 验证集
        test_samples: 测试集
        class_names: 类别名称列表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 train.txt
    with open(output_dir / 'train.txt', 'w', encoding='utf-8') as f:
        for img_path, class_idx in train_samples:
            f.write(f"{img_path} {class_idx}\n")
    
    # 保存 valid.txt
    with open(output_dir / 'valid.txt', 'w', encoding='utf-8') as f:
        for img_path, class_idx in valid_samples:
            f.write(f"{img_path} {class_idx}\n")
    
    # 保存 test.txt
    with open(output_dir / 'test.txt', 'w', encoding='utf-8') as f:
        for img_path, class_idx in test_samples:
            f.write(f"{img_path} {class_idx}\n")
    
    # 保存 class_names.txt
    with open(output_dir / 'class_names.txt', 'w', encoding='utf-8') as f:
        for idx, name in enumerate(class_names):
            f.write(f"{idx} {name}\n")
    
    print(f"\n划分文件已保存到: {output_dir}")
    print(f"  - train.txt ({len(train_samples)} 样本)")
    print(f"  - valid.txt ({len(valid_samples)} 样本)")
    print(f"  - test.txt ({len(test_samples)} 样本)")
    print(f"  - class_names.txt ({len(class_names)} 类别)")


def print_class_distribution(class_names: List[str], class_to_images: Dict[str, List[str]]) -> None:
    """打印类别分布"""
    print("\n类别分布 (按样本数量降序):")
    print("-" * 50)
    total = 0
    for idx, name in enumerate(class_names):
        count = len(class_to_images.get(name, []))
        total += count
        print(f"  {idx:2d}: {name:<30s} {count:>5d} 张")
    print("-" * 50)
    print(f"  总计: {len(class_names)} 类别, {total} 张图像")


def main():
    parser = argparse.ArgumentParser(
        description="Hyper-Kvasir 数据集划分",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, 
                        default='D:/codes/data/HH/hyper-kvasir-labeled-images/labeled-images',
                        help='Hyper-Kvasir labeled-images 目录路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认与 data_dir 相同）')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--valid_ratio', type=float, default=0.15, help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='测试集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--min_samples', type=int, default=3, help='最少样本数阈值')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.data_dir
    
    print("=" * 60)
    print("Hyper-Kvasir 数据集划分")
    print("=" * 60)
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"划分比例: train={args.train_ratio}, valid={args.valid_ratio}, test={args.test_ratio}")
    print(f"随机种子: {args.seed}")
    
    # 扫描数据集
    print("\n扫描数据集...")
    class_to_images = scan_hyperkvasir_dataset(args.data_dir)
    
    if not class_to_images:
        print("错误: 未找到任何图像文件")
        return
    
    # 打印类别分布
    # 先临时按数量排序用于显示
    sorted_classes = sorted(class_to_images.keys(), 
                           key=lambda x: len(class_to_images[x]), 
                           reverse=True)
    print_class_distribution(sorted_classes, class_to_images)
    
    # 划分数据集
    print("\n划分数据集...")
    train_samples, valid_samples, test_samples, class_names = split_dataset(
        class_to_images,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        min_samples=args.min_samples
    )
    
    # 保存划分文件
    save_split_files(args.output_dir, train_samples, valid_samples, test_samples, class_names)
    
    print("\n" + "=" * 60)
    print("划分完成!")
    print("=" * 60)
    print("\n使用方法:")
    print(f"  1. 在 Hyper-Kvasir 上预训练:")
    print(f"     python main.py --data_dir {args.output_dir} --epochs 30")
    print()
    print(f"  2. 在 Gastrovision 上微调 (替换 --data_dir):")
    print(f"     python main.py --data_dir D:/codes/work-projects/Gastrovision_model \\")
    print(f"                    --resume <hyper_kvasir_best_model.pth> --freeze_backbone")


if __name__ == "__main__":
    main()
