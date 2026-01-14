"""
Gastrovision 数据集分析工具

功能：
- 类别分布可视化
- 数据集划分统计
- 样本图像展示
"""

import os
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_split_stats(data_dir: str) -> Dict[str, Dict[str, int]]:
    """
    加载数据集划分统计
    
    Args:
        data_dir: 包含 train.txt, valid.txt, test.txt, class_names.txt 的目录
        
    Returns:
        {split_name: {class_name: count}}
    """
    data_dir = Path(data_dir)
    
    # 加载类别名称
    class_names = {}
    class_names_file = data_dir / "class_names.txt"
    if class_names_file.exists():
        with open(class_names_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    class_names[int(parts[0])] = parts[1]
    
    stats = {}
    
    for split in ['train', 'valid', 'test']:
        txt_file = data_dir / f"{split}.txt"
        if not txt_file.exists():
            continue
        
        class_counts = Counter()
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(' ', 1)
                if len(parts) == 2:
                    class_idx = int(parts[1])
                    class_name = class_names.get(class_idx, f"class_{class_idx}")
                    class_counts[class_name] += 1
        
        stats[split] = dict(class_counts)
    
    return stats


def plot_class_distribution(
    data_dir: str,
    output_path: str = None,
    figsize: Tuple[int, int] = (16, 10)
) -> None:
    """
    绘制类别分布柱状图
    
    Args:
        data_dir: 数据目录
        output_path: 输出路径
        figsize: 图形大小
    """
    data_dir = Path(data_dir)
    stats = load_split_stats(str(data_dir))
    
    if not stats:
        print("⚠ 没有找到数据集划分文件")
        return
    
    # 加载类别名称（按顺序）
    class_names = []
    class_names_file = data_dir / "class_names.txt"
    if class_names_file.exists():
        with open(class_names_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    class_names.append(parts[1])
    
    # 按 class_names.txt 顺序排列
    train_stats = stats.get('train', {})
    if class_names:
        all_classes = class_names
    else:
        all_classes = sorted(train_stats.keys())
    
    # 准备数据
    train_counts = [stats.get('train', {}).get(c, 0) for c in all_classes]
    valid_counts = [stats.get('valid', {}).get(c, 0) for c in all_classes]
    test_counts = [stats.get('test', {}).get(c, 0) for c in all_classes]
    
    # 使用类别索引作为 x 轴标签
    class_indices = list(range(len(all_classes)))
    x = np.arange(len(all_classes))
    width = 0.25
    
    # 创建图形（增加宽度以容纳 legend）
    fig, ax = plt.subplots(figsize=(figsize[0] + 4, figsize[1]))
    
    bars1 = ax.bar(x - width, train_counts, width, label='Train', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, valid_counts, width, label='Valid', color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x + width, test_counts, width, label='Test', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Class Index', fontsize=12)
    ax.set_ylabel('Sample Count', fontsize=12)
    ax.set_title('Dataset Class Distribution (ordered by class index)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_indices, fontsize=9)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加总数标注
    total_train = sum(train_counts)
    total_valid = sum(valid_counts)
    total_test = sum(test_counts)
    ax.text(0.02, 0.98, f'Total: Train={total_train}, Valid={total_valid}, Test={total_test}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 在图右侧添加类别名称图例表格
    legend_text = "Class Legend:\n" + "-" * 30 + "\n"
    for idx, name in enumerate(all_classes):
        display_name = name[:25] + "..." if len(name) > 25 else name
        legend_text += f"{idx:2d}: {display_name}\n"
    
    plt.gcf().text(0.82, 0.5, legend_text, fontsize=8, family='monospace',
                   verticalalignment='center', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.subplots_adjust(right=0.78)
    
    if output_path is None:
        output_path = data_dir / 'class_distribution.png'
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 类别分布图已保存到: {output_path}")


def plot_class_imbalance_analysis(
    data_dir: str,
    output_path: str = None,
    figsize: Tuple[int, int] = (18, 10)  # 增加宽度以容纳 legend
) -> None:
    """
    分析类别不平衡情况
    """
    data_dir = Path(data_dir)
    stats = load_split_stats(str(data_dir))
    train_stats = stats.get('train', {})
    
    if not train_stats:
        print("⚠ 没有找到训练集数据")
        return
    
    # 加载类别名称（按顺序）
    class_names = []
    class_names_file = data_dir / "class_names.txt"
    if class_names_file.exists():
        with open(class_names_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    class_names.append(parts[1])
    
    # 按 class_names.txt 顺序获取数据
    if class_names:
        classes = class_names
        counts = [train_stats.get(c, 0) for c in classes]
    else:
        sorted_classes = sorted(train_stats.items(), key=lambda x: x[1], reverse=True)
        classes = [c[0] for c in sorted_classes]
        counts = [c[1] for c in sorted_classes]
    
    # 计算统计量
    total = sum(counts)
    mean_count = total / len(counts) if counts else 0
    max_count = max(counts) if counts else 0
    min_count = min(counts) if counts else 0
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    # 使用类别索引
    class_indices = [str(i) for i in range(len(classes))]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 1. 柱状图（对数刻度）- 使用类别索引
    ax1 = axes[0]
    colors = ['#e74c3c' if c < mean_count * 0.3 else '#f39c12' if c < mean_count else '#2ecc71' 
              for c in counts]
    ax1.barh(class_indices, counts, color=colors, alpha=0.8)
    ax1.axvline(x=mean_count, color='blue', linestyle='--', alpha=0.7, 
                label=f'Mean ({mean_count:.0f})')
    ax1.set_xlabel('Sample Count', fontsize=12)
    ax1.set_ylabel('Class Index', fontsize=12)
    ax1.set_title('Class Sample Counts (Training Set)', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.set_xscale('log')
    ax1.invert_yaxis()
    ax1.tick_params(axis='y', labelsize=9)
    
    # 2. 饼图（只显示前10和其他）- 使用类别索引
    ax2 = axes[1]
    top_n = 10
    top_indices = list(range(min(top_n, len(classes))))
    top_counts = counts[:len(top_indices)]
    other_count = sum(counts[len(top_indices):])
    
    if other_count > 0:
        pie_labels = [str(i) for i in top_indices] + ['Others']
        pie_sizes = top_counts + [other_count]
    else:
        pie_labels = [str(i) for i in top_indices]
        pie_sizes = top_counts
    
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(pie_labels)))
    wedges, texts, autotexts = ax2.pie(
        pie_sizes, labels=pie_labels, autopct='%1.1f%%',
        colors=colors_pie, startangle=90, pctdistance=0.75
    )
    
    for text in texts:
        text.set_fontsize(9)
    for autotext in autotexts:
        autotext.set_fontsize(7)
    
    ax2.set_title(f'Dataset Composition\n(Imbalance Ratio: {imbalance_ratio:.1f}x)', 
                  fontsize=14, fontweight='bold')
    
    # 添加类别索引到名称的 legend 表格
    legend_text = "Class Legend:\n" + "-" * 30 + "\n"
    for idx, name in enumerate(classes):
        display_name = name[:22] + "..." if len(name) > 22 else name
        legend_text += f"{idx:2d}: {display_name}\n"
    
    plt.gcf().text(1.02, 0.5, legend_text, fontsize=7, family='monospace',
                   verticalalignment='center', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   transform=ax2.transAxes)
    
    # 添加统计信息
    stats_text = f"Total: {len(classes)} classes, {total} samples\nMax: {max_count} | Min: {min_count}"
    fig.text(0.5, 0.02, stats_text, fontsize=9, ha='center', 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.subplots_adjust(right=0.75)  # 为 legend 留出空间
    
    if output_path is None:
        output_path = data_dir / 'class_imbalance_analysis.png'
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 类别不平衡分析图已保存到: {output_path}")


def print_dataset_summary(data_dir: str) -> None:
    """打印数据集摘要"""
    stats = load_split_stats(data_dir)
    
    print("\n" + "=" * 60)
    print("数据集摘要")
    print("=" * 60)
    
    for split_name, split_stats in stats.items():
        total = sum(split_stats.values())
        num_classes = len(split_stats)
        print(f"\n{split_name.upper()}:")
        print(f"  总样本数: {total}")
        print(f"  类别数: {num_classes}")
        
        counts = list(split_stats.values())
        if counts:
            print(f"  每类平均: {np.mean(counts):.1f}")
            print(f"  最大类: {max(counts)}")
            print(f"  最小类: {min(counts)}")
    
    print("\n" + "=" * 60)


# 命令行入口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="分析数据集分布")
    parser.add_argument("--data_dir", type=str, 
                        default="D:/codes/work-projects/Gastrovision_model",
                        help="数据目录")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录")
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.data_dir
    
    print("=" * 50)
    print("Gastrovision 数据集分析")
    print("=" * 50)
    
    # 打印摘要
    print_dataset_summary(args.data_dir)
    
    # 绘制图表
    plot_class_distribution(
        args.data_dir,
        output_path=str(Path(output_dir) / 'class_distribution.png')
    )
    
    plot_class_imbalance_analysis(
        args.data_dir,
        output_path=str(Path(output_dir) / 'class_imbalance_analysis.png')
    )
    
    print("\n完成!")
