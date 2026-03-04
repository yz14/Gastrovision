"""
Gastrovision 可视化工具

功能：
- 训练曲线可视化（损失、准确率、学习率）
- 混淆矩阵热力图
- 分类报告可视化
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')


def load_class_names_from_file(class_names_path: str) -> List[str]:
    """
    从 class_names.txt 加载类别名称（按顺序）
    
    Args:
        class_names_path: class_names.txt 文件路径
        
    Returns:
        类别名称列表（按索引排序）
    """
    class_names = []
    with open(class_names_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2:
                class_names.append(parts[1])
    return class_names


def plot_training_curves(
    log_path: str,
    output_dir: str = None,
    figsize: Tuple[int, int] = (14, 10)
) -> None:
    """
    绘制训练曲线（损失、准确率、学习率）
    
    Args:
        log_path: training_log.json 路径
        output_dir: 输出目录（默认与 log_path 同目录）
        figsize: 图形大小
    """
    log_path = Path(log_path)
    if output_dir is None:
        output_dir = log_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载日志
    with open(log_path, 'r', encoding='utf-8') as f:
        log = json.load(f)
    
    history = log['history']
    epochs = list(range(1, len(history['train_loss']) + 1))
    best_epoch = log.get('best_epoch', 0)
    best_acc = log.get('best_valid_acc', 0)
    
    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. 损失曲线
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss')
    ax1.plot(epochs, history['valid_loss'], 'r-', linewidth=2, label='Validation Loss')
    ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. 准确率曲线
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Training Accuracy')
    ax2.plot(epochs, history['valid_acc'], 'r-', linewidth=2, label='Validation Accuracy')
    ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax2.axhline(y=best_acc, color='g', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'Accuracy Curves (Best: {best_acc:.4f})', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 3. 学习率曲线
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['learning_rate'], 'purple', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. 训练 vs 验证差距（过拟合分析）
    ax4 = axes[1, 1]
    gap = [t - v for t, v in zip(history['train_acc'], history['valid_acc'])]
    colors = ['green' if g < 0.05 else 'orange' if g < 0.1 else 'red' for g in gap]
    ax4.bar(epochs, gap, color=colors, alpha=0.7)
    ax4.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Warning (5%)')
    ax4.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Overfitting (10%)')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Accuracy Gap (Train - Valid)', fontsize=12)
    ax4.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    output_path = output_dir / 'training_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 训练曲线已保存到: {output_path}")


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str] = None,
    output_path: str = None,
    figsize: Tuple[int, int] = None,
    normalize: bool = True
) -> None:
    """
    绘制混淆矩阵热力图
    
    Args:
        confusion_matrix: 混淆矩阵 (numpy 数组)
        class_names: 类别名称列表（按索引顺序）
        output_path: 输出路径
        figsize: 图形大小
        normalize: 是否归一化（按行）
    """
    import seaborn as sns
    
    num_classes = len(confusion_matrix)
    
    # 增加图形宽度以容纳右侧的 legend 表格
    if figsize is None:
        figsize = (max(14, num_classes * 0.5) + 5, max(10, num_classes * 0.4))
    
    # 归一化
    if normalize:
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        cm_normalized = confusion_matrix.astype('float') / np.maximum(row_sums, 1)
    else:
        cm_normalized = confusion_matrix
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 使用类别索引作为标签
    class_indices = [str(i) for i in range(num_classes)]
    
    # 使用热力图
    show_annot = num_classes <= 20
    fmt = '.2f' if normalize else 'd'
    
    sns.heatmap(
        cm_normalized,
        annot=show_annot,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_indices if num_classes <= 30 else False,
        yticklabels=class_indices if num_classes <= 30 else False,
        ax=ax,
        cbar_kws={'label': 'Recall' if normalize else 'Count'}
    )
    
    ax.set_xlabel('Predicted Label (Index)', fontsize=12)
    ax.set_ylabel('True Label (Index)', fontsize=12)
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''), 
                 fontsize=14, fontweight='bold')
    
    if num_classes <= 30:
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
    
    # 在图右侧添加类别索引到名称的 legend 表格
    if class_names and num_classes <= 30:
        legend_text = "Class Legend:\n" + "-" * 32 + "\n"
        for idx, name in enumerate(class_names):
            display_name = name[:25] + "..." if len(name) > 25 else name
            legend_text += f"{idx:2d}: {display_name}\n"
        
        plt.gcf().text(0.82, 0.5, legend_text, fontsize=7, family='monospace',
                       verticalalignment='center', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.subplots_adjust(right=0.78)
    else:
        plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 混淆矩阵已保存到: {output_path}")
    
    plt.close()


def plot_per_class_metrics(
    classification_report: Dict,
    output_path: str = None,
    figsize: Tuple[int, int] = (14, 8),
    top_n: int = None,
    class_names_order: List[str] = None
) -> None:
    """
    绘制每个类别的指标（Precision, Recall, F1）
    
    Args:
        classification_report: sklearn 的分类报告字典
        output_path: 输出路径
        figsize: 图形大小
        top_n: 只显示前 N 个类别（按 F1 排序）
        class_names_order: 指定类别顺序（如果提供则按此顺序显示）
    """
    # 提取类别指标
    classes = []
    precision = []
    recall = []
    f1 = []
    support = []
    
    for key, value in classification_report.items():
        if isinstance(value, dict) and 'precision' in value:
            classes.append(key)
            precision.append(value['precision'])
            recall.append(value['recall'])
            f1.append(value['f1-score'])
            support.append(value.get('support', 0))
    
    if not classes:
        print("警告: 分类报告中没有有效的类别数据")
        return
    
    # 构建类别到指标的映射
    class_data = {c: {'precision': p, 'recall': r, 'f1': f, 'support': s}
                  for c, p, r, f, s in zip(classes, precision, recall, f1, support)}
    
    # 确定显示顺序
    if class_names_order:
        # 按提供的顺序显示
        ordered_classes = [c for c in class_names_order if c in class_data]
    else:
        # 按 F1 排序
        ordered_classes = sorted(classes, key=lambda c: class_data[c]['f1'], reverse=True)
    
    if top_n:
        ordered_classes = ordered_classes[:top_n]
    
    # 提取排序后的数据
    classes = ordered_classes
    precision = [class_data[c]['precision'] for c in classes]
    recall = [class_data[c]['recall'] for c in classes]
    f1_scores = [class_data[c]['f1'] for c in classes]
    support = [class_data[c]['support'] for c in classes]
    
    # 使用类别索引作为 x 轴标签
    class_indices = list(range(len(classes)))
    x = np.arange(len(classes))
    width = 0.25
    
    # 创建图形（增加宽度以容纳 legend）
    fig, ax = plt.subplots(figsize=(figsize[0] + 4, figsize[1]))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', color='#3498db', alpha=0.8)
    bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Class Index', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics (ordered by class index)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_indices, fontsize=9)
    ax.legend(loc='upper right')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加样本数标注
    for i, s in enumerate(support):
        ax.annotate(f'n={int(s)}', xy=(x[i], 1.02), ha='center', fontsize=7, color='gray')
    
    # 在图右侧添加类别名称图例表格
    legend_text = "Class Legend:\n" + "-" * 30 + "\n"
    for idx, name in enumerate(classes):
        # 截断过长的名称
        display_name = name[:25] + "..." if len(name) > 25 else name
        legend_text += f"{idx:2d}: {display_name}\n"
    
    # 在图外右侧添加文本框
    plt.gcf().text(0.82, 0.5, legend_text, fontsize=8, family='monospace',
                   verticalalignment='center', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.subplots_adjust(right=0.78)  # 为 legend 留出空间
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 类别指标图已保存到: {output_path}")
    
    plt.close()


def create_test_results_summary(
    results_path: str,
    output_dir: str = None,
    class_names: List[str] = None
) -> None:
    """
    创建测试结果汇总图
    
    Args:
        results_path: test_results.json 路径
        output_dir: 输出目录
        class_names: 类别名称列表（按顺序）
    """
    results_path = Path(results_path)
    if output_dir is None:
        output_dir = results_path.parent
    else:
        output_dir = Path(output_dir)
    
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 1. 混淆矩阵（使用传入的类别名称）
    if 'confusion_matrix' in results:
        cm = np.array(results['confusion_matrix'])
        plot_confusion_matrix(
            cm,
            class_names=class_names,  # 使用传入的类别名称
            output_path=str(output_dir / 'confusion_matrix_normalized.png'),
            normalize=True
        )
    
    # 2. 每类指标（按类别名称顺序）
    if 'classification_report' in results:
        plot_per_class_metrics(
            results['classification_report'],
            class_names_order=class_names,  # 传入类别顺序
            output_path=str(output_dir / 'per_class_metrics.png')
        )
    
    # 3. 整体指标摘要图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = {
        'Top-1 Accuracy': results.get('accuracy', 0),
        'Top-5 Accuracy': results.get('top5_accuracy', 0),
        'Precision (Macro)': results.get('precision_macro', 0),
        'Recall (Macro)': results.get('recall_macro', 0),
        'F1 (Macro)': results.get('f1_macro', 0),
        'Precision (Weighted)': results.get('precision_weighted', 0),
        'Recall (Weighted)': results.get('recall_weighted', 0),
        'F1 (Weighted)': results.get('f1_weighted', 0),
    }
    
    names = list(metrics.keys())
    values = list(metrics.values())
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6',
              '#1abc9c', '#e67e22', '#34495e']
    
    bars = ax.barh(names, values, color=colors, alpha=0.8)
    
    # 添加数值标注
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=10)
    
    ax.set_xlim([0, 1.15])
    ax.set_xlabel('Score', fontsize=12)
    ax.set_title(f'Test Results Summary (N={results.get("num_samples", "?")}, Classes={results.get("num_classes", "?")})',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 测试结果汇总已保存到: {output_dir / 'test_summary.png'}")


# 命令行入口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="可视化训练和测试结果")
    parser.add_argument("--log", type=str, default="D:/codes/work-projects/Gastrovision_results/convnext_tiny/training_log.json",
                        help="训练日志路径")
    parser.add_argument("--results", type=str, default="D:/codes/work-projects/Gastrovision_results/convnext_tiny/test_results.json",
                        help="测试结果路径")
    parser.add_argument("--output", type=str, default='D:/codes/work-projects/Gastrovision_results/convnext_tiny',
                        help="输出目录")
    parser.add_argument("--class_names", type=str, default="./class_names.txt",
                        help="类别名称文件路径")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Gastrovision 可视化工具")
    print("=" * 50)
    
    # 加载类别名称
    class_names = None
    if Path(args.class_names).exists():
        class_names = load_class_names_from_file(args.class_names)
        print(f"✓ 加载 {len(class_names)} 个类别名称")
    
    # 绘制训练曲线
    if Path(args.log).exists():
        plot_training_curves(args.log, args.output)
    else:
        print(f"⚠ 找不到训练日志: {args.log}")
    
    # 绘制测试结果
    if Path(args.results).exists():
        create_test_results_summary(args.results, args.output, class_names)
    else:
        print(f"⚠ 找不到测试结果: {args.results}")
    
    print("\n完成!")
