"""
Qwen3-VL-4B-Thinking 工具函数

包含：
- 类别-字符映射函数
- 日志配置
- 指标计算函数
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
from collections import Counter

from config import CLASS_TO_CHAR, CHAR_TO_CLASS, NUM_CLASSES


# ===================== 日志配置 =====================

def setup_logger(
    name: str = "qwen3vl",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    配置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 可选的日志文件路径
    
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # 文件处理器（可选）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


# ===================== 类别映射函数 =====================

def class_to_char(class_idx: int) -> str:
    """
    将类别索引转换为字符
    
    Args:
        class_idx: 类别索引 (0-26)
    
    Returns:
        对应的字符 (a-z 或 '0')
    
    Raises:
        ValueError: 如果类别索引超出范围
    """
    if class_idx not in CLASS_TO_CHAR:
        raise ValueError(f"类别索引 {class_idx} 超出范围 [0, {NUM_CLASSES-1}]")
    return CLASS_TO_CHAR[class_idx]


def char_to_class(char: str) -> int:
    """
    将字符转换为类别索引
    
    Args:
        char: 字符 (a-z 或 '0')
    
    Returns:
        对应的类别索引 (0-26)
    
    Raises:
        ValueError: 如果字符不在映射表中
    """
    char = char.lower().strip()
    if char not in CHAR_TO_CLASS:
        raise ValueError(f"字符 '{char}' 不在映射表中")
    return CHAR_TO_CLASS[char]


def parse_prediction(output_text: str) -> Tuple[int, str]:
    """
    解析模型输出，提取预测类别
    
    Args:
        output_text: 模型生成的文本
    
    Returns:
        (类别索引, 提取到的字符)
        如果无法解析，返回 (-1, '')
    """
    # 清理输出
    text = output_text.strip().lower()
    
    # 尝试提取第一个有效字符
    for char in text:
        if char in CHAR_TO_CLASS:
            return CHAR_TO_CLASS[char], char
    
    # 无法解析
    return -1, ''


# ===================== 数据加载函数 =====================

def load_txt_data(txt_path: Union[str, Path]) -> List[Tuple[str, int]]:
    """
    加载 txt 格式的数据文件
    
    Args:
        txt_path: txt 文件路径，每行格式: 图像路径 类别索引
    
    Returns:
        [(图像路径, 类别索引), ...] 列表
    """
    samples = []
    txt_path = Path(txt_path)
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.rsplit(' ', 1)  # 从右边分割，处理路径中可能有空格
            if len(parts) != 2:
                continue
            
            img_path, class_idx = parts
            try:
                class_idx = int(class_idx)
                samples.append((img_path, class_idx))
            except ValueError:
                continue
    
    return samples


def load_class_names(class_names_path: Union[str, Path]) -> List[str]:
    """
    加载类别名称文件
    
    Args:
        class_names_path: class_names.txt 路径
    
    Returns:
        类别名称列表
    """
    class_names = []
    with open(class_names_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                class_names.append(line)
    return class_names


# ===================== 评估指标 =====================

def compute_accuracy(
    predictions: List[int],
    labels: List[int]
) -> Dict[str, float]:
    """
    计算准确率指标
    
    Args:
        predictions: 预测类别列表
        labels: 真实类别列表
    
    Returns:
        包含各种准确率指标的字典
    """
    assert len(predictions) == len(labels), "预测和标签数量不匹配"
    
    total = len(predictions)
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    
    # 总体准确率
    accuracy = correct / total if total > 0 else 0.0
    
    # 按类别统计
    class_correct = Counter()
    class_total = Counter()
    
    for pred, label in zip(predictions, labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
    
    # 每类准确率
    per_class_acc = {}
    for cls in range(NUM_CLASSES):
        if class_total[cls] > 0:
            per_class_acc[cls] = class_correct[cls] / class_total[cls]
        else:
            per_class_acc[cls] = 0.0
    
    # 平均每类准确率（不考虑类别样本数）
    mean_class_acc = sum(per_class_acc.values()) / NUM_CLASSES
    
    return {
        'accuracy': accuracy,
        'mean_class_accuracy': mean_class_acc,
        'per_class_accuracy': per_class_acc,
        'total_samples': total,
        'correct_samples': correct
    }


def compute_confusion_matrix(
    predictions: List[int],
    labels: List[int]
) -> List[List[int]]:
    """
    计算混淆矩阵
    
    Args:
        predictions: 预测类别列表
        labels: 真实类别列表
    
    Returns:
        NUM_CLASSES x NUM_CLASSES 的混淆矩阵
        matrix[i][j] = 真实类别 i 被预测为 j 的次数
    """
    matrix = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    
    for pred, label in zip(predictions, labels):
        if 0 <= label < NUM_CLASSES and 0 <= pred < NUM_CLASSES:
            matrix[label][pred] += 1
    
    return matrix


def save_predictions(
    predictions: List[int],
    labels: List[int],
    image_paths: List[str],
    output_path: Union[str, Path],
    class_names: Optional[List[str]] = None
):
    """
    保存预测结果到 JSON 文件
    
    Args:
        predictions: 预测类别列表
        labels: 真实类别列表
        image_paths: 图像路径列表
        output_path: 输出文件路径
        class_names: 可选的类别名称列表
    """
    results = []
    for pred, label, img_path in zip(predictions, labels, image_paths):
        result = {
            'image_path': img_path,
            'predicted_class': pred,
            'true_class': label,
            'predicted_char': CLASS_TO_CHAR.get(pred, '?'),
            'true_char': CLASS_TO_CHAR.get(label, '?'),
            'correct': pred == label
        }
        
        if class_names:
            if 0 <= pred < len(class_names):
                result['predicted_name'] = class_names[pred]
            if 0 <= label < len(class_names):
                result['true_name'] = class_names[label]
        
        results.append(result)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # 测试工具函数
    print("=" * 50)
    print("工具函数测试")
    print("=" * 50)
    
    # 测试类别映射
    print("\n类别映射测试:")
    for i in [0, 13, 25, 26]:
        char = class_to_char(i)
        back = char_to_class(char)
        print(f"  类别 {i} -> 字符 '{char}' -> 类别 {back}")
    
    # 测试预测解析
    print("\n预测解析测试:")
    test_outputs = ["b", "The answer is c", "a\n", "  z  ", "invalid"]
    for output in test_outputs:
        cls, char = parse_prediction(output)
        print(f"  '{output}' -> 类别 {cls}, 字符 '{char}'")
    
    print("\n✓ 工具函数测试完成!")
