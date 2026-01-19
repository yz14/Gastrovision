"""
数据准备脚本

将 train.txt/valid.txt/test.txt 转换为 Qwen3-VL 训练格式的 JSONL 文件

用法:
    python prepare_data.py                    # 转换所有数据
    python prepare_data.py --validate         # 只验证不转换
    python prepare_data.py --max_samples 100  # 限制样本数（用于调试）
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from config import (
    PROJECT_ROOT, DATA_DIR,
    TRAIN_TXT, VALID_TXT, TEST_TXT, CLASS_NAMES_TXT,
    CLASS_TO_CHAR, NUM_CLASSES,
    CLASSIFICATION_PROMPT
)
from utils import (
    setup_logger,
    load_txt_data,
    load_class_names,
    class_to_char
)


logger = setup_logger("prepare_data")


def create_conversation_format(
    image_path: str,
    class_idx: int,
    prompt: str = CLASSIFICATION_PROMPT,
    include_response: bool = True
) -> Dict:
    """
    创建符合 Qwen3-VL 对话格式的数据样本
    
    Args:
        image_path: 图像文件路径
        class_idx: 类别索引
        prompt: 分类提示词
        include_response: 是否包含助手回复（训练时为True，推理时为False）
    
    Returns:
        对话格式的字典
    """
    # 用户消息：图像 + 提示词
    user_content = [
        {"type": "image", "image": image_path},
        {"type": "text", "text": prompt}
    ]
    
    messages = [
        {"role": "user", "content": user_content}
    ]
    
    # 训练数据需要包含助手回复
    if include_response:
        char_label = class_to_char(class_idx)
        messages.append({
            "role": "assistant",
            "content": char_label
        })
    
    return {"messages": messages}


def convert_txt_to_jsonl(
    txt_path: Path,
    output_path: Path,
    max_samples: Optional[int] = None,
    validate_images: bool = False
) -> Tuple[int, int, int]:
    """
    将 txt 数据文件转换为 JSONL 格式
    
    Args:
        txt_path: 输入 txt 文件路径
        output_path: 输出 JSONL 文件路径
        max_samples: 最大样本数（用于调试）
        validate_images: 是否验证图像文件存在
    
    Returns:
        (总样本数, 成功转换数, 跳过数)
    """
    logger.info(f"处理文件: {txt_path}")
    
    # 加载数据
    samples = load_txt_data(txt_path)
    logger.info(f"  加载样本数: {len(samples)}")
    
    if max_samples:
        samples = samples[:max_samples]
        logger.info(f"  限制为: {max_samples} 个样本")
    
    # 统计
    total = len(samples)
    success = 0
    skipped = 0
    class_counts = [0] * NUM_CLASSES
    
    # 转换并写入
    with open(output_path, 'w', encoding='utf-8') as f:
        for img_path, class_idx in tqdm(samples, desc=f"转换 {txt_path.name}"):
            # 验证类别
            if class_idx < 0 or class_idx >= NUM_CLASSES:
                logger.warning(f"  跳过无效类别 {class_idx}: {img_path}")
                skipped += 1
                continue
            
            # 验证图像（可选）
            if validate_images and not Path(img_path).exists():
                logger.warning(f"  跳过不存在的图像: {img_path}")
                skipped += 1
                continue
            
            # 创建对话格式
            sample = create_conversation_format(img_path, class_idx)
            
            # 写入 JSONL
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            success += 1
            class_counts[class_idx] += 1
    
    logger.info(f"  成功转换: {success}, 跳过: {skipped}")
    
    # 显示类别分布
    logger.info(f"  类别分布: 最小 {min(class_counts)}, 最大 {max(class_counts)}")
    
    return total, success, skipped


def validate_jsonl(jsonl_path: Path) -> Dict:
    """
    验证 JSONL 文件格式
    
    Args:
        jsonl_path: JSONL 文件路径
    
    Returns:
        验证结果字典
    """
    logger.info(f"验证文件: {jsonl_path}")
    
    results = {
        "valid": True,
        "total_lines": 0,
        "valid_lines": 0,
        "invalid_lines": 0,
        "unique_classes": set(),
        "errors": []
    }
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            results["total_lines"] += 1
            
            try:
                data = json.loads(line)
                
                # 验证基本结构
                if "messages" not in data:
                    results["errors"].append(f"Line {line_num}: 缺少 'messages' 字段")
                    results["invalid_lines"] += 1
                    continue
                
                messages = data["messages"]
                
                # 验证用户消息
                if len(messages) < 1:
                    results["errors"].append(f"Line {line_num}: messages 为空")
                    results["invalid_lines"] += 1
                    continue
                
                user_msg = messages[0]
                if user_msg.get("role") != "user":
                    results["errors"].append(f"Line {line_num}: 第一条消息应为 user")
                    results["invalid_lines"] += 1
                    continue
                
                # 验证用户消息内容
                content = user_msg.get("content", [])
                has_image = any(c.get("type") == "image" for c in content)
                has_text = any(c.get("type") == "text" for c in content)
                
                if not has_image:
                    results["errors"].append(f"Line {line_num}: 缺少图像")
                    results["invalid_lines"] += 1
                    continue
                
                if not has_text:
                    results["errors"].append(f"Line {line_num}: 缺少文本提示")
                    results["invalid_lines"] += 1
                    continue
                
                # 验证助手回复（训练数据）
                if len(messages) >= 2:
                    assistant_msg = messages[1]
                    if assistant_msg.get("role") != "assistant":
                        results["errors"].append(f"Line {line_num}: 第二条消息应为 assistant")
                        results["invalid_lines"] += 1
                        continue
                    
                    # 记录类别
                    response = assistant_msg.get("content", "").strip().lower()
                    if response:
                        results["unique_classes"].add(response)
                
                results["valid_lines"] += 1
                
            except json.JSONDecodeError as e:
                results["errors"].append(f"Line {line_num}: JSON 解析错误 - {e}")
                results["invalid_lines"] += 1
    
    # 转换 set 为 list 便于打印
    results["unique_classes"] = sorted(list(results["unique_classes"]))
    results["valid"] = results["invalid_lines"] == 0
    
    return results


def main():
    parser = argparse.ArgumentParser(description="准备 Qwen3-VL 训练数据")
    parser.add_argument("--validate", action="store_true",
                        help="只验证已有的 JSONL 文件")
    parser.add_argument("--validate_images", action="store_true",
                        help="验证图像文件是否存在")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="限制每个数据集的样本数（用于调试）")
    parser.add_argument("--output_dir", type=str, default=str(DATA_DIR),
                        help="输出目录")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 定义输入输出映射
    datasets = [
        ("train", TRAIN_TXT, output_dir / "train.jsonl"),
        ("valid", VALID_TXT, output_dir / "valid.jsonl"),
        ("test", TEST_TXT, output_dir / "test.jsonl"),
    ]
    
    if args.validate:
        # 验证模式
        logger.info("=" * 50)
        logger.info("验证 JSONL 文件")
        logger.info("=" * 50)
        
        all_valid = True
        for name, _, jsonl_path in datasets:
            if jsonl_path.exists():
                results = validate_jsonl(jsonl_path)
                
                logger.info(f"\n{name.upper()} 验证结果:")
                logger.info(f"  总行数: {results['total_lines']}")
                logger.info(f"  有效行数: {results['valid_lines']}")
                logger.info(f"  无效行数: {results['invalid_lines']}")
                logger.info(f"  唯一类别数: {len(results['unique_classes'])}")
                logger.info(f"  类别字符: {results['unique_classes']}")
                
                if results['errors']:
                    logger.warning(f"  错误示例（前5个）:")
                    for err in results['errors'][:5]:
                        logger.warning(f"    {err}")
                
                if not results['valid']:
                    all_valid = False
            else:
                logger.warning(f"{name}: JSONL 文件不存在: {jsonl_path}")
                all_valid = False
        
        if all_valid:
            logger.info("\n✓ 所有文件验证通过!")
        else:
            logger.error("\n✗ 存在验证错误，请检查!")
    
    else:
        # 转换模式
        logger.info("=" * 50)
        logger.info("转换数据为 JSONL 格式")
        logger.info("=" * 50)
        
        # 检查输入文件
        for name, txt_path, _ in datasets:
            if not txt_path.exists():
                logger.error(f"输入文件不存在: {txt_path}")
                return
        
        # 加载类别名称（用于日志）
        if CLASS_NAMES_TXT.exists():
            class_names = load_class_names(CLASS_NAMES_TXT)
            logger.info(f"加载类别名称: {len(class_names)} 个类别")
        
        # 转换每个数据集
        summary = []
        for name, txt_path, jsonl_path in datasets:
            total, success, skipped = convert_txt_to_jsonl(
                txt_path,
                jsonl_path,
                max_samples=args.max_samples,
                validate_images=args.validate_images
            )
            summary.append((name, total, success, skipped, jsonl_path))
        
        # 打印摘要
        logger.info("\n" + "=" * 50)
        logger.info("转换完成摘要")
        logger.info("=" * 50)
        for name, total, success, skipped, jsonl_path in summary:
            logger.info(f"  {name:6s}: {success:5d} 样本 -> {jsonl_path}")
        
        logger.info("\n✓ 数据准备完成!")
        logger.info(f"输出目录: {output_dir}")
        
        # 自动验证
        logger.info("\n自动验证生成的文件...")
        for name, _, jsonl_path in datasets:
            if jsonl_path.exists():
                results = validate_jsonl(jsonl_path)
                status = "✓" if results['valid'] else "✗"
                logger.info(f"  {status} {name}: {results['valid_lines']} 有效样本")


if __name__ == "__main__":
    main()
