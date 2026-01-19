"""
Qwen3-VL-4B-Thinking 推理与评估脚本

加载微调后的模型进行推理和评估

用法:
    python predict.py                                    # 使用默认模型评估测试集
    python predict.py --adapter_path output/checkpoints/final   # 指定 LoRA 适配器
    python predict.py --max_samples 100                  # 限制样本数
    python predict.py --split valid                      # 评估验证集
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from datetime import datetime

# ===================== 禁用 bitsandbytes（Windows 兼容性修复）=====================
from importlib.machinery import ModuleSpec
from unittest.mock import MagicMock

def _create_mock_module(name):
    """创建带有正确 __spec__ 的 mock 模块"""
    mock = MagicMock()
    mock.__name__ = name
    mock.__spec__ = ModuleSpec(name, None)
    mock.__path__ = []
    mock.__file__ = None
    mock.__loader__ = None
    mock.__package__ = name.rsplit('.', 1)[0] if '.' in name else name
    return mock

for mod_name in [
    'bitsandbytes', 'bitsandbytes.nn', 'bitsandbytes.nn.modules',
    'bitsandbytes.functional', 'bitsandbytes.autograd',
    'bitsandbytes.autograd._functions', 'bitsandbytes.optim',
    'bitsandbytes.cextension', 'bitsandbytes.cuda_setup',
    'bitsandbytes.cuda_setup.main',
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = _create_mock_module(mod_name)

import torch

from config import (
    MODEL_PATH, DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR,
    InferenceConfig, DEFAULT_INFERENCE_CONFIG,
    NUM_CLASSES, CLASS_TO_CHAR, CLASSIFICATION_PROMPT
)
from utils import (
    setup_logger,
    parse_prediction,
    compute_accuracy,
    compute_confusion_matrix,
    save_predictions,
    load_class_names,
    class_to_char
)


logger = setup_logger("predict")


def load_model_and_processor(config: InferenceConfig):
    """
    加载模型和处理器
    
    Args:
        config: 推理配置
    
    Returns:
        (model, processor)
    """
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    
    logger.info(f"加载模型: {config.model_path}")
    
    # 加载处理器（使用与训练一致的图像配置）
    processor = AutoProcessor.from_pretrained(
        config.model_path,
        min_pixels=config.image_config.min_pixels,
        max_pixels=config.image_config.max_pixels
    )
    logger.info(f"图像配置: min_pixels={config.image_config.min_pixels}, max_pixels={config.image_config.max_pixels}")
    
    # 模型加载参数（与训练保持一致）
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if config.torch_dtype == "bfloat16" else torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    
    # 加载基础模型
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.model_path,
        **model_kwargs
    )
    
    # 加载 LoRA 适配器（如果有）
    if config.adapter_path and Path(config.adapter_path).exists():
        logger.info(f"加载 LoRA 适配器: {config.adapter_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, config.adapter_path)
        model = model.merge_and_unload()  # 合并 LoRA 权重
        logger.info("✓ LoRA 适配器已合并")
    
    model.eval()
    
    return model, processor


def predict_single(
    model,
    processor,
    image_path: str,
    prompt: str = CLASSIFICATION_PROMPT,
    config: InferenceConfig = None
) -> Tuple[int, str, str]:
    """
    对单张图像进行预测
    
    Args:
        model: 模型
        processor: 处理器
        image_path: 图像路径
        prompt: 分类提示词
        config: 推理配置
    
    Returns:
        (预测类别, 预测字符, 原始输出)
    """
    from qwen_vl_utils import process_vision_info
    
    config = config or DEFAULT_INFERENCE_CONFIG
    
    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # 使用 apply_chat_template 生成文本（包含 generation prompt）
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 使用 qwen_vl_utils 处理图像
    image_inputs, video_inputs = process_vision_info(messages)
    
    # 使用 processor 处理图像和文本
    inputs = processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt"
    )
    
    # 移到设备
    inputs = inputs.to(model.device)
    
    # 生成
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            do_sample=config.do_sample,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    
    # 解码输出 - 只取新生成的 token
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    # 解析预测
    pred_class, pred_char = parse_prediction(output_text)
    
    return pred_class, pred_char, output_text


def evaluate_dataset(
    model,
    processor,
    jsonl_path: str,
    config: InferenceConfig = None,
    max_samples: Optional[int] = None
) -> Dict:
    """
    评估整个数据集
    
    Args:
        model: 模型
        processor: 处理器
        jsonl_path: JSONL 数据文件路径
        config: 推理配置
        max_samples: 最大样本数
    
    Returns:
        评估结果字典
    """
    config = config or DEFAULT_INFERENCE_CONFIG
    
    logger.info(f"评估数据集: {jsonl_path}")
    
    # 加载数据
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
            if max_samples and len(samples) >= max_samples:
                break
    
    logger.info(f"样本数: {len(samples)}")
    
    # 收集结果
    predictions = []
    labels = []
    image_paths = []
    raw_outputs = []
    failed_count = 0
    
    # 逐样本推理
    for sample in tqdm(samples, desc="推理"):
        messages = sample["messages"]
        
        # 提取图像路径和标签
        user_content = messages[0]["content"]
        image_path = None
        for item in user_content:
            if item.get("type") == "image":
                image_path = item.get("image")
                break
        
        if len(messages) >= 2:
            true_char = messages[1]["content"].strip().lower()
            from utils import char_to_class
            true_class = char_to_class(true_char)
        else:
            true_class = -1
        
        if not image_path:
            failed_count += 1
            continue
        
        try:
            pred_class, pred_char, output_text = predict_single(
                model, processor, image_path, config=config
            )
            
            predictions.append(pred_class)
            labels.append(true_class)
            image_paths.append(image_path)
            raw_outputs.append(output_text)
            
        except Exception as e:
            logger.warning(f"推理失败: {image_path}, 错误: {e}")
            predictions.append(-1)
            labels.append(true_class)
            image_paths.append(image_path)
            raw_outputs.append(str(e))
            failed_count += 1
    
    # 计算指标
    # 过滤无效预测
    valid_indices = [i for i, p in enumerate(predictions) if p >= 0]
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_labels = [labels[i] for i in valid_indices]
    
    if valid_predictions:
        metrics = compute_accuracy(valid_predictions, valid_labels)
        confusion_matrix = compute_confusion_matrix(valid_predictions, valid_labels)
    else:
        metrics = {"accuracy": 0.0, "mean_class_accuracy": 0.0}
        confusion_matrix = []
    
    # 汇总结果
    results = {
        "dataset": str(jsonl_path),
        "total_samples": len(samples),
        "valid_samples": len(valid_predictions),
        "failed_samples": failed_count,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix,
        "predictions_detail": {
            "predictions": predictions,
            "labels": labels,
            "image_paths": image_paths,
            "raw_outputs": raw_outputs[:20]  # 只保存前20个原始输出
        }
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL-4B-Thinking 推理评估")
    
    # 模型配置
    parser.add_argument("--model_path", type=str, default=str(MODEL_PATH),
                        help="基础模型路径")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="LoRA 适配器路径")
    
    # 数据配置
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR),
                        help="JSONL 数据目录")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "valid", "test"],
                        help="评估的数据集划分")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大样本数（用于调试）")
    
    # 生成配置
    parser.add_argument("--max_new_tokens", type=int, default=8,
                        help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="生成温度")
    
    # 输出配置
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR),
                        help="结果输出目录")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Qwen3-VL-4B-Thinking 推理评估")
    logger.info("=" * 60)
    
    # 配置
    config = InferenceConfig(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    logger.info(f"模型路径: {config.model_path}")
    logger.info(f"LoRA 适配器: {config.adapter_path or '无'}")
    logger.info(f"评估集: {args.split}")
    
    # 检查 CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"推理设备: {device}")
    
    # 数据文件
    jsonl_path = Path(args.data_dir) / f"{args.split}.jsonl"
    if not jsonl_path.exists():
        logger.error(f"数据文件不存在: {jsonl_path}")
        logger.error("请先运行 prepare_data.py 生成数据")
        return
    
    try:
        # 加载模型
        model, processor = load_model_and_processor(config)
        
        # 评估
        results = evaluate_dataset(
            model, processor, str(jsonl_path),
            config=config, max_samples=args.max_samples
        )
        
        # 打印结果
        logger.info("\n" + "=" * 60)
        logger.info("评估结果")
        logger.info("=" * 60)
        logger.info(f"总样本数: {results['total_samples']}")
        logger.info(f"有效样本: {results['valid_samples']}")
        logger.info(f"失败样本: {results['failed_samples']}")
        logger.info(f"准确率: {results['metrics']['accuracy']:.4f}")
        logger.info(f"平均类别准确率: {results['metrics']['mean_class_accuracy']:.4f}")
        
        # 保存结果
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_dir / f"eval_results_{args.split}_{timestamp}.json"
        
        # 移除不可序列化的内容
        results_to_save = {k: v for k, v in results.items() if k != "predictions_detail"}
        results_to_save["predictions_detail"] = {
            "predictions": results["predictions_detail"]["predictions"],
            "labels": results["predictions_detail"]["labels"]
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n结果已保存至: {result_file}")
        logger.info("\n✓ 评估完成!")
        
    except ImportError as e:
        logger.error(f"缺少依赖: {e}")
        logger.error("请安装: pip install transformers>=4.57.0 peft")
    except Exception as e:
        logger.error(f"评估失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
