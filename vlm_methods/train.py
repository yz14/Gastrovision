"""
Qwen3-VL-4B-Thinking 微调训练脚本

使用 LoRA 微调 Qwen3-VL-4B-Thinking 进行内镜图像分类

用法:
    python train.py                           # 使用默认配置训练
    python train.py --num_epochs 5            # 指定训练轮数
    python train.py --batch_size 4            # 指定批次大小
    python train.py --dry_run --max_samples 50  # 干运行测试
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# ===================== 禁用 bitsandbytes（Windows 兼容性修复） =====================
# 在导入 PEFT 之前 mock bitsandbytes 模块，避免 Windows 上的 CUDA 兼容性问题
import sys
from types import ModuleType
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

# 注册 mock 模块 - 需要在导入 PEFT 之前完成
for mod_name in [
    'bitsandbytes',
    'bitsandbytes.nn',
    'bitsandbytes.nn.modules',
    'bitsandbytes.functional',
    'bitsandbytes.autograd',
    'bitsandbytes.autograd._functions',
    'bitsandbytes.optim',
    'bitsandbytes.cextension',
    'bitsandbytes.cuda_setup',
    'bitsandbytes.cuda_setup.main',
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = _create_mock_module(mod_name)

import torch
from torch.utils.data import Dataset

from config import (
    MODEL_PATH, DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR,
    TrainingConfig, DEFAULT_TRAINING_CONFIG,
    NUM_CLASSES, CLASS_TO_CHAR
)
from utils import setup_logger, class_to_char, char_to_class, parse_prediction


logger = setup_logger("train")


# ===================== 数据集类 =====================

class VLMDataset(Dataset):
    """
    Qwen3-VL 训练数据集
    
    从 JSONL 文件加载数据
    """
    
    def __init__(self, jsonl_path: str, max_samples: Optional[int] = None):
        """
        Args:
            jsonl_path: JSONL 文件路径
            max_samples: 最大样本数（用于调试）
        """
        self.samples = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))
                if max_samples and len(self.samples) >= max_samples:
                    break
        
        logger.info(f"加载数据集: {jsonl_path}, 样本数: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


# ===================== 训练函数 =====================

def train(args):
    """主训练函数"""
    
    logger.info("=" * 60)
    logger.info("Qwen3-VL-4B-Thinking 微调训练")
    logger.info("=" * 60)
    
    # 配置 - 使用嵌套配置类
    from config import LoRAConfig, OptimizerConfig
    
    lora_cfg = LoRAConfig(
        use_lora=not args.full_finetune,
        r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    
    optimizer_cfg = OptimizerConfig(
        learning_rate=args.learning_rate
    )
    
    config = TrainingConfig(
        model_path=str(args.model_path),
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lora_config=lora_cfg,
        optimizer_config=optimizer_cfg,
        output_dir=str(args.output_dir)
    )
    
    logger.info(f"模型路径: {config.model_path}")
    logger.info(f"使用 LoRA: {config.use_lora}")
    if config.use_lora:
        logger.info(f"  LoRA r={config.lora_config.r}, alpha={config.lora_config.lora_alpha}, dropout={config.lora_config.lora_dropout}")
    logger.info(f"训练轮数: {config.num_epochs}")
    logger.info(f"批次大小: {config.batch_size}")
    logger.info(f"学习率: {config.learning_rate}")
    logger.info(f"数据增强: {config.augmentation_config.enable_augmentation}")
    if config.augmentation_config.enable_augmentation:
        logger.info(f"  水平翻转={config.augmentation_config.horizontal_flip}, 颜色抖动={config.augmentation_config.color_jitter}, 旋转={config.augmentation_config.random_rotation}")
    
    # 检查 CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"训练设备: {device}")
    
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 加载数据集
    train_jsonl = Path(args.data_dir) / "train.jsonl"
    valid_jsonl = Path(args.data_dir) / "valid.jsonl"
    
    if not train_jsonl.exists():
        logger.error(f"训练数据不存在: {train_jsonl}")
        logger.error("请先运行 prepare_data.py 生成训练数据")
        return
    
    # 干运行检查
    if args.dry_run:
        logger.info("\n[干运行模式] 验证配置和数据...")
        
        train_dataset = VLMDataset(str(train_jsonl), max_samples=args.max_samples or 10)
        logger.info(f"训练集样本示例: {json.dumps(train_dataset[0], ensure_ascii=False, indent=2)}")
        
        logger.info("\n[干运行模式] 尝试加载模型...")
        
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(
                config.model_path,
                min_pixels=config.image_config.min_pixels,
                max_pixels=config.image_config.max_pixels
            )
            logger.info(f"✓ Processor 加载成功 (min_pixels={config.image_config.min_pixels}, max_pixels={config.image_config.max_pixels})")
            
            from transformers import Qwen3VLForConditionalGeneration
            logger.info("尝试加载模型（这可能需要一些时间）...")
            
            # 模型加载参数
            model_kwargs = {
                "torch_dtype": torch.bfloat16 if config.torch_dtype == "bfloat16" else torch.float16,
                "device_map": "auto",
                "trust_remote_code": config.trust_remote_code,
            }
            if config.use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("启用 Flash Attention 2")
            
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.model_path,
                **model_kwargs
            )
            logger.info("✓ 模型加载成功")
            
            # 显示模型信息
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"模型参数量: {total_params / 1e9:.2f}B")
            
            if config.use_lora:
                from peft import LoraConfig, get_peft_model
                lora_config = LoraConfig(
                    r=config.lora_config.r,
                    lora_alpha=config.lora_config.lora_alpha,
                    lora_dropout=config.lora_config.lora_dropout,
                    target_modules=config.lora_config.target_modules,
                    task_type="CAUSAL_LM"
                )
                model = get_peft_model(model, lora_config)
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"✓ LoRA 配置成功，可训练参数: {trainable_params / 1e6:.2f}M")
            
            logger.info("\n✓ 干运行验证通过！可以开始正式训练。")
            
        except ImportError as e:
            logger.error(f"缺少依赖: {e}")
            logger.error("请安装: pip install transformers>=4.57.0 peft trl")
        except Exception as e:
            logger.error(f"加载失败: {e}")
            import traceback
            traceback.print_exc()
        
        return
    
    # 正式训练
    logger.info("\n开始正式训练...")
    
    try:
        # 导入训练依赖
        from transformers import (
            AutoProcessor,
            Qwen3VLForConditionalGeneration,
            TrainingArguments,
            Trainer
        )
        from peft import LoraConfig, get_peft_model
        
        # 加载模型和处理器
        logger.info("加载模型和处理器...")
        
        # 加载 processor 并应用图像配置
        processor = AutoProcessor.from_pretrained(
            config.model_path,
            min_pixels=config.image_config.min_pixels,
            max_pixels=config.image_config.max_pixels
        )
        logger.info(f"图像配置: min_pixels={config.image_config.min_pixels}, max_pixels={config.image_config.max_pixels}")
        
        # 模型加载参数
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if config.torch_dtype == "bfloat16" else torch.float16,
            "device_map": "auto",
            "trust_remote_code": config.trust_remote_code,
        }
        
        # Flash Attention 2 加速（如果可用）
        if config.use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("启用 Flash Attention 2 加速")
            except Exception as e:
                logger.warning(f"Flash Attention 2 不可用: {e}")
        
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.model_path,
            **model_kwargs
        )
        
        # 配置 LoRA
        if config.use_lora:
            logger.info("配置 LoRA 适配器...")
            lora_config = LoraConfig(
                r=config.lora_config.r,
                lora_alpha=config.lora_config.lora_alpha,
                lora_dropout=config.lora_config.lora_dropout,
                target_modules=config.lora_config.target_modules,
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        # 启用输入梯度（解决 gradient_checkpointing 与 VLM 输入不兼容的问题）
        if config.gradient_checkpointing:
            # 对于 PEFT 模型，需要启用输入梯度
            if hasattr(model, 'enable_input_require_grads'):
                model.enable_input_require_grads()
            else:
                # 手动设置
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
        # 加载数据集
        train_dataset = VLMDataset(str(train_jsonl), max_samples=args.max_samples)
        valid_dataset = VLMDataset(str(valid_jsonl), max_samples=args.max_samples) if valid_jsonl.exists() else None
        
        # 数据增强函数
        def apply_augmentation(image, aug_config):
            """对图像应用数据增强"""
            if not aug_config.enable_augmentation:
                return image
            
            from PIL import Image
            import random
            from torchvision import transforms
            
            # 确保是 PIL Image
            if not isinstance(image, Image.Image):
                return image
            
            # 水平翻转
            if aug_config.horizontal_flip and random.random() < aug_config.horizontal_flip_prob:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
            # 颜色抖动
            if aug_config.color_jitter:
                color_jitter = transforms.ColorJitter(
                    brightness=aug_config.brightness,
                    contrast=aug_config.contrast,
                    saturation=aug_config.saturation,
                    hue=aug_config.hue
                )
                image = color_jitter(image)
            
            # 随机旋转
            if aug_config.random_rotation:
                angle = random.uniform(-aug_config.rotation_degrees, aug_config.rotation_degrees)
                image = image.rotate(angle, fillcolor=(0, 0, 0))
            
            return image
        
        # 数据整理函数
        def collate_fn(batch):
            """将批次数据转换为模型输入格式"""
            from qwen_vl_utils import process_vision_info
            from PIL import Image
            
            all_input_ids = []
            all_attention_mask = []
            all_labels = []
            all_pixel_values = []
            all_image_grid_thw = []
            
            for item in batch:
                messages = item["messages"]
                
                # 分别生成带答案和不带答案的文本，用于计算 prompt 长度
                # 完整对话（包含答案）
                full_text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                
                # 只有用户消息（用于计算 prompt 长度）
                user_messages = [messages[0]]  # 只取 user message
                prompt_text = processor.apply_chat_template(
                    user_messages,
                    tokenize=False,
                    add_generation_prompt=True  # 添加生成提示
                )
                
                # 使用 qwen-vl-utils 处理图像
                image_inputs, video_inputs = process_vision_info(messages)
                
                # 应用数据增强（仅对训练数据）
                if image_inputs and config.augmentation_config.enable_augmentation:
                    augmented_images = []
                    for img in image_inputs:
                        if isinstance(img, Image.Image):
                            img = apply_augmentation(img, config.augmentation_config)
                        augmented_images.append(img)
                    image_inputs = augmented_images
                
                # 使用 processor 处理完整文本（包含答案）
                full_inputs = processor(
                    text=[full_text],
                    images=image_inputs if image_inputs else None,
                    videos=video_inputs if video_inputs else None,
                    padding=True,
                    return_tensors="pt"
                )
                
                # 处理 prompt（不包含答案）来计算长度
                prompt_inputs = processor(
                    text=[prompt_text],
                    images=image_inputs if image_inputs else None,
                    videos=video_inputs if video_inputs else None,
                    padding=True,
                    return_tensors="pt"
                )
                
                # 计算 prompt 长度
                prompt_len = prompt_inputs["input_ids"].size(1)
                
                # 创建 labels：prompt 部分设为 -100（忽略），只计算答案部分的 loss
                labels = full_inputs["input_ids"].clone()
                labels[0, :prompt_len] = -100  # 忽略 prompt 部分
                
                all_input_ids.append(full_inputs["input_ids"])
                all_attention_mask.append(full_inputs["attention_mask"])
                all_labels.append(labels)
                
                if "pixel_values" in full_inputs:
                    all_pixel_values.append(full_inputs["pixel_values"])
                if "image_grid_thw" in full_inputs:
                    all_image_grid_thw.append(full_inputs["image_grid_thw"])
            
            # 合并批次
            max_len = max(x.size(1) for x in all_input_ids)
            
            # Pad sequences
            padded_input_ids = []
            padded_attention_mask = []
            padded_labels = []
            
            for i in range(len(all_input_ids)):
                pad_len = max_len - all_input_ids[i].size(1)
                if pad_len > 0:
                    padded_input_ids.append(torch.cat([
                        all_input_ids[i],
                        torch.full((1, pad_len), processor.tokenizer.pad_token_id)
                    ], dim=1))
                    padded_attention_mask.append(torch.cat([
                        all_attention_mask[i],
                        torch.zeros((1, pad_len), dtype=torch.long)
                    ], dim=1))
                    padded_labels.append(torch.cat([
                        all_labels[i],
                        torch.full((1, pad_len), -100)  # -100 for ignored tokens
                    ], dim=1))
                else:
                    padded_input_ids.append(all_input_ids[i])
                    padded_attention_mask.append(all_attention_mask[i])
                    padded_labels.append(all_labels[i])
            
            result = {
                "input_ids": torch.cat(padded_input_ids, dim=0),
                "attention_mask": torch.cat(padded_attention_mask, dim=0),
                "labels": torch.cat(padded_labels, dim=0)
            }
            
            # 处理图像数据
            if all_pixel_values:
                result["pixel_values"] = torch.cat(all_pixel_values, dim=0)
            if all_image_grid_thw:
                result["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)
            
            return result
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.optimizer_config.learning_rate,
            weight_decay=config.optimizer_config.weight_decay,
            warmup_ratio=config.scheduler_config.warmup_ratio,
            max_grad_norm=config.optimizer_config.max_grad_norm,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            eval_steps=config.eval_steps if valid_dataset else None,
            eval_strategy="steps" if valid_dataset else "no",
            save_total_limit=config.save_total_limit,
            bf16=config.bf16,
            fp16=config.fp16,
            gradient_checkpointing=config.gradient_checkpointing,
            dataloader_pin_memory=config.dataloader_pin_memory,
            dataloader_num_workers=config.dataloader_num_workers,
            remove_unused_columns=config.remove_unused_columns,
            report_to=config.report_to,
            seed=config.seed,
        )
        
        # 定义评估指标计算函数
        def compute_metrics(eval_preds):
            """计算评估指标（准确率）
            
            注意：对于自回归语言模型，logits[i] 预测的是 labels[i+1]
            因此需要进行偏移对齐
            """
            predictions, labels = eval_preds
            
            # predictions 已经是 preprocess_logits_for_metrics 处理后的 argmax 结果
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            
            pred_ids = predictions  # 已经是 token ids，shape: [batch, seq_len]
            
            # 对于自回归模型：pred_ids[i] 对应 labels[i+1]
            # 需要偏移对齐：比较 pred_ids[:-1] 与 labels[1:]
            correct = 0
            total = 0
            
            for pred, label in zip(pred_ids, labels):
                # 偏移：pred[i] 预测的是 label[i+1]
                # 所以比较 pred[:-1] 与 label[1:]
                shifted_pred = pred[:-1]
                shifted_label = label[1:]
                
                # 找到标签中非 -100 的位置（即答案部分）
                valid_mask = shifted_label != -100
                if valid_mask.sum() > 0:
                    valid_pred = shifted_pred[valid_mask]
                    valid_label = shifted_label[valid_mask]
                    correct += (valid_pred == valid_label).sum().item()
                    total += valid_mask.sum().item()
            
            accuracy = correct / total if total > 0 else 0.0
            
            return {
                "accuracy": accuracy,
                "correct_tokens": correct,
                "total_tokens": total
            }
        
        # 定义 preprocess_logits_for_metrics 以减少内存使用
        def preprocess_logits_for_metrics(logits, labels):
            """预处理 logits 以减少内存使用"""
            if isinstance(logits, tuple):
                logits = logits[0]
            # 只保留 argmax 结果，不保留完整 logits
            return logits.argmax(dim=-1)
        
        # 创建 Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=collate_fn,
            compute_metrics=compute_metrics if valid_dataset else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics if valid_dataset else None,
        )
        
        # 开始训练
        logger.info("开始训练...")
        trainer.train()
        
        # 保存模型
        logger.info("保存模型...")
        final_dir = Path(config.output_dir) / "final"
        
        if config.use_lora:
            model.save_pretrained(final_dir)
            logger.info(f"LoRA 适配器已保存到: {final_dir}")
        else:
            trainer.save_model(final_dir)
            logger.info(f"模型已保存到: {final_dir}")
        
        # 保存处理器
        processor.save_pretrained(final_dir)
        
        logger.info("\n✓ 训练完成!")
        
    except ImportError as e:
        logger.error(f"缺少依赖: {e}")
        logger.error("请安装: pip install transformers>=4.57.0 peft trl")
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL-4B-Thinking 微调训练")
    
    # 模型配置
    parser.add_argument("--model_path", type=str, default=str(MODEL_PATH),
                        help="模型路径")
    parser.add_argument("--full_finetune", action="store_true",
                        help="全量微调（不使用 LoRA）")
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128,
                        help="LoRA alpha")
    
    # 训练配置
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率")
    
    # 数据配置
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR),
                        help="JSONL 数据目录")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大样本数（用于调试）")
    
    # 输出配置
    parser.add_argument("--output_dir", type=str, default=str(CHECKPOINT_DIR),
                        help="检查点输出目录")
    
    # 调试选项
    parser.add_argument("--dry_run", action="store_true",
                        help="干运行：只验证配置和数据，不实际训练")
    
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()
