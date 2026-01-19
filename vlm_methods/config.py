"""
Qwen3-VL-4B-Thinking 内镜图像分类配置

包含：
- 模型路径配置
- 类别映射表
- 图像处理配置
- 训练超参数（完整版）
- 数据路径配置
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ===================== 路径配置 =====================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# VLM 方法目录
VLM_DIR = Path(__file__).parent

# 模型路径（已下载的本地模型）
MODEL_PATH = VLM_DIR / "qw3vl4B" / "Qwen" / "Qwen3-VL-4B-Thinking"

# 数据文件路径
TRAIN_TXT = PROJECT_ROOT / "train.txt"
VALID_TXT = PROJECT_ROOT / "valid.txt"
TEST_TXT = PROJECT_ROOT / "test.txt"
CLASS_NAMES_TXT = PROJECT_ROOT / "class_names.txt"

# 输出目录
OUTPUT_DIR = VLM_DIR / "output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
DATA_DIR = VLM_DIR / "data"
LOG_DIR = OUTPUT_DIR / "logs"

# 确保目录存在
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


# ===================== 类别映射 =====================

# 类别索引到字符的映射 (0-25 -> a-z, 26 -> '0')
CLASS_TO_CHAR: Dict[int, str] = {
    0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h',
    8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p',
    16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x',
    24: 'y', 25: 'z', 26: '0'
}

# 字符到类别索引的映射（反向）
CHAR_TO_CLASS: Dict[str, int] = {v: k for k, v in CLASS_TO_CHAR.items()}

# 类别数量
NUM_CLASSES = 27


# ===================== 提示词模板 =====================

# 用于训练和推理的提示词（详细版，推荐）
CLASSIFICATION_PROMPT = (
    "This is an endoscopic image. Classify it into one of 27 categories. "
    "Output ONLY a single character (a-z for classes 0-25, or '0' for class 26). "
    "Do not output any other text or explanation."
)

# 简化版提示词（可用于对比实验）
SIMPLE_PROMPT = "Classify this endoscopic image. Output only one character (a-z or 0)."

# 中文提示词（可选）
CHINESE_PROMPT = (
    "这是一张内镜图像。请将其分类为27个类别之一。"
    "只输出一个字符（a-z对应类别0-25，'0'对应类别26）。"
    "不要输出任何其他文字。"
)


# ===================== 图像处理配置 =====================

@dataclass
class ImageConfig:
    """图像处理配置
    
    Qwen3-VL 使用 patch_size=16，图像尺寸会被调整为 32 的倍数
    """
    
    # 图像尺寸控制（基于像素总数）
    # Qwen3-VL 推荐通过 min_pixels/max_pixels 控制
    min_pixels: int = 224 * 224  # 最小像素数 (256x256 = 65536)
    max_pixels: int = 256 * 256  # 最大像素数 (512x512 = 262144)
    
    # 或者直接指定尺寸（会被调整为 32 的倍数）
    # 如果设置了这两个值，将覆盖 min_pixels/max_pixels
    resized_height: Optional[int] = None  # 例如: 448
    resized_width: Optional[int] = None   # 例如: 448
    
    # 图像预处理
    do_resize: bool = True
    do_center_crop: bool = False
    do_normalize: bool = True
    
    # 归一化参数（ImageNet 默认值）
    image_mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    image_std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
    
    # 图像格式
    image_format: str = "RGB"


# ===================== 数据增强配置 =====================

@dataclass
class AugmentationConfig:
    """数据增强配置
    
    注意：VLM 微调通常使用较轻的数据增强，避免破坏图像语义
    """
    
    # 是否启用数据增强
    enable_augmentation: bool = True
    
    # 水平翻转
    horizontal_flip: bool = True
    horizontal_flip_prob: float = 0.5
    
    # 颜色抖动（轻微）
    color_jitter: bool = True
    brightness: float = 0.1
    contrast: float = 0.1
    saturation: float = 0.1
    hue: float = 0.05
    
    # 旋转（小角度）
    random_rotation: bool = True
    rotation_degrees: int = 15
    
    # 随机裁剪和缩放
    random_resized_crop: bool = False  # VLM 通常不用，可能影响识别
    crop_scale: Tuple[float, float] = (0.8, 1.0)
    
    # 高斯模糊（轻微）
    gaussian_blur: bool = False
    blur_kernel_size: int = 3
    blur_prob: float = 0.1


# ===================== LoRA 配置 =====================

@dataclass
class LoRAConfig:
    """LoRA 低秩适配配置
    
    参考最佳实践：
    - r=16-64 通常足够，较大的 r 可能过拟合
    - lora_alpha 通常设为 r 的 1-2 倍
    - dropout 在小数据集上设 0.05-0.1，大数据集可设 0
    """
    
    # 是否使用 LoRA
    use_lora: bool = True
    
    # LoRA 秩（低秩矩阵的维度）
    # 越大越接近全量微调效果，但参数量增加
    r: int = 32  # 推荐 16-64
    
    # LoRA 缩放因子
    # 实际缩放 = lora_alpha / r
    lora_alpha: int = 64  # 通常设为 r 的 1-2 倍
    
    # LoRA Dropout（正则化）
    lora_dropout: float = 0.05
    
    # 目标模块（要应用 LoRA 的层）
    # Qwen3-VL 中的注意力和 MLP 层
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
        "gate_proj", "up_proj", "down_proj"       # MLP 层
    ])
    
    # 是否也训练 bias
    bias: str = "none"  # "none", "all", "lora_only"
    
    # 任务类型
    task_type: str = "CAUSAL_LM"
    
    # 是否在 Vision Encoder 上也应用 LoRA
    # 通常冻结视觉编码器，只训练语言模型部分
    train_vision_encoder: bool = False
    
    # 模块排除（不应用 LoRA 的模块正则表达式）
    modules_to_save: Optional[List[str]] = None


# ===================== 优化器配置 =====================

@dataclass
class OptimizerConfig:
    """优化器配置"""
    
    # 优化器类型
    # "adamw" - 标准 AdamW
    # "adamw_8bit" - 8-bit AdamW (需要 bitsandbytes，节省内存)
    # "sgd" - 随机梯度下降
    # "adafactor" - Adafactor (内存效率高)
    optimizer_type: str = "adamw"
    
    # 学习率
    learning_rate: float = 2e-5  # LoRA 微调推荐 1e-5 ~ 5e-5
    
    # 权重衰减（L2 正则化）
    weight_decay: float = 0.01
    
    # Adam 参数
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # 梯度裁剪
    max_grad_norm: float = 1.0


# ===================== 学习率调度配置 =====================

@dataclass
class SchedulerConfig:
    """学习率调度器配置"""
    
    # 调度器类型
    # "linear" - 线性衰减
    # "cosine" - 余弦衰减（推荐）
    # "cosine_with_restarts" - 带重启的余弦衰减
    # "polynomial" - 多项式衰减
    # "constant" - 常数
    # "constant_with_warmup" - 常数带预热
    scheduler_type: str = "cosine"
    
    # 预热配置
    warmup_ratio: float = 0.1  # 预热步数占总步数的比例
    warmup_steps: Optional[int] = None  # 直接指定预热步数（优先）
    
    # 余弦调度特定参数
    num_cycles: float = 0.5  # cosine_with_restarts 的周期数


# ===================== 训练配置（主配置）=====================

@dataclass
class TrainingConfig:
    """训练配置（完整版）"""
    
    # ========== 模型配置 ==========
    model_path: str = str(MODEL_PATH)
    torch_dtype: str = "bfloat16"  # "bfloat16", "float16", "float32"
    use_flash_attention: bool = False  # Windows 上不支持，Linux 上可设为 True
    trust_remote_code: bool = True
    
    # ========== 图像配置 ==========
    image_config: ImageConfig = field(default_factory=ImageConfig)
    
    # ========== 数据增强 ==========
    augmentation_config: AugmentationConfig = field(default_factory=AugmentationConfig)
    
    # ========== LoRA 配置 ==========
    lora_config: LoRAConfig = field(default_factory=LoRAConfig)
    
    # ========== 优化器配置 ==========
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    
    # ========== 学习率调度 ==========
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # ========== 训练超参数 ==========
    num_epochs: int = 3
    batch_size: int = 2  # 每 GPU 的批次大小
    gradient_accumulation_steps: int = 8  # 等效批次 = batch_size * grad_accum
    
    # ========== 序列长度 ==========
    max_seq_length: int = 512  # 最大序列长度
    
    # ========== 混合精度训练 ==========
    bf16: bool = True   # 使用 bfloat16 混合精度
    fp16: bool = False  # 使用 float16 混合精度（二选一）
    
    # ========== 梯度检查点（节省显存）==========
    gradient_checkpointing: bool = True
    
    # ========== 保存和日志 ==========
    output_dir: str = str(CHECKPOINT_DIR)
    logging_dir: str = str(LOG_DIR)
    save_steps: int = 500
    eval_steps: int = 10  # 每 100 步评估一次，显示准确率
    logging_steps: int = 10
    save_total_limit: int = 3  # 最多保存的检查点数
    save_strategy: str = "steps"  # "steps", "epoch", "no"
    evaluation_strategy: str = "steps"  # "steps", "epoch", "no"
    
    # ========== 早停 ==========
    early_stopping: bool = False
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    
    # ========== 数据加载 ==========
    dataloader_num_workers: int = 0  # Windows 上设为 0 避免多进程问题
    dataloader_pin_memory: bool = True
    dataloader_drop_last: bool = False
    
    # ========== 随机种子 ==========
    seed: int = 42
    
    # ========== 其他 ==========
    remove_unused_columns: bool = False
    report_to: str = "none"  # "none", "tensorboard", "wandb"
    
    # ========== 类别权重（处理不平衡）==========
    use_class_weights: bool = False  # VLM 生成任务通常不使用
    
    # ========== 便捷属性 ==========
    @property
    def effective_batch_size(self) -> int:
        """等效批次大小"""
        return self.batch_size * self.gradient_accumulation_steps
    
    @property
    def use_lora(self) -> bool:
        """是否使用 LoRA"""
        return self.lora_config.use_lora
    
    @property
    def learning_rate(self) -> float:
        """学习率"""
        return self.optimizer_config.learning_rate


# ===================== 推理配置 =====================

@dataclass
class InferenceConfig:
    """推理配置"""
    
    # 模型配置
    model_path: str = str(MODEL_PATH)
    adapter_path: Optional[str] = None  # LoRA 适配器路径
    torch_dtype: str = "bfloat16"
    
    # 图像配置（与训练保持一致）
    image_config: ImageConfig = field(default_factory=ImageConfig)
    
    # 生成配置
    max_new_tokens: int = 8  # 分类任务只需要生成 1 个字符
    
    # Thinking 模型推荐的生成参数
    temperature: float = 0.1  # 分类任务使用低温度
    top_p: float = 0.9
    top_k: int = 20
    repetition_penalty: float = 1.0
    
    # 采样策略
    do_sample: bool = False  # 分类任务使用贪婪解码
    
    # 批处理配置
    batch_size: int = 8


# ===================== 默认配置实例 =====================

DEFAULT_IMAGE_CONFIG = ImageConfig()
DEFAULT_AUGMENTATION_CONFIG = AugmentationConfig()
DEFAULT_LORA_CONFIG = LoRAConfig()
DEFAULT_OPTIMIZER_CONFIG = OptimizerConfig()
DEFAULT_SCHEDULER_CONFIG = SchedulerConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_INFERENCE_CONFIG = InferenceConfig()


# ===================== 配置验证 =====================

def validate_config(config: TrainingConfig) -> List[str]:
    """验证配置的合理性，返回警告列表"""
    warnings = []
    
    # 检查学习率
    lr = config.optimizer_config.learning_rate
    if lr > 1e-4:
        warnings.append(f"学习率 {lr} 较高，LoRA 微调建议使用 1e-5 ~ 5e-5")
    if lr < 1e-6:
        warnings.append(f"学习率 {lr} 较低，可能训练过慢")
    
    # 检查 LoRA 配置
    if config.lora_config.use_lora:
        if config.lora_config.r > 128:
            warnings.append(f"LoRA rank {config.lora_config.r} 较大，可能导致过拟合")
        if config.lora_config.r < 8:
            warnings.append(f"LoRA rank {config.lora_config.r} 较小，表达能力可能不足")
    
    # 检查批次大小
    effective_bs = config.effective_batch_size
    if effective_bs < 8:
        warnings.append(f"等效批次大小 {effective_bs} 较小，可能影响收敛稳定性")
    if effective_bs > 64:
        warnings.append(f"等效批次大小 {effective_bs} 较大，可能需要调整学习率")
    
    # 检查图像配置
    img_cfg = config.image_config
    if img_cfg.max_pixels > 1024 * 1024:
        warnings.append("最大像素数超过 1M，可能导致显存不足")
    
    return warnings


# ===================== 测试 =====================

if __name__ == "__main__":
    print("=" * 60)
    print("Qwen3-VL-4B-Thinking 完整配置信息")
    print("=" * 60)
    
    config = DEFAULT_TRAINING_CONFIG
    
    print("\n【基础路径】")
    print(f"  项目根目录: {PROJECT_ROOT}")
    print(f"  模型路径: {MODEL_PATH}")
    print(f"  模型存在: {MODEL_PATH.exists()}")
    
    print("\n【类别映射】")
    print(f"  类别数量: {NUM_CLASSES}")
    print(f"  映射示例: 0 -> '{CLASS_TO_CHAR[0]}', 26 -> '{CLASS_TO_CHAR[26]}'")
    
    print("\n【图像配置】")
    print(f"  最小像素: {config.image_config.min_pixels} ({int(config.image_config.min_pixels**0.5)}x{int(config.image_config.min_pixels**0.5)})")
    print(f"  最大像素: {config.image_config.max_pixels} ({int(config.image_config.max_pixels**0.5)}x{int(config.image_config.max_pixels**0.5)})")
    
    print("\n【LoRA 配置】")
    print(f"  使用 LoRA: {config.lora_config.use_lora}")
    print(f"  LoRA rank: {config.lora_config.r}")
    print(f"  LoRA alpha: {config.lora_config.lora_alpha}")
    print(f"  目标模块: {config.lora_config.target_modules}")
    
    print("\n【训练配置】")
    print(f"  训练轮数: {config.num_epochs}")
    print(f"  批次大小: {config.batch_size}")
    print(f"  梯度累积: {config.gradient_accumulation_steps}")
    print(f"  等效批次: {config.effective_batch_size}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  混合精度: bf16={config.bf16}, fp16={config.fp16}")
    print(f"  梯度检查点: {config.gradient_checkpointing}")
    
    print("\n【配置验证】")
    warnings = validate_config(config)
    if warnings:
        for w in warnings:
            print(f"  ⚠ {w}")
    else:
        print("  ✓ 配置验证通过")
    
    print("\n" + "=" * 60)
