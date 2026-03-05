"""
配置加载器

纯 YAML 配置驱动，所有参数在 YAML 文件中定义。

用法:
    from gastrovision.utils.config import load_config
    
    cfg = load_config('configs/train_cls.yaml')
    print(cfg.model)       # "resnet50"
    print(cfg.batch_size)  # 32
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import yaml


def load_config(yaml_path: str) -> SimpleNamespace:
    """
    从 YAML 文件加载配置，返回可通过 . 访问的命名空间对象。

    Args:
        yaml_path: YAML 配置文件路径

    Returns:
        SimpleNamespace 对象，所有配置值可通过属性访问

    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML 格式错误
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {yaml_path}")

    with open(path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    if config_dict is None:
        config_dict = {}

    cfg = SimpleNamespace(**config_dict)

    # 保存配置文件路径，供日志使用
    cfg._config_path = str(path.resolve())

    return cfg


def get_config_from_cli() -> SimpleNamespace:
    """
    从命令行参数获取 YAML 配置路径并加载。

    用法:
        python train_cls.py configs/train_cls.yaml

    Returns:
        SimpleNamespace 配置对象
    """
    if len(sys.argv) < 2:
        print("用法: python train_cls.py <config.yaml>")
        print("示例: python train_cls.py configs/train_cls.yaml")
        sys.exit(1)

    yaml_path = sys.argv[1]
    return load_config(yaml_path)


def save_config(cfg: SimpleNamespace, output_path: str):
    """
    将配置保存为可读文本文件（用于记录训练参数）。

    Args:
        cfg: 配置对象
        output_path: 输出文件路径
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        for key, value in sorted(vars(cfg).items()):
            if not key.startswith('_'):
                f.write(f"{key}: {value}\n")
