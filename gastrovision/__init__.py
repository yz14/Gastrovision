"""
Gastrovision - 胃镜图像分类框架

模块结构:
- data: 数据集、增强、采样器
- models: 模型定义
- losses: 损失函数
- trainers: 训练器
- utils: 工具函数
"""

from . import data
from . import models
from . import losses
from . import trainers
from . import utils

__version__ = "1.0.0"
