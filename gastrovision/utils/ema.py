"""
Exponential Moving Average (EMA) 模型参数

原理:
  维护模型参数的指数滑动平均。EMA 参数通常比直接训练的参数
  具有更好的泛化能力，因为它相当于隐式的模型集成。

  EMA_param = decay * EMA_param + (1 - decay) * current_param

使用:
  from gastrovision.utils.ema import ModelEMA

  ema = ModelEMA(model, decay=0.9999)

  for batch in train_loader:
      loss.backward()
      optimizer.step()
      ema.update(model)  # 每步更新 EMA

  # 验证或测试时:
  ema.apply_shadow()    # 将 EMA 参数应用到模型
  validate(model)
  ema.restore()         # 恢复原始训练参数
"""

import copy
from typing import Optional

import torch
import torch.nn as nn


class ModelEMA:
    """
    模型参数的指数滑动平均

    Args:
        model: 需要跟踪的模型
        decay: 衰减系数 (推荐 0.999-0.9999，值越大越平滑)
        warmup_steps: 预热步数，预热期间 decay 从 0 线性增加到目标值
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 0
    ):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.num_updates = 0

        # 保存 EMA 参数的 shadow 副本
        self.shadow: dict = {}
        self.backup: dict = {}

        # 初始化 shadow 为当前参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def _get_decay(self) -> float:
        """获取当前衰减率（考虑预热）"""
        if self.warmup_steps > 0 and self.num_updates < self.warmup_steps:
            # 线性预热: decay 从 0 增加到目标值
            return self.decay * self.num_updates / self.warmup_steps
        return self.decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        在每个优化步骤后调用，更新 EMA 参数

        Args:
            model: 当前模型（包含最新参数）
        """
        self.num_updates += 1
        decay = self._get_decay()

        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # EMA: shadow = decay * shadow + (1-decay) * param
                self.shadow[name].mul_(decay).add_(param.data, alpha=1.0 - decay)

    def apply_shadow(self, model: nn.Module):
        """
        将 EMA 参数应用到模型（用于验证/测试）

        同时备份当前模型参数，之后可以用 restore() 恢复。

        Args:
            model: 目标模型
        """
        self.backup.clear()
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """
        恢复被 apply_shadow() 替换的原始训练参数

        Args:
            model: 目标模型
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self) -> dict:
        """保存 EMA 状态（用于 checkpoint）"""
        return {
            'shadow': self.shadow,
            'num_updates': self.num_updates,
            'decay': self.decay,
        }

    def load_state_dict(self, state: dict):
        """加载 EMA 状态"""
        self.shadow = state['shadow']
        self.num_updates = state['num_updates']
        self.decay = state.get('decay', self.decay)
