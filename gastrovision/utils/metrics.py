"""
Gastrovision 通用指标工具

提供:
- AverageMeter: 计算和存储运行平均值
"""


class AverageMeter:
    """计算和存储运行平均值

    用法:
        meter = AverageMeter()
        for batch_loss in losses:
            meter.update(batch_loss, batch_size)
        print(meter.avg)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
