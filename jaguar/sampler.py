"""
PK Sampler for Re-ID

标准 Re-ID 训练采样器：每个 batch 包含 P 个类别，每个类别 K 个样本。
确保 pair-based 损失 (Triplet/Contrastive/Circle/Lifted/NPair) 在每个 batch 中
都能构造足够的正样本对。

参考: In Defense of the Triplet Loss for Person Re-Identification (arXiv 2017)
"""

import random
from collections import defaultdict
from typing import List, Iterator

from torch.utils.data import Sampler


class PKSampler(Sampler):
    """
    PK Sampler: 每个 batch 采样 P 个类别，每个类别 K 个实例

    Args:
        labels: 数据集所有样本的标签列表
        p: 每个 batch 的类别数 (默认 8)
        k: 每个类别的实例数 (默认 4)
        seed: 随机种子 (None 表示不固定)
    """

    def __init__(
        self,
        labels: List[int],
        p: int = 8,
        k: int = 4,
        seed: int = None,
    ):
        self.labels = labels
        self.p = p
        self.k = k
        self.seed = seed
        self._epoch_counter = 0

        # 构建 label -> [indices] 映射
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

        # 过滤掉样本数不足 2 的类别 (无法构成正样本对)
        self.valid_labels = [
            label for label, indices in self.label_to_indices.items()
            if len(indices) >= 2
        ]

        if len(self.valid_labels) < self.p:
            # 如果有效类别不足 P 个，降低 P
            self.p = max(2, len(self.valid_labels))

        # 每个 epoch 的采样数 (近似原始数据集大小)
        total_samples = len(labels)
        self.num_batches = max(1, total_samples // (self.p * self.k))

    def __iter__(self) -> Iterator[int]:
        # 每次迭代 (每个 epoch) 使用不同的随机序列
        # seed=None 时完全随机；seed!=None 时基于 epoch 计数产生可复现但不同的序列
        if self.seed is not None:
            rng = random.Random(self.seed + self._epoch_counter)
            self._epoch_counter += 1
        else:
            rng = random.Random()

        for _ in range(self.num_batches):
            # 随机选择 P 个类别
            selected_labels = rng.sample(self.valid_labels, self.p)

            batch_indices = []
            for label in selected_labels:
                indices = self.label_to_indices[label]
                if len(indices) >= self.k:
                    # 随机选 K 个 (不放回)
                    selected = rng.sample(indices, self.k)
                else:
                    # 样本不足 K 个，有放回采样补齐
                    selected = rng.choices(indices, k=self.k)
                batch_indices.extend(selected)

            yield from batch_indices

    def __len__(self) -> int:
        return self.num_batches * self.p * self.k
