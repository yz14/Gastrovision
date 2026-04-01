# Jaguar Re-ID 训练流程总结

> 最终成绩：**0.920 Private Score (AUC)**  
> 评估指标：AUC（正负样本对的相似度区分能力）

---

## 一、关键技术点及贡献估计

| 技术点 | 代码位置 | 贡献估计 |
|--------|----------|----------|
| BNNeck 架构 | `model.py: ReIDModel` | 基础性，+0.03~0.05 |
| GeM Pooling | `model.py: GeMPooling` | 对比 AvgPool，+0.01~0.02 |
| ArcFace 主损失 | `train.py`, `LOSS_DEFAULTS` | 核心度量，+0.05~0.08 |
| ArcFace + Triplet 组合 | `trainer.py: _train_epoch` | 对比单一损失，+0.01~0.02 |
| PK Sampler | `sampler.py: PKSampler` | 保证 Triplet 有意义，+0.01 |
| 高分辨率输入 (384px) | `jaguar_reid.yaml` | 对比 224，+0.02~0.03 |
| Heavy 数据增强 + Random Erasing | `transforms.py` | 减少过拟合，+0.01~0.02 |
| 分离学习率（backbone×0.1）| `train.py: param_groups` | 保护预训练，+0.01 |
| Warmup Cosine 调度器 | `jaguar_reid.yaml` | 训练稳定，+0.01 |
| EMA（decay=0.999）| `trainer.py: ModelEMA` | 减少噪声，+0.005~0.01 |
| 水平翻转 TTA | `inference.py: extract_embeddings` | 推理提升，+0.005~0.01 |
| 自适应 Sigmoid 校准 | `inference.py: predict_similarity_fast` | 输出分布，~0 |

---

## 二、架构设计要点

```
Image (384×384)
  → Backbone (ResNet50, ImageNet pretrained)
  → GeM Pooling (p≈3, 可学习)
  → Dropout(0.2)
  → FC → raw_feat (512-d)       ← Triplet 损失作用于此
  → BNNeck                       ← 分离度量与分类空间
  → bn_feat (512-d)              ← ArcFace 损失作用于此
  → L2 Normalize                 ← 推理时用 raw_feat 归一化
```

**BNNeck 的作用**：BNNeck 后的特征更适合 ArcFace 分类目标；推理时用 BNNeck 前的 `raw_feat` 做余弦相似度，避免 BN 引入分布偏差。这是 "Bag of Tricks for ReID" (CVPR 2019) 的核心贡献之一。

---

## 三、损失函数组合

```
total_loss = arcface(bn_feat, labels)
           + 1.0 × triplet(raw_feat, labels)
```

- **ArcFace**（scale=30, margin=0.5）：全局分类监督，学习类间间隔
- **Triplet + Batch Hard Mining**（margin=0.3）：直接优化 Embedding 距离空间
- **PK Sampler**（P=8, K=4）：每 batch 32 张图，保证每类至少 4 张供 Triplet 挖掘
- **Label Smoothing = 0.1**：防止小数据集过拟合

---

## 四、数据增强策略（heavy 模式）

```
RandomResizedCrop(scale=0.6~1.0) → HorizontalFlip
→ Affine(rotate±25°, scale 0.8~1.2)
→ GridDistortion / ElasticTransform (p=0.2)
→ BrightnessContrast + HueSaturation
→ GaussNoise / GaussianBlur / MotionBlur (p=0.3)
→ CoarseDropout (Random Erasing, 1~4 blocks, p=0.5)
→ Normalize(ImageNet)
```

Random Erasing 对 Re-ID 至关重要，模拟自然场景中豹子被草木遮挡的情况。

---

## 五、训练策略细节

- **分组学习率**：backbone lr = 3e-5，head + 损失头 lr = 3e-4，保护 ImageNet 预训练特征
- **Warmup Cosine**：前 5 epoch warmup 防止初期梯度爆炸
- **EMA decay=0.999**：自动计算合理衰减，验证时用 EMA 模型选最佳 checkpoint
- **验证指标**：按 mAP 选模型（与 Kaggle AUC 相关性高于 Rank-1）
- **梯度裁剪**：max_norm=5.0 防止 ArcFace 初期不稳定

---

## 六、推理流程

```
提取 embedding: 正向 + 水平翻转 → 平均 → L2 归一化
计算相似度: 余弦点积
后处理: 基于 IQR 的 Robust Z-score + Sigmoid 校准 → [0, 1]
```

---

## 七、对比竞赛 Top 方案的差距分析

根据 Kaggle Discussion，Top 方案的额外优势：

| Top 方案技术 | 本方案状态 | 潜在提升空间 |
|-------------|-----------|------------|
| DINOv2 / EVA-02 大模型 backbone | ❌ 使用 ResNet50 | +0.03~0.05 |
| k-Reciprocal Re-ranking | ❌ 未实现 | +0.02~0.04 |
| 5-Fold 集成 | ❌ 单模型 | +0.01~0.02 |
| Pseudo-labeling | ❌ 未实现 | +0.01~0.02 |
| 更大分辨率（448+）| ⚠️ 使用 384 | +0.005~0.01 |

**结论**：在 ResNet50 单模型框架下，0.920 的成绩已非常接近该规格的上限。主要提升空间来自更换更强的 backbone（如 DINOv2）以及加入 Re-ranking 后处理。
