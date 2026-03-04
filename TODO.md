# 核心原则

## 质量第一
- 宁可多花时间，也要保证代码质量
- 充分思考、分析后再动手实现
- 不要为了快速完成而牺牲代码质量

## 分步完成
- 如果当前对话无法完成所有功能，主动拆分为多轮对话
- 每轮只专注完成一个清晰的目标
- 不贪多，确保每一步都高质量完成

## 充分调研
- 如有需要，充分、彻底地搜索和调研
- 分析和掌握现有的高质量功能实现和算法
- 借鉴业界最佳实践，不要闭门造车

## 调试支持
- 如有需要，可以加入 debug/logging 函数辅助开发
- 通过日志输出帮助定位和解决问题
- 调试代码可在功能稳定后标注或移除

## 沟通规范
- **开始前**：说明你理解的任务目标和将遵守的规则
- **进行中**：如需拆分，明确告知本轮将完成什么
- **完成后**：总结本轮成果，说明后续计划（如有）  


**测试环境为conda activate torch27_env**


# TODO   

1. 全面代码审查：main.py是训练入口，先全面彻底的检查所有的代码，是否有算法写错，损失函数写错，训练过程，逻辑写错等等  
2. 实现以下损失函数，并提供参数来调用  
1.1 Contrastive Loss — 孪生网络
论文
Dimensionality Reduction by Learning an Invariant Mapping (DrLIM)
作者
Hadsell, Chopra, LeCun
会议
CVPR 2006
链接
http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
核心贡献：

提出最早的对比损失，将正对（相似样本）拉近，负对（不相似样本）推远
损失形式：$$L = (1-y)\frac{1}{2}D^2 + y\frac{1}{2}\max(0,, m-D)^$$
奠定后续所有对比学习的基本思想


1.2 Triplet Loss
论文
FaceNet: A Unified Embedding for Face Recognition and Clustering
作者
Schroff, Kalenichenko, Philbin
会议
CVPR 2015
链接
https://arxiv.org/abs/1503.03832
核心贡献：

三元组 (Anchor, Positive, Negative) 形式，直接优化相对距离
损失形式：$$L = \max(0,; d(a,p) - d(a,n) + \alpha$$
提出 Online Hard Negative Mining，极大加速收敛
在人脸识别上达到当时 SOTA，推动大规模度量学习实用化


1.3 Lifted Structure Loss / Proxy NCA
论文
Deep Metric Learning via Lifted Structured Feature Embedding
会议
CVPR 2016
链接
https://arxiv.org/abs/1511.06452
论文
No Fuss Distance Metric Learning using Proxies
会议
ICCV 2017
链接
https://arxiv.org/abs/1703.07464
核心贡献：

Lifted Structure：批内所有正负对同时参与，充分利用 mini-batch 信息
Proxy NCA：每类学一个代理向量（Proxy），大幅降低计算复杂度，训练更稳定


1.4 N-pair Loss
论文
Improved Deep Metric Learning with Multi-class N-pair Loss Objective
作者
Sohn
会议
NeurIPS 2016
链接
https://papers.nips.cc/paper/2016/hash/6b180037abbebea991d8b1232f8a8ca9-Abstract.html
核心贡献：

将 Triplet 扩展为 N 对，每次更新同时考虑多个负样本
等价于 InfoNCE 的前身，批次构造更高效
避免 Triplet Loss 中大量无效三元组导致的训练崩溃


1.5 SphereFace / CosFace / ArcFace — Angular Margin 系列
SphereFace
Deep Hypersphere Embedding for Face Recognition — CVPR 2017 · https://arxiv.org/abs/1704.08063
CosFace
Large Margin Cosine Loss — CVPR 2018 · https://arxiv.org/abs/1801.09414
ArcFace
Additive Angular Margin Loss — CVPR 2019 · https://arxiv.org/abs/1801.07698
核心贡献（以 ArcFace 为代表）：

在超球面上施加角度间隔（Angular Margin），几何意义明确
损失形式：$$L = -\log\dfrac{e^{s\cos(\theta_{y_i}+m)}}{e^{s\cos(\theta_{y_i}+m)}+\sum_{j\neq y_i}e^{s\cos\theta_j}$$
成为人脸识别事实标准，同样适用于医学图像检索、细粒度识别等任务


1.6 Circle Loss
论文
Circle Loss: A Unified Perspective of Pair Similarity Optimization
会议
CVPR 2020
链接
https://arxiv.org/abs/2002.10857
核心贡献：

统一框架：将 Triplet、Softmax、N-pair 等多种损失纳入同一视角
对每个相似度分数施加自适应权重，正负对梯度独立调节
在度量学习与分类任务上均取得强基线性能