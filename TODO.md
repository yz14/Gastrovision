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

## 代码质量  
- 注意代码尽可能模块化设计，职责尽可能的分离，不要把所有代码写在一个文件里，不方便后续理解和维护  
- 注意代码的复用性，不要写重复的代码  

## 沟通规范
- **开始前**：说明你理解的任务目标和将遵守的规则
- **进行中**：如需拆分，明确告知本轮将完成什么
- **完成后**：总结本轮成果，说明后续计划（如有）  


**测试环境为conda activate torch27_env**


# TODO   

1. D:\codes\work-projects\Gastrovision_models\train_cls.py这是我训练内镜分类的代码。目前看起来都正常。我复用这套方案来解决https://www.kaggle.com/competitions/jaguar-re-id/overview这个kaggle比赛。我数据都在服务器上/data0/yzhen/data/jaguar_reid下面有csv，train和test文件夹。我把csv文件D:\codes\work-projects\Gastrovision_models\data\train.csv放到这里了，还有test.csv和提交的csv样例。我写好的方案在D:\codes\work-projects\Gastrovision_models\jaguar这里。


2. 我用最新的代码在服务器上训练，将测试的结果提交kaggle，结果resnet50只有0.3，convnext只有0.4和0.48（两次实验）。保存最佳模型的验证集指标是否合理？是用val_rank1还是val_map?还是val_auc?我觉得整个流程需要细致，彻底的审查，我感觉肯定哪里有问题？或者还有更好的策略和技巧？如果实在找不到问题，请在所有可能的地方加入debug来获取关键信息。  

python jaguar/train.py configs/jaguar_reid.yaml
/data0/yzhen/py3/envs/py310/lib/python3.10/site-packages/albumentations/check_version.py:107: UserWarning: Error fetching version info <urlopen error [Errno 101] Network is unreachable>
  data = fetch_version_info()
============================================================
Jaguar Re-Identification Training
============================================================
时间:   2026-03-20 09:41:06
设备:   cuda
GPU:    NVIDIA GeForce RTX 4090
配置:   /data0/yzhen/projects/gastrovision_v3/configs/jaguar_reid.yaml

========================================
数据划分
========================================
总样本数: 1895
类别数:   31
样本数范围: 13 ~ 183

划分结果:
  训练集: 1625 样本
  验证集: 270 样本
  已保存到: /data0/yzhen/projects/gastrovision_v3/output/jaguar_arcface

类别数: 31

训练集: 1625 样本
验证集: 270 样本
  PK Sampler: P=8, K=4 (batch_size=32)

创建模型: resnet50
  加载本地预训练权重: /data0/data/resnet50.pth
  模型参数: 24.56M, 可训练: 24.56M

主损失: arcface
  [primary] arcface: scale=30.0, margin=0.5
辅助损失: triplet (weight=1.0)
  [aux_] triplet: scale=1.0, margin=0.3
优化器: adamw (backbone_lr=0.000030, head_lr=0.000300)
调度器: warmup_cosine (warmup=5)
  EMA 已启用 (decay=0.9999)

========================================
开始训练
========================================

Epoch [1/80] lr=0.000006 train_loss=17.2541 train_acc=0.0569 aux_loss=0.4241 val_rank1=0.7556 val_mAP=0.3180 val_auc=0.6442 val_coverage=270/270 time=61.1s
  ★ 新最佳! Rank-1=0.7556
Epoch [2/80] lr=0.000012 train_loss=16.4448 train_acc=0.2344 aux_loss=0.4139 val_rank1=0.7444 val_mAP=0.3176 val_auc=0.6435 val_coverage=270/270 time=62.9s
Epoch [3/80] lr=0.000018 train_loss=15.5132 train_acc=0.4081 aux_loss=0.3998 val_rank1=0.7481 val_mAP=0.3213 val_auc=0.6452 val_coverage=270/270 time=64.3s
Epoch [4/80] lr=0.000024 train_loss=14.3286 train_acc=0.5238 aux_loss=0.3788 val_rank1=0.7519 val_mAP=0.3223 val_auc=0.6462 val_coverage=270/270 time=69.2s
Epoch [5/80] lr=0.000030 train_loss=12.7641 train_acc=0.6388 aux_loss=0.3595 val_rank1=0.7407 val_mAP=0.3239 val_auc=0.6465 val_coverage=270/270 time=65.1s
Epoch [6/80] lr=0.000030 train_loss=11.5645 train_acc=0.6825 aux_loss=0.3495 val_rank1=0.7481 val_mAP=0.3261 val_auc=0.6473 val_coverage=270/270 time=63.3s
Epoch [7/80] lr=0.000030 train_loss=10.2175 train_acc=0.7350 aux_loss=0.3266 val_rank1=0.7481 val_mAP=0.3259 val_auc=0.6475 val_coverage=270/270 time=58.3s
Epoch [8/80] lr=0.000030 train_loss=8.9596 train_acc=0.7781 aux_loss=0.3090 val_rank1=0.7481 val_mAP=0.3257 val_auc=0.6480 val_coverage=270/270 time=63.6s
Epoch [9/80] lr=0.000030 train_loss=7.8955 train_acc=0.8306 aux_loss=0.2837 val_rank1=0.7333 val_mAP=0.3204 val_auc=0.6472 val_coverage=270/270 time=68.7s
Epoch [10/80] lr=0.000030 train_loss=7.0174 train_acc=0.8456 aux_loss=0.2614 val_rank1=0.7407 val_mAP=0.3220 val_auc=0.6443 val_coverage=270/270 time=64.7s
Epoch [11/80] lr=0.000030 train_loss=6.0722 train_acc=0.8831 aux_loss=0.2312 val_rank1=0.7370 val_mAP=0.3238 val_auc=0.6449 val_coverage=270/270 time=64.9s
Epoch [12/80] lr=0.000030 train_loss=5.8634 train_acc=0.8888 aux_loss=0.2158 val_rank1=0.7296 val_mAP=0.3196 val_auc=0.6473 val_coverage=270/270 time=62.1s
Epoch [13/80] lr=0.000029 train_loss=5.2552 train_acc=0.8944 aux_loss=0.1877 val_rank1=0.7444 val_mAP=0.3204 val_auc=0.6466 val_coverage=270/270 time=67.7s
Epoch [14/80] lr=0.000029 train_loss=4.8473 train_acc=0.9131 aux_loss=0.1667 val_rank1=0.7481 val_mAP=0.3220 val_auc=0.6502 val_coverage=270/270 time=65.6s
Epoch [15/80] lr=0.000029 train_loss=4.4132 train_acc=0.9263 aux_loss=0.1478 val_rank1=0.7333 val_mAP=0.3236 val_auc=0.6495 val_coverage=270/270 time=64.1s
Epoch [16/80] lr=0.000029 train_loss=4.1910 train_acc=0.9356 aux_loss=0.1334 val_rank1=0.7444 val_mAP=0.3234 val_auc=0.6494 val_coverage=270/270 time=59.8s
Epoch [17/80] lr=0.000028 train_loss=3.9602 train_acc=0.9344 aux_loss=0.1237 val_rank1=0.7407 val_mAP=0.3226 val_auc=0.6493 val_coverage=270/270 time=62.6s
Epoch [18/80] lr=0.000028 train_loss=3.6536 train_acc=0.9450 aux_loss=0.1052 val_rank1=0.7333 val_mAP=0.3246 val_auc=0.6542 val_coverage=270/270 time=62.7s
Epoch [19/80] lr=0.000028 train_loss=3.5003 train_acc=0.9537 aux_loss=0.0956 val_rank1=0.7296 val_mAP=0.3242 val_auc=0.6570 val_coverage=270/270 time=63.6s
Epoch [20/80] lr=0.000028 train_loss=3.3751 train_acc=0.9525 aux_loss=0.0896 val_rank1=0.7296 val_mAP=0.3266 val_auc=0.6587 val_coverage=270/270 time=65.6s
Epoch [21/80] lr=0.000027 train_loss=3.0996 train_acc=0.9644 aux_loss=0.0785 val_rank1=0.7333 val_mAP=0.3274 val_auc=0.6577 val_coverage=270/270 time=66.2s

早停触发 (连续 20 轮无改善)

训练完成! 总时间: 22.6 分钟
最佳 Rank-1: 0.7556 (Epoch 1)

========================================
加载最佳模型进行推理
========================================
  加载 checkpoint: /data0/yzhen/projects/gastrovision_v3/output/jaguar_arcface/best_model.pth
  恢复自 Epoch 1, 最佳 Rank-1=0.7556
  已应用 EMA 权重
测试图片数: 371
提取 embedding: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:31<00:00,  5.30s/it]
读取测试对: /data0/yzhen/data/jaguar_reid/test.csv
共 137270 对
校准参数: median=0.8206, IQR=0.0685
提交文件已保存: /data0/yzhen/projects/gastrovision_v3/output/jaguar_arcface/submission.csv
相似度统计: mean=0.4843, std=0.2065, min=0.0046, max=0.9692

完成!


python jaguar/train.py configs/jaguar_reid.yaml
/data0/yzhen/py3/envs/py310/lib/python3.10/site-packages/albumentations/check_version.py:107: UserWarning: Error fetching version info <urlopen error [Errno 101] Network is unreachable>
  data = fetch_version_info()
============================================================
Jaguar Re-Identification Training
============================================================
时间:   2026-03-20 10:08:50
设备:   cuda
GPU:    NVIDIA GeForce RTX 4090
配置:   /data0/yzhen/projects/gastrovision_v3/configs/jaguar_reid.yaml

========================================
数据划分
========================================
总样本数: 1895
类别数:   31
样本数范围: 13 ~ 183

划分结果:
  训练集: 1625 样本
  验证集: 270 样本
  已保存到: /data0/yzhen/projects/gastrovision_v3/output/jaguar_arcface

类别数: 31

训练集: 1625 样本
验证集: 270 样本
  PK Sampler: P=8, K=4 (batch_size=32)

创建模型: convnext_base
  加载本地预训练权重: /data0/yzhen/data/pretrained/pretrain/convnext_base-6075fbad.pth
  模型参数: 88.09M, 可训练: 88.09M

主损失: arcface
  [primary] arcface: scale=30.0, margin=0.5
辅助损失: triplet (weight=1.0)
  [aux_] triplet: scale=1.0, margin=0.3
优化器: adamw (backbone_lr=0.000030, head_lr=0.000300)
调度器: warmup_cosine (warmup=5)
  EMA 已启用 (decay=0.9999)

========================================
开始训练
========================================

Epoch [1/80] lr=0.000006 train_loss=17.5625 train_acc=0.0400 aux_loss=0.5839 val_rank1=0.6852 val_mAP=0.2825 val_auc=0.6009 val_coverage=270/270 time=62.0s
  ★ 新最佳! Rank-1=0.6852
Epoch [2/80] lr=0.000012 train_loss=16.9134 train_acc=0.1462 aux_loss=0.5372 val_rank1=0.6852 val_mAP=0.2825 val_auc=0.6010 val_coverage=270/270 time=63.1s
Epoch [3/80] lr=0.000018 train_loss=15.9936 train_acc=0.3312 aux_loss=0.4853 val_rank1=0.6852 val_mAP=0.2827 val_auc=0.6010 val_coverage=270/270 time=65.1s
Epoch [4/80] lr=0.000024 train_loss=14.6608 train_acc=0.5144 aux_loss=0.4389 val_rank1=0.6852 val_mAP=0.2832 val_auc=0.6011 val_coverage=270/270 time=68.7s
Epoch [5/80] lr=0.000030 train_loss=12.9905 train_acc=0.6469 aux_loss=0.4001 val_rank1=0.6852 val_mAP=0.2836 val_auc=0.6011 val_coverage=270/270 time=64.6s
Epoch [6/80] lr=0.000030 train_loss=11.4916 train_acc=0.7075 aux_loss=0.3799 val_rank1=0.6852 val_mAP=0.2841 val_auc=0.6012 val_coverage=270/270 time=65.5s
Epoch [7/80] lr=0.000030 train_loss=10.0794 train_acc=0.7475 aux_loss=0.3422 val_rank1=0.6852 val_mAP=0.2847 val_auc=0.6012 val_coverage=270/270 time=63.4s
Epoch [8/80] lr=0.000030 train_loss=8.9706 train_acc=0.7956 aux_loss=0.3200 val_rank1=0.6852 val_mAP=0.2853 val_auc=0.6013 val_coverage=270/270 time=63.5s
Epoch [9/80] lr=0.000030 train_loss=7.9064 train_acc=0.8438 aux_loss=0.2886 val_rank1=0.6852 val_mAP=0.2864 val_auc=0.6013 val_coverage=270/270 time=70.7s
Epoch [10/80] lr=0.000030 train_loss=6.9827 train_acc=0.8575 aux_loss=0.2586 val_rank1=0.6852 val_mAP=0.2872 val_auc=0.6014 val_coverage=270/270 time=64.3s
Epoch [11/80] lr=0.000030 train_loss=6.2502 train_acc=0.8794 aux_loss=0.2353 val_rank1=0.6815 val_mAP=0.2873 val_auc=0.6016 val_coverage=270/270 time=64.5s
Epoch [12/80] lr=0.000030 train_loss=5.7999 train_acc=0.9069 aux_loss=0.2180 val_rank1=0.6852 val_mAP=0.2884 val_auc=0.6017 val_coverage=270/270 time=64.1s
Epoch [13/80] lr=0.000029 train_loss=5.1825 train_acc=0.9131 aux_loss=0.2030 val_rank1=0.6889 val_mAP=0.2895 val_auc=0.6018 val_coverage=270/270 time=67.0s
  ★ 新最佳! Rank-1=0.6889
Epoch [14/80] lr=0.000029 train_loss=4.8286 train_acc=0.9200 aux_loss=0.1769 val_rank1=0.6926 val_mAP=0.2901 val_auc=0.6020 val_coverage=270/270 time=65.1s
  ★ 新最佳! Rank-1=0.6926
Epoch [15/80] lr=0.000029 train_loss=4.4364 train_acc=0.9394 aux_loss=0.1707 val_rank1=0.6963 val_mAP=0.2911 val_auc=0.6022 val_coverage=270/270 time=64.3s
  ★ 新最佳! Rank-1=0.6963
Epoch [16/80] lr=0.000029 train_loss=4.0984 train_acc=0.9425 aux_loss=0.1556 val_rank1=0.6963 val_mAP=0.2919 val_auc=0.6024 val_coverage=270/270 time=60.8s
Epoch [17/80] lr=0.000028 train_loss=4.1405 train_acc=0.9400 aux_loss=0.1577 val_rank1=0.6963 val_mAP=0.2927 val_auc=0.6027 val_coverage=270/270 time=62.4s
Epoch [18/80] lr=0.000028 train_loss=3.6022 train_acc=0.9537 aux_loss=0.1307 val_rank1=0.6963 val_mAP=0.2937 val_auc=0.6029 val_coverage=270/270 time=64.3s
Epoch [19/80] lr=0.000028 train_loss=3.4008 train_acc=0.9619 aux_loss=0.1237 val_rank1=0.7000 val_mAP=0.2952 val_auc=0.6032 val_coverage=270/270 time=62.9s
  ★ 新最佳! Rank-1=0.7000
Epoch [20/80] lr=0.000028 train_loss=3.2311 train_acc=0.9594 aux_loss=0.1112 val_rank1=0.7037 val_mAP=0.2965 val_auc=0.6035 val_coverage=270/270 time=65.2s
  ★ 新最佳! Rank-1=0.7037
Epoch [21/80] lr=0.000027 train_loss=2.9891 train_acc=0.9656 aux_loss=0.0963 val_rank1=0.7074 val_mAP=0.2982 val_auc=0.6038 val_coverage=270/270 time=65.5s
  ★ 新最佳! Rank-1=0.7074
Epoch [22/80] lr=0.000027 train_loss=3.0481 train_acc=0.9619 aux_loss=0.1027 val_rank1=0.7111 val_mAP=0.2991 val_auc=0.6042 val_coverage=270/270 time=65.6s
  ★ 新最佳! Rank-1=0.7111
Epoch [23/80] lr=0.000026 train_loss=2.7746 train_acc=0.9681 aux_loss=0.0910 val_rank1=0.7111 val_mAP=0.2999 val_auc=0.6045 val_coverage=270/270 time=69.1s
Epoch [24/80] lr=0.000026 train_loss=2.7166 train_acc=0.9744 aux_loss=0.0893 val_rank1=0.7111 val_mAP=0.3009 val_auc=0.6049 val_coverage=270/270 time=60.5s
Epoch [25/80] lr=0.000026 train_loss=2.5821 train_acc=0.9769 aux_loss=0.0794 val_rank1=0.7074 val_mAP=0.3020 val_auc=0.6054 val_coverage=270/270 time=63.7s
Epoch [26/80] lr=0.000025 train_loss=2.5263 train_acc=0.9769 aux_loss=0.0779 val_rank1=0.7037 val_mAP=0.3029 val_auc=0.6058 val_coverage=270/270 time=64.6s
Epoch [27/80] lr=0.000025 train_loss=2.3883 train_acc=0.9788 aux_loss=0.0664 val_rank1=0.7111 val_mAP=0.3043 val_auc=0.6062 val_coverage=270/270 time=66.0s
Epoch [28/80] lr=0.000024 train_loss=2.1774 train_acc=0.9831 aux_loss=0.0565 val_rank1=0.7111 val_mAP=0.3056 val_auc=0.6067 val_coverage=270/270 time=63.1s
Epoch [29/80] lr=0.000024 train_loss=2.1142 train_acc=0.9856 aux_loss=0.0530 val_rank1=0.7111 val_mAP=0.3065 val_auc=0.6071 val_coverage=270/270 time=64.7s
Epoch [30/80] lr=0.000023 train_loss=2.3099 train_acc=0.9806 aux_loss=0.0662 val_rank1=0.7111 val_mAP=0.3077 val_auc=0.6076 val_coverage=270/270 time=63.2s
Epoch [31/80] lr=0.000023 train_loss=2.0461 train_acc=0.9912 aux_loss=0.0494 val_rank1=0.7111 val_mAP=0.3090 val_auc=0.6082 val_coverage=270/270 time=63.5s
Epoch [32/80] lr=0.000022 train_loss=2.0256 train_acc=0.9906 aux_loss=0.0481 val_rank1=0.7148 val_mAP=0.3105 val_auc=0.6087 val_coverage=270/270 time=65.2s
  ★ 新最佳! Rank-1=0.7148
Epoch [33/80] lr=0.000022 train_loss=1.9846 train_acc=0.9888 aux_loss=0.0466 val_rank1=0.7111 val_mAP=0.3115 val_auc=0.6093 val_coverage=270/270 time=68.3s
Epoch [34/80] lr=0.000021 train_loss=1.9887 train_acc=0.9850 aux_loss=0.0497 val_rank1=0.7148 val_mAP=0.3127 val_auc=0.6099 val_coverage=270/270 time=64.0s
Epoch [35/80] lr=0.000021 train_loss=1.9050 train_acc=0.9869 aux_loss=0.0423 val_rank1=0.7148 val_mAP=0.3139 val_auc=0.6105 val_coverage=270/270 time=65.2s
Epoch [36/80] lr=0.000020 train_loss=1.8798 train_acc=0.9894 aux_loss=0.0414 val_rank1=0.7185 val_mAP=0.3155 val_auc=0.6112 val_coverage=270/270 time=63.9s
  ★ 新最佳! Rank-1=0.7185
Epoch [37/80] lr=0.000019 train_loss=1.7788 train_acc=0.9912 aux_loss=0.0360 val_rank1=0.7185 val_mAP=0.3172 val_auc=0.6119 val_coverage=270/270 time=70.0s
Epoch [38/80] lr=0.000019 train_loss=1.8442 train_acc=0.9900 aux_loss=0.0400 val_rank1=0.7185 val_mAP=0.3186 val_auc=0.6126 val_coverage=270/270 time=64.1s
Epoch [39/80] lr=0.000018 train_loss=1.5824 train_acc=0.9938 aux_loss=0.0265 val_rank1=0.7185 val_mAP=0.3203 val_auc=0.6133 val_coverage=270/270 time=61.5s
Epoch [40/80] lr=0.000018 train_loss=1.6448 train_acc=0.9925 aux_loss=0.0272 val_rank1=0.7185 val_mAP=0.3219 val_auc=0.6141 val_coverage=270/270 time=65.6s
Epoch [41/80] lr=0.000017 train_loss=1.6297 train_acc=0.9944 aux_loss=0.0268 val_rank1=0.7185 val_mAP=0.3239 val_auc=0.6149 val_coverage=270/270 time=62.8s
Epoch [42/80] lr=0.000016 train_loss=1.6241 train_acc=0.9931 aux_loss=0.0258 val_rank1=0.7222 val_mAP=0.3260 val_auc=0.6157 val_coverage=270/270 time=69.4s
  ★ 新最佳! Rank-1=0.7222
Epoch [43/80] lr=0.000016 train_loss=1.6724 train_acc=0.9931 aux_loss=0.0305 val_rank1=0.7222 val_mAP=0.3279 val_auc=0.6165 val_coverage=270/270 time=64.7s
Epoch [44/80] lr=0.000015 train_loss=1.5394 train_acc=0.9969 aux_loss=0.0245 val_rank1=0.7259 val_mAP=0.3314 val_auc=0.6173 val_coverage=270/270 time=64.1s
  ★ 新最佳! Rank-1=0.7259
Epoch [45/80] lr=0.000015 train_loss=1.6024 train_acc=0.9925 aux_loss=0.0250 val_rank1=0.7259 val_mAP=0.3328 val_auc=0.6182 val_coverage=270/270 time=67.7s
Epoch [46/80] lr=0.000014 train_loss=1.5691 train_acc=0.9938 aux_loss=0.0279 val_rank1=0.7296 val_mAP=0.3346 val_auc=0.6191 val_coverage=270/270 time=65.2s
  ★ 新最佳! Rank-1=0.7296
Epoch [47/80] lr=0.000013 train_loss=1.5353 train_acc=0.9900 aux_loss=0.0255 val_rank1=0.7333 val_mAP=0.3374 val_auc=0.6200 val_coverage=270/270 time=65.2s
  ★ 新最佳! Rank-1=0.7333
Epoch [48/80] lr=0.000013 train_loss=1.4792 train_acc=0.9919 aux_loss=0.0200 val_rank1=0.7370 val_mAP=0.3392 val_auc=0.6210 val_coverage=270/270 time=61.8s
  ★ 新最佳! Rank-1=0.7370
Epoch [49/80] lr=0.000012 train_loss=1.4820 train_acc=0.9931 aux_loss=0.0197 val_rank1=0.7370 val_mAP=0.3420 val_auc=0.6220 val_coverage=270/270 time=64.1s
Epoch [50/80] lr=0.000012 train_loss=1.4767 train_acc=0.9962 aux_loss=0.0206 val_rank1=0.7370 val_mAP=0.3441 val_auc=0.6230 val_coverage=270/270 time=63.9s
Epoch [51/80] lr=0.000011 train_loss=1.4424 train_acc=0.9969 aux_loss=0.0217 val_rank1=0.7370 val_mAP=0.3457 val_auc=0.6240 val_coverage=270/270 time=66.1s
Epoch [52/80] lr=0.000010 train_loss=1.4140 train_acc=0.9969 aux_loss=0.0172 val_rank1=0.7407 val_mAP=0.3478 val_auc=0.6250 val_coverage=270/270 time=64.1s
  ★ 新最佳! Rank-1=0.7407
Epoch [53/80] lr=0.000010 train_loss=1.4108 train_acc=0.9962 aux_loss=0.0159 val_rank1=0.7407 val_mAP=0.3499 val_auc=0.6261 val_coverage=270/270 time=69.5s
Epoch [54/80] lr=0.000009 train_loss=1.3666 train_acc=0.9962 aux_loss=0.0168 val_rank1=0.7407 val_mAP=0.3518 val_auc=0.6272 val_coverage=270/270 time=67.0s
Epoch [55/80] lr=0.000009 train_loss=1.3913 train_acc=0.9969 aux_loss=0.0193 val_rank1=0.7370 val_mAP=0.3534 val_auc=0.6283 val_coverage=270/270 time=61.8s
Epoch [56/80] lr=0.000008 train_loss=1.4082 train_acc=0.9962 aux_loss=0.0164 val_rank1=0.7370 val_mAP=0.3551 val_auc=0.6295 val_coverage=270/270 time=64.7s
Epoch [57/80] lr=0.000008 train_loss=1.3377 train_acc=0.9988 aux_loss=0.0134 val_rank1=0.7370 val_mAP=0.3574 val_auc=0.6307 val_coverage=270/270 time=64.7s
Epoch [58/80] lr=0.000007 train_loss=1.4902 train_acc=0.9931 aux_loss=0.0191 val_rank1=0.7444 val_mAP=0.3603 val_auc=0.6319 val_coverage=270/270 time=69.1s
  ★ 新最佳! Rank-1=0.7444
Epoch [59/80] lr=0.000007 train_loss=1.4362 train_acc=0.9944 aux_loss=0.0194 val_rank1=0.7481 val_mAP=0.3627 val_auc=0.6331 val_coverage=270/270 time=63.6s
  ★ 新最佳! Rank-1=0.7481
Epoch [60/80] lr=0.000006 train_loss=1.4105 train_acc=0.9956 aux_loss=0.0189 val_rank1=0.7481 val_mAP=0.3652 val_auc=0.6344 val_coverage=270/270 time=64.3s
Epoch [61/80] lr=0.000006 train_loss=1.3833 train_acc=0.9950 aux_loss=0.0168 val_rank1=0.7481 val_mAP=0.3677 val_auc=0.6357 val_coverage=270/270 time=63.0s
Epoch [62/80] lr=0.000005 train_loss=1.3759 train_acc=0.9944 aux_loss=0.0171 val_rank1=0.7481 val_mAP=0.3696 val_auc=0.6371 val_coverage=270/270 time=65.7s
Epoch [63/80] lr=0.000005 train_loss=1.3012 train_acc=0.9969 aux_loss=0.0126 val_rank1=0.7481 val_mAP=0.3727 val_auc=0.6385 val_coverage=270/270 time=65.4s
Epoch [64/80] lr=0.000005 train_loss=1.4152 train_acc=0.9944 aux_loss=0.0225 val_rank1=0.7481 val_mAP=0.3756 val_auc=0.6399 val_coverage=270/270 time=65.4s
Epoch [65/80] lr=0.000004 train_loss=1.3039 train_acc=0.9988 aux_loss=0.0136 val_rank1=0.7481 val_mAP=0.3780 val_auc=0.6414 val_coverage=270/270 time=64.0s
Epoch [66/80] lr=0.000004 train_loss=1.3376 train_acc=0.9975 aux_loss=0.0149 val_rank1=0.7481 val_mAP=0.3806 val_auc=0.6429 val_coverage=270/270 time=66.3s
Epoch [67/80] lr=0.000003 train_loss=1.2929 train_acc=0.9994 aux_loss=0.0116 val_rank1=0.7481 val_mAP=0.3834 val_auc=0.6445 val_coverage=270/270 time=63.7s
Epoch [68/80] lr=0.000003 train_loss=1.3441 train_acc=0.9981 aux_loss=0.0157 val_rank1=0.7481 val_mAP=0.3858 val_auc=0.6460 val_coverage=270/270 time=64.1s
Epoch [69/80] lr=0.000003 train_loss=1.3761 train_acc=0.9969 aux_loss=0.0190 val_rank1=0.7556 val_mAP=0.3884 val_auc=0.6477 val_coverage=270/270 time=64.9s
  ★ 新最佳! Rank-1=0.7556
Epoch [70/80] lr=0.000003 train_loss=1.3365 train_acc=0.9975 aux_loss=0.0156 val_rank1=0.7519 val_mAP=0.3908 val_auc=0.6494 val_coverage=270/270 time=65.8s
Epoch [71/80] lr=0.000002 train_loss=1.2993 train_acc=0.9969 aux_loss=0.0141 val_rank1=0.7519 val_mAP=0.3938 val_auc=0.6511 val_coverage=270/270 time=66.5s
Epoch [72/80] lr=0.000002 train_loss=1.3189 train_acc=0.9969 aux_loss=0.0136 val_rank1=0.7519 val_mAP=0.3967 val_auc=0.6529 val_coverage=270/270 time=67.1s
Epoch [73/80] lr=0.000002 train_loss=1.3124 train_acc=0.9969 aux_loss=0.0155 val_rank1=0.7519 val_mAP=0.3996 val_auc=0.6547 val_coverage=270/270 time=63.4s
Epoch [74/80] lr=0.000002 train_loss=1.3631 train_acc=0.9975 aux_loss=0.0174 val_rank1=0.7556 val_mAP=0.4026 val_auc=0.6566 val_coverage=270/270 time=61.8s
Epoch [75/80] lr=0.000001 train_loss=1.3587 train_acc=0.9962 aux_loss=0.0153 val_rank1=0.7556 val_mAP=0.4055 val_auc=0.6585 val_coverage=270/270 time=68.1s
Epoch [76/80] lr=0.000001 train_loss=1.3390 train_acc=0.9981 aux_loss=0.0165 val_rank1=0.7556 val_mAP=0.4084 val_auc=0.6605 val_coverage=270/270 time=63.0s
Epoch [77/80] lr=0.000001 train_loss=1.2619 train_acc=0.9962 aux_loss=0.0129 val_rank1=0.7556 val_mAP=0.4126 val_auc=0.6625 val_coverage=270/270 time=66.9s
Epoch [78/80] lr=0.000001 train_loss=1.3445 train_acc=0.9969 aux_loss=0.0163 val_rank1=0.7556 val_mAP=0.4156 val_auc=0.6646 val_coverage=270/270 time=66.7s
Epoch [79/80] lr=0.000001 train_loss=1.3056 train_acc=0.9962 aux_loss=0.0148 val_rank1=0.7556 val_mAP=0.4187 val_auc=0.6667 val_coverage=270/270 time=65.9s
Epoch [80/80] lr=0.000001 train_loss=1.3214 train_acc=0.9975 aux_loss=0.0179 val_rank1=0.7556 val_mAP=0.4222 val_auc=0.6689 val_coverage=270/270 time=62.3s

训练完成! 总时间: 90.7 分钟
最佳 Rank-1: 0.7556 (Epoch 69)

========================================
加载最佳模型进行推理
========================================
  加载 checkpoint: /data0/yzhen/projects/gastrovision_v3/output/jaguar_arcface/best_model.pth
  恢复自 Epoch 69, 最佳 Rank-1=0.7556
  已应用 EMA 权重
测试图片数: 371
提取 embedding: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:30<00:00,  5.00s/it]
读取测试对: /data0/yzhen/data/jaguar_reid/test.csv
共 137270 对
校准参数: median=0.5410, IQR=0.2238
提交文件已保存: /data0/yzhen/projects/gastrovision_v3/output/jaguar_arcface/submission.csv
相似度统计: mean=0.4981, std=0.2027, min=0.0295, max=0.9383

完成!




python jaguar/train.py configs/jaguar_reid.yaml
/data0/yzhen/py3/envs/py310/lib/python3.10/site-packages/albumentations/check_version.py:107: UserWarning: Error fetching version info <urlopen error [Errno 101] Network is unreachable>
  data = fetch_version_info()
============================================================
Jaguar Re-Identification Training
============================================================
时间:   2026-03-20 12:28:52
设备:   cuda
GPU:    NVIDIA GeForce RTX 4090
配置:   /data0/yzhen/projects/gastrovision_v3/configs/jaguar_reid.yaml

========================================
数据划分
========================================
总样本数: 1895
类别数:   31
样本数范围: 13 ~ 183

划分结果:
  训练集: 1625 样本
  验证集: 270 样本
  已保存到: /data0/yzhen/projects/gastrovision_v3/output/jaguar_arcface

类别数: 31

训练集: 1625 样本
验证集: 270 样本
  PK Sampler: P=8, K=4 (batch_size=32)

创建模型: convnext_base
  加载本地预训练权重: /data0/yzhen/data/pretrained/pretrain/convnext_base-6075fbad.pth
  模型参数: 88.09M, 可训练: 88.09M

主损失: arcface
  [primary] arcface: scale=30.0, margin=0.5
辅助损失: triplet (weight=1.0)
  [aux_] triplet: scale=1.0, margin=0.3
优化器: adamw (backbone_lr=0.000030, head_lr=0.000300)
调度器: warmup_cosine (warmup=5)
  EMA 已启用 (decay=0.9999)

========================================
开始训练
========================================

Epoch [1/100] lr=0.000006 train_loss=17.4712 train_acc=0.0531 aux_loss=0.5658 val_rank1=0.6852 val_mAP=0.2825 val_auc=0.6009 val_coverage=270/270 time=61.8s
  ★ 新最佳! Rank-1=0.6852
Epoch [2/100] lr=0.000012 train_loss=16.6269 train_acc=0.2056 aux_loss=0.5122 val_rank1=0.6852 val_mAP=0.2826 val_auc=0.6010 val_coverage=270/270 time=62.5s
Epoch [3/100] lr=0.000018 train_loss=15.4019 train_acc=0.4456 aux_loss=0.4520 val_rank1=0.6852 val_mAP=0.2827 val_auc=0.6011 val_coverage=270/270 time=64.0s
Epoch [4/100] lr=0.000024 train_loss=13.7778 train_acc=0.5956 aux_loss=0.4204 val_rank1=0.6852 val_mAP=0.2832 val_auc=0.6012 val_coverage=270/270 time=68.0s
Epoch [5/100] lr=0.000030 train_loss=11.7092 train_acc=0.7219 aux_loss=0.3692 val_rank1=0.6852 val_mAP=0.2837 val_auc=0.6013 val_coverage=270/270 time=64.2s
Epoch [6/100] lr=0.000030 train_loss=9.9115 train_acc=0.7594 aux_loss=0.3394 val_rank1=0.6852 val_mAP=0.2845 val_auc=0.6015 val_coverage=270/270 time=64.8s
Epoch [7/100] lr=0.000030 train_loss=8.4596 train_acc=0.8131 aux_loss=0.2964 val_rank1=0.6852 val_mAP=0.2852 val_auc=0.6016 val_coverage=270/270 time=61.4s
Epoch [8/100] lr=0.000030 train_loss=7.3800 train_acc=0.8575 aux_loss=0.2727 val_rank1=0.6852 val_mAP=0.2862 val_auc=0.6017 val_coverage=270/270 time=63.2s
Epoch [9/100] lr=0.000030 train_loss=6.2558 train_acc=0.9038 aux_loss=0.2327 val_rank1=0.6852 val_mAP=0.2875 val_auc=0.6018 val_coverage=270/270 time=68.2s
Epoch [10/100] lr=0.000030 train_loss=5.6170 train_acc=0.9069 aux_loss=0.2118 val_rank1=0.6889 val_mAP=0.2888 val_auc=0.6019 val_coverage=270/270 time=64.5s
  ★ 新最佳! Rank-1=0.6889
Epoch [11/100] lr=0.000030 train_loss=4.8056 train_acc=0.9319 aux_loss=0.1795 val_rank1=0.6889 val_mAP=0.2895 val_auc=0.6022 val_coverage=270/270 time=63.6s
Epoch [12/100] lr=0.000030 train_loss=4.4732 train_acc=0.9406 aux_loss=0.1686 val_rank1=0.6926 val_mAP=0.2904 val_auc=0.6024 val_coverage=270/270 time=63.0s
  ★ 新最佳! Rank-1=0.6926
Epoch [13/100] lr=0.000030 train_loss=3.8591 train_acc=0.9487 aux_loss=0.1461 val_rank1=0.6963 val_mAP=0.2912 val_auc=0.6027 val_coverage=270/270 time=67.3s
  ★ 新最佳! Rank-1=0.6963
Epoch [14/100] lr=0.000029 train_loss=3.4903 train_acc=0.9550 aux_loss=0.1248 val_rank1=0.6963 val_mAP=0.2918 val_auc=0.6029 val_coverage=270/270 time=65.0s
Epoch [15/100] lr=0.000029 train_loss=3.2076 train_acc=0.9619 aux_loss=0.1074 val_rank1=0.6926 val_mAP=0.2917 val_auc=0.6032 val_coverage=270/270 time=63.2s
Epoch [16/100] lr=0.000029 train_loss=3.1731 train_acc=0.9613 aux_loss=0.1104 val_rank1=0.6926 val_mAP=0.2929 val_auc=0.6035 val_coverage=270/270 time=60.5s
Epoch [17/100] lr=0.000029 train_loss=3.0024 train_acc=0.9669 aux_loss=0.1011 val_rank1=0.6963 val_mAP=0.2940 val_auc=0.6039 val_coverage=270/270 time=61.8s
Epoch [18/100] lr=0.000029 train_loss=2.6664 train_acc=0.9700 aux_loss=0.0817 val_rank1=0.6963 val_mAP=0.2950 val_auc=0.6042 val_coverage=270/270 time=64.6s
Epoch [19/100] lr=0.000029 train_loss=2.4009 train_acc=0.9812 aux_loss=0.0708 val_rank1=0.7000 val_mAP=0.2961 val_auc=0.6045 val_coverage=270/270 time=62.7s
  ★ 新最佳! Rank-1=0.7000
Epoch [20/100] lr=0.000028 train_loss=2.4420 train_acc=0.9788 aux_loss=0.0741 val_rank1=0.7000 val_mAP=0.2972 val_auc=0.6049 val_coverage=270/270 time=65.1s
Epoch [21/100] lr=0.000028 train_loss=2.2493 train_acc=0.9812 aux_loss=0.0607 val_rank1=0.7000 val_mAP=0.2982 val_auc=0.6053 val_coverage=270/270 time=64.9s
Epoch [22/100] lr=0.000028 train_loss=2.2535 train_acc=0.9794 aux_loss=0.0671 val_rank1=0.7037 val_mAP=0.2994 val_auc=0.6058 val_coverage=270/270 time=64.7s
  ★ 新最佳! Rank-1=0.7037
Epoch [23/100] lr=0.000028 train_loss=2.0865 train_acc=0.9806 aux_loss=0.0546 val_rank1=0.7037 val_mAP=0.3004 val_auc=0.6063 val_coverage=270/270 time=69.1s
Epoch [24/100] lr=0.000028 train_loss=2.0014 train_acc=0.9862 aux_loss=0.0503 val_rank1=0.7074 val_mAP=0.3019 val_auc=0.6067 val_coverage=270/270 time=59.8s
  ★ 新最佳! Rank-1=0.7074
Epoch [25/100] lr=0.000027 train_loss=1.8351 train_acc=0.9912 aux_loss=0.0423 val_rank1=0.7074 val_mAP=0.3029 val_auc=0.6072 val_coverage=270/270 time=62.8s
Epoch [26/100] lr=0.000027 train_loss=1.8665 train_acc=0.9862 aux_loss=0.0458 val_rank1=0.7074 val_mAP=0.3040 val_auc=0.6077 val_coverage=270/270 time=64.8s
Epoch [27/100] lr=0.000027 train_loss=1.8823 train_acc=0.9862 aux_loss=0.0414 val_rank1=0.7148 val_mAP=0.3059 val_auc=0.6082 val_coverage=270/270 time=65.3s
  ★ 新最佳! Rank-1=0.7148
Epoch [28/100] lr=0.000026 train_loss=1.7319 train_acc=0.9925 aux_loss=0.0366 val_rank1=0.7148 val_mAP=0.3070 val_auc=0.6087 val_coverage=270/270 time=62.6s
Epoch [29/100] lr=0.000026 train_loss=1.6081 train_acc=0.9931 aux_loss=0.0284 val_rank1=0.7111 val_mAP=0.3083 val_auc=0.6093 val_coverage=270/270 time=65.3s
Epoch [30/100] lr=0.000026 train_loss=1.7792 train_acc=0.9912 aux_loss=0.0372 val_rank1=0.7111 val_mAP=0.3098 val_auc=0.6098 val_coverage=270/270 time=61.8s
Epoch [31/100] lr=0.000025 train_loss=1.6305 train_acc=0.9944 aux_loss=0.0284 val_rank1=0.7111 val_mAP=0.3109 val_auc=0.6105 val_coverage=270/270 time=64.2s
Epoch [32/100] lr=0.000025 train_loss=1.4783 train_acc=0.9956 aux_loss=0.0225 val_rank1=0.7111 val_mAP=0.3124 val_auc=0.6111 val_coverage=270/270 time=64.3s
Epoch [33/100] lr=0.000025 train_loss=1.5023 train_acc=0.9938 aux_loss=0.0232 val_rank1=0.7111 val_mAP=0.3138 val_auc=0.6118 val_coverage=270/270 time=68.1s
Epoch [34/100] lr=0.000024 train_loss=1.4520 train_acc=0.9962 aux_loss=0.0190 val_rank1=0.7111 val_mAP=0.3153 val_auc=0.6125 val_coverage=270/270 time=63.7s
Epoch [35/100] lr=0.000024 train_loss=1.4841 train_acc=0.9975 aux_loss=0.0216 val_rank1=0.7148 val_mAP=0.3171 val_auc=0.6132 val_coverage=270/270 time=64.4s
Epoch [36/100] lr=0.000023 train_loss=1.4242 train_acc=0.9950 aux_loss=0.0182 val_rank1=0.7111 val_mAP=0.3184 val_auc=0.6139 val_coverage=270/270 time=63.4s
Epoch [37/100] lr=0.000023 train_loss=1.3931 train_acc=0.9962 aux_loss=0.0191 val_rank1=0.7185 val_mAP=0.3219 val_auc=0.6146 val_coverage=270/270 time=69.5s
  ★ 新最佳! Rank-1=0.7185
Epoch [38/100] lr=0.000023 train_loss=1.4377 train_acc=0.9950 aux_loss=0.0191 val_rank1=0.7185 val_mAP=0.3240 val_auc=0.6154 val_coverage=270/270 time=63.8s
Epoch [39/100] lr=0.000022 train_loss=1.3548 train_acc=0.9969 aux_loss=0.0165 val_rank1=0.7185 val_mAP=0.3255 val_auc=0.6163 val_coverage=270/270 time=61.1s
Epoch [40/100] lr=0.000022 train_loss=1.3682 train_acc=0.9950 aux_loss=0.0184 val_rank1=0.7222 val_mAP=0.3272 val_auc=0.6171 val_coverage=270/270 time=65.5s
  ★ 新最佳! Rank-1=0.7222
Epoch [41/100] lr=0.000021 train_loss=1.3333 train_acc=0.9962 aux_loss=0.0149 val_rank1=0.7259 val_mAP=0.3295 val_auc=0.6180 val_coverage=270/270 time=63.7s
  ★ 新最佳! Rank-1=0.7259
Epoch [42/100] lr=0.000021 train_loss=1.3320 train_acc=0.9988 aux_loss=0.0144 val_rank1=0.7259 val_mAP=0.3310 val_auc=0.6189 val_coverage=270/270 time=68.7s
Epoch [43/100] lr=0.000020 train_loss=1.3047 train_acc=0.9981 aux_loss=0.0135 val_rank1=0.7259 val_mAP=0.3330 val_auc=0.6198 val_coverage=270/270 time=63.9s
Epoch [44/100] lr=0.000020 train_loss=1.2664 train_acc=0.9969 aux_loss=0.0131 val_rank1=0.7259 val_mAP=0.3352 val_auc=0.6208 val_coverage=270/270 time=62.8s
Epoch [45/100] lr=0.000020 train_loss=1.2644 train_acc=0.9975 aux_loss=0.0118 val_rank1=0.7259 val_mAP=0.3372 val_auc=0.6217 val_coverage=270/270 time=67.0s
Epoch [46/100] lr=0.000019 train_loss=1.2464 train_acc=0.9981 aux_loss=0.0119 val_rank1=0.7259 val_mAP=0.3390 val_auc=0.6227 val_coverage=270/270 time=63.7s
Epoch [47/100] lr=0.000019 train_loss=1.2230 train_acc=0.9988 aux_loss=0.0103 val_rank1=0.7296 val_mAP=0.3412 val_auc=0.6237 val_coverage=270/270 time=65.1s
  ★ 新最佳! Rank-1=0.7296
Epoch [48/100] lr=0.000018 train_loss=1.2162 train_acc=0.9988 aux_loss=0.0109 val_rank1=0.7333 val_mAP=0.3432 val_auc=0.6248 val_coverage=270/270 time=60.9s
  ★ 新最佳! Rank-1=0.7333
Epoch [49/100] lr=0.000018 train_loss=1.2504 train_acc=0.9981 aux_loss=0.0113 val_rank1=0.7370 val_mAP=0.3452 val_auc=0.6258 val_coverage=270/270 time=63.4s
  ★ 新最佳! Rank-1=0.7370
Epoch [50/100] lr=0.000017 train_loss=1.1941 train_acc=0.9962 aux_loss=0.0092 val_rank1=0.7407 val_mAP=0.3477 val_auc=0.6269 val_coverage=270/270 time=63.3s
  ★ 新最佳! Rank-1=0.7407
Epoch [51/100] lr=0.000017 train_loss=1.1421 train_acc=0.9994 aux_loss=0.0065 val_rank1=0.7444 val_mAP=0.3495 val_auc=0.6280 val_coverage=270/270 time=65.0s
  ★ 新最佳! Rank-1=0.7444
Epoch [52/100] lr=0.000016 train_loss=1.1198 train_acc=1.0000 aux_loss=0.0054 val_rank1=0.7481 val_mAP=0.3522 val_auc=0.6292 val_coverage=270/270 time=63.7s
  ★ 新最佳! Rank-1=0.7481
Epoch [53/100] lr=0.000016 train_loss=1.1529 train_acc=0.9981 aux_loss=0.0074 val_rank1=0.7481 val_mAP=0.3548 val_auc=0.6304 val_coverage=270/270 time=68.6s
Epoch [54/100] lr=0.000015 train_loss=1.0865 train_acc=0.9994 aux_loss=0.0063 val_rank1=0.7481 val_mAP=0.3572 val_auc=0.6315 val_coverage=270/270 time=67.2s
Epoch [55/100] lr=0.000015 train_loss=1.1059 train_acc=0.9981 aux_loss=0.0078 val_rank1=0.7481 val_mAP=0.3592 val_auc=0.6328 val_coverage=270/270 time=61.6s
Epoch [56/100] lr=0.000014 train_loss=1.1470 train_acc=0.9981 aux_loss=0.0094 val_rank1=0.7481 val_mAP=0.3617 val_auc=0.6340 val_coverage=270/270 time=63.0s
Epoch [57/100] lr=0.000014 train_loss=1.0924 train_acc=0.9975 aux_loss=0.0068 val_rank1=0.7444 val_mAP=0.3640 val_auc=0.6353 val_coverage=270/270 time=64.0s
Epoch [58/100] lr=0.000013 train_loss=1.0853 train_acc=1.0000 aux_loss=0.0052 val_rank1=0.7444 val_mAP=0.3664 val_auc=0.6367 val_coverage=270/270 time=69.1s
Epoch [59/100] lr=0.000013 train_loss=1.1486 train_acc=0.9988 aux_loss=0.0075 val_rank1=0.7444 val_mAP=0.3688 val_auc=0.6381 val_coverage=270/270 time=63.0s
Epoch [60/100] lr=0.000012 train_loss=1.0532 train_acc=0.9994 aux_loss=0.0046 val_rank1=0.7481 val_mAP=0.3711 val_auc=0.6395 val_coverage=270/270 time=64.1s
Epoch [61/100] lr=0.000012 train_loss=1.1129 train_acc=0.9962 aux_loss=0.0080 val_rank1=0.7481 val_mAP=0.3737 val_auc=0.6410 val_coverage=270/270 time=62.2s
Epoch [62/100] lr=0.000011 train_loss=1.1038 train_acc=0.9969 aux_loss=0.0081 val_rank1=0.7481 val_mAP=0.3761 val_auc=0.6425 val_coverage=270/270 time=64.7s
Epoch [63/100] lr=0.000011 train_loss=1.0696 train_acc=0.9981 aux_loss=0.0057 val_rank1=0.7519 val_mAP=0.3787 val_auc=0.6441 val_coverage=270/270 time=64.5s
  ★ 新最佳! Rank-1=0.7519
Epoch [64/100] lr=0.000011 train_loss=1.1131 train_acc=0.9969 aux_loss=0.0094 val_rank1=0.7556 val_mAP=0.3817 val_auc=0.6457 val_coverage=270/270 time=65.0s
  ★ 新最佳! Rank-1=0.7556
Epoch [65/100] lr=0.000010 train_loss=1.0825 train_acc=0.9988 aux_loss=0.0069 val_rank1=0.7556 val_mAP=0.3850 val_auc=0.6474 val_coverage=270/270 time=62.9s
Epoch [66/100] lr=0.000010 train_loss=1.0941 train_acc=0.9988 aux_loss=0.0071 val_rank1=0.7556 val_mAP=0.3882 val_auc=0.6492 val_coverage=270/270 time=65.3s
Epoch [67/100] lr=0.000009 train_loss=1.0917 train_acc=0.9981 aux_loss=0.0076 val_rank1=0.7519 val_mAP=0.3909 val_auc=0.6510 val_coverage=270/270 time=63.1s
Epoch [68/100] lr=0.000009 train_loss=0.9834 train_acc=1.0000 aux_loss=0.0023 val_rank1=0.7519 val_mAP=0.3941 val_auc=0.6528 val_coverage=270/270 time=62.7s
Epoch [69/100] lr=0.000008 train_loss=1.0222 train_acc=0.9994 aux_loss=0.0036 val_rank1=0.7556 val_mAP=0.3979 val_auc=0.6547 val_coverage=270/270 time=64.4s
Epoch [70/100] lr=0.000008 train_loss=1.0031 train_acc=0.9994 aux_loss=0.0036 val_rank1=0.7556 val_mAP=0.4008 val_auc=0.6566 val_coverage=270/270 time=65.4s
Epoch [71/100] lr=0.000008 train_loss=1.0367 train_acc=0.9994 aux_loss=0.0061 val_rank1=0.7519 val_mAP=0.4036 val_auc=0.6586 val_coverage=270/270 time=65.0s
Epoch [72/100] lr=0.000007 train_loss=0.9943 train_acc=0.9994 aux_loss=0.0041 val_rank1=0.7519 val_mAP=0.4064 val_auc=0.6606 val_coverage=270/270 time=66.1s
Epoch [73/100] lr=0.000007 train_loss=1.0301 train_acc=0.9981 aux_loss=0.0045 val_rank1=0.7556 val_mAP=0.4100 val_auc=0.6627 val_coverage=270/270 time=62.6s
Epoch [74/100] lr=0.000006 train_loss=1.0645 train_acc=0.9981 aux_loss=0.0068 val_rank1=0.7593 val_mAP=0.4137 val_auc=0.6649 val_coverage=270/270 time=61.4s
  ★ 新最佳! Rank-1=0.7593
Epoch [75/100] lr=0.000006 train_loss=1.0437 train_acc=0.9969 aux_loss=0.0065 val_rank1=0.7593 val_mAP=0.4160 val_auc=0.6671 val_coverage=270/270 time=68.5s
Epoch [76/100] lr=0.000006 train_loss=1.0158 train_acc=1.0000 aux_loss=0.0040 val_rank1=0.7593 val_mAP=0.4191 val_auc=0.6694 val_coverage=270/270 time=62.2s
Epoch [77/100] lr=0.000005 train_loss=0.9984 train_acc=0.9994 aux_loss=0.0035 val_rank1=0.7593 val_mAP=0.4226 val_auc=0.6717 val_coverage=270/270 time=65.9s
Epoch [78/100] lr=0.000005 train_loss=1.0351 train_acc=0.9988 aux_loss=0.0047 val_rank1=0.7593 val_mAP=0.4254 val_auc=0.6741 val_coverage=270/270 time=66.6s
Epoch [79/100] lr=0.000005 train_loss=1.0454 train_acc=0.9988 aux_loss=0.0051 val_rank1=0.7593 val_mAP=0.4283 val_auc=0.6766 val_coverage=270/270 time=64.2s
Epoch [80/100] lr=0.000004 train_loss=1.0219 train_acc=0.9994 aux_loss=0.0044 val_rank1=0.7593 val_mAP=0.4316 val_auc=0.6791 val_coverage=270/270 time=62.2s
Epoch [81/100] lr=0.000004 train_loss=1.0027 train_acc=0.9994 aux_loss=0.0038 val_rank1=0.7593 val_mAP=0.4358 val_auc=0.6817 val_coverage=270/270 time=62.9s
Epoch [82/100] lr=0.000004 train_loss=1.0055 train_acc=0.9981 aux_loss=0.0044 val_rank1=0.7630 val_mAP=0.4395 val_auc=0.6843 val_coverage=270/270 time=63.2s
  ★ 新最佳! Rank-1=0.7630
Epoch [83/100] lr=0.000003 train_loss=0.9955 train_acc=1.0000 aux_loss=0.0039 val_rank1=0.7667 val_mAP=0.4429 val_auc=0.6871 val_coverage=270/270 time=67.9s
  ★ 新最佳! Rank-1=0.7667
Epoch [84/100] lr=0.000003 train_loss=0.9333 train_acc=1.0000 aux_loss=0.0011 val_rank1=0.7667 val_mAP=0.4467 val_auc=0.6898 val_coverage=270/270 time=67.7s
Epoch [85/100] lr=0.000003 train_loss=0.9491 train_acc=0.9994 aux_loss=0.0022 val_rank1=0.7667 val_mAP=0.4506 val_auc=0.6927 val_coverage=270/270 time=65.3s
Epoch [86/100] lr=0.000003 train_loss=0.9987 train_acc=0.9988 aux_loss=0.0049 val_rank1=0.7704 val_mAP=0.4551 val_auc=0.6956 val_coverage=270/270 time=61.2s
  ★ 新最佳! Rank-1=0.7704
Epoch [87/100] lr=0.000003 train_loss=1.0023 train_acc=0.9988 aux_loss=0.0049 val_rank1=0.7704 val_mAP=0.4589 val_auc=0.6985 val_coverage=270/270 time=65.5s
Epoch [88/100] lr=0.000002 train_loss=0.9564 train_acc=0.9981 aux_loss=0.0024 val_rank1=0.7741 val_mAP=0.4637 val_auc=0.7016 val_coverage=270/270 time=62.8s
  ★ 新最佳! Rank-1=0.7741
Epoch [89/100] lr=0.000002 train_loss=0.9655 train_acc=0.9988 aux_loss=0.0031 val_rank1=0.7741 val_mAP=0.4680 val_auc=0.7046 val_coverage=270/270 time=65.8s
Epoch [90/100] lr=0.000002 train_loss=0.9465 train_acc=1.0000 aux_loss=0.0021 val_rank1=0.7741 val_mAP=0.4716 val_auc=0.7078 val_coverage=270/270 time=62.3s
Epoch [91/100] lr=0.000002 train_loss=0.9501 train_acc=0.9988 aux_loss=0.0029 val_rank1=0.7815 val_mAP=0.4765 val_auc=0.7110 val_coverage=270/270 time=61.1s
  ★ 新最佳! Rank-1=0.7815
Epoch [92/100] lr=0.000002 train_loss=0.9653 train_acc=0.9988 aux_loss=0.0036 val_rank1=0.7815 val_mAP=0.4802 val_auc=0.7143 val_coverage=270/270 time=62.4s
Epoch [93/100] lr=0.000002 train_loss=0.9563 train_acc=1.0000 aux_loss=0.0016 val_rank1=0.7889 val_mAP=0.4846 val_auc=0.7177 val_coverage=270/270 time=63.0s
  ★ 新最佳! Rank-1=0.7889
Epoch [94/100] lr=0.000001 train_loss=0.9604 train_acc=1.0000 aux_loss=0.0023 val_rank1=0.7889 val_mAP=0.4888 val_auc=0.7211 val_coverage=270/270 time=65.7s
Epoch [95/100] lr=0.000001 train_loss=0.9630 train_acc=0.9994 aux_loss=0.0026 val_rank1=0.7889 val_mAP=0.4933 val_auc=0.7246 val_coverage=270/270 time=69.3s
Epoch [96/100] lr=0.000001 train_loss=0.9548 train_acc=0.9994 aux_loss=0.0023 val_rank1=0.7889 val_mAP=0.4976 val_auc=0.7281 val_coverage=270/270 time=67.4s
Epoch [97/100] lr=0.000001 train_loss=0.9432 train_acc=0.9994 aux_loss=0.0030 val_rank1=0.7963 val_mAP=0.5041 val_auc=0.7317 val_coverage=270/270 time=63.3s
  ★ 新最佳! Rank-1=0.7963
Epoch [98/100] lr=0.000001 train_loss=0.9293 train_acc=1.0000 aux_loss=0.0010 val_rank1=0.7963 val_mAP=0.5086 val_auc=0.7354 val_coverage=270/270 time=64.5s
Epoch [99/100] lr=0.000001 train_loss=0.9850 train_acc=0.9975 aux_loss=0.0046 val_rank1=0.7963 val_mAP=0.5130 val_auc=0.7391 val_coverage=270/270 time=63.1s
Epoch [100/100] lr=0.000001 train_loss=0.9355 train_acc=0.9994 aux_loss=0.0018 val_rank1=0.7963 val_mAP=0.5169 val_auc=0.7429 val_coverage=270/270 time=62.6s

训练完成! 总时间: 112.4 分钟
最佳 Rank-1: 0.7963 (Epoch 97)

========================================
加载最佳模型进行推理
========================================
  加载 checkpoint: /data0/yzhen/projects/gastrovision_v3/output/jaguar_arcface/best_model.pth
  恢复自 Epoch 97, 最佳 Rank-1=0.7963
  已应用 EMA 权重
测试图片数: 371
提取 embedding: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:29<00:00,  4.95s/it]
读取测试对: /data0/yzhen/data/jaguar_reid/test.csv
共 137270 对
校准参数: median=0.4010, IQR=0.2307
提交文件已保存: /data0/yzhen/projects/gastrovision_v3/output/jaguar_arcface/submission.csv
相似度统计: mean=0.5115, std=0.2021, min=0.0508, max=0.9692

完成!