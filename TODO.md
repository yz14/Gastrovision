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

1. D:\codes\work-projects\Gastrovision_models\train_cls.py这是我训练内镜分类的代码。目前看起来都正常。我复用这套方案来解决https://www.kaggle.com/competitions/jaguar-re-id/overview这个kaggle比赛。我数据都在服务器上/data0/yzhen/data/jaguar_reid下面有csv，train和test文件夹。我把csv文件D:\codes\work-projects\Gastrovision_models\data\train.csv放到这里了，还有test.csv和提交的csv样例。我写好的方案在D:\codes\work-projects\Gastrovision_models\jaguar这里。请先进行一次彻底的，细致的代码审查。我在服务器上进行了训练，训练集损失一直下降，但是验证集指标一直不变，我感觉有问题。其次，度量学习损失需要支持切换，我要全面的对比D:\codes\work-projects\Gastrovision_models\gastrovision\losses\metric_learning.py这里的损失的效果，不同的损失中需要额外参数的请提前给定合理的默认参数。


2. 我用最新的代码在服务器上训练，将测试的结果提交kaggle，结果只有0.4和验证集指标接近。这个代码是不是仍然哪里存在问题?我看模型输出的是512维度的embedding，用这个特征做# Forward: 始终获取 bn_feat 和 raw_feat
            bn_feat, raw_feat = self.model(images, return_both=True)
            print(bn_feat.shape, raw_feat.shape)
            raise

            # ---- 主损失 ----
            # Proxy-based: 使用 bn_feat (BNNeck 后特征, 适合分类目标)
            # Pair-based:  使用 raw_feat (BNNeck 前特征, 适合度量目标)
            if self.is_proxy:
                primary_loss = self.criterion(bn_feat, labels)
            else:
                primary_loss = self.criterion(raw_feat, labels)分类这样是合适的吗？
整个模型，整个训练过程，损失加入的位置是否合适？数据采样等等是否合适？由于这个比赛的数据有一定的特殊性，是否需要预处理数据？以下是训练过程：

 python jaguar/train.py configs/jaguar_reid.yaml
/data0/yzhen/py3/envs/py310/lib/python3.10/site-packages/albumentations/check_version.py:107: UserWarning: Error fetching version info <urlopen error [Errno 101] Network is unreachable>
  data = fetch_version_info()
============================================================
Jaguar Re-Identification Training
============================================================
时间:   2026-03-19 17:49:20
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

创建模型: resnet50
  加载本地预训练权重: /data0/data/resnet50.pth
  模型参数: 24.56M, 可训练: 24.56M

主损失: arcface
  [primary] arcface: scale=30.0, margin=0.5
优化器: adamw (backbone_lr=0.000030, head_lr=0.000300)
调度器: warmup_cosine (warmup=5)
  EMA 已启用 (decay=0.9999)

========================================
开始训练
========================================

Epoch [1/80] lr=0.000006 train_loss=17.6891 train_acc=0.1663 val_rank1=0.7407 val_mAP=0.3120 val_coverage=270/270 time=65.0s
  ★ 新最佳! Rank-1=0.7407
Epoch [2/80] lr=0.000012 train_loss=15.7400 train_acc=0.4562 val_rank1=0.7370 val_mAP=0.3105 val_coverage=270/270 time=67.5s
Epoch [3/80] lr=0.000018 train_loss=13.2394 train_acc=0.6188 val_rank1=0.7296 val_mAP=0.3141 val_coverage=270/270 time=67.6s
Epoch [4/80] lr=0.000024 train_loss=10.4008 train_acc=0.7144 val_rank1=0.7333 val_mAP=0.3141 val_coverage=270/270 time=66.1s
Epoch [5/80] lr=0.000030 train_loss=8.3240 train_acc=0.7694 val_rank1=0.7222 val_mAP=0.3180 val_coverage=270/270 time=68.7s
Epoch [6/80] lr=0.000030 train_loss=6.4879 train_acc=0.8331 val_rank1=0.7296 val_mAP=0.3176 val_coverage=270/270 time=69.6s
Epoch [7/80] lr=0.000030 train_loss=5.2998 train_acc=0.8662 val_rank1=0.7407 val_mAP=0.3249 val_coverage=270/270 time=68.0s
Epoch [8/80] lr=0.000030 train_loss=4.6065 train_acc=0.8881 val_rank1=0.7333 val_mAP=0.3243 val_coverage=270/270 time=66.9s
Epoch [9/80] lr=0.000030 train_loss=3.9682 train_acc=0.8994 val_rank1=0.7333 val_mAP=0.3257 val_coverage=270/270 time=65.0s
Epoch [10/80] lr=0.000030 train_loss=3.5462 train_acc=0.9194 val_rank1=0.7259 val_mAP=0.3198 val_coverage=270/270 time=69.5s
Epoch [11/80] lr=0.000030 train_loss=3.1946 train_acc=0.9287 val_rank1=0.7296 val_mAP=0.3224 val_coverage=270/270 time=66.1s
Epoch [12/80] lr=0.000030 train_loss=2.7218 train_acc=0.9306 val_rank1=0.7296 val_mAP=0.3229 val_coverage=270/270 time=67.0s
Epoch [13/80] lr=0.000029 train_loss=2.5361 train_acc=0.9419 val_rank1=0.7333 val_mAP=0.3245 val_coverage=270/270 time=64.1s
Epoch [14/80] lr=0.000029 train_loss=2.3018 train_acc=0.9419 val_rank1=0.7296 val_mAP=0.3244 val_coverage=270/270 time=69.7s
Epoch [15/80] lr=0.000029 train_loss=2.0155 train_acc=0.9581 val_rank1=0.7296 val_mAP=0.3255 val_coverage=270/270 time=67.6s
Epoch [16/80] lr=0.000029 train_loss=1.7206 train_acc=0.9644 val_rank1=0.7333 val_mAP=0.3253 val_coverage=270/270 time=68.7s
Epoch [17/80] lr=0.000028 train_loss=1.7105 train_acc=0.9631 val_rank1=0.7333 val_mAP=0.3231 val_coverage=270/270 time=64.0s
Epoch [18/80] lr=0.000028 train_loss=1.5959 train_acc=0.9656 val_rank1=0.7333 val_mAP=0.3239 val_coverage=270/270 time=69.3s
Epoch [19/80] lr=0.000028 train_loss=1.5201 train_acc=0.9694 val_rank1=0.7444 val_mAP=0.3285 val_coverage=270/270 time=63.1s
  ★ 新最佳! Rank-1=0.7444
Epoch [20/80] lr=0.000028 train_loss=1.2642 train_acc=0.9750 val_rank1=0.7333 val_mAP=0.3303 val_coverage=270/270 time=63.9s
Epoch [21/80] lr=0.000027 train_loss=1.0902 train_acc=0.9831 val_rank1=0.7370 val_mAP=0.3280 val_coverage=270/270 time=67.1s
Epoch [22/80] lr=0.000027 train_loss=1.1745 train_acc=0.9794 val_rank1=0.7259 val_mAP=0.3289 val_coverage=270/270 time=69.2s
Epoch [23/80] lr=0.000026 train_loss=1.2152 train_acc=0.9800 val_rank1=0.7259 val_mAP=0.3313 val_coverage=270/270 time=72.8s
Epoch [24/80] lr=0.000026 train_loss=0.9604 train_acc=0.9812 val_rank1=0.7296 val_mAP=0.3346 val_coverage=270/270 time=69.3s
Epoch [25/80] lr=0.000026 train_loss=0.8953 train_acc=0.9812 val_rank1=0.7222 val_mAP=0.3330 val_coverage=270/270 time=67.8s
Epoch [26/80] lr=0.000025 train_loss=0.8631 train_acc=0.9862 val_rank1=0.7296 val_mAP=0.3360 val_coverage=270/270 time=67.1s
Epoch [27/80] lr=0.000025 train_loss=0.9375 train_acc=0.9831 val_rank1=0.7296 val_mAP=0.3383 val_coverage=270/270 time=67.2s
Epoch [28/80] lr=0.000024 train_loss=0.7817 train_acc=0.9881 val_rank1=0.7333 val_mAP=0.3366 val_coverage=270/270 time=66.2s
Epoch [29/80] lr=0.000024 train_loss=0.7546 train_acc=0.9881 val_rank1=0.7333 val_mAP=0.3394 val_coverage=270/270 time=66.0s
Epoch [30/80] lr=0.000023 train_loss=0.7506 train_acc=0.9869 val_rank1=0.7407 val_mAP=0.3415 val_coverage=270/270 time=70.8s
Epoch [31/80] lr=0.000023 train_loss=0.8358 train_acc=0.9844 val_rank1=0.7407 val_mAP=0.3451 val_coverage=270/270 time=69.0s
Epoch [32/80] lr=0.000022 train_loss=0.6183 train_acc=0.9912 val_rank1=0.7333 val_mAP=0.3448 val_coverage=270/270 time=67.7s
Epoch [33/80] lr=0.000022 train_loss=0.5073 train_acc=0.9919 val_rank1=0.7370 val_mAP=0.3434 val_coverage=270/270 time=67.7s
Epoch [34/80] lr=0.000021 train_loss=0.5323 train_acc=0.9931 val_rank1=0.7370 val_mAP=0.3484 val_coverage=270/270 time=69.4s
Epoch [35/80] lr=0.000021 train_loss=0.5193 train_acc=0.9906 val_rank1=0.7333 val_mAP=0.3457 val_coverage=270/270 time=66.2s
Epoch [36/80] lr=0.000020 train_loss=0.5262 train_acc=0.9912 val_rank1=0.7444 val_mAP=0.3471 val_coverage=270/270 time=63.7s
Epoch [37/80] lr=0.000019 train_loss=0.5327 train_acc=0.9869 val_rank1=0.7407 val_mAP=0.3487 val_coverage=270/270 time=67.1s
Epoch [38/80] lr=0.000019 train_loss=0.5275 train_acc=0.9906 val_rank1=0.7444 val_mAP=0.3539 val_coverage=270/270 time=69.0s
Epoch [39/80] lr=0.000018 train_loss=0.4416 train_acc=0.9931 val_rank1=0.7519 val_mAP=0.3556 val_coverage=270/270 time=63.8s
  ★ 新最佳! Rank-1=0.7519
Epoch [40/80] lr=0.000018 train_loss=0.3913 train_acc=0.9906 val_rank1=0.7333 val_mAP=0.3522 val_coverage=270/270 time=66.9s
Epoch [41/80] lr=0.000017 train_loss=0.4091 train_acc=0.9944 val_rank1=0.7407 val_mAP=0.3528 val_coverage=270/270 time=63.8s
Epoch [42/80] lr=0.000016 train_loss=0.3585 train_acc=0.9969 val_rank1=0.7444 val_mAP=0.3531 val_coverage=270/270 time=65.2s
Epoch [43/80] lr=0.000016 train_loss=0.3004 train_acc=0.9969 val_rank1=0.7481 val_mAP=0.3552 val_coverage=270/270 time=70.5s
Epoch [44/80] lr=0.000015 train_loss=0.3201 train_acc=0.9950 val_rank1=0.7444 val_mAP=0.3553 val_coverage=270/270 time=67.9s
Epoch [45/80] lr=0.000015 train_loss=0.4066 train_acc=0.9919 val_rank1=0.7481 val_mAP=0.3559 val_coverage=270/270 time=64.1s
Epoch [46/80] lr=0.000014 train_loss=0.2743 train_acc=0.9975 val_rank1=0.7519 val_mAP=0.3593 val_coverage=270/270 time=63.1s
Epoch [47/80] lr=0.000013 train_loss=0.2146 train_acc=0.9975 val_rank1=0.7519 val_mAP=0.3601 val_coverage=270/270 time=67.8s
Epoch [48/80] lr=0.000013 train_loss=0.2596 train_acc=0.9956 val_rank1=0.7519 val_mAP=0.3622 val_coverage=270/270 time=65.1s
Epoch [49/80] lr=0.000012 train_loss=0.2750 train_acc=0.9956 val_rank1=0.7519 val_mAP=0.3631 val_coverage=270/270 time=70.4s
Epoch [50/80] lr=0.000012 train_loss=0.2838 train_acc=0.9962 val_rank1=0.7556 val_mAP=0.3649 val_coverage=270/270 time=66.5s
  ★ 新最佳! Rank-1=0.7556
Epoch [51/80] lr=0.000011 train_loss=0.2407 train_acc=0.9962 val_rank1=0.7519 val_mAP=0.3669 val_coverage=270/270 time=66.4s
Epoch [52/80] lr=0.000010 train_loss=0.1895 train_acc=0.9988 val_rank1=0.7556 val_mAP=0.3683 val_coverage=270/270 time=64.2s
Epoch [53/80] lr=0.000010 train_loss=0.1989 train_acc=0.9969 val_rank1=0.7556 val_mAP=0.3704 val_coverage=270/270 time=68.9s
Epoch [54/80] lr=0.000009 train_loss=0.2696 train_acc=0.9962 val_rank1=0.7556 val_mAP=0.3712 val_coverage=270/270 time=68.1s
Epoch [55/80] lr=0.000009 train_loss=0.2091 train_acc=0.9962 val_rank1=0.7593 val_mAP=0.3737 val_coverage=270/270 time=65.6s
  ★ 新最佳! Rank-1=0.7593
Epoch [56/80] lr=0.000008 train_loss=0.2395 train_acc=0.9956 val_rank1=0.7593 val_mAP=0.3746 val_coverage=270/270 time=66.4s
Epoch [57/80] lr=0.000008 train_loss=0.2805 train_acc=0.9956 val_rank1=0.7556 val_mAP=0.3763 val_coverage=270/270 time=64.1s
Epoch [58/80] lr=0.000007 train_loss=0.2811 train_acc=0.9950 val_rank1=0.7593 val_mAP=0.3771 val_coverage=270/270 time=67.7s
Epoch [59/80] lr=0.000007 train_loss=0.1728 train_acc=0.9981 val_rank1=0.7593 val_mAP=0.3802 val_coverage=270/270 time=66.0s
Epoch [60/80] lr=0.000006 train_loss=0.1395 train_acc=0.9988 val_rank1=0.7593 val_mAP=0.3801 val_coverage=270/270 time=68.7s
Epoch [61/80] lr=0.000006 train_loss=0.1563 train_acc=0.9994 val_rank1=0.7519 val_mAP=0.3841 val_coverage=270/270 time=68.9s
Epoch [62/80] lr=0.000005 train_loss=0.2222 train_acc=0.9975 val_rank1=0.7556 val_mAP=0.3845 val_coverage=270/270 time=65.4s
Epoch [63/80] lr=0.000005 train_loss=0.1813 train_acc=0.9994 val_rank1=0.7519 val_mAP=0.3863 val_coverage=270/270 time=61.9s
Epoch [64/80] lr=0.000005 train_loss=0.1341 train_acc=0.9988 val_rank1=0.7519 val_mAP=0.3893 val_coverage=270/270 time=71.5s
Epoch [65/80] lr=0.000004 train_loss=0.1825 train_acc=0.9988 val_rank1=0.7556 val_mAP=0.3918 val_coverage=270/270 time=64.3s
Epoch [66/80] lr=0.000004 train_loss=0.1675 train_acc=0.9975 val_rank1=0.7519 val_mAP=0.3928 val_coverage=270/270 time=66.8s
Epoch [67/80] lr=0.000003 train_loss=0.1755 train_acc=0.9981 val_rank1=0.7519 val_mAP=0.3949 val_coverage=270/270 time=67.9s
Epoch [68/80] lr=0.000003 train_loss=0.2086 train_acc=0.9962 val_rank1=0.7519 val_mAP=0.3977 val_coverage=270/270 time=68.2s
Epoch [69/80] lr=0.000003 train_loss=0.1236 train_acc=0.9981 val_rank1=0.7519 val_mAP=0.4005 val_coverage=270/270 time=67.1s
Epoch [70/80] lr=0.000003 train_loss=0.0817 train_acc=1.0000 val_rank1=0.7593 val_mAP=0.4028 val_coverage=270/270 time=66.7s
Epoch [71/80] lr=0.000002 train_loss=0.1132 train_acc=0.9988 val_rank1=0.7593 val_mAP=0.4042 val_coverage=270/270 time=66.1s
Epoch [72/80] lr=0.000002 train_loss=0.1816 train_acc=0.9962 val_rank1=0.7593 val_mAP=0.4079 val_coverage=270/270 time=68.2s
Epoch [73/80] lr=0.000002 train_loss=0.1350 train_acc=0.9981 val_rank1=0.7630 val_mAP=0.4128 val_coverage=270/270 time=66.7s
  ★ 新最佳! Rank-1=0.7630
Epoch [74/80] lr=0.000002 train_loss=0.1403 train_acc=0.9981 val_rank1=0.7556 val_mAP=0.4124 val_coverage=270/270 time=68.4s
Epoch [75/80] lr=0.000001 train_loss=0.0964 train_acc=0.9994 val_rank1=0.7593 val_mAP=0.4146 val_coverage=270/270 time=67.0s
Epoch [76/80] lr=0.000001 train_loss=0.1571 train_acc=0.9988 val_rank1=0.7704 val_mAP=0.4215 val_coverage=270/270 time=69.0s
  ★ 新最佳! Rank-1=0.7704
Epoch [77/80] lr=0.000001 train_loss=0.0978 train_acc=0.9994 val_rank1=0.7667 val_mAP=0.4225 val_coverage=270/270 time=68.5s
Epoch [78/80] lr=0.000001 train_loss=0.1297 train_acc=0.9981 val_rank1=0.7667 val_mAP=0.4265 val_coverage=270/270 time=69.0s
Epoch [79/80] lr=0.000001 train_loss=0.1072 train_acc=0.9988 val_rank1=0.7704 val_mAP=0.4304 val_coverage=270/270 time=67.2s
Epoch [80/80] lr=0.000001 train_loss=0.1283 train_acc=0.9988 val_rank1=0.7778 val_mAP=0.4329 val_coverage=270/270 time=67.7s
  ★ 新最佳! Rank-1=0.7778

训练完成! 总时间: 89.9 分钟
最佳 Rank-1: 0.7778 (Epoch 80)

========================================
加载最佳模型进行推理
========================================
  加载 checkpoint: /data0/yzhen/projects/gastrovision_v3/output/jaguar_arcface/best_model.pth
  恢复自 Epoch 80, 最佳 Rank-1=0.7778
  已应用 EMA 权重
测试图片数: 371
提取 embedding: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:31<00:00,  5.23s/it]
读取测试对: /data0/yzhen/data/jaguar_reid/test.csv
共 137270 对
提交文件已保存: /data0/yzhen/projects/gastrovision_v3/output/jaguar_arcface/submission.csv
相似度统计: mean=0.8821, std=0.0299, min=0.7480, max=0.9986

完成!