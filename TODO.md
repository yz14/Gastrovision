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

2. mlp训练的指标反而低了，提交测试只有0.909
  data = fetch_version_info()
============================================================
Jaguar Re-Identification Training
============================================================
时间:   2026-03-31 16:25:48
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
  已保存到: /data0/yzhen/projects/gastrovision_v3/output/jaguar_arcface_mlp

类别数: 31

训练集: 1625 样本
验证集: 270 样本
  PK Sampler: P=8, K=4 (batch_size=32)

创建模型: convnext_base
  加载本地预训练权重: /data0/yzhen/data/pretrained/pretrain/convnext_base-6075fbad.pth
  [统计] 成功加载: 344/344 个参数
  Embedding Head: MLP (1024→512→512)
  模型参数: 88.35M, 可训练: 88.35M
  [解冻] Backbone 与 Head 一起训练

主损失: arcface
  [primary] arcface: scale=30.0, margin=0.5
辅助损失: triplet (weight=1.0)
  [aux_] triplet: scale=1.0, margin=0.3
优化器: adamw (backbone_lr=0.000030, head_lr=0.000300)
调度器: warmup_cosine (warmup=5)
  EMA 已启用 (decay=0.9)

========================================
开始训练
========================================

  最佳模型选择指标: val_mAP
  [DEBUG] EMA 诊断: decay=0.9, steps/epoch=50, total_steps=10000, 初始权重保留率=0.0000 (训练结束时 EMA 仅吸收 100.0% 的新参数)
  [DEBUG] Embedding: sim_all=[0.074, 0.515, 0.972] pos_sim=0.536 neg_sim=0.514 gap=0.022 emb_norm=1.000
Epoch [1/200] lr=0.000006 train_loss=17.5165 train_acc=0.0431 aux_loss=0.5518 val_rank1=0.5593 val_mAP=0.1969 val_auc=0.5302 val_coverage=270/270 time=69.5s
  ★ 新最佳! mAP=0.1969
  [DEBUG] Embedding: sim_all=[0.063, 0.415, 0.956] pos_sim=0.454 neg_sim=0.413 gap=0.042 emb_norm=1.000
Epoch [2/200] lr=0.000012 train_loss=17.0727 train_acc=0.1125 aux_loss=0.5256 val_rank1=0.5667 val_mAP=0.2015 val_auc=0.5716 val_coverage=270/270 time=70.3s
  ★ 新最佳! mAP=0.2015
  [DEBUG] Embedding: sim_all=[0.079, 0.412, 0.967] pos_sim=0.468 neg_sim=0.409 gap=0.059 emb_norm=1.000
Epoch [3/200] lr=0.000018 train_loss=16.4625 train_acc=0.2375 aux_loss=0.4860 val_rank1=0.6296 val_mAP=0.2422 val_auc=0.6042 val_coverage=270/270 time=69.9s
  ★ 新最佳! mAP=0.2422
Epoch [4/200] lr=0.000024 train_loss=15.3447 train_acc=0.4306 aux_loss=0.4425 val_rank1=0.7074 val_mAP=0.3175 val_auc=0.6631 val_coverage=270/270 time=72.8s
  ★ 新最佳! mAP=0.3175
Epoch [5/200] lr=0.000030 train_loss=13.4963 train_acc=0.6238 aux_loss=0.3872 val_rank1=0.7593 val_mAP=0.4232 val_auc=0.7490 val_coverage=270/270 time=69.2s
  ★ 新最佳! mAP=0.4232
Epoch [6/200] lr=0.000030 train_loss=11.5387 train_acc=0.7125 aux_loss=0.3536 val_rank1=0.8148 val_mAP=0.5167 val_auc=0.8189 val_coverage=270/270 time=71.8s
  ★ 新最佳! mAP=0.5167
Epoch [7/200] lr=0.000030 train_loss=9.5639 train_acc=0.7762 aux_loss=0.3030 val_rank1=0.8000 val_mAP=0.5887 val_auc=0.8734 val_coverage=270/270 time=70.9s
  ★ 新最佳! mAP=0.5887
Epoch [8/200] lr=0.000030 train_loss=7.9986 train_acc=0.8244 aux_loss=0.2739 val_rank1=0.8222 val_mAP=0.6537 val_auc=0.9061 val_coverage=270/270 time=69.3s
  ★ 新最佳! mAP=0.6537
Epoch [9/200] lr=0.000030 train_loss=6.6699 train_acc=0.8831 aux_loss=0.2270 val_rank1=0.8593 val_mAP=0.7093 val_auc=0.9210 val_coverage=270/270 time=71.3s
  ★ 新最佳! mAP=0.7093
  [DEBUG] Embedding: sim_all=[-0.079, 0.259, 0.990] pos_sim=0.598 neg_sim=0.241 gap=0.357 emb_norm=1.000
Epoch [10/200] lr=0.000030 train_loss=5.9395 train_acc=0.8819 aux_loss=0.2073 val_rank1=0.8778 val_mAP=0.7619 val_auc=0.9434 val_coverage=270/270 time=70.9s
  ★ 新最佳! mAP=0.7619
Epoch [11/200] lr=0.000030 train_loss=4.8636 train_acc=0.9275 aux_loss=0.1689 val_rank1=0.8852 val_mAP=0.7958 val_auc=0.9577 val_coverage=270/270 time=70.4s
  ★ 新最佳! mAP=0.7958
Epoch [12/200] lr=0.000030 train_loss=4.3619 train_acc=0.9363 aux_loss=0.1463 val_rank1=0.8963 val_mAP=0.8128 val_auc=0.9623 val_coverage=270/270 time=70.6s
  ★ 新最佳! mAP=0.8128
Epoch [13/200] lr=0.000030 train_loss=3.8991 train_acc=0.9444 aux_loss=0.1329 val_rank1=0.8963 val_mAP=0.8240 val_auc=0.9665 val_coverage=270/270 time=72.2s
  ★ 新最佳! mAP=0.8240
Epoch [14/200] lr=0.000030 train_loss=3.6593 train_acc=0.9413 aux_loss=0.1134 val_rank1=0.9000 val_mAP=0.8400 val_auc=0.9722 val_coverage=270/270 time=69.7s
  ★ 新最佳! mAP=0.8400
Epoch [15/200] lr=0.000030 train_loss=3.2282 train_acc=0.9575 aux_loss=0.0961 val_rank1=0.9111 val_mAP=0.8409 val_auc=0.9713 val_coverage=270/270 time=70.2s
  ★ 新最佳! mAP=0.8409
Epoch [16/200] lr=0.000030 train_loss=3.0315 train_acc=0.9625 aux_loss=0.0881 val_rank1=0.9074 val_mAP=0.8574 val_auc=0.9782 val_coverage=270/270 time=68.9s
  ★ 新最佳! mAP=0.8574
Epoch [17/200] lr=0.000030 train_loss=2.8054 train_acc=0.9700 aux_loss=0.0742 val_rank1=0.9111 val_mAP=0.8654 val_auc=0.9821 val_coverage=270/270 time=68.8s
  ★ 新最佳! mAP=0.8654
Epoch [18/200] lr=0.000030 train_loss=2.6635 train_acc=0.9656 aux_loss=0.0690 val_rank1=0.9148 val_mAP=0.8763 val_auc=0.9845 val_coverage=270/270 time=71.2s
  ★ 新最佳! mAP=0.8763
Epoch [19/200] lr=0.000030 train_loss=2.3669 train_acc=0.9800 aux_loss=0.0537 val_rank1=0.9185 val_mAP=0.8791 val_auc=0.9832 val_coverage=270/270 time=70.7s
  ★ 新最佳! mAP=0.8791
  [DEBUG] Embedding: sim_all=[-0.259, 0.139, 0.994] pos_sim=0.703 neg_sim=0.108 gap=0.596 emb_norm=1.000
Epoch [20/200] lr=0.000030 train_loss=2.3935 train_acc=0.9750 aux_loss=0.0562 val_rank1=0.9296 val_mAP=0.8899 val_auc=0.9874 val_coverage=270/270 time=72.8s
  ★ 新最佳! mAP=0.8899
Epoch [21/200] lr=0.000030 train_loss=2.2018 train_acc=0.9806 aux_loss=0.0439 val_rank1=0.9259 val_mAP=0.8905 val_auc=0.9873 val_coverage=270/270 time=70.4s
  ★ 新最佳! mAP=0.8905
Epoch [22/200] lr=0.000030 train_loss=2.1564 train_acc=0.9800 aux_loss=0.0466 val_rank1=0.9259 val_mAP=0.8938 val_auc=0.9884 val_coverage=270/270 time=72.6s
  ★ 新最佳! mAP=0.8938
Epoch [23/200] lr=0.000029 train_loss=2.0949 train_acc=0.9819 aux_loss=0.0416 val_rank1=0.9259 val_mAP=0.8960 val_auc=0.9902 val_coverage=270/270 time=74.9s
  ★ 新最佳! mAP=0.8960
Epoch [24/200] lr=0.000029 train_loss=2.0084 train_acc=0.9900 aux_loss=0.0389 val_rank1=0.9259 val_mAP=0.8969 val_auc=0.9892 val_coverage=270/270 time=69.6s
  ★ 新最佳! mAP=0.8969
Epoch [25/200] lr=0.000029 train_loss=1.7864 train_acc=0.9944 aux_loss=0.0245 val_rank1=0.9296 val_mAP=0.9036 val_auc=0.9899 val_coverage=270/270 time=72.8s
  ★ 新最佳! mAP=0.9036
Epoch [26/200] lr=0.000029 train_loss=1.8582 train_acc=0.9900 aux_loss=0.0272 val_rank1=0.9333 val_mAP=0.9036 val_auc=0.9903 val_coverage=270/270 time=71.2s
  ★ 新最佳! mAP=0.9036
Epoch [27/200] lr=0.000029 train_loss=1.8289 train_acc=0.9894 aux_loss=0.0262 val_rank1=0.9333 val_mAP=0.9056 val_auc=0.9902 val_coverage=270/270 time=73.4s
  ★ 新最佳! mAP=0.9056
Epoch [28/200] lr=0.000029 train_loss=1.7910 train_acc=0.9912 aux_loss=0.0244 val_rank1=0.9333 val_mAP=0.9045 val_auc=0.9892 val_coverage=270/270 time=72.1s
Epoch [29/200] lr=0.000029 train_loss=1.6476 train_acc=0.9931 aux_loss=0.0182 val_rank1=0.9333 val_mAP=0.9095 val_auc=0.9890 val_coverage=270/270 time=70.7s
  ★ 新最佳! mAP=0.9095
  [DEBUG] Embedding: sim_all=[-0.330, 0.099, 0.995] pos_sim=0.736 neg_sim=0.064 gap=0.672 emb_norm=1.000
Epoch [30/200] lr=0.000029 train_loss=1.8777 train_acc=0.9881 aux_loss=0.0291 val_rank1=0.9333 val_mAP=0.9037 val_auc=0.9886 val_coverage=270/270 time=72.2s
Epoch [31/200] lr=0.000029 train_loss=1.6347 train_acc=0.9944 aux_loss=0.0180 val_rank1=0.9370 val_mAP=0.9099 val_auc=0.9870 val_coverage=270/270 time=70.7s
  ★ 新最佳! mAP=0.9099
Epoch [32/200] lr=0.000029 train_loss=1.5954 train_acc=0.9938 aux_loss=0.0188 val_rank1=0.9333 val_mAP=0.9118 val_auc=0.9889 val_coverage=270/270 time=71.8s
  ★ 新最佳! mAP=0.9118
Epoch [33/200] lr=0.000029 train_loss=1.5099 train_acc=0.9944 aux_loss=0.0143 val_rank1=0.9333 val_mAP=0.9124 val_auc=0.9901 val_coverage=270/270 time=76.9s
  ★ 新最佳! mAP=0.9124
Epoch [34/200] lr=0.000029 train_loss=1.6191 train_acc=0.9931 aux_loss=0.0197 val_rank1=0.9444 val_mAP=0.9142 val_auc=0.9887 val_coverage=270/270 time=72.2s
  ★ 新最佳! mAP=0.9142
Epoch [35/200] lr=0.000028 train_loss=1.5569 train_acc=0.9931 aux_loss=0.0169 val_rank1=0.9407 val_mAP=0.9143 val_auc=0.9877 val_coverage=270/270 time=72.9s
  ★ 新最佳! mAP=0.9143
Epoch [36/200] lr=0.000028 train_loss=1.4904 train_acc=0.9969 aux_loss=0.0100 val_rank1=0.9407 val_mAP=0.9181 val_auc=0.9898 val_coverage=270/270 time=71.7s
  ★ 新最佳! mAP=0.9181
Epoch [37/200] lr=0.000028 train_loss=1.5134 train_acc=0.9956 aux_loss=0.0129 val_rank1=0.9407 val_mAP=0.9184 val_auc=0.9905 val_coverage=270/270 time=73.8s
  ★ 新最佳! mAP=0.9184
Epoch [38/200] lr=0.000028 train_loss=1.5511 train_acc=0.9919 aux_loss=0.0145 val_rank1=0.9407 val_mAP=0.9193 val_auc=0.9931 val_coverage=270/270 time=70.4s
  ★ 新最佳! mAP=0.9193
Epoch [39/200] lr=0.000028 train_loss=1.4338 train_acc=0.9956 aux_loss=0.0107 val_rank1=0.9370 val_mAP=0.9148 val_auc=0.9888 val_coverage=270/270 time=70.9s
  [DEBUG] Embedding: sim_all=[-0.371, 0.075, 0.995] pos_sim=0.757 neg_sim=0.038 gap=0.719 emb_norm=1.000
Epoch [40/200] lr=0.000028 train_loss=1.4253 train_acc=0.9969 aux_loss=0.0098 val_rank1=0.9407 val_mAP=0.9150 val_auc=0.9893 val_coverage=270/270 time=73.3s
Epoch [41/200] lr=0.000028 train_loss=1.3524 train_acc=0.9975 aux_loss=0.0073 val_rank1=0.9370 val_mAP=0.9159 val_auc=0.9914 val_coverage=270/270 time=72.4s
Epoch [42/200] lr=0.000028 train_loss=1.3422 train_acc=0.9962 aux_loss=0.0079 val_rank1=0.9407 val_mAP=0.9130 val_auc=0.9884 val_coverage=270/270 time=71.8s
Epoch [43/200] lr=0.000027 train_loss=1.3558 train_acc=0.9988 aux_loss=0.0080 val_rank1=0.9444 val_mAP=0.9168 val_auc=0.9888 val_coverage=270/270 time=71.5s
Epoch [44/200] lr=0.000027 train_loss=1.4234 train_acc=0.9969 aux_loss=0.0092 val_rank1=0.9407 val_mAP=0.9162 val_auc=0.9909 val_coverage=270/270 time=70.9s
Epoch [45/200] lr=0.000027 train_loss=1.3357 train_acc=0.9969 aux_loss=0.0078 val_rank1=0.9444 val_mAP=0.9194 val_auc=0.9909 val_coverage=270/270 time=71.5s
  ★ 新最佳! mAP=0.9194
Epoch [46/200] lr=0.000027 train_loss=1.3726 train_acc=0.9969 aux_loss=0.0085 val_rank1=0.9370 val_mAP=0.9143 val_auc=0.9898 val_coverage=270/270 time=71.5s
Epoch [47/200] lr=0.000027 train_loss=1.2853 train_acc=0.9994 aux_loss=0.0049 val_rank1=0.9370 val_mAP=0.9179 val_auc=0.9924 val_coverage=270/270 time=70.7s
Epoch [48/200] lr=0.000027 train_loss=1.2772 train_acc=0.9975 aux_loss=0.0041 val_rank1=0.9370 val_mAP=0.9190 val_auc=0.9930 val_coverage=270/270 time=69.4s
Epoch [49/200] lr=0.000027 train_loss=1.3049 train_acc=0.9981 aux_loss=0.0042 val_rank1=0.9444 val_mAP=0.9166 val_auc=0.9912 val_coverage=270/270 time=70.8s
  [DEBUG] Embedding: sim_all=[-0.388, 0.065, 0.996] pos_sim=0.780 neg_sim=0.026 gap=0.754 emb_norm=1.000
Epoch [50/200] lr=0.000027 train_loss=1.2514 train_acc=0.9988 aux_loss=0.0040 val_rank1=0.9370 val_mAP=0.9174 val_auc=0.9922 val_coverage=270/270 time=71.8s
Epoch [51/200] lr=0.000026 train_loss=1.3052 train_acc=0.9956 aux_loss=0.0063 val_rank1=0.9407 val_mAP=0.9189 val_auc=0.9922 val_coverage=270/270 time=73.5s
Epoch [52/200] lr=0.000026 train_loss=1.2155 train_acc=0.9981 aux_loss=0.0042 val_rank1=0.9444 val_mAP=0.9240 val_auc=0.9920 val_coverage=270/270 time=73.4s
  ★ 新最佳! mAP=0.9240
Epoch [53/200] lr=0.000026 train_loss=1.2560 train_acc=0.9975 aux_loss=0.0064 val_rank1=0.9444 val_mAP=0.9223 val_auc=0.9929 val_coverage=270/270 time=74.8s
Epoch [54/200] lr=0.000026 train_loss=1.1880 train_acc=0.9981 aux_loss=0.0033 val_rank1=0.9407 val_mAP=0.9211 val_auc=0.9925 val_coverage=270/270 time=71.1s
Epoch [55/200] lr=0.000026 train_loss=1.2361 train_acc=0.9969 aux_loss=0.0058 val_rank1=0.9444 val_mAP=0.9224 val_auc=0.9934 val_coverage=270/270 time=69.9s
Epoch [56/200] lr=0.000026 train_loss=1.2672 train_acc=0.9962 aux_loss=0.0072 val_rank1=0.9444 val_mAP=0.9209 val_auc=0.9918 val_coverage=270/270 time=71.5s
Epoch [57/200] lr=0.000025 train_loss=1.2272 train_acc=0.9962 aux_loss=0.0053 val_rank1=0.9407 val_mAP=0.9191 val_auc=0.9911 val_coverage=270/270 time=71.0s
Epoch [58/200] lr=0.000025 train_loss=1.1781 train_acc=0.9994 aux_loss=0.0025 val_rank1=0.9407 val_mAP=0.9186 val_auc=0.9917 val_coverage=270/270 time=72.5s
Epoch [59/200] lr=0.000025 train_loss=1.2036 train_acc=1.0000 aux_loss=0.0034 val_rank1=0.9444 val_mAP=0.9202 val_auc=0.9923 val_coverage=270/270 time=70.3s
  [DEBUG] Embedding: sim_all=[-0.414, 0.052, 0.996] pos_sim=0.796 neg_sim=0.012 gap=0.784 emb_norm=1.000
Epoch [60/200] lr=0.000025 train_loss=1.1416 train_acc=0.9994 aux_loss=0.0028 val_rank1=0.9481 val_mAP=0.9247 val_auc=0.9937 val_coverage=270/270 time=74.1s
  ★ 新最佳! mAP=0.9247
Epoch [61/200] lr=0.000025 train_loss=1.1697 train_acc=0.9981 aux_loss=0.0024 val_rank1=0.9481 val_mAP=0.9167 val_auc=0.9908 val_coverage=270/270 time=69.3s
Epoch [62/200] lr=0.000024 train_loss=1.1569 train_acc=0.9988 aux_loss=0.0028 val_rank1=0.9370 val_mAP=0.9136 val_auc=0.9902 val_coverage=270/270 time=71.9s
Epoch [63/200] lr=0.000024 train_loss=1.1233 train_acc=1.0000 aux_loss=0.0015 val_rank1=0.9407 val_mAP=0.9176 val_auc=0.9905 val_coverage=270/270 time=69.4s
Epoch [64/200] lr=0.000024 train_loss=1.1296 train_acc=0.9994 aux_loss=0.0027 val_rank1=0.9370 val_mAP=0.9134 val_auc=0.9901 val_coverage=270/270 time=70.3s
Epoch [65/200] lr=0.000024 train_loss=1.1301 train_acc=1.0000 aux_loss=0.0015 val_rank1=0.9444 val_mAP=0.9156 val_auc=0.9898 val_coverage=270/270 time=70.0s
Epoch [66/200] lr=0.000024 train_loss=1.1138 train_acc=0.9988 aux_loss=0.0017 val_rank1=0.9407 val_mAP=0.9156 val_auc=0.9905 val_coverage=270/270 time=73.9s
Epoch [67/200] lr=0.000024 train_loss=1.0943 train_acc=0.9994 aux_loss=0.0022 val_rank1=0.9407 val_mAP=0.9200 val_auc=0.9929 val_coverage=270/270 time=73.0s
Epoch [68/200] lr=0.000023 train_loss=1.1125 train_acc=0.9988 aux_loss=0.0023 val_rank1=0.9444 val_mAP=0.9210 val_auc=0.9927 val_coverage=270/270 time=70.7s
Epoch [69/200] lr=0.000023 train_loss=1.1103 train_acc=0.9981 aux_loss=0.0020 val_rank1=0.9444 val_mAP=0.9216 val_auc=0.9935 val_coverage=270/270 time=73.3s
  [DEBUG] Embedding: sim_all=[-0.422, 0.049, 0.998] pos_sim=0.809 neg_sim=0.007 gap=0.802 emb_norm=1.000
Epoch [70/200] lr=0.000023 train_loss=1.1025 train_acc=0.9981 aux_loss=0.0029 val_rank1=0.9444 val_mAP=0.9219 val_auc=0.9938 val_coverage=270/270 time=73.7s
Epoch [71/200] lr=0.000023 train_loss=1.0878 train_acc=1.0000 aux_loss=0.0011 val_rank1=0.9444 val_mAP=0.9181 val_auc=0.9904 val_coverage=270/270 time=74.2s
Epoch [72/200] lr=0.000023 train_loss=1.0652 train_acc=0.9994 aux_loss=0.0010 val_rank1=0.9444 val_mAP=0.9175 val_auc=0.9884 val_coverage=270/270 time=70.3s
Epoch [73/200] lr=0.000022 train_loss=1.0546 train_acc=0.9994 aux_loss=0.0012 val_rank1=0.9444 val_mAP=0.9201 val_auc=0.9899 val_coverage=270/270 time=70.3s
Epoch [74/200] lr=0.000022 train_loss=1.0497 train_acc=0.9988 aux_loss=0.0012 val_rank1=0.9444 val_mAP=0.9190 val_auc=0.9905 val_coverage=270/270 time=71.9s
Epoch [75/200] lr=0.000022 train_loss=1.0624 train_acc=0.9988 aux_loss=0.0019 val_rank1=0.9370 val_mAP=0.9158 val_auc=0.9905 val_coverage=270/270 time=73.9s
Epoch [76/200] lr=0.000022 train_loss=1.0731 train_acc=0.9994 aux_loss=0.0024 val_rank1=0.9444 val_mAP=0.9200 val_auc=0.9893 val_coverage=270/270 time=73.4s
Epoch [77/200] lr=0.000022 train_loss=1.0427 train_acc=0.9994 aux_loss=0.0015 val_rank1=0.9481 val_mAP=0.9209 val_auc=0.9923 val_coverage=270/270 time=74.2s
Epoch [78/200] lr=0.000021 train_loss=1.0594 train_acc=0.9988 aux_loss=0.0011 val_rank1=0.9481 val_mAP=0.9207 val_auc=0.9898 val_coverage=270/270 time=72.7s
Epoch [79/200] lr=0.000021 train_loss=0.9885 train_acc=0.9994 aux_loss=0.0006 val_rank1=0.9444 val_mAP=0.9217 val_auc=0.9917 val_coverage=270/270 time=73.4s
  [DEBUG] Embedding: sim_all=[-0.433, 0.049, 0.998] pos_sim=0.823 neg_sim=0.007 gap=0.816 emb_norm=1.000
Epoch [80/200] lr=0.000021 train_loss=1.0112 train_acc=0.9994 aux_loss=0.0016 val_rank1=0.9407 val_mAP=0.9175 val_auc=0.9898 val_coverage=270/270 time=71.7s
Epoch [81/200] lr=0.000021 train_loss=1.0283 train_acc=1.0000 aux_loss=0.0006 val_rank1=0.9407 val_mAP=0.9214 val_auc=0.9937 val_coverage=270/270 time=72.4s
Epoch [82/200] lr=0.000020 train_loss=1.0025 train_acc=0.9994 aux_loss=0.0014 val_rank1=0.9370 val_mAP=0.9202 val_auc=0.9939 val_coverage=270/270 time=72.4s
Epoch [83/200] lr=0.000020 train_loss=0.9785 train_acc=0.9994 aux_loss=0.0010 val_rank1=0.9333 val_mAP=0.9179 val_auc=0.9922 val_coverage=270/270 time=73.0s
Epoch [84/200] lr=0.000020 train_loss=0.9941 train_acc=0.9994 aux_loss=0.0003 val_rank1=0.9333 val_mAP=0.9170 val_auc=0.9902 val_coverage=270/270 time=74.2s
Epoch [85/200] lr=0.000020 train_loss=0.9845 train_acc=0.9994 aux_loss=0.0005 val_rank1=0.9407 val_mAP=0.9164 val_auc=0.9915 val_coverage=270/270 time=71.5s
Epoch [86/200] lr=0.000020 train_loss=0.9921 train_acc=0.9994 aux_loss=0.0018 val_rank1=0.9370 val_mAP=0.9175 val_auc=0.9906 val_coverage=270/270 time=72.2s
Epoch [87/200] lr=0.000019 train_loss=1.0147 train_acc=0.9988 aux_loss=0.0018 val_rank1=0.9370 val_mAP=0.9159 val_auc=0.9886 val_coverage=270/270 time=72.0s
Epoch [88/200] lr=0.000019 train_loss=0.9842 train_acc=0.9988 aux_loss=0.0012 val_rank1=0.9370 val_mAP=0.9187 val_auc=0.9926 val_coverage=270/270 time=70.9s
Epoch [89/200] lr=0.000019 train_loss=0.9879 train_acc=0.9994 aux_loss=0.0017 val_rank1=0.9444 val_mAP=0.9218 val_auc=0.9904 val_coverage=270/270 time=72.9s
  [DEBUG] Embedding: sim_all=[-0.421, 0.037, 0.999] pos_sim=0.833 neg_sim=-0.007 gap=0.840 emb_norm=1.000
Epoch [90/200] lr=0.000019 train_loss=0.9767 train_acc=1.0000 aux_loss=0.0009 val_rank1=0.9407 val_mAP=0.9200 val_auc=0.9929 val_coverage=270/270 time=70.3s
Epoch [91/200] lr=0.000018 train_loss=0.9527 train_acc=0.9994 aux_loss=0.0004 val_rank1=0.9370 val_mAP=0.9187 val_auc=0.9918 val_coverage=270/270 time=71.3s
Epoch [92/200] lr=0.000018 train_loss=0.9622 train_acc=0.9994 aux_loss=0.0005 val_rank1=0.9407 val_mAP=0.9217 val_auc=0.9923 val_coverage=270/270 time=70.7s
Epoch [93/200] lr=0.000018 train_loss=0.9552 train_acc=1.0000 aux_loss=0.0002 val_rank1=0.9407 val_mAP=0.9206 val_auc=0.9915 val_coverage=270/270 time=70.9s
Epoch [94/200] lr=0.000018 train_loss=0.9813 train_acc=0.9994 aux_loss=0.0006 val_rank1=0.9407 val_mAP=0.9216 val_auc=0.9918 val_coverage=270/270 time=73.9s
Epoch [95/200] lr=0.000017 train_loss=0.9740 train_acc=1.0000 aux_loss=0.0009 val_rank1=0.9407 val_mAP=0.9187 val_auc=0.9903 val_coverage=270/270 time=73.5s
Epoch [96/200] lr=0.000017 train_loss=0.9132 train_acc=1.0000 aux_loss=0.0001 val_rank1=0.9407 val_mAP=0.9179 val_auc=0.9882 val_coverage=270/270 time=72.3s
Epoch [97/200] lr=0.000017 train_loss=0.9208 train_acc=1.0000 aux_loss=0.0001 val_rank1=0.9407 val_mAP=0.9171 val_auc=0.9892 val_coverage=270/270 time=74.4s
Epoch [98/200] lr=0.000017 train_loss=0.9400 train_acc=0.9994 aux_loss=0.0008 val_rank1=0.9407 val_mAP=0.9193 val_auc=0.9881 val_coverage=270/270 time=70.6s
Epoch [99/200] lr=0.000017 train_loss=0.9094 train_acc=1.0000 aux_loss=0.0002 val_rank1=0.9407 val_mAP=0.9176 val_auc=0.9877 val_coverage=270/270 time=71.2s
  [DEBUG] Embedding: sim_all=[-0.369, 0.038, 0.998] pos_sim=0.844 neg_sim=-0.006 gap=0.850 emb_norm=1.000
Epoch [100/200] lr=0.000016 train_loss=0.9197 train_acc=0.9994 aux_loss=0.0001 val_rank1=0.9444 val_mAP=0.9178 val_auc=0.9899 val_coverage=270/270 time=67.9s
Epoch [101/200] lr=0.000016 train_loss=0.9094 train_acc=0.9994 aux_loss=0.0003 val_rank1=0.9481 val_mAP=0.9218 val_auc=0.9922 val_coverage=270/270 time=73.8s
Epoch [102/200] lr=0.000016 train_loss=0.9418 train_acc=0.9994 aux_loss=0.0005 val_rank1=0.9444 val_mAP=0.9185 val_auc=0.9926 val_coverage=270/270 time=68.5s
Epoch [103/200] lr=0.000016 train_loss=0.9713 train_acc=0.9988 aux_loss=0.0011 val_rank1=0.9444 val_mAP=0.9208 val_auc=0.9913 val_coverage=270/270 time=72.5s
Epoch [104/200] lr=0.000015 train_loss=0.9309 train_acc=0.9994 aux_loss=0.0008 val_rank1=0.9444 val_mAP=0.9190 val_auc=0.9870 val_coverage=270/270 time=75.1s
Epoch [105/200] lr=0.000015 train_loss=0.9258 train_acc=1.0000 aux_loss=0.0003 val_rank1=0.9407 val_mAP=0.9175 val_auc=0.9885 val_coverage=270/270 time=69.3s
Epoch [106/200] lr=0.000015 train_loss=0.9097 train_acc=1.0000 aux_loss=0.0005 val_rank1=0.9444 val_mAP=0.9168 val_auc=0.9896 val_coverage=270/270 time=73.9s
Epoch [107/200] lr=0.000015 train_loss=0.9103 train_acc=1.0000 aux_loss=0.0001 val_rank1=0.9407 val_mAP=0.9197 val_auc=0.9924 val_coverage=270/270 time=69.7s
Epoch [108/200] lr=0.000014 train_loss=0.8940 train_acc=0.9994 aux_loss=0.0005 val_rank1=0.9444 val_mAP=0.9219 val_auc=0.9926 val_coverage=270/270 time=71.6s
Epoch [109/200] lr=0.000014 train_loss=0.8920 train_acc=1.0000 aux_loss=0.0002 val_rank1=0.9444 val_mAP=0.9230 val_auc=0.9898 val_coverage=270/270 time=71.3s
  [DEBUG] Embedding: sim_all=[-0.414, 0.036, 0.999] pos_sim=0.850 neg_sim=-0.008 gap=0.858 emb_norm=1.000
Epoch [110/200] lr=0.000014 train_loss=0.9115 train_acc=0.9988 aux_loss=0.0008 val_rank1=0.9444 val_mAP=0.9203 val_auc=0.9906 val_coverage=270/270 time=70.6s
Epoch [111/200] lr=0.000014 train_loss=0.9088 train_acc=0.9994 aux_loss=0.0003 val_rank1=0.9407 val_mAP=0.9188 val_auc=0.9905 val_coverage=270/270 time=71.7s
Epoch [112/200] lr=0.000014 train_loss=0.8988 train_acc=1.0000 aux_loss=0.0002 val_rank1=0.9444 val_mAP=0.9198 val_auc=0.9907 val_coverage=270/270 time=72.0s
Epoch [113/200] lr=0.000013 train_loss=0.8829 train_acc=0.9994 aux_loss=0.0004 val_rank1=0.9444 val_mAP=0.9217 val_auc=0.9908 val_coverage=270/270 time=73.8s
Epoch [114/200] lr=0.000013 train_loss=0.8794 train_acc=0.9994 aux_loss=0.0005 val_rank1=0.9444 val_mAP=0.9232 val_auc=0.9916 val_coverage=270/270 time=72.6s
Epoch [115/200] lr=0.000013 train_loss=0.9040 train_acc=0.9994 aux_loss=0.0003 val_rank1=0.9407 val_mAP=0.9200 val_auc=0.9917 val_coverage=270/270 time=70.4s
Epoch [116/200] lr=0.000013 train_loss=0.9084 train_acc=1.0000 aux_loss=0.0002 val_rank1=0.9407 val_mAP=0.9206 val_auc=0.9913 val_coverage=270/270 time=71.5s
Epoch [117/200] lr=0.000012 train_loss=0.8797 train_acc=1.0000 aux_loss=0.0001 val_rank1=0.9407 val_mAP=0.9223 val_auc=0.9942 val_coverage=270/270 time=70.0s
Epoch [118/200] lr=0.000012 train_loss=0.8873 train_acc=0.9994 aux_loss=0.0002 val_rank1=0.9407 val_mAP=0.9206 val_auc=0.9932 val_coverage=270/270 time=72.5s
Epoch [119/200] lr=0.000012 train_loss=0.8874 train_acc=0.9994 aux_loss=0.0009 val_rank1=0.9407 val_mAP=0.9208 val_auc=0.9924 val_coverage=270/270 time=73.3s
  [DEBUG] Embedding: sim_all=[-0.404, 0.037, 0.999] pos_sim=0.863 neg_sim=-0.008 gap=0.871 emb_norm=1.000
Epoch [120/200] lr=0.000012 train_loss=0.8943 train_acc=0.9981 aux_loss=0.0017 val_rank1=0.9444 val_mAP=0.9231 val_auc=0.9936 val_coverage=270/270 time=74.4s
Epoch [121/200] lr=0.000011 train_loss=0.8625 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9208 val_auc=0.9924 val_coverage=270/270 time=71.1s
Epoch [122/200] lr=0.000011 train_loss=0.8639 train_acc=1.0000 aux_loss=0.0001 val_rank1=0.9407 val_mAP=0.9203 val_auc=0.9933 val_coverage=270/270 time=73.1s
Epoch [123/200] lr=0.000011 train_loss=0.8650 train_acc=1.0000 aux_loss=0.0001 val_rank1=0.9407 val_mAP=0.9220 val_auc=0.9946 val_coverage=270/270 time=75.3s
Epoch [124/200] lr=0.000011 train_loss=0.8630 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9228 val_auc=0.9946 val_coverage=270/270 time=71.1s
Epoch [125/200] lr=0.000011 train_loss=0.8691 train_acc=0.9994 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9210 val_auc=0.9929 val_coverage=270/270 time=70.9s
Epoch [126/200] lr=0.000010 train_loss=0.8757 train_acc=0.9994 aux_loss=0.0005 val_rank1=0.9407 val_mAP=0.9240 val_auc=0.9952 val_coverage=270/270 time=69.9s
Epoch [127/200] lr=0.000010 train_loss=0.8744 train_acc=0.9994 aux_loss=0.0006 val_rank1=0.9407 val_mAP=0.9217 val_auc=0.9932 val_coverage=270/270 time=73.3s
Epoch [128/200] lr=0.000010 train_loss=0.8646 train_acc=1.0000 aux_loss=0.0002 val_rank1=0.9407 val_mAP=0.9238 val_auc=0.9938 val_coverage=270/270 time=70.1s
Epoch [129/200] lr=0.000010 train_loss=0.8493 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9234 val_auc=0.9947 val_coverage=270/270 time=68.6s
  [DEBUG] Embedding: sim_all=[-0.450, 0.037, 0.999] pos_sim=0.864 neg_sim=-0.008 gap=0.872 emb_norm=1.000
Epoch [130/200] lr=0.000009 train_loss=0.8833 train_acc=0.9994 aux_loss=0.0008 val_rank1=0.9407 val_mAP=0.9230 val_auc=0.9930 val_coverage=270/270 time=73.4s
Epoch [131/200] lr=0.000009 train_loss=0.8423 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9444 val_mAP=0.9240 val_auc=0.9935 val_coverage=270/270 time=70.6s
Epoch [132/200] lr=0.000009 train_loss=0.8716 train_acc=0.9988 aux_loss=0.0013 val_rank1=0.9407 val_mAP=0.9228 val_auc=0.9940 val_coverage=270/270 time=69.7s
Epoch [133/200] lr=0.000009 train_loss=0.8798 train_acc=0.9994 aux_loss=0.0002 val_rank1=0.9370 val_mAP=0.9195 val_auc=0.9918 val_coverage=270/270 time=69.9s
Epoch [134/200] lr=0.000009 train_loss=0.8635 train_acc=0.9994 aux_loss=0.0004 val_rank1=0.9370 val_mAP=0.9196 val_auc=0.9913 val_coverage=270/270 time=72.4s
Epoch [135/200] lr=0.000008 train_loss=0.8476 train_acc=0.9994 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9222 val_auc=0.9933 val_coverage=270/270 time=72.4s
Epoch [136/200] lr=0.000008 train_loss=0.8364 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9221 val_auc=0.9930 val_coverage=270/270 time=70.7s
Epoch [137/200] lr=0.000008 train_loss=0.8727 train_acc=0.9994 aux_loss=0.0006 val_rank1=0.9407 val_mAP=0.9229 val_auc=0.9923 val_coverage=270/270 time=73.8s
Epoch [138/200] lr=0.000008 train_loss=0.8449 train_acc=1.0000 aux_loss=0.0003 val_rank1=0.9407 val_mAP=0.9231 val_auc=0.9951 val_coverage=270/270 time=71.1s
Epoch [139/200] lr=0.000008 train_loss=0.8337 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9227 val_auc=0.9936 val_coverage=270/270 time=72.9s
  [DEBUG] Embedding: sim_all=[-0.402, 0.035, 0.999] pos_sim=0.866 neg_sim=-0.011 gap=0.877 emb_norm=1.000
Epoch [140/200] lr=0.000007 train_loss=0.8388 train_acc=1.0000 aux_loss=0.0002 val_rank1=0.9407 val_mAP=0.9223 val_auc=0.9941 val_coverage=270/270 time=69.6s
Epoch [141/200] lr=0.000007 train_loss=0.8383 train_acc=0.9994 aux_loss=0.0006 val_rank1=0.9407 val_mAP=0.9219 val_auc=0.9915 val_coverage=270/270 time=72.7s
Epoch [142/200] lr=0.000007 train_loss=0.8464 train_acc=1.0000 aux_loss=0.0004 val_rank1=0.9407 val_mAP=0.9229 val_auc=0.9935 val_coverage=270/270 time=68.2s
Epoch [143/200] lr=0.000007 train_loss=0.8376 train_acc=1.0000 aux_loss=0.0001 val_rank1=0.9407 val_mAP=0.9234 val_auc=0.9923 val_coverage=270/270 time=70.6s
Epoch [144/200] lr=0.000007 train_loss=0.8365 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9219 val_auc=0.9925 val_coverage=270/270 time=71.2s
Epoch [145/200] lr=0.000007 train_loss=0.8423 train_acc=0.9994 aux_loss=0.0008 val_rank1=0.9444 val_mAP=0.9243 val_auc=0.9937 val_coverage=270/270 time=73.0s
Epoch [146/200] lr=0.000006 train_loss=0.8261 train_acc=0.9994 aux_loss=0.0005 val_rank1=0.9407 val_mAP=0.9214 val_auc=0.9923 val_coverage=270/270 time=70.0s
Epoch [147/200] lr=0.000006 train_loss=0.8471 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9229 val_auc=0.9947 val_coverage=270/270 time=70.2s
Epoch [148/200] lr=0.000006 train_loss=0.8160 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9370 val_mAP=0.9221 val_auc=0.9932 val_coverage=270/270 time=72.4s
Epoch [149/200] lr=0.000006 train_loss=0.8243 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9218 val_auc=0.9928 val_coverage=270/270 time=70.9s
  [DEBUG] Embedding: sim_all=[-0.409, 0.035, 0.999] pos_sim=0.875 neg_sim=-0.011 gap=0.886 emb_norm=1.000
Epoch [150/200] lr=0.000006 train_loss=0.8254 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9444 val_mAP=0.9230 val_auc=0.9933 val_coverage=270/270 time=70.2s
Epoch [151/200] lr=0.000005 train_loss=0.8331 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9230 val_auc=0.9917 val_coverage=270/270 time=73.5s
Epoch [152/200] lr=0.000005 train_loss=0.8537 train_acc=0.9994 aux_loss=0.0001 val_rank1=0.9407 val_mAP=0.9222 val_auc=0.9903 val_coverage=270/270 time=72.5s
Epoch [153/200] lr=0.000005 train_loss=0.8245 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9247 val_auc=0.9947 val_coverage=270/270 time=72.3s
  ★ 新最佳! mAP=0.9247
Epoch [154/200] lr=0.000005 train_loss=0.8256 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9444 val_mAP=0.9224 val_auc=0.9926 val_coverage=270/270 time=69.6s
Epoch [155/200] lr=0.000005 train_loss=0.8483 train_acc=1.0000 aux_loss=0.0001 val_rank1=0.9407 val_mAP=0.9220 val_auc=0.9930 val_coverage=270/270 time=72.0s
Epoch [156/200] lr=0.000005 train_loss=0.8577 train_acc=1.0000 aux_loss=0.0003 val_rank1=0.9407 val_mAP=0.9224 val_auc=0.9928 val_coverage=270/270 time=69.7s
Epoch [157/200] lr=0.000004 train_loss=0.8560 train_acc=0.9994 aux_loss=0.0010 val_rank1=0.9444 val_mAP=0.9232 val_auc=0.9928 val_coverage=270/270 time=71.0s
Epoch [158/200] lr=0.000004 train_loss=0.8397 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9444 val_mAP=0.9233 val_auc=0.9938 val_coverage=270/270 time=69.9s
Epoch [159/200] lr=0.000004 train_loss=0.8405 train_acc=0.9988 aux_loss=0.0007 val_rank1=0.9407 val_mAP=0.9219 val_auc=0.9935 val_coverage=270/270 time=69.6s
  [DEBUG] Embedding: sim_all=[-0.385, 0.035, 0.999] pos_sim=0.877 neg_sim=-0.011 gap=0.887 emb_norm=1.000
Epoch [160/200] lr=0.000004 train_loss=0.8138 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9226 val_auc=0.9940 val_coverage=270/270 time=68.7s
Epoch [161/200] lr=0.000004 train_loss=0.8226 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9230 val_auc=0.9937 val_coverage=270/270 time=71.4s
Epoch [162/200] lr=0.000004 train_loss=0.8207 train_acc=0.9994 aux_loss=0.0005 val_rank1=0.9407 val_mAP=0.9234 val_auc=0.9942 val_coverage=270/270 time=70.2s
Epoch [163/200] lr=0.000004 train_loss=0.8289 train_acc=1.0000 aux_loss=0.0001 val_rank1=0.9407 val_mAP=0.9220 val_auc=0.9923 val_coverage=270/270 time=73.0s
Epoch [164/200] lr=0.000004 train_loss=0.8065 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9235 val_auc=0.9932 val_coverage=270/270 time=72.5s
Epoch [165/200] lr=0.000003 train_loss=0.8360 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9234 val_auc=0.9945 val_coverage=270/270 time=70.1s
Epoch [166/200] lr=0.000003 train_loss=0.8308 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9239 val_auc=0.9945 val_coverage=270/270 time=69.8s
Epoch [167/200] lr=0.000003 train_loss=0.8098 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9230 val_auc=0.9908 val_coverage=270/270 time=70.2s
Epoch [168/200] lr=0.000003 train_loss=0.8190 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9240 val_auc=0.9936 val_coverage=270/270 time=69.1s
Epoch [169/200] lr=0.000003 train_loss=0.8127 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9256 val_auc=0.9950 val_coverage=270/270 time=70.6s
  ★ 新最佳! mAP=0.9256
  [DEBUG] Embedding: sim_all=[-0.397, 0.037, 0.999] pos_sim=0.878 neg_sim=-0.009 gap=0.887 emb_norm=1.000
Epoch [170/200] lr=0.000003 train_loss=0.8272 train_acc=1.0000 aux_loss=0.0001 val_rank1=0.9370 val_mAP=0.9226 val_auc=0.9921 val_coverage=270/270 time=72.7s
Epoch [171/200] lr=0.000003 train_loss=0.8145 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9248 val_auc=0.9929 val_coverage=270/270 time=69.8s
Epoch [172/200] lr=0.000003 train_loss=0.8188 train_acc=0.9994 aux_loss=0.0005 val_rank1=0.9407 val_mAP=0.9248 val_auc=0.9945 val_coverage=270/270 time=72.0s
Epoch [173/200] lr=0.000002 train_loss=0.8244 train_acc=1.0000 aux_loss=0.0001 val_rank1=0.9407 val_mAP=0.9245 val_auc=0.9932 val_coverage=270/270 time=76.4s
Epoch [174/200] lr=0.000002 train_loss=0.8598 train_acc=0.9981 aux_loss=0.0011 val_rank1=0.9407 val_mAP=0.9236 val_auc=0.9941 val_coverage=270/270 time=70.3s
Epoch [175/200] lr=0.000002 train_loss=0.8165 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9233 val_auc=0.9924 val_coverage=270/270 time=72.2s
Epoch [176/200] lr=0.000002 train_loss=0.8275 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9239 val_auc=0.9930 val_coverage=270/270 time=71.7s
Epoch [177/200] lr=0.000002 train_loss=0.8104 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9230 val_auc=0.9923 val_coverage=270/270 time=73.3s
Epoch [178/200] lr=0.000002 train_loss=0.8233 train_acc=1.0000 aux_loss=0.0004 val_rank1=0.9407 val_mAP=0.9238 val_auc=0.9933 val_coverage=270/270 time=69.6s
Epoch [179/200] lr=0.000002 train_loss=0.8112 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9444 val_mAP=0.9253 val_auc=0.9923 val_coverage=270/270 time=70.6s
  [DEBUG] Embedding: sim_all=[-0.401, 0.035, 0.999] pos_sim=0.879 neg_sim=-0.011 gap=0.890 emb_norm=1.000
Epoch [180/200] lr=0.000002 train_loss=0.8402 train_acc=0.9988 aux_loss=0.0007 val_rank1=0.9407 val_mAP=0.9231 val_auc=0.9919 val_coverage=270/270 time=72.3s
Epoch [181/200] lr=0.000002 train_loss=0.8081 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9234 val_auc=0.9923 val_coverage=270/270 time=70.7s
Epoch [182/200] lr=0.000002 train_loss=0.8274 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9232 val_auc=0.9926 val_coverage=270/270 time=72.8s
Epoch [183/200] lr=0.000002 train_loss=0.8163 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9253 val_auc=0.9951 val_coverage=270/270 time=71.3s
Epoch [184/200] lr=0.000002 train_loss=0.8251 train_acc=1.0000 aux_loss=0.0001 val_rank1=0.9407 val_mAP=0.9243 val_auc=0.9937 val_coverage=270/270 time=70.9s
Epoch [185/200] lr=0.000001 train_loss=0.8165 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9238 val_auc=0.9936 val_coverage=270/270 time=70.7s
Epoch [186/200] lr=0.000001 train_loss=0.8022 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9250 val_auc=0.9938 val_coverage=270/270 time=71.8s
Epoch [187/200] lr=0.000001 train_loss=0.8154 train_acc=1.0000 aux_loss=0.0001 val_rank1=0.9407 val_mAP=0.9230 val_auc=0.9931 val_coverage=270/270 time=72.0s
Epoch [188/200] lr=0.000001 train_loss=0.8365 train_acc=0.9988 aux_loss=0.0008 val_rank1=0.9407 val_mAP=0.9226 val_auc=0.9928 val_coverage=270/270 time=71.9s
Epoch [189/200] lr=0.000001 train_loss=0.8049 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9247 val_auc=0.9923 val_coverage=270/270 time=69.0s
  [DEBUG] Embedding: sim_all=[-0.394, 0.037, 0.999] pos_sim=0.880 neg_sim=-0.009 gap=0.889 emb_norm=1.000
Epoch [190/200] lr=0.000001 train_loss=0.8091 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9230 val_auc=0.9931 val_coverage=270/270 time=70.7s
Epoch [191/200] lr=0.000001 train_loss=0.8235 train_acc=0.9994 aux_loss=0.0004 val_rank1=0.9444 val_mAP=0.9253 val_auc=0.9934 val_coverage=270/270 time=69.2s
Epoch [192/200] lr=0.000001 train_loss=0.8127 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9444 val_mAP=0.9261 val_auc=0.9950 val_coverage=270/270 time=73.3s
  ★ 新最佳! mAP=0.9261
Epoch [193/200] lr=0.000001 train_loss=0.8273 train_acc=0.9994 aux_loss=0.0004 val_rank1=0.9444 val_mAP=0.9243 val_auc=0.9940 val_coverage=270/270 time=71.5s
Epoch [194/200] lr=0.000001 train_loss=0.8251 train_acc=0.9994 aux_loss=0.0011 val_rank1=0.9444 val_mAP=0.9231 val_auc=0.9933 val_coverage=270/270 time=69.0s
Epoch [195/200] lr=0.000001 train_loss=0.8173 train_acc=0.9994 aux_loss=0.0004 val_rank1=0.9407 val_mAP=0.9261 val_auc=0.9960 val_coverage=270/270 time=71.0s
Epoch [196/200] lr=0.000001 train_loss=0.8066 train_acc=1.0000 aux_loss=0.0001 val_rank1=0.9444 val_mAP=0.9241 val_auc=0.9929 val_coverage=270/270 time=72.7s
Epoch [197/200] lr=0.000001 train_loss=0.8229 train_acc=1.0000 aux_loss=0.0001 val_rank1=0.9407 val_mAP=0.9239 val_auc=0.9928 val_coverage=270/270 time=71.7s
Epoch [198/200] lr=0.000001 train_loss=0.8153 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9246 val_auc=0.9943 val_coverage=270/270 time=71.4s
Epoch [199/200] lr=0.000001 train_loss=0.8164 train_acc=1.0000 aux_loss=0.0001 val_rank1=0.9407 val_mAP=0.9232 val_auc=0.9939 val_coverage=270/270 time=68.7s
  [DEBUG] Embedding: sim_all=[-0.388, 0.034, 0.999] pos_sim=0.879 neg_sim=-0.012 gap=0.892 emb_norm=1.000
Epoch [200/200] lr=0.000001 train_loss=0.8122 train_acc=0.9994 aux_loss=0.0000 val_rank1=0.9444 val_mAP=0.9263 val_auc=0.9940 val_coverage=270/270 time=70.0s
  ★ 新最佳! mAP=0.9263

训练完成! 总时间: 244.4 分钟
最佳 mAP: 0.9263 (Epoch 200)


3. 先训练head再训练backbone指标低了非常多，提交只有0.85
(py312) yuanzhen@g8600v7:/data0/yzhen/projects/gastrovision_v3$ CUDA_VISIBLE_DEVICES=1 python jaguar/train.py configs/jaguar_reid.yaml
============================================================
Jaguar Re-Identification Training
============================================================
时间:   2026-03-31 17:36:12
设备:   cuda
GPU:    NVIDIA Graphics Device
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
  已保存到: /data0/yzhen/projects/gastrovision_v3/output/jaguar_arcface_backbone

类别数: 31

训练集: 1625 样本
验证集: 270 样本
  PK Sampler: P=8, K=4 (batch_size=32)

创建模型: convnext_large
  加载本地预训练权重: /data0/yzhen/projects/Gastrovision_results/pretrain/convnext_large-ea097f82.pth
  [统计] 成功加载: 344/344 个参数
  Embedding Head: Linear (1536→512)
  模型参数: 197.02M, 可训练: 197.02M
  [解冻] Backbone 与 Head 一起训练

主损失: arcface
  [primary] arcface: scale=30.0, margin=0.5
辅助损失: triplet (weight=1.0)
  [aux_] triplet: scale=1.0, margin=0.3
优化器: adamw (backbone_lr=0.000300, head_lr=0.003000)
调度器: warmup_cosine (warmup=5)
  EMA 已启用 (decay=0.9)
  加载 checkpoint: /data0/yzhen/projects/gastrovision_v3/output/jaguar_arcface_head/best_model.pth
  ✓ 模型权重完全加载
  [跳过] 优化器参数组数量不匹配: checkpoint=1 组 vs 当前=2 组
  [提示] 可能是 freeze_backbone 开关发生了变化，优化器将从当前配置重新开始（模型权重已正常加载）
  [两阶段训练] EMA decay 保持为 0.9 (未从 checkpoint 覆盖)
  恢复自 Epoch 154, 最佳指标=0.6993
  [两阶段训练] best_metric 已重置为 0，Phase2 将从头选择最佳模型
  [两阶段训练] Scheduler 已重建，Phase2 将从新的 LR 周期开始
  [两阶段训练] 当前 LR = 0.000012

========================================
开始训练
========================================

  最佳模型选择指标: val_mAP
  [DEBUG] EMA 诊断: decay=0.9, steps/epoch=50, total_steps=20000, 初始权重保留率=0.0000 (训练结束时 EMA 仅吸收 100.0% 的新参数)
Epoch [155/400] lr=0.000012 train_loss=15.7713 train_acc=0.4419 aux_loss=0.4315 val_rank1=0.7852 val_mAP=0.5076 val_auc=0.8148 val_coverage=270/270 time=197.6s
  ★ 新最佳! mAP=0.5076
Epoch [156/400] lr=0.000024 train_loss=13.2564 train_acc=0.6244 aux_loss=0.3897 val_rank1=0.7963 val_mAP=0.5722 val_auc=0.8680 val_coverage=270/270 time=193.7s
  ★ 新最佳! mAP=0.5722
Epoch [157/400] lr=0.000036 train_loss=10.7712 train_acc=0.7106 aux_loss=0.3633 val_rank1=0.8074 val_mAP=0.6031 val_auc=0.8919 val_coverage=270/270 time=208.8s
  ★ 新最佳! mAP=0.6031
Epoch [158/400] lr=0.000048 train_loss=9.0229 train_acc=0.7656 aux_loss=0.3376 val_rank1=0.8185 val_mAP=0.6406 val_auc=0.9059 val_coverage=270/270 time=215.3s
  ★ 新最佳! mAP=0.6406
Epoch [159/400] lr=0.000060 train_loss=7.4803 train_acc=0.8306 aux_loss=0.3037 val_rank1=0.8407 val_mAP=0.6797 val_auc=0.9240 val_coverage=270/270 time=193.0s
  ★ 新最佳! mAP=0.6797
  [DEBUG] Embedding: sim_all=[0.218, 0.601, 0.998] pos_sim=0.828 neg_sim=0.589 gap=0.240 emb_norm=1.000
Epoch [160/400] lr=0.000060 train_loss=6.6498 train_acc=0.8519 aux_loss=0.2867 val_rank1=0.8556 val_mAP=0.7299 val_auc=0.9445 val_coverage=270/270 time=194.9s
  ★ 新最佳! mAP=0.7299
Epoch [161/400] lr=0.000060 train_loss=5.6900 train_acc=0.8856 aux_loss=0.2416 val_rank1=0.8481 val_mAP=0.7634 val_auc=0.9500 val_coverage=270/270 time=212.0s
  ★ 新最佳! mAP=0.7634
Epoch [162/400] lr=0.000060 train_loss=4.9484 train_acc=0.9119 aux_loss=0.2011 val_rank1=0.8704 val_mAP=0.7878 val_auc=0.9567 val_coverage=270/270 time=204.1s
  ★ 新最佳! mAP=0.7878
Epoch [163/400] lr=0.000060 train_loss=4.2744 train_acc=0.9256 aux_loss=0.1642 val_rank1=0.8741 val_mAP=0.7993 val_auc=0.9605 val_coverage=270/270 time=192.8s
  ★ 新最佳! mAP=0.7993
Epoch [164/400] lr=0.000060 train_loss=3.8930 train_acc=0.9363 aux_loss=0.1471 val_rank1=0.8852 val_mAP=0.8230 val_auc=0.9685 val_coverage=270/270 time=197.9s
  ★ 新最佳! mAP=0.8230
Epoch [165/400] lr=0.000060 train_loss=3.4978 train_acc=0.9469 aux_loss=0.1218 val_rank1=0.8963 val_mAP=0.8358 val_auc=0.9727 val_coverage=270/270 time=206.1s
  ★ 新最佳! mAP=0.8358
Epoch [166/400] lr=0.000060 train_loss=3.2726 train_acc=0.9544 aux_loss=0.1003 val_rank1=0.8926 val_mAP=0.8402 val_auc=0.9761 val_coverage=270/270 time=193.0s
  ★ 新最佳! mAP=0.8402
Epoch [167/400] lr=0.000060 train_loss=2.8478 train_acc=0.9675 aux_loss=0.0904 val_rank1=0.8889 val_mAP=0.8436 val_auc=0.9745 val_coverage=270/270 time=198.3s
  ★ 新最佳! mAP=0.8436
Epoch [168/400] lr=0.000060 train_loss=2.8805 train_acc=0.9669 aux_loss=0.0844 val_rank1=0.8926 val_mAP=0.8484 val_auc=0.9739 val_coverage=270/270 time=210.2s
  ★ 新最佳! mAP=0.8484
Epoch [169/400] lr=0.000060 train_loss=2.6954 train_acc=0.9712 aux_loss=0.0760 val_rank1=0.9000 val_mAP=0.8550 val_auc=0.9801 val_coverage=270/270 time=211.2s
  ★ 新最佳! mAP=0.8550
  [DEBUG] Embedding: sim_all=[-0.392, 0.135, 0.999] pos_sim=0.760 neg_sim=0.101 gap=0.659 emb_norm=1.000
Epoch [170/400] lr=0.000060 train_loss=2.5447 train_acc=0.9762 aux_loss=0.0680 val_rank1=0.9074 val_mAP=0.8603 val_auc=0.9837 val_coverage=270/270 time=197.1s
  ★ 新最佳! mAP=0.8603
Epoch [171/400] lr=0.000060 train_loss=2.4010 train_acc=0.9812 aux_loss=0.0566 val_rank1=0.9111 val_mAP=0.8660 val_auc=0.9742 val_coverage=270/270 time=189.4s
  ★ 新最佳! mAP=0.8660
Epoch [172/400] lr=0.000060 train_loss=2.3539 train_acc=0.9756 aux_loss=0.0544 val_rank1=0.9074 val_mAP=0.8791 val_auc=0.9854 val_coverage=270/270 time=209.0s
  ★ 新最佳! mAP=0.8791
Epoch [173/400] lr=0.000060 train_loss=2.2700 train_acc=0.9850 aux_loss=0.0470 val_rank1=0.9185 val_mAP=0.8753 val_auc=0.9820 val_coverage=270/270 time=211.3s
Epoch [174/400] lr=0.000060 train_loss=2.2170 train_acc=0.9850 aux_loss=0.0485 val_rank1=0.9148 val_mAP=0.8789 val_auc=0.9851 val_coverage=270/270 time=194.7s
Epoch [175/400] lr=0.000060 train_loss=1.9969 train_acc=0.9881 aux_loss=0.0311 val_rank1=0.9259 val_mAP=0.8858 val_auc=0.9869 val_coverage=270/270 time=200.0s
  ★ 新最佳! mAP=0.8858
Epoch [176/400] lr=0.000060 train_loss=2.0117 train_acc=0.9875 aux_loss=0.0333 val_rank1=0.9222 val_mAP=0.8846 val_auc=0.9849 val_coverage=270/270 time=215.2s
Epoch [177/400] lr=0.000060 train_loss=1.8971 train_acc=0.9900 aux_loss=0.0256 val_rank1=0.9259 val_mAP=0.8900 val_auc=0.9866 val_coverage=270/270 time=220.2s
  ★ 新最佳! mAP=0.8900
Epoch [178/400] lr=0.000060 train_loss=1.8116 train_acc=0.9925 aux_loss=0.0271 val_rank1=0.9185 val_mAP=0.8816 val_auc=0.9853 val_coverage=270/270 time=197.8s
Epoch [179/400] lr=0.000060 train_loss=1.7631 train_acc=0.9925 aux_loss=0.0218 val_rank1=0.9259 val_mAP=0.8968 val_auc=0.9853 val_coverage=270/270 time=195.5s
  ★ 新最佳! mAP=0.8968
  [DEBUG] Embedding: sim_all=[-0.431, 0.090, 0.999] pos_sim=0.782 neg_sim=0.052 gap=0.730 emb_norm=1.000
Epoch [180/400] lr=0.000060 train_loss=1.8623 train_acc=0.9919 aux_loss=0.0252 val_rank1=0.9185 val_mAP=0.8995 val_auc=0.9866 val_coverage=270/270 time=203.7s
  ★ 新最佳! mAP=0.8995
Epoch [181/400] lr=0.000060 train_loss=1.7451 train_acc=0.9912 aux_loss=0.0215 val_rank1=0.9222 val_mAP=0.8985 val_auc=0.9868 val_coverage=270/270 time=212.5s
Epoch [182/400] lr=0.000060 train_loss=1.6587 train_acc=0.9969 aux_loss=0.0172 val_rank1=0.9259 val_mAP=0.8931 val_auc=0.9849 val_coverage=270/270 time=208.2s
Epoch [183/400] lr=0.000060 train_loss=1.5908 train_acc=0.9962 aux_loss=0.0152 val_rank1=0.9296 val_mAP=0.9018 val_auc=0.9853 val_coverage=270/270 time=197.9s
  ★ 新最佳! mAP=0.9018
Epoch [184/400] lr=0.000059 train_loss=1.6652 train_acc=0.9956 aux_loss=0.0172 val_rank1=0.9185 val_mAP=0.8978 val_auc=0.9870 val_coverage=270/270 time=194.9s
Epoch [185/400] lr=0.000059 train_loss=1.6003 train_acc=0.9944 aux_loss=0.0154 val_rank1=0.9148 val_mAP=0.9053 val_auc=0.9901 val_coverage=270/270 time=201.4s
  ★ 新最佳! mAP=0.9053
Epoch [186/400] lr=0.000059 train_loss=1.4940 train_acc=0.9975 aux_loss=0.0099 val_rank1=0.9259 val_mAP=0.9052 val_auc=0.9905 val_coverage=270/270 time=210.4s
Epoch [187/400] lr=0.000059 train_loss=1.5882 train_acc=0.9919 aux_loss=0.0159 val_rank1=0.9222 val_mAP=0.9054 val_auc=0.9888 val_coverage=270/270 time=217.4s
  ★ 新最佳! mAP=0.9054
Epoch [188/400] lr=0.000059 train_loss=1.5161 train_acc=0.9956 aux_loss=0.0128 val_rank1=0.9259 val_mAP=0.9118 val_auc=0.9920 val_coverage=270/270 time=205.8s
  ★ 新最佳! mAP=0.9118
Epoch [189/400] lr=0.000059 train_loss=1.5034 train_acc=0.9944 aux_loss=0.0104 val_rank1=0.9148 val_mAP=0.8993 val_auc=0.9923 val_coverage=270/270 time=200.1s
  [DEBUG] Embedding: sim_all=[-0.421, 0.100, 0.999] pos_sim=0.812 neg_sim=0.061 gap=0.750 emb_norm=1.000
Epoch [190/400] lr=0.000059 train_loss=1.4338 train_acc=0.9975 aux_loss=0.0081 val_rank1=0.9222 val_mAP=0.9018 val_auc=0.9861 val_coverage=270/270 time=193.8s
Epoch [191/400] lr=0.000059 train_loss=1.4388 train_acc=0.9975 aux_loss=0.0101 val_rank1=0.9222 val_mAP=0.8941 val_auc=0.9859 val_coverage=270/270 time=206.0s
Epoch [192/400] lr=0.000059 train_loss=1.3969 train_acc=0.9981 aux_loss=0.0075 val_rank1=0.9222 val_mAP=0.8994 val_auc=0.9857 val_coverage=270/270 time=213.2s
Epoch [193/400] lr=0.000059 train_loss=1.2930 train_acc=0.9994 aux_loss=0.0034 val_rank1=0.9259 val_mAP=0.8965 val_auc=0.9860 val_coverage=270/270 time=202.5s
Epoch [194/400] lr=0.000059 train_loss=1.3405 train_acc=0.9988 aux_loss=0.0061 val_rank1=0.9259 val_mAP=0.9051 val_auc=0.9864 val_coverage=270/270 time=195.3s
Epoch [195/400] lr=0.000059 train_loss=1.3395 train_acc=0.9975 aux_loss=0.0081 val_rank1=0.9259 val_mAP=0.9063 val_auc=0.9876 val_coverage=270/270 time=201.4s
Epoch [196/400] lr=0.000059 train_loss=1.2204 train_acc=0.9988 aux_loss=0.0041 val_rank1=0.9222 val_mAP=0.9035 val_auc=0.9867 val_coverage=270/270 time=216.5s
Epoch [197/400] lr=0.000059 train_loss=1.2449 train_acc=0.9975 aux_loss=0.0030 val_rank1=0.9259 val_mAP=0.9147 val_auc=0.9872 val_coverage=270/270 time=208.2s
  ★ 新最佳! mAP=0.9147
Epoch [198/400] lr=0.000059 train_loss=1.2825 train_acc=0.9981 aux_loss=0.0035 val_rank1=0.9259 val_mAP=0.9057 val_auc=0.9863 val_coverage=270/270 time=197.9s
Epoch [199/400] lr=0.000059 train_loss=1.2321 train_acc=0.9975 aux_loss=0.0037 val_rank1=0.9222 val_mAP=0.9084 val_auc=0.9883 val_coverage=270/270 time=192.5s
  [DEBUG] Embedding: sim_all=[-0.369, 0.063, 0.999] pos_sim=0.818 neg_sim=0.022 gap=0.796 emb_norm=1.000
Epoch [200/400] lr=0.000059 train_loss=1.2001 train_acc=1.0000 aux_loss=0.0031 val_rank1=0.9222 val_mAP=0.8980 val_auc=0.9863 val_coverage=270/270 time=209.5s
Epoch [201/400] lr=0.000058 train_loss=1.1748 train_acc=0.9988 aux_loss=0.0022 val_rank1=0.9148 val_mAP=0.9011 val_auc=0.9854 val_coverage=270/270 time=217.1s
Epoch [202/400] lr=0.000058 train_loss=1.1666 train_acc=1.0000 aux_loss=0.0021 val_rank1=0.9259 val_mAP=0.9045 val_auc=0.9861 val_coverage=270/270 time=211.9s
Epoch [203/400] lr=0.000058 train_loss=1.2299 train_acc=0.9969 aux_loss=0.0041 val_rank1=0.9222 val_mAP=0.9050 val_auc=0.9864 val_coverage=270/270 time=191.2s
Epoch [204/400] lr=0.000058 train_loss=1.0971 train_acc=0.9994 aux_loss=0.0012 val_rank1=0.9333 val_mAP=0.9179 val_auc=0.9883 val_coverage=270/270 time=195.8s
  ★ 新最佳! mAP=0.9179
Epoch [205/400] lr=0.000058 train_loss=1.1464 train_acc=0.9994 aux_loss=0.0028 val_rank1=0.9259 val_mAP=0.9074 val_auc=0.9886 val_coverage=270/270 time=212.0s
Epoch [206/400] lr=0.000058 train_loss=1.1042 train_acc=0.9994 aux_loss=0.0011 val_rank1=0.9296 val_mAP=0.9186 val_auc=0.9893 val_coverage=270/270 time=217.2s
  ★ 新最佳! mAP=0.9186
Epoch [207/400] lr=0.000058 train_loss=1.1250 train_acc=1.0000 aux_loss=0.0010 val_rank1=0.9333 val_mAP=0.9174 val_auc=0.9869 val_coverage=270/270 time=211.0s
Epoch [208/400] lr=0.000058 train_loss=1.0514 train_acc=1.0000 aux_loss=0.0006 val_rank1=0.9370 val_mAP=0.9227 val_auc=0.9890 val_coverage=270/270 time=192.9s
  ★ 新最佳! mAP=0.9227
Epoch [209/400] lr=0.000058 train_loss=1.0661 train_acc=0.9994 aux_loss=0.0021 val_rank1=0.9185 val_mAP=0.9103 val_auc=0.9885 val_coverage=270/270 time=198.2s
  [DEBUG] Embedding: sim_all=[-0.414, 0.059, 1.000] pos_sim=0.855 neg_sim=0.016 gap=0.839 emb_norm=1.000
Epoch [210/400] lr=0.000058 train_loss=1.0886 train_acc=0.9981 aux_loss=0.0021 val_rank1=0.9259 val_mAP=0.9115 val_auc=0.9868 val_coverage=270/270 time=213.1s
Epoch [211/400] lr=0.000058 train_loss=1.0242 train_acc=1.0000 aux_loss=0.0005 val_rank1=0.9222 val_mAP=0.9098 val_auc=0.9883 val_coverage=270/270 time=216.9s
Epoch [212/400] lr=0.000058 train_loss=1.0739 train_acc=0.9988 aux_loss=0.0022 val_rank1=0.9259 val_mAP=0.9155 val_auc=0.9899 val_coverage=270/270 time=201.8s
Epoch [213/400] lr=0.000057 train_loss=1.0363 train_acc=1.0000 aux_loss=0.0011 val_rank1=0.9259 val_mAP=0.9178 val_auc=0.9881 val_coverage=270/270 time=192.1s
Epoch [214/400] lr=0.000057 train_loss=0.9808 train_acc=1.0000 aux_loss=0.0002 val_rank1=0.9296 val_mAP=0.9164 val_auc=0.9882 val_coverage=270/270 time=205.0s
Epoch [215/400] lr=0.000057 train_loss=1.0055 train_acc=1.0000 aux_loss=0.0002 val_rank1=0.9296 val_mAP=0.9187 val_auc=0.9825 val_coverage=270/270 time=217.9s
Epoch [216/400] lr=0.000057 train_loss=1.0177 train_acc=0.9981 aux_loss=0.0015 val_rank1=0.9296 val_mAP=0.9144 val_auc=0.9874 val_coverage=270/270 time=207.5s
Epoch [217/400] lr=0.000057 train_loss=1.0007 train_acc=0.9994 aux_loss=0.0007 val_rank1=0.9296 val_mAP=0.9155 val_auc=0.9874 val_coverage=270/270 time=191.1s
Epoch [218/400] lr=0.000057 train_loss=1.0401 train_acc=0.9981 aux_loss=0.0017 val_rank1=0.9407 val_mAP=0.9282 val_auc=0.9901 val_coverage=270/270 time=204.0s
  ★ 新最佳! mAP=0.9282
Epoch [219/400] lr=0.000057 train_loss=0.9852 train_acc=1.0000 aux_loss=0.0003 val_rank1=0.9296 val_mAP=0.9198 val_auc=0.9867 val_coverage=270/270 time=212.3s
  [DEBUG] Embedding: sim_all=[-0.447, 0.042, 1.000] pos_sim=0.865 neg_sim=-0.002 gap=0.867 emb_norm=1.000
Epoch [220/400] lr=0.000057 train_loss=0.9670 train_acc=1.0000 aux_loss=0.0003 val_rank1=0.9296 val_mAP=0.9204 val_auc=0.9893 val_coverage=270/270 time=215.5s
Epoch [221/400] lr=0.000057 train_loss=0.9695 train_acc=0.9994 aux_loss=0.0002 val_rank1=0.9333 val_mAP=0.9234 val_auc=0.9902 val_coverage=270/270 time=202.5s
Epoch [222/400] lr=0.000056 train_loss=0.9490 train_acc=0.9981 aux_loss=0.0005 val_rank1=0.9296 val_mAP=0.9195 val_auc=0.9882 val_coverage=270/270 time=191.3s
Epoch [223/400] lr=0.000056 train_loss=0.9463 train_acc=0.9994 aux_loss=0.0005 val_rank1=0.9296 val_mAP=0.9180 val_auc=0.9878 val_coverage=270/270 time=210.1s
Epoch [224/400] lr=0.000056 train_loss=0.9590 train_acc=0.9981 aux_loss=0.0013 val_rank1=0.9370 val_mAP=0.9220 val_auc=0.9871 val_coverage=270/270 time=220.4s
Epoch [225/400] lr=0.000056 train_loss=0.9687 train_acc=0.9981 aux_loss=0.0003 val_rank1=0.9259 val_mAP=0.9190 val_auc=0.9902 val_coverage=270/270 time=213.6s
Epoch [226/400] lr=0.000056 train_loss=0.9403 train_acc=0.9988 aux_loss=0.0012 val_rank1=0.9296 val_mAP=0.9197 val_auc=0.9875 val_coverage=270/270 time=194.3s
Epoch [227/400] lr=0.000056 train_loss=0.8869 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9196 val_auc=0.9874 val_coverage=270/270 time=195.3s
Epoch [228/400] lr=0.000056 train_loss=0.9288 train_acc=0.9994 aux_loss=0.0002 val_rank1=0.9296 val_mAP=0.9184 val_auc=0.9858 val_coverage=270/270 time=210.0s
Epoch [229/400] lr=0.000056 train_loss=0.9486 train_acc=0.9981 aux_loss=0.0015 val_rank1=0.9185 val_mAP=0.9120 val_auc=0.9847 val_coverage=270/270 time=215.5s
  [DEBUG] Embedding: sim_all=[-0.460, 0.032, 1.000] pos_sim=0.871 neg_sim=-0.014 gap=0.885 emb_norm=1.000
Epoch [230/400] lr=0.000056 train_loss=0.8837 train_acc=0.9994 aux_loss=0.0003 val_rank1=0.9296 val_mAP=0.9145 val_auc=0.9844 val_coverage=270/270 time=199.5s
Epoch [231/400] lr=0.000055 train_loss=0.8687 train_acc=0.9994 aux_loss=0.0006 val_rank1=0.9296 val_mAP=0.9175 val_auc=0.9814 val_coverage=270/270 time=193.1s
Epoch [232/400] lr=0.000055 train_loss=0.9065 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9116 val_auc=0.9863 val_coverage=270/270 time=198.5s
Epoch [233/400] lr=0.000055 train_loss=0.8870 train_acc=1.0000 aux_loss=0.0001 val_rank1=0.9333 val_mAP=0.9225 val_auc=0.9857 val_coverage=270/270 time=218.2s
Epoch [234/400] lr=0.000055 train_loss=0.8824 train_acc=0.9994 aux_loss=0.0000 val_rank1=0.9222 val_mAP=0.9167 val_auc=0.9841 val_coverage=270/270 time=207.5s
Epoch [235/400] lr=0.000055 train_loss=0.8692 train_acc=0.9994 aux_loss=0.0004 val_rank1=0.9296 val_mAP=0.9141 val_auc=0.9794 val_coverage=270/270 time=191.2s
Epoch [236/400] lr=0.000055 train_loss=0.8739 train_acc=1.0000 aux_loss=0.0002 val_rank1=0.9222 val_mAP=0.9155 val_auc=0.9863 val_coverage=270/270 time=203.4s
Epoch [237/400] lr=0.000055 train_loss=0.8478 train_acc=0.9994 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9165 val_auc=0.9867 val_coverage=270/270 time=214.4s
Epoch [238/400] lr=0.000055 train_loss=0.8723 train_acc=0.9988 aux_loss=0.0005 val_rank1=0.9259 val_mAP=0.9204 val_auc=0.9826 val_coverage=270/270 time=207.8s
Epoch [239/400] lr=0.000054 train_loss=0.8503 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9222 val_mAP=0.9174 val_auc=0.9877 val_coverage=270/270 time=191.9s
  [DEBUG] Embedding: sim_all=[-0.405, 0.031, 1.000] pos_sim=0.891 neg_sim=-0.015 gap=0.906 emb_norm=1.000
Epoch [240/400] lr=0.000054 train_loss=0.8805 train_acc=0.9994 aux_loss=0.0005 val_rank1=0.9259 val_mAP=0.9219 val_auc=0.9903 val_coverage=270/270 time=207.7s
Epoch [241/400] lr=0.000054 train_loss=0.8466 train_acc=0.9994 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9236 val_auc=0.9883 val_coverage=270/270 time=215.9s
Epoch [242/400] lr=0.000054 train_loss=0.8243 train_acc=0.9988 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9246 val_auc=0.9890 val_coverage=270/270 time=218.0s
Epoch [243/400] lr=0.000054 train_loss=0.8307 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9155 val_auc=0.9830 val_coverage=270/270 time=215.3s
Epoch [244/400] lr=0.000054 train_loss=0.8373 train_acc=0.9988 aux_loss=0.0007 val_rank1=0.9296 val_mAP=0.9224 val_auc=0.9897 val_coverage=270/270 time=194.0s
Epoch [245/400] lr=0.000054 train_loss=0.8061 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9192 val_auc=0.9885 val_coverage=270/270 time=193.6s
Epoch [246/400] lr=0.000053 train_loss=0.8451 train_acc=1.0000 aux_loss=0.0002 val_rank1=0.9222 val_mAP=0.9179 val_auc=0.9864 val_coverage=270/270 time=211.8s
Epoch [247/400] lr=0.000053 train_loss=0.8307 train_acc=0.9994 aux_loss=0.0002 val_rank1=0.9222 val_mAP=0.9172 val_auc=0.9845 val_coverage=270/270 time=216.4s
Epoch [248/400] lr=0.000053 train_loss=0.7957 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9212 val_auc=0.9865 val_coverage=270/270 time=213.2s
Epoch [249/400] lr=0.000053 train_loss=0.8184 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9222 val_mAP=0.9191 val_auc=0.9885 val_coverage=270/270 time=194.3s
  [DEBUG] Embedding: sim_all=[-0.406, 0.024, 1.000] pos_sim=0.904 neg_sim=-0.024 gap=0.928 emb_norm=1.000
Epoch [250/400] lr=0.000053 train_loss=0.8040 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9251 val_auc=0.9885 val_coverage=270/270 time=195.5s
Epoch [251/400] lr=0.000053 train_loss=0.8076 train_acc=0.9994 aux_loss=0.0000 val_rank1=0.9333 val_mAP=0.9254 val_auc=0.9884 val_coverage=270/270 time=207.6s
Epoch [252/400] lr=0.000052 train_loss=0.8018 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9198 val_auc=0.9856 val_coverage=270/270 time=219.3s
Epoch [253/400] lr=0.000052 train_loss=0.7732 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9219 val_auc=0.9829 val_coverage=270/270 time=217.3s
Epoch [254/400] lr=0.000052 train_loss=0.8200 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9220 val_auc=0.9875 val_coverage=270/270 time=205.0s
Epoch [255/400] lr=0.000052 train_loss=0.8353 train_acc=0.9988 aux_loss=0.0007 val_rank1=0.9259 val_mAP=0.9196 val_auc=0.9873 val_coverage=270/270 time=191.7s
Epoch [256/400] lr=0.000052 train_loss=0.8046 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9208 val_auc=0.9878 val_coverage=270/270 time=207.0s
Epoch [257/400] lr=0.000052 train_loss=0.8029 train_acc=0.9994 aux_loss=0.0000 val_rank1=0.9370 val_mAP=0.9263 val_auc=0.9866 val_coverage=270/270 time=216.0s
Epoch [258/400] lr=0.000051 train_loss=0.7828 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9333 val_mAP=0.9228 val_auc=0.9863 val_coverage=270/270 time=215.3s
Epoch [259/400] lr=0.000051 train_loss=0.8046 train_acc=0.9994 aux_loss=0.0004 val_rank1=0.9296 val_mAP=0.9222 val_auc=0.9859 val_coverage=270/270 time=198.9s
  [DEBUG] Embedding: sim_all=[-0.405, 0.027, 1.000] pos_sim=0.905 neg_sim=-0.021 gap=0.926 emb_norm=1.000
Epoch [260/400] lr=0.000051 train_loss=0.7859 train_acc=0.9994 aux_loss=0.0003 val_rank1=0.9296 val_mAP=0.9184 val_auc=0.9807 val_coverage=270/270 time=196.0s
Epoch [261/400] lr=0.000051 train_loss=0.8078 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9176 val_auc=0.9851 val_coverage=270/270 time=201.7s
Epoch [262/400] lr=0.000051 train_loss=0.7740 train_acc=1.0000 aux_loss=0.0001 val_rank1=0.9222 val_mAP=0.9162 val_auc=0.9848 val_coverage=270/270 time=219.3s
Epoch [263/400] lr=0.000051 train_loss=0.7985 train_acc=0.9988 aux_loss=0.0009 val_rank1=0.9259 val_mAP=0.9174 val_auc=0.9869 val_coverage=270/270 time=216.9s
Epoch [264/400] lr=0.000050 train_loss=0.7757 train_acc=0.9994 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9185 val_auc=0.9845 val_coverage=270/270 time=207.9s
Epoch [265/400] lr=0.000050 train_loss=0.7805 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9177 val_auc=0.9815 val_coverage=270/270 time=191.9s
Epoch [266/400] lr=0.000050 train_loss=0.7886 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9194 val_auc=0.9873 val_coverage=270/270 time=150.3s
Epoch [267/400] lr=0.000050 train_loss=0.7674 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9185 val_mAP=0.9158 val_auc=0.9885 val_coverage=270/270 time=139.3s
Epoch [268/400] lr=0.000050 train_loss=0.7918 train_acc=0.9988 aux_loss=0.0000 val_rank1=0.9222 val_mAP=0.9187 val_auc=0.9881 val_coverage=270/270 time=137.8s
Epoch [269/400] lr=0.000050 train_loss=0.7759 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9185 val_mAP=0.9142 val_auc=0.9865 val_coverage=270/270 time=136.3s
  [DEBUG] Embedding: sim_all=[-0.513, 0.024, 1.000] pos_sim=0.898 neg_sim=-0.023 gap=0.922 emb_norm=1.000
Epoch [270/400] lr=0.000049 train_loss=0.7827 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9207 val_auc=0.9855 val_coverage=270/270 time=134.8s
Epoch [271/400] lr=0.000049 train_loss=0.7622 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9222 val_mAP=0.9174 val_auc=0.9878 val_coverage=270/270 time=136.3s
Epoch [272/400] lr=0.000049 train_loss=0.7602 train_acc=0.9994 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9200 val_auc=0.9861 val_coverage=270/270 time=138.1s
Epoch [273/400] lr=0.000049 train_loss=0.7787 train_acc=0.9994 aux_loss=0.0000 val_rank1=0.9222 val_mAP=0.9186 val_auc=0.9867 val_coverage=270/270 time=136.9s
Epoch [274/400] lr=0.000049 train_loss=0.7685 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9185 val_mAP=0.9164 val_auc=0.9878 val_coverage=270/270 time=139.5s
Epoch [275/400] lr=0.000048 train_loss=0.7760 train_acc=0.9994 aux_loss=0.0001 val_rank1=0.9259 val_mAP=0.9152 val_auc=0.9876 val_coverage=270/270 time=137.1s
Epoch [276/400] lr=0.000048 train_loss=0.7772 train_acc=0.9994 aux_loss=0.0005 val_rank1=0.9222 val_mAP=0.9174 val_auc=0.9826 val_coverage=270/270 time=139.8s
Epoch [277/400] lr=0.000048 train_loss=0.7509 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9194 val_auc=0.9838 val_coverage=270/270 time=140.5s
Epoch [278/400] lr=0.000048 train_loss=0.7679 train_acc=0.9994 aux_loss=0.0000 val_rank1=0.9148 val_mAP=0.9066 val_auc=0.9835 val_coverage=270/270 time=137.0s
Epoch [279/400] lr=0.000048 train_loss=0.7593 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9145 val_auc=0.9781 val_coverage=270/270 time=138.1s
  [DEBUG] Embedding: sim_all=[-0.366, 0.024, 1.000] pos_sim=0.908 neg_sim=-0.024 gap=0.932 emb_norm=1.000
Epoch [280/400] lr=0.000048 train_loss=0.7486 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9222 val_mAP=0.9146 val_auc=0.9808 val_coverage=270/270 time=136.3s
Epoch [281/400] lr=0.000047 train_loss=0.7502 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9222 val_mAP=0.9175 val_auc=0.9878 val_coverage=270/270 time=139.2s
Epoch [282/400] lr=0.000047 train_loss=0.7552 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9173 val_auc=0.9899 val_coverage=270/270 time=137.0s
Epoch [283/400] lr=0.000047 train_loss=0.7449 train_acc=1.0000 aux_loss=0.0001 val_rank1=0.9296 val_mAP=0.9218 val_auc=0.9877 val_coverage=270/270 time=134.7s
Epoch [284/400] lr=0.000047 train_loss=0.7652 train_acc=0.9994 aux_loss=0.0001 val_rank1=0.9222 val_mAP=0.9117 val_auc=0.9829 val_coverage=270/270 time=140.4s
Epoch [285/400] lr=0.000047 train_loss=0.7614 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9185 val_mAP=0.9081 val_auc=0.9753 val_coverage=270/270 time=135.6s
Epoch [286/400] lr=0.000046 train_loss=0.7493 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9148 val_mAP=0.9028 val_auc=0.9742 val_coverage=270/270 time=136.0s
Epoch [287/400] lr=0.000046 train_loss=0.7582 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9148 val_mAP=0.9069 val_auc=0.9759 val_coverage=270/270 time=135.0s
Epoch [288/400] lr=0.000046 train_loss=0.7495 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9222 val_mAP=0.9135 val_auc=0.9740 val_coverage=270/270 time=135.4s
Epoch [289/400] lr=0.000046 train_loss=0.7463 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9222 val_mAP=0.9100 val_auc=0.9812 val_coverage=270/270 time=135.9s
  [DEBUG] Embedding: sim_all=[-0.362, 0.033, 1.000] pos_sim=0.909 neg_sim=-0.014 gap=0.924 emb_norm=1.000
Epoch [290/400] lr=0.000046 train_loss=0.7514 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9222 val_mAP=0.9134 val_auc=0.9766 val_coverage=270/270 time=136.4s
Epoch [291/400] lr=0.000045 train_loss=0.7502 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9162 val_auc=0.9824 val_coverage=270/270 time=138.8s
Epoch [292/400] lr=0.000045 train_loss=0.7567 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9154 val_auc=0.9841 val_coverage=270/270 time=135.7s
Epoch [293/400] lr=0.000045 train_loss=0.7396 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9139 val_auc=0.9827 val_coverage=270/270 time=139.1s
Epoch [294/400] lr=0.000045 train_loss=0.7533 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9085 val_auc=0.9798 val_coverage=270/270 time=136.1s
Epoch [295/400] lr=0.000045 train_loss=0.7580 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9222 val_mAP=0.9127 val_auc=0.9812 val_coverage=270/270 time=136.9s
Epoch [296/400] lr=0.000044 train_loss=0.7573 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9142 val_auc=0.9813 val_coverage=270/270 time=133.4s
Epoch [297/400] lr=0.000044 train_loss=0.7483 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9179 val_auc=0.9868 val_coverage=270/270 time=135.6s
Epoch [298/400] lr=0.000044 train_loss=0.7371 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9222 val_mAP=0.9162 val_auc=0.9854 val_coverage=270/270 time=138.5s
Epoch [299/400] lr=0.000044 train_loss=0.7415 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9176 val_auc=0.9842 val_coverage=270/270 time=140.4s
  [DEBUG] Embedding: sim_all=[-0.377, 0.032, 1.000] pos_sim=0.928 neg_sim=-0.017 gap=0.945 emb_norm=1.000
Epoch [300/400] lr=0.000044 train_loss=0.7474 train_acc=0.9994 aux_loss=0.0003 val_rank1=0.9296 val_mAP=0.9238 val_auc=0.9886 val_coverage=270/270 time=138.0s
Epoch [301/400] lr=0.000043 train_loss=0.7620 train_acc=0.9994 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9156 val_auc=0.9811 val_coverage=270/270 time=134.8s
Epoch [302/400] lr=0.000043 train_loss=0.7361 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9169 val_auc=0.9792 val_coverage=270/270 time=138.4s
Epoch [303/400] lr=0.000043 train_loss=0.7405 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9370 val_mAP=0.9215 val_auc=0.9763 val_coverage=270/270 time=136.1s
Epoch [304/400] lr=0.000043 train_loss=0.7402 train_acc=1.0000 aux_loss=0.0001 val_rank1=0.9296 val_mAP=0.9165 val_auc=0.9830 val_coverage=270/270 time=134.9s
Epoch [305/400] lr=0.000042 train_loss=0.7522 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9175 val_auc=0.9756 val_coverage=270/270 time=140.7s
Epoch [306/400] lr=0.000042 train_loss=0.7338 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9160 val_auc=0.9767 val_coverage=270/270 time=138.9s
Epoch [307/400] lr=0.000042 train_loss=0.7659 train_acc=0.9994 aux_loss=0.0001 val_rank1=0.9222 val_mAP=0.9100 val_auc=0.9720 val_coverage=270/270 time=138.3s
Epoch [308/400] lr=0.000042 train_loss=0.7567 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9167 val_auc=0.9739 val_coverage=270/270 time=135.4s
Epoch [309/400] lr=0.000042 train_loss=0.7580 train_acc=0.9994 aux_loss=0.0001 val_rank1=0.9259 val_mAP=0.9155 val_auc=0.9794 val_coverage=270/270 time=137.5s
  [DEBUG] Embedding: sim_all=[-0.348, 0.031, 1.000] pos_sim=0.921 neg_sim=-0.018 gap=0.939 emb_norm=1.000
Epoch [310/400] lr=0.000041 train_loss=0.7473 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9200 val_auc=0.9789 val_coverage=270/270 time=135.4s
Epoch [311/400] lr=0.000041 train_loss=0.7404 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9333 val_mAP=0.9224 val_auc=0.9769 val_coverage=270/270 time=134.4s
Epoch [312/400] lr=0.000041 train_loss=0.7398 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9333 val_mAP=0.9253 val_auc=0.9882 val_coverage=270/270 time=134.3s
Epoch [313/400] lr=0.000041 train_loss=0.7421 train_acc=1.0000 aux_loss=0.0001 val_rank1=0.9259 val_mAP=0.9211 val_auc=0.9825 val_coverage=270/270 time=134.7s
Epoch [314/400] lr=0.000041 train_loss=0.7288 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9161 val_auc=0.9762 val_coverage=270/270 time=134.8s
Epoch [315/400] lr=0.000040 train_loss=0.7200 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9230 val_auc=0.9812 val_coverage=270/270 time=137.6s
Epoch [316/400] lr=0.000040 train_loss=0.7384 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9185 val_mAP=0.9105 val_auc=0.9806 val_coverage=270/270 time=136.6s
Epoch [317/400] lr=0.000040 train_loss=0.7551 train_acc=0.9994 aux_loss=0.0007 val_rank1=0.9259 val_mAP=0.9207 val_auc=0.9839 val_coverage=270/270 time=139.5s
Epoch [318/400] lr=0.000040 train_loss=0.7269 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9230 val_auc=0.9877 val_coverage=270/270 time=138.5s
Epoch [319/400] lr=0.000039 train_loss=0.7257 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9238 val_auc=0.9834 val_coverage=270/270 time=138.4s
  [DEBUG] Embedding: sim_all=[-0.331, 0.032, 1.000] pos_sim=0.918 neg_sim=-0.016 gap=0.934 emb_norm=1.000
Epoch [320/400] lr=0.000039 train_loss=0.7519 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9165 val_auc=0.9751 val_coverage=270/270 time=134.8s
Epoch [321/400] lr=0.000039 train_loss=0.7364 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9135 val_auc=0.9731 val_coverage=270/270 time=136.3s
Epoch [322/400] lr=0.000039 train_loss=0.7270 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9162 val_auc=0.9765 val_coverage=270/270 time=135.8s
Epoch [323/400] lr=0.000038 train_loss=0.7276 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9153 val_auc=0.9747 val_coverage=270/270 time=134.8s
Epoch [324/400] lr=0.000038 train_loss=0.7253 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9188 val_auc=0.9737 val_coverage=270/270 time=136.1s
Epoch [325/400] lr=0.000038 train_loss=0.7325 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9164 val_auc=0.9725 val_coverage=270/270 time=135.5s
Epoch [326/400] lr=0.000038 train_loss=0.7213 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9135 val_auc=0.9709 val_coverage=270/270 time=137.1s
Epoch [327/400] lr=0.000038 train_loss=0.7177 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9214 val_auc=0.9787 val_coverage=270/270 time=141.1s
Epoch [328/400] lr=0.000037 train_loss=0.7445 train_acc=0.9994 aux_loss=0.0003 val_rank1=0.9222 val_mAP=0.9170 val_auc=0.9796 val_coverage=270/270 time=135.7s
Epoch [329/400] lr=0.000037 train_loss=0.7372 train_acc=0.9994 aux_loss=0.0004 val_rank1=0.9333 val_mAP=0.9258 val_auc=0.9806 val_coverage=270/270 time=137.6s
  [DEBUG] Embedding: sim_all=[-0.437, 0.034, 1.000] pos_sim=0.925 neg_sim=-0.015 gap=0.940 emb_norm=1.000
Epoch [330/400] lr=0.000037 train_loss=0.7333 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9333 val_mAP=0.9249 val_auc=0.9793 val_coverage=270/270 time=137.4s
Epoch [331/400] lr=0.000037 train_loss=0.7143 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9226 val_auc=0.9795 val_coverage=270/270 time=139.7s
Epoch [332/400] lr=0.000036 train_loss=0.7290 train_acc=0.9994 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9227 val_auc=0.9801 val_coverage=270/270 time=136.0s
Epoch [333/400] lr=0.000036 train_loss=0.7298 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9198 val_auc=0.9817 val_coverage=270/270 time=138.1s
Epoch [334/400] lr=0.000036 train_loss=0.7258 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9200 val_auc=0.9837 val_coverage=270/270 time=139.9s
Epoch [335/400] lr=0.000036 train_loss=0.7235 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9185 val_mAP=0.9114 val_auc=0.9764 val_coverage=270/270 time=135.4s
Epoch [336/400] lr=0.000036 train_loss=0.7265 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9150 val_auc=0.9778 val_coverage=270/270 time=140.0s
Epoch [337/400] lr=0.000035 train_loss=0.7330 train_acc=0.9994 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9196 val_auc=0.9795 val_coverage=270/270 time=137.9s
Epoch [338/400] lr=0.000035 train_loss=0.7463 train_acc=0.9994 aux_loss=0.0007 val_rank1=0.9296 val_mAP=0.9155 val_auc=0.9796 val_coverage=270/270 time=134.3s
Epoch [339/400] lr=0.000035 train_loss=0.7272 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9205 val_auc=0.9822 val_coverage=270/270 time=134.6s
  [DEBUG] Embedding: sim_all=[-0.391, 0.032, 1.000] pos_sim=0.915 neg_sim=-0.016 gap=0.931 emb_norm=1.000
Epoch [340/400] lr=0.000035 train_loss=0.7203 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9160 val_auc=0.9779 val_coverage=270/270 time=137.4s
Epoch [341/400] lr=0.000034 train_loss=0.7216 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9185 val_mAP=0.9138 val_auc=0.9802 val_coverage=270/270 time=136.9s
Epoch [342/400] lr=0.000034 train_loss=0.7157 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9222 val_mAP=0.9134 val_auc=0.9742 val_coverage=270/270 time=137.2s
Epoch [343/400] lr=0.000034 train_loss=0.7222 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9222 val_mAP=0.9163 val_auc=0.9810 val_coverage=270/270 time=135.1s
Epoch [344/400] lr=0.000034 train_loss=0.7235 train_acc=0.9994 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9198 val_auc=0.9807 val_coverage=270/270 time=135.2s
Epoch [345/400] lr=0.000033 train_loss=0.7304 train_acc=0.9994 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9136 val_auc=0.9717 val_coverage=270/270 time=137.2s
Epoch [346/400] lr=0.000033 train_loss=0.7227 train_acc=0.9994 aux_loss=0.0000 val_rank1=0.9185 val_mAP=0.9144 val_auc=0.9779 val_coverage=270/270 time=137.4s
Epoch [347/400] lr=0.000033 train_loss=0.7127 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9222 val_mAP=0.9175 val_auc=0.9795 val_coverage=270/270 time=137.2s
Epoch [348/400] lr=0.000033 train_loss=0.7162 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9222 val_mAP=0.9164 val_auc=0.9777 val_coverage=270/270 time=133.8s
Epoch [349/400] lr=0.000032 train_loss=0.7159 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9222 val_mAP=0.9106 val_auc=0.9707 val_coverage=270/270 time=134.0s
  [DEBUG] Embedding: sim_all=[-0.298, 0.038, 1.000] pos_sim=0.918 neg_sim=-0.010 gap=0.928 emb_norm=1.000
Epoch [350/400] lr=0.000032 train_loss=0.7106 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9185 val_mAP=0.9027 val_auc=0.9685 val_coverage=270/270 time=139.2s
Epoch [351/400] lr=0.000032 train_loss=0.7245 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9147 val_auc=0.9748 val_coverage=270/270 time=136.1s
Epoch [352/400] lr=0.000032 train_loss=0.7185 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9185 val_mAP=0.9121 val_auc=0.9764 val_coverage=270/270 time=138.5s
Epoch [353/400] lr=0.000032 train_loss=0.7117 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9222 val_mAP=0.9148 val_auc=0.9782 val_coverage=270/270 time=134.0s
Epoch [354/400] lr=0.000031 train_loss=0.7282 train_acc=0.9994 aux_loss=0.0001 val_rank1=0.9222 val_mAP=0.9121 val_auc=0.9764 val_coverage=270/270 time=135.6s
Epoch [355/400] lr=0.000031 train_loss=0.7189 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9222 val_mAP=0.9076 val_auc=0.9749 val_coverage=270/270 time=135.7s
Epoch [356/400] lr=0.000031 train_loss=0.7111 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9185 val_mAP=0.9123 val_auc=0.9776 val_coverage=270/270 time=138.5s
Epoch [357/400] lr=0.000031 train_loss=0.7118 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9222 val_mAP=0.9125 val_auc=0.9769 val_coverage=270/270 time=137.5s
Epoch [358/400] lr=0.000030 train_loss=0.7154 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9186 val_auc=0.9810 val_coverage=270/270 time=137.4s
Epoch [359/400] lr=0.000030 train_loss=0.7199 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9095 val_auc=0.9791 val_coverage=270/270 time=135.0s
  [DEBUG] Embedding: sim_all=[-0.410, 0.041, 1.000] pos_sim=0.921 neg_sim=-0.007 gap=0.929 emb_norm=1.000
Epoch [360/400] lr=0.000030 train_loss=0.7093 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9222 val_mAP=0.9099 val_auc=0.9804 val_coverage=270/270 time=136.8s
Epoch [361/400] lr=0.000030 train_loss=0.7161 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9152 val_auc=0.9757 val_coverage=270/270 time=137.4s
Epoch [362/400] lr=0.000029 train_loss=0.7143 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9179 val_auc=0.9815 val_coverage=270/270 time=136.8s
Epoch [363/400] lr=0.000029 train_loss=0.7106 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9333 val_mAP=0.9243 val_auc=0.9794 val_coverage=270/270 time=137.8s
Epoch [364/400] lr=0.000029 train_loss=0.7179 train_acc=0.9994 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9202 val_auc=0.9765 val_coverage=270/270 time=138.3s
Epoch [365/400] lr=0.000029 train_loss=0.7126 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9150 val_auc=0.9765 val_coverage=270/270 time=140.1s
Epoch [366/400] lr=0.000029 train_loss=0.7122 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9206 val_auc=0.9772 val_coverage=270/270 time=138.4s
Epoch [367/400] lr=0.000028 train_loss=0.7140 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9333 val_mAP=0.9203 val_auc=0.9777 val_coverage=270/270 time=138.0s
Epoch [368/400] lr=0.000028 train_loss=0.7233 train_acc=0.9994 aux_loss=0.0000 val_rank1=0.9333 val_mAP=0.9212 val_auc=0.9785 val_coverage=270/270 time=137.0s
Epoch [369/400] lr=0.000028 train_loss=0.7142 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9370 val_mAP=0.9271 val_auc=0.9793 val_coverage=270/270 time=137.2s
  [DEBUG] Embedding: sim_all=[-0.392, 0.044, 1.000] pos_sim=0.924 neg_sim=-0.004 gap=0.928 emb_norm=1.000
Epoch [370/400] lr=0.000028 train_loss=0.7127 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9135 val_auc=0.9767 val_coverage=270/270 time=137.8s
Epoch [371/400] lr=0.000027 train_loss=0.7249 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9144 val_auc=0.9753 val_coverage=270/270 time=138.8s
Epoch [372/400] lr=0.000027 train_loss=0.7161 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9176 val_auc=0.9774 val_coverage=270/270 time=138.6s
Epoch [373/400] lr=0.000027 train_loss=0.7086 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9204 val_auc=0.9775 val_coverage=270/270 time=136.4s
Epoch [374/400] lr=0.000027 train_loss=0.7129 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9407 val_mAP=0.9281 val_auc=0.9782 val_coverage=270/270 time=138.1s
Epoch [375/400] lr=0.000026 train_loss=0.7116 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9370 val_mAP=0.9245 val_auc=0.9778 val_coverage=270/270 time=139.5s
Epoch [376/400] lr=0.000026 train_loss=0.7089 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9370 val_mAP=0.9235 val_auc=0.9770 val_coverage=270/270 time=137.4s
Epoch [377/400] lr=0.000026 train_loss=0.7073 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9370 val_mAP=0.9236 val_auc=0.9762 val_coverage=270/270 time=140.0s
Epoch [378/400] lr=0.000026 train_loss=0.7127 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9370 val_mAP=0.9260 val_auc=0.9781 val_coverage=270/270 time=136.9s
Epoch [379/400] lr=0.000025 train_loss=0.7107 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9444 val_mAP=0.9315 val_auc=0.9776 val_coverage=270/270 time=137.3s
  ★ 新最佳! mAP=0.9315
  [DEBUG] Embedding: sim_all=[-0.374, 0.046, 1.000] pos_sim=0.931 neg_sim=-0.002 gap=0.934 emb_norm=1.000
Epoch [380/400] lr=0.000025 train_loss=0.7107 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9178 val_auc=0.9776 val_coverage=270/270 time=137.6s
Epoch [381/400] lr=0.000025 train_loss=0.7100 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9333 val_mAP=0.9245 val_auc=0.9795 val_coverage=270/270 time=136.2s
Epoch [382/400] lr=0.000025 train_loss=0.7109 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9159 val_auc=0.9685 val_coverage=270/270 time=136.4s
Epoch [383/400] lr=0.000025 train_loss=0.7061 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9333 val_mAP=0.9245 val_auc=0.9774 val_coverage=270/270 time=135.7s
Epoch [384/400] lr=0.000024 train_loss=0.7333 train_acc=0.9994 aux_loss=0.0021 val_rank1=0.9333 val_mAP=0.9229 val_auc=0.9809 val_coverage=270/270 time=136.2s
Epoch [385/400] lr=0.000024 train_loss=0.7120 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9206 val_auc=0.9758 val_coverage=270/270 time=135.2s
Epoch [386/400] lr=0.000024 train_loss=0.7152 train_acc=0.9994 aux_loss=0.0000 val_rank1=0.9370 val_mAP=0.9231 val_auc=0.9756 val_coverage=270/270 time=136.5s
Epoch [387/400] lr=0.000024 train_loss=0.7082 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9333 val_mAP=0.9213 val_auc=0.9759 val_coverage=270/270 time=136.4s
Epoch [388/400] lr=0.000023 train_loss=0.7188 train_acc=0.9994 aux_loss=0.0000 val_rank1=0.9370 val_mAP=0.9244 val_auc=0.9797 val_coverage=270/270 time=134.5s
Epoch [389/400] lr=0.000023 train_loss=0.7115 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9333 val_mAP=0.9211 val_auc=0.9763 val_coverage=270/270 time=135.2s
  [DEBUG] Embedding: sim_all=[-0.327, 0.047, 1.000] pos_sim=0.935 neg_sim=-0.002 gap=0.937 emb_norm=1.000
Epoch [390/400] lr=0.000023 train_loss=0.7119 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9155 val_auc=0.9750 val_coverage=270/270 time=136.7s
Epoch [391/400] lr=0.000023 train_loss=0.7082 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9333 val_mAP=0.9196 val_auc=0.9763 val_coverage=270/270 time=137.4s
Epoch [392/400] lr=0.000023 train_loss=0.7102 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9216 val_auc=0.9765 val_coverage=270/270 time=136.9s
Epoch [393/400] lr=0.000022 train_loss=0.7110 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9144 val_auc=0.9757 val_coverage=270/270 time=136.3s
Epoch [394/400] lr=0.000022 train_loss=0.7038 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9333 val_mAP=0.9222 val_auc=0.9765 val_coverage=270/270 time=141.3s
Epoch [395/400] lr=0.000022 train_loss=0.7203 train_acc=0.9994 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9198 val_auc=0.9764 val_coverage=270/270 time=136.2s
Epoch [396/400] lr=0.000022 train_loss=0.7105 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9259 val_mAP=0.9161 val_auc=0.9780 val_coverage=270/270 time=139.3s
Epoch [397/400] lr=0.000021 train_loss=0.7034 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9193 val_auc=0.9770 val_coverage=270/270 time=137.0s
Epoch [398/400] lr=0.000021 train_loss=0.7159 train_acc=0.9994 aux_loss=0.0000 val_rank1=0.9333 val_mAP=0.9228 val_auc=0.9775 val_coverage=270/270 time=139.4s
Epoch [399/400] lr=0.000021 train_loss=0.7050 train_acc=1.0000 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9185 val_auc=0.9778 val_coverage=270/270 time=138.3s
  [DEBUG] Embedding: sim_all=[-0.349, 0.047, 1.000] pos_sim=0.932 neg_sim=-0.001 gap=0.933 emb_norm=1.000
Epoch [400/400] lr=0.000021 train_loss=0.7150 train_acc=0.9994 aux_loss=0.0000 val_rank1=0.9296 val_mAP=0.9170 val_auc=0.9770 val_coverage=270/270 time=139.9s

训练完成! 总时间: 690.3 分钟
最佳 mAP: 0.9315 (Epoch 379)

========================================
加载最佳模型进行推理
========================================
  加载 checkpoint: /data0/yzhen/projects/gastrovision_v3/output/jaguar_arcface_backbone/best_model.pth
  ✓ 模型权重完全加载
  恢复自 Epoch 379, 最佳指标=0.9315
  已应用 EMA 权重
测试图片数: 371
提取 embedding: 100%|███████████████████████████████████████████████████████████████████████████| 6/6 [00:32<00:00,  5.49s/it]
读取测试对: /data0/yzhen/projects/data/jaguar_reid/test.csv
共 137270 对
校准参数: median=0.0033, IQR=0.0812
提交文件已保存: /data0/yzhen/projects/gastrovision_v3/output/jaguar_arcface_backbone/submission.csv
相似度统计: mean=0.5010, std=0.2387, min=0.0043, max=1.0000

完成