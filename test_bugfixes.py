"""
验证 7 个 BUG 修复的测试脚本

测试环境: conda activate torch27_env
运行: python test_bugfixes.py
"""

import torch
import torch.nn as nn
import numpy as np
import sys

PASSED = 0
FAILED = 0

def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✓ {name}")
    else:
        FAILED += 1
        print(f"  ✗ {name} — {detail}")


def test_bug1_focal_loss_no_label_smoothing():
    """BUG-1: FocalLoss 不接受 label_smoothing 参数"""
    print("\n[BUG-1] FocalLoss 构造函数参数检查")
    from gastrovision.losses.classification import FocalLoss
    
    # 应该可以正常创建
    fl = FocalLoss(gamma=2.0)
    pred = torch.randn(4, 10)
    target = torch.tensor([0, 1, 2, 3])
    loss = fl(pred, target)
    check("FocalLoss(gamma=2.0) 正常创建", loss.item() > 0)
    
    # 不应接受 label_smoothing
    import inspect
    sig = inspect.signature(FocalLoss.__init__)
    check("FocalLoss 无 label_smoothing 参数", 
          'label_smoothing' not in sig.parameters,
          "参数列表中不应包含 label_smoothing")


def test_bug2_class_balanced_loss_no_device():
    """BUG-2: ClassBalancedLoss 不接受 device 参数"""
    print("\n[BUG-2] ClassBalancedLoss 构造函数参数检查")
    from gastrovision.losses.classification import ClassBalancedLoss
    
    samples = [100, 50, 30, 20, 15, 10, 8, 5, 3, 2]
    cb = ClassBalancedLoss(samples_per_class=samples, beta=0.9999, gamma=2.0)
    pred = torch.randn(4, 10)
    target = torch.tensor([0, 1, 2, 3])
    loss = cb(pred, target)
    check("ClassBalancedLoss 正常创建和前向传播", loss.item() > 0)
    
    # main.py 不应传入 device 参数 — 检查源码
    import re
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    # 查找 ClassBalancedLoss 调用中是否还有 device=
    pattern = r'ClassBalancedLoss\([^)]*device='
    check("main.py 不传 device 给 ClassBalancedLoss",
          not re.search(pattern, content),
          "main.py 中仍传入 device 参数")


def test_bug3_resume_default_empty():
    """BUG-3: --resume 默认值应为空"""
    print("\n[BUG-3] --resume 默认值检查")
    import re
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找 --resume 参数定义
    match = re.search(r"add_argument\('--resume'.*?default='(.*?)'", content)
    check("--resume 参数存在", match is not None)
    if match:
        default_val = match.group(1)
        check("--resume 默认值为空字符串", 
              default_val == '',
              f"当前默认值: '{default_val}'")


def test_bug4_asymmetric_loss_focal_weight():
    """BUG-4: AsymmetricLoss 负样本 focal 权重方向验证"""
    print("\n[BUG-4] AsymmetricLoss focal 权重方向验证")
    from gastrovision.losses.multilabel import AsymmetricLoss
    
    # 创建 ASL (禁用 clip 以简化验证)
    asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.0)
    
    # 构造已知场景:
    # 简单负样本: logit 很小 (sigmoid ≈ 0), 标签=0 → p≈0, 应该被 down-weight
    # 困难负样本: logit 很大 (sigmoid ≈ 1), 标签=0 → p≈1, 应该保留高权重
    
    # 测试1: 简单负样本 (低 logit, 标签=0) 应该产生小损失
    easy_neg_logits = torch.tensor([[-5.0]])  # sigmoid ≈ 0.007
    easy_neg_targets = torch.tensor([[0.0]])
    loss_easy = asl(easy_neg_logits, easy_neg_targets)
    
    # 测试2: 困难负样本 (高 logit, 标签=0) 应该产生大损失
    hard_neg_logits = torch.tensor([[3.0]])   # sigmoid ≈ 0.95
    hard_neg_targets = torch.tensor([[0.0]])
    loss_hard = asl(hard_neg_logits, hard_neg_targets)
    
    check("困难负样本损失 > 简单负样本损失",
          loss_hard.item() > loss_easy.item(),
          f"hard={loss_hard.item():.4f} vs easy={loss_easy.item():.4f}")
    
    # 额外验证: 与官方 ASL 公式对比
    # 对于负样本 (clip=0): p = sigmoid(logit), weight = p^γ-
    # easy neg: p≈0.007, weight = 0.007^4 ≈ 2.4e-9 (tiny!)
    # hard neg: p≈0.95, weight = 0.95^4 ≈ 0.815
    p_easy = torch.sigmoid(easy_neg_logits).item()
    p_hard = torch.sigmoid(hard_neg_logits).item()
    ratio = loss_hard.item() / max(loss_easy.item(), 1e-12)
    check("损失比值合理 (困难/简单 >> 1)",
          ratio > 10,
          f"ratio={ratio:.1f}, 预期 >> 1")


def test_bug5_multilabel_trainer_tuple_outputs():
    """BUG-5: MultilabelTrainer 处理 tuple 输出"""
    print("\n[BUG-5] MultilabelTrainer tuple 输出处理")
    
    # 检查源码中 validate/test/optimize_thresholds 是否有 tuple 处理
    with open('gastrovision/trainers/multilabel.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 计算 "isinstance(outputs, tuple)" 出现次数
    count = content.count("isinstance(outputs, tuple)")
    # train_epoch(1) + validate(1) + optimize_thresholds(1) + test(1) = 4
    check("tuple 输出处理在 4 个方法中",
          count >= 4,
          f"只找到 {count} 处 isinstance(outputs, tuple)")
    
    # 检查 "outputs[0]" 出现次数 (应该至少有 4 处)
    count_extract = content.count("outputs[0]")
    check("outputs[0] 提取 logits 至少 4 处",
          count_extract >= 4,
          f"只找到 {count_extract} 处 outputs[0]")


def test_bug6_warmup_scheduler_initial_lr():
    """BUG-6: WarmupCosineScheduler 应在构造时设置 warmup LR"""
    print("\n[BUG-6] WarmupCosineScheduler 初始 LR 检查")
    from gastrovision.data.augmentation import WarmupCosineScheduler
    
    model = nn.Linear(10, 10)
    base_lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    
    warmup_epochs = 5
    total_epochs = 50
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=warmup_epochs, 
                                       total_epochs=total_epochs)
    
    # 构造后, LR 应该是 warmup 的 epoch-0 值, 不是 base_lr
    initial_lr = optimizer.param_groups[0]['lr']
    expected_lr = base_lr * 1 / warmup_epochs  # epoch 0: base_lr * 1/5
    
    check("构造后 LR 为 warmup 值",
          abs(initial_lr - expected_lr) < 1e-8,
          f"实际={initial_lr:.6f}, 预期={expected_lr:.6f}")
    
    check("构造后 LR < base_lr",
          initial_lr < base_lr,
          f"LR={initial_lr:.6f} 应该 < base_lr={base_lr}")
    
    # step() 后应该递增
    scheduler.step()  # current_epoch → 1
    lr_after_step1 = optimizer.param_groups[0]['lr']
    expected_lr_step1 = base_lr * 2 / warmup_epochs
    check("step() 后 LR 递增",
          abs(lr_after_step1 - expected_lr_step1) < 1e-8,
          f"实际={lr_after_step1:.6f}, 预期={expected_lr_step1:.6f}")


def test_bug7_onecycle_per_step():
    """BUG-7: OneCycleLR 在 train_epoch 中按 step 调用"""
    print("\n[BUG-7] OneCycleLR per-step 调用检查")
    
    # 检查 trainer.py 源码
    with open('gastrovision/trainers/trainer.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    check("trainer.py 有 OneCycleLR per-step 调用",
          "isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR)" in content
          and "pass" in content,
          "未找到 OneCycleLR 特殊处理")
    
    # 检查 multilabel.py 源码
    with open('gastrovision/trainers/multilabel.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    check("multilabel.py 有 OneCycleLR per-step 调用",
          "isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR)" in content
          and "pass" in content,
          "未找到 OneCycleLR 特殊处理")


def test_bug1_main_focal_no_crash():
    """BUG-1 补充: main.py create_loss_function 不再传 label_smoothing 给 FocalLoss"""
    print("\n[BUG-1 补充] main.py create_loss_function 源码检查")
    import re
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # FocalLoss 调用中不应包含 label_smoothing
    pattern = r'FocalLoss\([^)]*label_smoothing'
    check("main.py 不传 label_smoothing 给 FocalLoss",
          not re.search(pattern, content),
          "main.py 中仍传入 label_smoothing")


if __name__ == "__main__":
    print("=" * 60)
    print("BUG 修复验证测试")
    print("=" * 60)
    
    test_bug1_focal_loss_no_label_smoothing()
    test_bug2_class_balanced_loss_no_device()
    test_bug3_resume_default_empty()
    test_bug4_asymmetric_loss_focal_weight()
    test_bug5_multilabel_trainer_tuple_outputs()
    test_bug6_warmup_scheduler_initial_lr()
    test_bug7_onecycle_per_step()
    test_bug1_main_focal_no_crash()
    
    print("\n" + "=" * 60)
    print(f"结果: {PASSED} 通过, {FAILED} 失败")
    print("=" * 60)
    
    sys.exit(0 if FAILED == 0 else 1)
