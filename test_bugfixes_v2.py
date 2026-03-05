"""
验证 BUG-9/10/11 修复的测试脚本

测试环境: conda activate torch27_env
运行: python test_bugfixes_v2.py
"""

import torch
import torch.nn as nn
import numpy as np
import sys

PASSED = 0
FAILED = 0
FAILED_CASES = []


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        FAILED_CASES.append((name, detail))
        print(f"  [FAIL] {name} -- {detail}")


# =============================================================================
# BUG-9: CosFace/SphereFace CLI margin 参数失效
# =============================================================================

def test_bug9_cosface_margin():
    """BUG-9: CosFace margin 参数应映射到 m3"""
    print("\n[BUG-9a] CosFace margin → m3 参数映射")
    from gastrovision.losses.metric_learning import create_metric_loss

    # 默认 margin
    loss_fn = create_metric_loss('cosface', num_classes=10, embedding_dim=128)
    check("CosFace 默认 m3=0.35", loss_fn.m3 == 0.35, f"m3={loss_fn.m3}")
    check("CosFace 默认 margin(m2)=0.0", loss_fn.margin == 0.0, f"margin={loss_fn.margin}")

    # 用户指定 margin=0.5 → 应映射到 m3
    loss_fn2 = create_metric_loss('cosface', num_classes=10, embedding_dim=128, margin=0.5)
    check("CosFace 用户 margin=0.5 → m3=0.5", loss_fn2.m3 == 0.5, f"m3={loss_fn2.m3}")
    check("CosFace 用户 margin=0.5 → m2 仍为 0", loss_fn2.margin == 0.0, f"margin={loss_fn2.margin}")

    # 功能测试：不同 margin 产生不同 loss
    features = torch.randn(8, 128, requires_grad=True)
    labels = torch.randint(0, 10, (8,))
    loss1 = create_metric_loss('cosface', num_classes=10, embedding_dim=128, margin=0.1)(features, labels)
    loss2 = create_metric_loss('cosface', num_classes=10, embedding_dim=128, margin=0.5)(features.detach().requires_grad_(True), labels)
    check("CosFace 不同 margin 产生不同 loss", abs(loss1.item() - loss2.item()) > 0.01,
          f"loss(m=0.1)={loss1.item():.4f}, loss(m=0.5)={loss2.item():.4f}")


def test_bug9_sphereface_margin():
    """BUG-9: SphereFace margin 参数应映射到 m1"""
    print("\n[BUG-9b] SphereFace margin → m1 参数映射")
    from gastrovision.losses.metric_learning import create_metric_loss

    # 默认 m1
    loss_fn = create_metric_loss('sphereface', num_classes=10, embedding_dim=128)
    check("SphereFace 默认 m1=4.0", loss_fn.m1 == 4.0, f"m1={loss_fn.m1}")

    # 用户指定 margin=2.0 → 应映射到 m1
    loss_fn2 = create_metric_loss('sphereface', num_classes=10, embedding_dim=128, margin=2.0)
    check("SphereFace 用户 margin=2.0 → m1=2.0", loss_fn2.m1 == 2.0, f"m1={loss_fn2.m1}")

    # ArcFace 不受影响
    loss_fn3 = create_metric_loss('arcface', num_classes=10, embedding_dim=128, margin=0.3)
    check("ArcFace margin=0.3 不受影响", loss_fn3.margin == 0.3, f"margin={loss_fn3.margin}")


# =============================================================================
# BUG-10: tta_predict 不处理 tuple 模型输出
# =============================================================================

def test_bug10_tta_tuple_output():
    """BUG-10: tta_predict 应处理返回 tuple 的模型"""
    print("\n[BUG-10] tta_predict tuple 输出处理")
    from gastrovision.data.augmentation import tta_predict

    # 模拟返回 tensor 的模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3 * 32 * 32, 10)

        def forward(self, x):
            return self.fc(x.flatten(1))

    model = SimpleModel()
    images = torch.randn(2, 3, 32, 32)
    pred = tta_predict(model, images, num_augments=3)
    check("tensor 输出模型 TTA 成功", pred.shape == (2, 10), f"shape={pred.shape}")

    # 模拟返回 tuple 的模型
    class TupleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3 * 32 * 32, 10)
            self.feat = nn.Linear(3 * 32 * 32, 128)

        def forward(self, x):
            flat = x.flatten(1)
            return self.fc(flat), self.feat(flat)

    model2 = TupleModel()
    pred2 = tta_predict(model2, images, num_augments=3)
    check("tuple 输出模型 TTA 成功", pred2.shape == (2, 10), f"shape={pred2.shape}")
    check("TTA 结果无 nan", not torch.isnan(pred2).any().item())


# =============================================================================
# BUG-11: Mixup/CutMix 未接入训练循环
# =============================================================================

def test_bug11_trainer_mixup_params():
    """BUG-11: Trainer 应接受 mixup/cutmix 参数"""
    print("\n[BUG-11a] Trainer mixup/cutmix 参数")
    import inspect
    from gastrovision.trainers.trainer import Trainer

    sig = inspect.signature(Trainer.__init__)
    params = list(sig.parameters.keys())
    check("Trainer.__init__ 有 mixup_alpha", 'mixup_alpha' in params, f"params={params}")
    check("Trainer.__init__ 有 cutmix_alpha", 'cutmix_alpha' in params, f"params={params}")


def test_bug11_multilabel_trainer_mixup_params():
    """BUG-11: MultilabelTrainer 应接受 mixup/cutmix 参数"""
    print("\n[BUG-11b] MultilabelTrainer mixup/cutmix 参数")
    import inspect
    from gastrovision.trainers.multilabel import MultilabelTrainer

    sig = inspect.signature(MultilabelTrainer.__init__)
    params = list(sig.parameters.keys())
    check("MultilabelTrainer.__init__ 有 mixup_alpha", 'mixup_alpha' in params, f"params={params}")
    check("MultilabelTrainer.__init__ 有 cutmix_alpha", 'cutmix_alpha' in params, f"params={params}")


def test_bug11_trainer_mixup_functional():
    """BUG-11: Trainer 的 Mixup 功能应正常工作"""
    print("\n[BUG-11c] Trainer Mixup 功能测试")
    from gastrovision.trainers.trainer import Trainer
    from torch.utils.data import TensorDataset, DataLoader

    # 创建简单模型和数据
    model = nn.Linear(3 * 32 * 32, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 包装为合适的模型
    class FlatModel(nn.Module):
        def __init__(self, fc):
            super().__init__()
            self.fc = fc

        def forward(self, x):
            return self.fc(x.flatten(1))

    flat_model = FlatModel(model)

    # 创建假数据
    images = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, 10, (16,))
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=8, drop_last=True)

    # 测试 Mixup Trainer
    trainer = Trainer(
        model=flat_model,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device('cpu'),
        mixup_alpha=0.4,
        cutmix_alpha=0.0
    )
    check("Trainer 创建成功 (mixup_alpha=0.4)", trainer.mixup_alpha == 0.4)

    # 训练一个 epoch（不应崩溃）
    try:
        metrics = trainer.train_epoch(loader, epoch=1, total_epochs=1)
        check("Mixup train_epoch 成功", 'loss' in metrics, f"metrics keys={list(metrics.keys())}")
        check("Mixup loss 有效", metrics['loss'] > 0 and not np.isnan(metrics['loss']),
              f"loss={metrics['loss']}")
    except Exception as e:
        check("Mixup train_epoch 无异常", False, str(e))


def test_bug11_trainer_cutmix_functional():
    """BUG-11: Trainer 的 CutMix 功能应正常工作"""
    print("\n[BUG-11d] Trainer CutMix 功能测试")
    from gastrovision.trainers.trainer import Trainer
    from torch.utils.data import TensorDataset, DataLoader

    class FlatModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3 * 32 * 32, 10)

        def forward(self, x):
            return self.fc(x.flatten(1))

    model = FlatModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    images = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, 10, (16,))
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=8, drop_last=True)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device('cpu'),
        mixup_alpha=0.0,
        cutmix_alpha=1.0
    )
    check("Trainer 创建成功 (cutmix_alpha=1.0)", trainer.cutmix_alpha == 1.0)

    try:
        metrics = trainer.train_epoch(loader, epoch=1, total_epochs=1)
        check("CutMix train_epoch 成功", 'loss' in metrics)
    except Exception as e:
        check("CutMix train_epoch 无异常", False, str(e))


def test_bug11_trainer_both_mixup_cutmix():
    """BUG-11: Mixup + CutMix 同时启用"""
    print("\n[BUG-11e] Trainer Mixup+CutMix 同时启用")
    from gastrovision.trainers.trainer import Trainer
    from torch.utils.data import TensorDataset, DataLoader

    class FlatModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3 * 32 * 32, 10)

        def forward(self, x):
            return self.fc(x.flatten(1))

    model = FlatModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    images = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, 10, (16,))
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=8, drop_last=True)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device('cpu'),
        mixup_alpha=0.4,
        cutmix_alpha=1.0
    )

    try:
        metrics = trainer.train_epoch(loader, epoch=1, total_epochs=1)
        check("Mixup+CutMix train_epoch 成功", 'loss' in metrics)
    except Exception as e:
        check("Mixup+CutMix train_epoch 无异常", False, str(e))


def test_bug11_multilabel_mixup_functional():
    """BUG-11: MultilabelTrainer 的 Mixup 功能"""
    print("\n[BUG-11f] MultilabelTrainer Mixup 功能测试")
    from gastrovision.trainers.multilabel import MultilabelTrainer
    from torch.utils.data import TensorDataset, DataLoader

    class FlatModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3 * 32 * 32, 16)

        def forward(self, x):
            return self.fc(x.flatten(1))

    model = FlatModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    images = torch.randn(16, 3, 32, 32)
    # 多标签: 每个样本 1-3 个正标签
    labels = torch.zeros(16, 16)
    for i in range(16):
        pos_count = np.random.randint(1, 4)
        pos_indices = np.random.choice(16, pos_count, replace=False)
        labels[i, pos_indices] = 1.0

    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=8, drop_last=True)

    trainer = MultilabelTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device('cpu'),
        mixup_alpha=0.4,
        cutmix_alpha=0.0
    )

    try:
        metrics = trainer.train_epoch(loader, epoch=1, total_epochs=1)
        check("MultilabelTrainer Mixup train_epoch 成功", 'loss' in metrics)
        check("MultilabelTrainer Mixup loss 有效",
              metrics['loss'] > 0 and not np.isnan(metrics['loss']),
              f"loss={metrics['loss']}")
    except Exception as e:
        check("MultilabelTrainer Mixup 无异常", False, str(e))


def test_bug11_no_mixup_still_works():
    """BUG-11: 不启用 Mixup/CutMix 时训练应正常"""
    print("\n[BUG-11g] 无 Mixup/CutMix 时回归验证")
    from gastrovision.trainers.trainer import Trainer
    from torch.utils.data import TensorDataset, DataLoader

    class FlatModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3 * 32 * 32, 10)

        def forward(self, x):
            return self.fc(x.flatten(1))

    model = FlatModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    images = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, 10, (16,))
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=8, drop_last=True)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device('cpu'),
        mixup_alpha=0.0,
        cutmix_alpha=0.0
    )

    try:
        metrics = trainer.train_epoch(loader, epoch=1, total_epochs=1)
        check("无 Mixup/CutMix train_epoch 正常", 'loss' in metrics)
    except Exception as e:
        check("无 Mixup/CutMix 无异常", False, str(e))


def test_bug11_mixup_skips_metric_loss():
    """BUG-11: Mixup 激活时应跳过度量学习损失"""
    print("\n[BUG-11h] Mixup 跳过度量学习损失")

    # 验证源码中有 'not mixed' 条件
    with open('gastrovision/trainers/trainer.py', 'r', encoding='utf-8') as f:
        content = f.read()
    check("Trainer: metric loss 受 mixed 保护",
          "self.metric_loss is not None and not mixed" in content)

    with open('gastrovision/trainers/multilabel.py', 'r', encoding='utf-8') as f:
        content = f.read()
    check("MultilabelTrainer: metric loss 受 mixed 保护",
          "self.metric_loss is not None and not mixed" in content)
    check("MultilabelTrainer: triplet loss 受 mixed 保护",
          "self.triplet_loss is not None and not mixed" in content)


def test_bug11_main_passes_mixup_args():
    """BUG-11: main.py 传递 mixup/cutmix 参数"""
    print("\n[BUG-11i] main.py 参数传递")

    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()

    check("main.py 传递 mixup_alpha 给 Trainer",
          "mixup_alpha=args.mixup" in content)
    check("main.py 传递 cutmix_alpha 给 Trainer",
          "cutmix_alpha=args.cutmix" in content)
    # 两处传递（Trainer + MultilabelTrainer）
    check("main.py 两处传递 mixup_alpha",
          content.count("mixup_alpha=args.mixup") >= 2,
          f"count={content.count('mixup_alpha=args.mixup')}")


# =============================================================================
# 回归测试：确保之前的 BUG 修复仍有效
# =============================================================================

def test_regression_previous_fixes():
    """回归测试：之前 7 个 BUG 的修复仍然有效"""
    print("\n[回归] 之前的修复验证")

    # BUG-1: FocalLoss 不接受 label_smoothing
    from gastrovision.losses.classification import FocalLoss
    import inspect
    sig = inspect.signature(FocalLoss.__init__)
    check("BUG-1: FocalLoss 无 label_smoothing 参数",
          'label_smoothing' not in sig.parameters)

    # BUG-2: ClassBalancedLoss 不接受 device
    from gastrovision.losses.classification import ClassBalancedLoss
    sig = inspect.signature(ClassBalancedLoss.__init__)
    check("BUG-2: ClassBalancedLoss 无 device 参数",
          'device' not in sig.parameters)

    # BUG-5: MultilabelTrainer validate 处理 tuple 输出
    with open('gastrovision/trainers/multilabel.py', 'r', encoding='utf-8') as f:
        content = f.read()
    # validate 方法应有 isinstance check
    check("BUG-5: MultilabelTrainer validate 处理 tuple",
          "isinstance(outputs, tuple)" in content)

    # BUG-6: WarmupCosineScheduler 第一个 epoch warmup
    from gastrovision.data.augmentation import WarmupCosineScheduler
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, total_epochs=50)
    lr_epoch0 = optimizer.param_groups[0]['lr']
    check("BUG-6: WarmupCosineScheduler epoch-0 LR < base_lr",
          lr_epoch0 < 0.01, f"lr={lr_epoch0}")

    # BUG-7: OneCycleLR 在 train_epoch 中按 step 调用
    with open('gastrovision/trainers/trainer.py', 'r', encoding='utf-8') as f:
        trainer_content = f.read()
    check("BUG-7: OneCycleLR per-step in train_epoch",
          "isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR)" in trainer_content)


# =============================================================================
# 运行所有测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BUG-9/10/11 修复验证测试")
    print("=" * 60)

    test_bug9_cosface_margin()
    test_bug9_sphereface_margin()
    test_bug10_tta_tuple_output()
    test_bug11_trainer_mixup_params()
    test_bug11_multilabel_trainer_mixup_params()
    test_bug11_trainer_mixup_functional()
    test_bug11_trainer_cutmix_functional()
    test_bug11_trainer_both_mixup_cutmix()
    test_bug11_multilabel_mixup_functional()
    test_bug11_no_mixup_still_works()
    test_bug11_mixup_skips_metric_loss()
    test_bug11_main_passes_mixup_args()
    test_regression_previous_fixes()

    print("\n" + "=" * 60)
    print(f"结果: {PASSED} 通过, {FAILED} 失败")
    if FAILED_CASES:
        print("失败用例:")
        for idx, (name, detail) in enumerate(FAILED_CASES, 1):
            print(f"  {idx}. {name} :: {detail}")
    print("=" * 60)

    sys.exit(0 if FAILED == 0 else 1)
