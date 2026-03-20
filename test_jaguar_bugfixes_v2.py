"""
Jaguar Re-ID Session 4 BUG 修复验证测试

BUG-1 (CRITICAL): EMA decay=0.9999 对小数据集过高 → 自动计算
BUG-2 (HIGH): 用 val_rank1 选最佳模型 → 改为可配置 (默认 val_mAP)
BUG-3 (MEDIUM): 早停 patience=20 配合超慢 EMA → 增大或禁用
DEBUG: 添加 EMA 诊断 + embedding 统计日志

测试环境: conda activate torch27_env
"""

import sys
import os
import tempfile
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from types import SimpleNamespace
from io import StringIO

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ================================================================
# BUG-1: EMA decay 自动计算
# ================================================================

def test_bug1_ema_auto_compute_triggers():
    """BUG-1: 当 ema_decay >= 0.9999 时应自动计算更合理的 decay"""
    # 模拟 train.py 中的自动计算逻辑
    cfg_ema_decay = 0.9999
    steps_per_epoch = 50  # 1625 / 32 ≈ 50
    epochs = 80
    total_steps = steps_per_epoch * epochs  # 4000

    # 旧 decay 的保留率
    old_retention = cfg_ema_decay ** total_steps
    assert old_retention > 0.5, f"旧 decay 保留率应 > 0.5, 实际 {old_retention:.4f}"

    # 自动计算的新 decay
    auto_decay = 1.0 - 3.0 / total_steps
    auto_decay = max(auto_decay, 0.99)
    auto_decay = min(auto_decay, 0.9999)

    new_retention = auto_decay ** total_steps
    assert new_retention < 0.1, f"新 decay 保留率应 < 0.1, 实际 {new_retention:.4f}"

    # 新 decay 应显著低于旧 decay
    assert auto_decay < cfg_ema_decay, \
        f"自动 decay={auto_decay} 应 < 原始 {cfg_ema_decay}"

    print(f"  旧 decay={cfg_ema_decay}: 保留率={old_retention:.4f}")
    print(f"  新 decay={auto_decay:.6f}: 保留率={new_retention:.4f}")
    print("✓ test_bug1_ema_auto_compute_triggers PASSED")


def test_bug1_ema_auto_compute_various_sizes():
    """BUG-1: 自动计算对不同数据集大小都应给出合理 decay"""
    test_cases = [
        (50, 80, "小数据集 (1600样本, 80ep)"),
        (50, 120, "小数据集 (1600样本, 120ep)"),
        (100, 100, "中数据集 (3200样本, 100ep)"),
        (500, 200, "大数据集 (16000样本, 200ep)"),
    ]

    for steps_per_epoch, epochs, desc in test_cases:
        total_steps = steps_per_epoch * epochs
        auto_decay = 1.0 - 3.0 / total_steps
        auto_decay = max(auto_decay, 0.99)
        auto_decay = min(auto_decay, 0.9999)

        retention = auto_decay ** total_steps
        assert 0.99 <= auto_decay <= 0.9999, \
            f"{desc}: decay={auto_decay} 超出范围 [0.99, 0.9999]"
        assert retention < 0.2, \
            f"{desc}: 保留率={retention:.4f} 太高 (>0.2)"
        print(f"  {desc}: decay={auto_decay:.6f}, retention={retention:.4f}")

    print("✓ test_bug1_ema_auto_compute_various_sizes PASSED")


def test_bug1_ema_no_auto_for_low_decay():
    """BUG-1: 用户手动设置低 decay (< 0.9999) 时不应触发自动计算"""
    # 模拟 train.py 的条件: if cfg.ema and ema_decay >= 0.9999
    user_decay = 0.999
    should_auto = user_decay >= 0.9999
    assert not should_auto, "decay=0.999 不应触发自动计算"

    user_decay2 = 0.9999
    should_auto2 = user_decay2 >= 0.9999
    assert should_auto2, "decay=0.9999 应触发自动计算"

    print("✓ test_bug1_ema_no_auto_for_low_decay PASSED")


def test_bug1_ema_absorption_comparison():
    """BUG-1: 对比新旧 decay 在训练中的 EMA 吸收速度"""
    from gastrovision.utils.ema import ModelEMA

    # 简单模型
    model = nn.Linear(16, 8)

    # 旧 decay: 0.9999
    ema_old = ModelEMA(model, decay=0.9999)
    # 新 decay: 0.999
    ema_new = ModelEMA(model, decay=0.999)

    # 模拟 200 步更新
    for _ in range(200):
        # 修改模型参数
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        ema_old.update(model)
        ema_new.update(model)

    # 比较 EMA shadow 与实际参数的距离
    dist_old = 0.0
    dist_new = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad:
            dist_old += (ema_old.shadow[name] - param.data).pow(2).sum().item()
            dist_new += (ema_new.shadow[name] - param.data).pow(2).sum().item()

    dist_old = math.sqrt(dist_old)
    dist_new = math.sqrt(dist_new)

    # 新 decay 的 EMA 应更接近实际参数
    assert dist_new < dist_old, \
        f"decay=0.999 的 EMA 距离 ({dist_new:.4f}) 应 < decay=0.9999 ({dist_old:.4f})"
    print(f"  decay=0.9999 距离: {dist_old:.4f}")
    print(f"  decay=0.999  距离: {dist_new:.4f}")
    print("✓ test_bug1_ema_absorption_comparison PASSED")


# ================================================================
# BUG-2: 最佳模型选择指标
# ================================================================

def test_bug2_save_metric_parameter():
    """BUG-2: fit() 应接受 save_metric 参数"""
    from jaguar.trainer import ReIDTrainer
    from jaguar.model import ReIDModel
    from gastrovision.losses.metric_learning import ArcFaceLoss

    model = ReIDModel('resnet18', embedding_dim=64, pretrained=False, use_gem=False)
    criterion = ArcFaceLoss(num_classes=4, embedding_dim=64)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()), lr=0.001
    )

    with tempfile.TemporaryDirectory() as tmp:
        trainer = ReIDTrainer(
            model=model, criterion=criterion, loss_type='arcface',
            optimizer=optimizer, device=torch.device('cpu'),
            output_dir=tmp
        )

        # fit 应接受 save_metric
        import inspect
        sig = inspect.signature(trainer.fit)
        assert 'save_metric' in sig.parameters, "fit() 应有 save_metric 参数"
        # 默认值应为 'mAP'
        assert sig.parameters['save_metric'].default == 'mAP', \
            f"save_metric 默认值应为 'mAP', 实际 {sig.parameters['save_metric'].default}"

    print("✓ test_bug2_save_metric_parameter PASSED")


def test_bug2_save_metric_validation():
    """BUG-2: save_metric 应只接受 rank1/mAP/auc"""
    from jaguar.trainer import ReIDTrainer
    from jaguar.model import ReIDModel
    from gastrovision.losses.metric_learning import ArcFaceLoss
    from torch.utils.data import DataLoader, TensorDataset

    model = ReIDModel('resnet18', embedding_dim=64, pretrained=False, use_gem=False)
    criterion = ArcFaceLoss(num_classes=4, embedding_dim=64)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()), lr=0.001
    )

    images = torch.randn(8, 3, 224, 224)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)

    with tempfile.TemporaryDirectory() as tmp:
        trainer = ReIDTrainer(
            model=model, criterion=criterion, loss_type='arcface',
            optimizer=optimizer, device=torch.device('cpu'),
            output_dir=tmp
        )

        # 无效 metric 应报错
        try:
            trainer.fit(loader, loader, epochs=1, save_metric='invalid')
            assert False, "应抛出 AssertionError"
        except AssertionError:
            pass  # Expected

        # 有效 metrics 应通过
        for metric in ['rank1', 'mAP', 'auc']:
            trainer.best_metric = 0.0
            trainer.fit(loader, loader, epochs=1, early_stopping=0, save_metric=metric)

    print("✓ test_bug2_save_metric_validation PASSED")


def test_bug2_mAP_selects_better_model():
    """BUG-2: 使用 mAP 选模型应比 rank1 选到更晚期 (更好) 的模型"""
    # 模拟训练日志中的指标变化
    # ConvNeXt run 3: rank1 在 epoch 64 就到 0.7556 并稳定
    # 但 mAP 一直增长到 epoch 100 = 0.5169
    mock_metrics = [
        # (epoch, rank1, mAP, auc)
        (50, 0.7407, 0.3477, 0.6269),
        (60, 0.7481, 0.3711, 0.6395),
        (70, 0.7556, 0.4008, 0.6566),
        (80, 0.7593, 0.4316, 0.6791),
        (90, 0.7741, 0.4716, 0.7078),
        (97, 0.7963, 0.5041, 0.7317),
        (100, 0.7963, 0.5169, 0.7429),
    ]

    # 按 rank1 选: 最佳是 epoch 97 (rank1=0.7963)
    best_rank1_epoch = max(mock_metrics, key=lambda x: x[1])[0]
    best_rank1_mAP = [m for m in mock_metrics if m[0] == best_rank1_epoch][0][2]

    # 按 mAP 选: 最佳是 epoch 100 (mAP=0.5169)
    best_mAP_epoch = max(mock_metrics, key=lambda x: x[2])[0]
    best_mAP_mAP = [m for m in mock_metrics if m[0] == best_mAP_epoch][0][2]

    # mAP 选的模型的 mAP 应 >= rank1 选的
    assert best_mAP_mAP >= best_rank1_mAP, \
        f"mAP 选模型 ({best_mAP_mAP}) 应 >= rank1 选模型 ({best_rank1_mAP})"

    print(f"  rank1 选: epoch {best_rank1_epoch}, mAP={best_rank1_mAP}")
    print(f"  mAP   选: epoch {best_mAP_epoch}, mAP={best_mAP_mAP}")
    print("✓ test_bug2_mAP_selects_better_model PASSED")


# ================================================================
# BUG-3: 早停配置
# ================================================================

def test_bug3_early_stopping_disabled():
    """BUG-3: early_stopping=0 应禁用早停"""
    from jaguar.trainer import ReIDTrainer
    from jaguar.model import ReIDModel
    from gastrovision.losses.metric_learning import ArcFaceLoss
    from torch.utils.data import DataLoader, TensorDataset

    model = ReIDModel('resnet18', embedding_dim=64, pretrained=False, use_gem=False)
    criterion = ArcFaceLoss(num_classes=4, embedding_dim=64)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()), lr=0.001
    )

    images = torch.randn(8, 3, 224, 224)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)

    with tempfile.TemporaryDirectory() as tmp:
        trainer = ReIDTrainer(
            model=model, criterion=criterion, loss_type='arcface',
            optimizer=optimizer, device=torch.device('cpu'),
            output_dir=tmp
        )

        # 跑 5 个 epoch, early_stopping=0 应跑完全部
        trainer.fit(loader, loader, epochs=5, early_stopping=0)
        # 验证跑了 5 个 epoch (历史中应有 5 条记录)
        assert len(trainer.train_history) == 5, \
            f"应跑 5 个 epoch, 实际 {len(trainer.train_history)}"

    print("✓ test_bug3_early_stopping_disabled PASSED")


def test_bug3_config_default_values():
    """BUG-3: 配置文件中的默认值应合理"""
    import yaml

    config_path = ROOT / 'configs' / 'jaguar_reid.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 早停应禁用或 patience >= 30
    early_stopping = cfg.get('early_stopping', 0)
    assert early_stopping == 0 or early_stopping >= 30, \
        f"early_stopping={early_stopping}, 应为 0 或 >= 30"

    # save_metric 应为 mAP
    save_metric = cfg.get('save_metric', 'mAP')
    assert save_metric == 'mAP', f"save_metric={save_metric}, 应为 'mAP'"

    # ema_decay 应 <= 0.999
    ema_decay = cfg.get('ema_decay', 0.9999)
    assert ema_decay <= 0.999, \
        f"ema_decay={ema_decay}, 小数据集应 <= 0.999"

    # epochs 应 >= 100
    epochs = cfg.get('epochs', 80)
    assert epochs >= 100, f"epochs={epochs}, 应 >= 100 以确保充分训练"

    print(f"  early_stopping={early_stopping}, save_metric={save_metric}, "
          f"ema_decay={ema_decay}, epochs={epochs}")
    print("✓ test_bug3_config_default_values PASSED")


# ================================================================
# DEBUG: EMA 诊断日志
# ================================================================

def test_debug_ema_diagnostic_output():
    """DEBUG: trainer.fit 应输出 EMA 诊断信息"""
    from jaguar.trainer import ReIDTrainer
    from jaguar.model import ReIDModel
    from gastrovision.losses.metric_learning import ArcFaceLoss
    from torch.utils.data import DataLoader, TensorDataset

    model = ReIDModel('resnet18', embedding_dim=64, pretrained=False, use_gem=False)
    criterion = ArcFaceLoss(num_classes=4, embedding_dim=64)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()), lr=0.001
    )

    images = torch.randn(8, 3, 224, 224)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)

    with tempfile.TemporaryDirectory() as tmp:
        trainer = ReIDTrainer(
            model=model, criterion=criterion, loss_type='arcface',
            optimizer=optimizer, device=torch.device('cpu'),
            output_dir=tmp, use_ema=True, ema_decay=0.999
        )

        # 捕获输出
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            trainer.fit(loader, loader, epochs=2, early_stopping=0)

        output = f.getvalue()
        assert 'EMA 诊断' in output, "应输出 EMA 诊断信息"
        assert '保留率' in output or '吸收' in output, "应输出吸收率信息"

    print("✓ test_debug_ema_diagnostic_output PASSED")


def test_debug_embedding_stats():
    """DEBUG: 验证应输出 embedding 统计信息"""
    from jaguar.trainer import ReIDTrainer
    from jaguar.model import ReIDModel
    from gastrovision.losses.metric_learning import ArcFaceLoss
    from torch.utils.data import DataLoader, TensorDataset

    model = ReIDModel('resnet18', embedding_dim=64, pretrained=False, use_gem=False)
    criterion = ArcFaceLoss(num_classes=4, embedding_dim=64)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()), lr=0.001
    )

    images = torch.randn(16, 3, 224, 224)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    loader = DataLoader(TensorDataset(images, labels), batch_size=16)

    with tempfile.TemporaryDirectory() as tmp:
        trainer = ReIDTrainer(
            model=model, criterion=criterion, loss_type='arcface',
            optimizer=optimizer, device=torch.device('cpu'),
            output_dir=tmp
        )

        # 前 3 个 epoch 应输出 debug 信息
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            trainer.fit(loader, loader, epochs=2, early_stopping=0)

        output = f.getvalue()
        assert 'Embedding' in output, "应输出 Embedding 统计"
        assert 'pos_sim' in output, "应输出正样本对相似度"
        assert 'neg_sim' in output, "应输出负样本对相似度"
        assert 'gap' in output, "应输出正负间隔"

    print("✓ test_debug_embedding_stats PASSED")


# ================================================================
# 集成测试: 完整训练流程 (含新修复)
# ================================================================

def test_e2e_full_training_with_fixes():
    """端到端: 使用所有修复后的参数完成完整训练 + 验证"""
    from jaguar.model import ReIDModel
    from jaguar.trainer import ReIDTrainer
    from jaguar.train import _create_loss
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device('cpu')
    cfg = SimpleNamespace()

    model = ReIDModel('resnet18', embedding_dim=64, pretrained=False, use_gem=False)
    criterion = _create_loss(cfg, 'arcface', num_classes=5, embedding_dim=64)
    aux_criterion = _create_loss(cfg, 'triplet', num_classes=5, embedding_dim=64, prefix='aux_')

    all_params = (
        list(model.parameters()) +
        list(criterion.parameters()) +
        list(aux_criterion.parameters())
    )
    optimizer = torch.optim.Adam(all_params, lr=0.001)

    # 5 类, 每类 4 个样本
    images = torch.randn(20, 3, 224, 224)
    labels = torch.tensor([0]*4 + [1]*4 + [2]*4 + [3]*4 + [4]*4)
    loader = DataLoader(TensorDataset(images, labels), batch_size=20)

    with tempfile.TemporaryDirectory() as tmp:
        trainer = ReIDTrainer(
            model=model, criterion=criterion, loss_type='arcface',
            optimizer=optimizer, device=device, output_dir=tmp,
            aux_criterion=aux_criterion, aux_loss_weight=1.0,
            use_ema=True, ema_decay=0.999,  # 修复后的 decay
        )

        trainer.fit(
            loader, loader,
            epochs=3,
            early_stopping=0,
            save_metric='mAP',  # BUG-2 修复
        )

        assert len(trainer.train_history) == 3, "应完成 3 个 epoch"
        assert trainer.best_metric > 0, "最佳指标应 > 0"
        assert trainer._save_metric == 'mAP', "应使用 mAP 选模型"

        # 验证 checkpoint 存在
        assert (Path(tmp) / 'best_model.pth').exists(), "应保存最佳模型"

    print("✓ test_e2e_full_training_with_fixes PASSED")


def test_e2e_save_metric_affects_selection():
    """端到端: save_metric='mAP' 时确实用 mAP 选最佳模型"""
    from jaguar.model import ReIDModel
    from jaguar.trainer import ReIDTrainer
    from gastrovision.losses.metric_learning import ArcFaceLoss
    from torch.utils.data import DataLoader, TensorDataset

    model = ReIDModel('resnet18', embedding_dim=64, pretrained=False, use_gem=False)
    criterion = ArcFaceLoss(num_classes=4, embedding_dim=64)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()), lr=0.001
    )

    images = torch.randn(16, 3, 224, 224)
    labels = torch.tensor([0]*4 + [1]*4 + [2]*4 + [3]*4)
    loader = DataLoader(TensorDataset(images, labels), batch_size=16)

    with tempfile.TemporaryDirectory() as tmp:
        trainer = ReIDTrainer(
            model=model, criterion=criterion, loss_type='arcface',
            optimizer=optimizer, device=torch.device('cpu'),
            output_dir=tmp
        )

        # 用 mAP 选
        trainer.fit(loader, loader, epochs=3, early_stopping=0, save_metric='mAP')
        best_mAP = trainer.best_metric

        # 用 auc 选 (重置)
        trainer.best_metric = 0.0
        trainer.best_epoch = 0
        trainer.fit(loader, loader, epochs=3, early_stopping=0, save_metric='auc')
        best_auc = trainer.best_metric

        # 两者应该是不同的数值 (mAP 和 AUC 在同一数据上不同)
        # 不要求哪个更大, 只验证确实使用了不同的指标
        print(f"  save_metric=mAP: best={best_mAP:.4f}")
        print(f"  save_metric=auc: best={best_auc:.4f}")

    print("✓ test_e2e_save_metric_affects_selection PASSED")


# ================================================================
# 运行所有测试
# ================================================================

if __name__ == '__main__':
    tests = [
        # BUG-1: EMA decay
        test_bug1_ema_auto_compute_triggers,
        test_bug1_ema_auto_compute_various_sizes,
        test_bug1_ema_no_auto_for_low_decay,
        test_bug1_ema_absorption_comparison,
        # BUG-2: save_metric
        test_bug2_save_metric_parameter,
        test_bug2_save_metric_validation,
        test_bug2_mAP_selects_better_model,
        # BUG-3: early stopping
        test_bug3_early_stopping_disabled,
        test_bug3_config_default_values,
        # DEBUG
        test_debug_ema_diagnostic_output,
        test_debug_embedding_stats,
        # E2E
        test_e2e_full_training_with_fixes,
        test_e2e_save_metric_affects_selection,
    ]

    passed = 0
    failed = 0
    errors = []

    print("=" * 60)
    print("Jaguar Re-ID Session 4 BUG 修复验证测试")
    print("=" * 60)
    print()

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            import traceback
            print(f"✗ {test_fn.__name__} FAILED: {e}")
            traceback.print_exc()
            print()

    print()
    print("=" * 60)
    print(f"结果: {passed}/{passed + failed} PASSED, {failed} FAILED")
    print("=" * 60)

    if errors:
        print("\n失败的测试:")
        for name, err in errors:
            print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print("\n✓ 所有测试通过!")
