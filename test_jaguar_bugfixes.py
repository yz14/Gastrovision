"""
Jaguar Re-ID BUG 修复验证测试

BUG-1: Missing imports in train.py (extract_embeddings, predict_similarity_fast)
BUG-2: _create_loss 全局 scale/margin 覆盖 per-loss-type 默认值
BUG-3: Pair-based 损失缺少 PK Sampler
BUG-4: 非 ResNet backbone 的 GeM pool params 被加入 optimizer
BUG-5: 验证指标缺少 coverage 信息

测试环境: conda activate torch27_env
"""

import sys
import os
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from types import SimpleNamespace
from collections import Counter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ================================================================
# BUG-1: Missing imports
# ================================================================

def test_bug1_imports_exist():
    """BUG-1: train.py 应正确导入 extract_embeddings 和 predict_similarity_fast"""
    from jaguar.train import extract_embeddings, predict_similarity_fast
    assert callable(extract_embeddings), "extract_embeddings 应可调用"
    assert callable(predict_similarity_fast), "predict_similarity_fast 应可调用"
    print("✓ test_bug1_imports_exist PASSED")


def test_bug1_inference_functions_accessible():
    """BUG-1: 确认 _run_inference 中使用的函数可正常访问"""
    import jaguar.train as train_module
    assert hasattr(train_module, 'extract_embeddings')
    assert hasattr(train_module, 'predict_similarity_fast')
    print("✓ test_bug1_inference_functions_accessible PASSED")


# ================================================================
# BUG-2: Per-loss-type default parameters
# ================================================================

def test_bug2_loss_defaults_table():
    """BUG-2: LOSS_DEFAULTS 应包含所有 10 种损失的合理默认参数"""
    from jaguar.train import LOSS_DEFAULTS

    expected_losses = [
        'arcface', 'cosface', 'sphereface', 'proxy_nca', 'circle_cls',
        'triplet', 'contrastive', 'circle', 'lifted', 'npair'
    ]
    for lt in expected_losses:
        assert lt in LOSS_DEFAULTS, f"LOSS_DEFAULTS 缺少 {lt}"
        assert 'scale' in LOSS_DEFAULTS[lt], f"{lt} 缺少 scale"
        assert 'margin' in LOSS_DEFAULTS[lt], f"{lt} 缺少 margin"

    # 验证关键默认值
    assert LOSS_DEFAULTS['arcface']['scale'] == 30.0
    assert LOSS_DEFAULTS['arcface']['margin'] == 0.5
    assert LOSS_DEFAULTS['circle']['scale'] == 256.0
    assert LOSS_DEFAULTS['circle']['margin'] == 0.25
    assert LOSS_DEFAULTS['proxy_nca']['scale'] == 8.0
    assert LOSS_DEFAULTS['triplet']['margin'] == 0.3
    assert LOSS_DEFAULTS['sphereface']['margin'] == 1.5  # m1=1.5 (modern practical default)

    print("✓ test_bug2_loss_defaults_table PASSED")


def test_bug2_create_loss_uses_defaults():
    """BUG-2: _create_loss 在无 loss-type-specific 配置时应使用 LOSS_DEFAULTS"""
    from jaguar.train import _create_loss

    # 空配置 — 无任何 loss 参数
    cfg = SimpleNamespace()

    # ArcFace: 应得到 scale=30.0, margin=0.5
    loss = _create_loss(cfg, 'arcface', num_classes=10, embedding_dim=128)
    assert loss.scale == 30.0, f"ArcFace scale={loss.scale}, expected 30.0"
    assert loss.margin == 0.5, f"ArcFace margin={loss.margin}, expected 0.5"

    # CircleLoss (pair-level): 应得到 scale=256.0, margin=0.25
    loss_c = _create_loss(cfg, 'circle', num_classes=0, embedding_dim=0)
    assert loss_c.scale == 256.0, f"Circle scale={loss_c.scale}, expected 256.0"
    assert loss_c.margin == 0.25, f"Circle margin={loss_c.margin}, expected 0.25"

    # ProxyNCA: 应得到 scale=8.0
    loss_p = _create_loss(cfg, 'proxy_nca', num_classes=10, embedding_dim=128)
    assert loss_p.scale == 8.0, f"ProxyNCA scale={loss_p.scale}, expected 8.0"

    # Triplet: 应得到 margin=0.3
    loss_t = _create_loss(cfg, 'triplet', num_classes=0, embedding_dim=0)
    assert loss_t.margin == 0.3, f"Triplet margin={loss_t.margin}, expected 0.3"

    print("✓ test_bug2_create_loss_uses_defaults PASSED")


def test_bug2_create_loss_override():
    """BUG-2: 用户 loss-type-specific 配置应覆盖 LOSS_DEFAULTS"""
    from jaguar.train import _create_loss

    # 用户显式设置 arcface_scale=64.0, arcface_margin=0.3
    cfg = SimpleNamespace(arcface_scale=64.0, arcface_margin=0.3)
    loss = _create_loss(cfg, 'arcface', num_classes=10, embedding_dim=128)
    assert loss.scale == 64.0, f"ArcFace scale={loss.scale}, expected 64.0"
    assert loss.margin == 0.3, f"ArcFace margin={loss.margin}, expected 0.3"

    # 用户设置 circle_scale=128.0 但不设置 circle_margin → margin 应回退到 LOSS_DEFAULTS
    cfg2 = SimpleNamespace(circle_scale=128.0)
    loss2 = _create_loss(cfg2, 'circle', num_classes=0, embedding_dim=0)
    assert loss2.scale == 128.0, f"Circle scale={loss2.scale}, expected 128.0"
    assert loss2.margin == 0.25, f"Circle margin={loss2.margin}, expected 0.25"

    print("✓ test_bug2_create_loss_override PASSED")


def test_bug2_aux_loss_prefix():
    """BUG-2: 辅助损失使用 prefix='aux_' 读取 loss-type-specific 配置"""
    from jaguar.train import _create_loss

    # aux_triplet_margin=0.5 应覆盖 triplet 默认 margin=0.3
    cfg = SimpleNamespace(aux_triplet_margin=0.5)
    loss = _create_loss(cfg, 'triplet', num_classes=0, embedding_dim=0, prefix='aux_')
    assert loss.margin == 0.5, f"Aux triplet margin={loss.margin}, expected 0.5"

    # 无 aux_ 前缀参数时，应使用 LOSS_DEFAULTS
    cfg2 = SimpleNamespace()
    loss2 = _create_loss(cfg2, 'triplet', num_classes=0, embedding_dim=0, prefix='aux_')
    assert loss2.margin == 0.3, f"Aux triplet margin={loss2.margin}, expected 0.3"

    print("✓ test_bug2_aux_loss_prefix PASSED")


def test_bug2_no_cross_contamination():
    """BUG-2: 切换 loss_type 时，旧 loss 的 config 不应影响新 loss"""
    from jaguar.train import _create_loss

    # 模拟：YAML 中有 arcface_scale=30.0（旧配置），现在切换到 circle
    cfg = SimpleNamespace(arcface_scale=30.0, arcface_margin=0.5)

    # Circle 不应受 arcface_scale 影响
    loss = _create_loss(cfg, 'circle', num_classes=0, embedding_dim=0)
    assert loss.scale == 256.0, f"Circle scale={loss.scale}, expected 256.0 (not contaminated by arcface)"
    assert loss.margin == 0.25, f"Circle margin={loss.margin}, expected 0.25"

    print("✓ test_bug2_no_cross_contamination PASSED")


def test_bug2_all_losses_create_successfully():
    """BUG-2: 所有 10 种损失都能用默认参数正确创建"""
    from jaguar.train import _create_loss, LOSS_DEFAULTS

    cfg = SimpleNamespace()
    num_classes = 10
    embedding_dim = 128

    for lt in LOSS_DEFAULTS:
        loss = _create_loss(cfg, lt, num_classes=num_classes, embedding_dim=embedding_dim)
        assert loss is not None, f"Failed to create {lt}"

        # 验证损失可前向传播
        features = torch.randn(8, embedding_dim)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss_val = loss(features, labels)
        assert loss_val.dim() == 0, f"{lt}: loss should be scalar"
        assert torch.isfinite(loss_val), f"{lt}: loss={loss_val.item()} is not finite"

    print("✓ test_bug2_all_losses_create_successfully PASSED")


# ================================================================
# BUG-3: PK Sampler
# ================================================================

def test_bug3_pk_sampler_basic():
    """BUG-3: PKSampler 基本功能"""
    from jaguar.sampler import PKSampler

    # 5 类，每类 10 个样本
    labels = []
    for c in range(5):
        labels.extend([c] * 10)

    sampler = PKSampler(labels, p=4, k=3, seed=42)

    # 每 batch 应有 p*k = 12 个样本
    indices = list(sampler)
    assert len(indices) > 0, "Sampler 应产生索引"
    assert len(indices) == len(sampler), f"长度不一致: {len(indices)} vs {len(sampler)}"

    # 验证每个 batch 有正确的类别结构
    batch_size = 4 * 3  # p * k
    for batch_start in range(0, len(indices), batch_size):
        batch = indices[batch_start:batch_start + batch_size]
        batch_labels = [labels[i] for i in batch]
        label_counts = Counter(batch_labels)
        # 应有 4 个不同类别
        assert len(label_counts) == 4, \
            f"Batch 应有 4 类，实际 {len(label_counts)}: {label_counts}"
        # 每个类别应有 3 个样本
        for label, count in label_counts.items():
            assert count == 3, f"类 {label} 应有 3 个样本，实际 {count}"

    print("✓ test_bug3_pk_sampler_basic PASSED")


def test_bug3_pk_sampler_small_classes():
    """BUG-3: PKSampler 处理小类别 (样本数 < K)"""
    from jaguar.sampler import PKSampler

    # 3 类有很多样本，2 类只有 2 个样本
    labels = [0]*20 + [1]*20 + [2]*20 + [3]*2 + [4]*2

    sampler = PKSampler(labels, p=4, k=4, seed=42)
    indices = list(sampler)

    # 验证所有索引都有效
    for idx in indices:
        assert 0 <= idx < len(labels), f"索引越界: {idx}"

    # 验证小类别通过有放回采样补齐到 K 个
    batch_size = 4 * 4
    for batch_start in range(0, len(indices), batch_size):
        batch = indices[batch_start:batch_start + batch_size]
        batch_labels = [labels[i] for i in batch]
        label_counts = Counter(batch_labels)
        for label, count in label_counts.items():
            assert count == 4, f"类 {label} 应有 4 个样本 (可能有重复)，实际 {count}"

    print("✓ test_bug3_pk_sampler_small_classes PASSED")


def test_bug3_pk_sampler_filters_singletons():
    """BUG-3: PKSampler 应过滤掉只有 1 个样本的类别"""
    from jaguar.sampler import PKSampler

    # 3 类有多个样本，2 类只有 1 个样本
    labels = [0]*10 + [1]*10 + [2]*10 + [3]*1 + [4]*1

    sampler = PKSampler(labels, p=3, k=4, seed=42)
    indices = list(sampler)

    # 验证 singleton 类别 (3, 4) 永远不会出现
    for idx in indices:
        assert labels[idx] not in [3, 4], \
            f"Singleton 类 {labels[idx]} 不应被采样 (idx={idx})"

    print("✓ test_bug3_pk_sampler_filters_singletons PASSED")


def test_bug3_pk_sampler_auto_reduce_p():
    """BUG-3: 有效类别不足 P 时应自动降低 P"""
    from jaguar.sampler import PKSampler

    # 只有 3 个有效类 (>=2 样本)，但 P=8
    labels = [0]*5 + [1]*5 + [2]*5 + [3]*1

    sampler = PKSampler(labels, p=8, k=4, seed=42)
    assert sampler.p == 3, f"P 应自动降至 3，实际 {sampler.p}"

    indices = list(sampler)
    assert len(indices) > 0

    print("✓ test_bug3_pk_sampler_auto_reduce_p PASSED")


def test_bug3_pair_loss_uses_pk_sampler():
    """BUG-3: pair-based loss 在 train.py 中应使用 PK sampler"""
    from jaguar.trainer import is_proxy_loss

    pair_losses = ['triplet', 'contrastive', 'circle', 'lifted', 'npair']
    proxy_losses = ['arcface', 'cosface', 'sphereface', 'proxy_nca', 'circle_cls']

    for lt in pair_losses:
        assert not is_proxy_loss(lt), f"{lt} 应为 pair-based"
        # In train.py: use_pk_sampler = not is_proxy_loss(loss_type)
        # So pair-based losses will get PK sampler

    for lt in proxy_losses:
        assert is_proxy_loss(lt), f"{lt} 应为 proxy-based"

    print("✓ test_bug3_pair_loss_uses_pk_sampler PASSED")


# ================================================================
# BUG-4: GeM pool params in optimizer
# ================================================================

def test_bug4_gem_not_in_optimizer_for_convnext():
    """BUG-4: ConvNeXt 不使用 GeM pool → 其参数不应在 optimizer 中"""
    from jaguar.model import ReIDModel

    model = ReIDModel(
        backbone_name='convnext_tiny',
        embedding_dim=128,
        pretrained=False,
        use_gem=True,  # GeM 被创建但不应被 ConvNeXt 使用
    )

    assert not model._needs_pool, "ConvNeXt 不应需要手动 pool"

    # 模拟 train.py 中的参数收集逻辑
    head_params = (
        list(model.embedding.parameters()) +
        list(model.bn_neck.parameters())
    )
    # BUG-4 修复后: 只有 _needs_pool=True 时才添加 pool 参数
    if model._needs_pool and hasattr(model, 'pool') and hasattr(model.pool, 'parameters'):
        head_params += list(model.pool.parameters())

    # GeM pool 的 p 参数不应在 head_params 中
    pool_param_ids = {id(p) for p in model.pool.parameters()}
    head_param_ids = {id(p) for p in head_params}
    assert pool_param_ids.isdisjoint(head_param_ids), \
        "ConvNeXt 的 GeM pool 参数不应在 optimizer 参数中"

    print("✓ test_bug4_gem_not_in_optimizer_for_convnext PASSED")


def test_bug4_gem_in_optimizer_for_resnet():
    """BUG-4: ResNet 使用 GeM pool → 其参数应在 optimizer 中"""
    from jaguar.model import ReIDModel

    model = ReIDModel(
        backbone_name='resnet18',
        embedding_dim=128,
        pretrained=False,
        use_gem=True,
    )

    assert model._needs_pool, "ResNet 应需要手动 pool"

    head_params = (
        list(model.embedding.parameters()) +
        list(model.bn_neck.parameters())
    )
    if model._needs_pool and hasattr(model, 'pool') and hasattr(model.pool, 'parameters'):
        head_params += list(model.pool.parameters())

    pool_param_ids = {id(p) for p in model.pool.parameters()}
    head_param_ids = {id(p) for p in head_params}
    assert pool_param_ids.issubset(head_param_ids), \
        "ResNet 的 GeM pool 参数应在 optimizer 参数中"

    print("✓ test_bug4_gem_in_optimizer_for_resnet PASSED")


# ================================================================
# BUG-5: Validation coverage info
# ================================================================

def test_bug5_validate_returns_coverage():
    """BUG-5: _validate 应返回 valid_queries 和 total"""
    from jaguar.model import ReIDModel
    from jaguar.trainer import ReIDTrainer
    from jaguar.dataset import JaguarTrainDataset
    from jaguar.transforms import get_val_transforms
    from jaguar.split import build_label_map
    from gastrovision.losses.metric_learning import ArcFaceLoss
    from torch.utils.data import DataLoader
    from PIL import Image

    device = torch.device('cpu')

    with tempfile.TemporaryDirectory() as tmp:
        # 创建虚拟数据
        img_dir = Path(tmp) / 'imgs'
        img_dir.mkdir()
        rows = []
        for i in range(20):
            fname = f'img_{i:03d}.png'
            label = f'cls_{i % 4}'
            rows.append({'filename': fname, 'ground_truth': label})
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(img_dir / fname)

        df = pd.DataFrame(rows)
        label_map = build_label_map(df)
        val_t = get_val_transforms(image_size=224)
        ds = JaguarTrainDataset(df, str(img_dir), label_map, val_t)
        loader = DataLoader(ds, batch_size=8, shuffle=False)

        model = ReIDModel('resnet18', embedding_dim=128, pretrained=False, use_gem=False)
        arcface = ArcFaceLoss(num_classes=4, embedding_dim=128)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(arcface.parameters()), lr=0.001
        )

        trainer = ReIDTrainer(
            model=model, criterion=arcface, loss_type='arcface',
            optimizer=optimizer, device=device, output_dir=tmp
        )

        metrics = trainer._validate(loader)

        # 验证返回 coverage 信息
        assert 'valid_queries' in metrics, "缺少 valid_queries"
        assert 'total' in metrics, "缺少 total"
        assert metrics['total'] == 20, f"total={metrics['total']}, expected 20"
        assert metrics['valid_queries'] > 0, "valid_queries 应 > 0"
        assert metrics['valid_queries'] <= metrics['total']

    print("✓ test_bug5_validate_returns_coverage PASSED")


# ================================================================
# 端到端集成测试：完整损失切换流程
# ================================================================

def test_e2e_loss_switching_with_defaults():
    """端到端: 所有损失类型用默认参数完成单步训练 + 验证"""
    from jaguar.model import ReIDModel
    from jaguar.trainer import ReIDTrainer
    from jaguar.train import _create_loss, LOSS_DEFAULTS
    from jaguar.dataset import JaguarTrainDataset
    from jaguar.transforms import get_train_transforms, get_val_transforms
    from jaguar.split import build_label_map
    from torch.utils.data import DataLoader
    from PIL import Image

    device = torch.device('cpu')
    num_classes = 5
    embedding_dim = 128

    with tempfile.TemporaryDirectory() as tmp:
        # 创建虚拟数据 — 确保每类至少 4 个样本
        img_dir = Path(tmp) / 'imgs'
        img_dir.mkdir()
        rows = []
        for i in range(40):
            fname = f'img_{i:03d}.png'
            label = f'cls_{i % num_classes}'
            rows.append({'filename': fname, 'ground_truth': label})
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(img_dir / fname)

        df = pd.DataFrame(rows)
        label_map = build_label_map(df)
        val_t = get_val_transforms(image_size=224)
        val_ds = JaguarTrainDataset(df, str(img_dir), label_map, val_t)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

        # 空配置，只依赖 LOSS_DEFAULTS
        cfg = SimpleNamespace()

        # 确保每个 batch 有足够正样本对
        images = torch.randn(8, 3, 224, 224)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

        for lt in LOSS_DEFAULTS:
            model = ReIDModel('resnet18', embedding_dim=embedding_dim,
                              pretrained=False, use_gem=False)
            criterion = _create_loss(cfg, lt, num_classes=num_classes,
                                     embedding_dim=embedding_dim)

            all_params = list(model.parameters()) + list(criterion.parameters())
            optimizer = torch.optim.Adam(all_params, lr=0.001)

            out_dir = Path(tmp) / f'out_{lt}'
            out_dir.mkdir()

            trainer = ReIDTrainer(
                model=model, criterion=criterion, loss_type=lt,
                optimizer=optimizer, device=device, output_dir=str(out_dir)
            )

            # 训练一步
            train_metrics = trainer._train_epoch(
                DataLoader(
                    torch.utils.data.TensorDataset(images, labels),
                    batch_size=8
                ),
                epoch=0, total_epochs=1
            )
            assert train_metrics['loss'] > 0, f"{lt}: loss should > 0"

            # 验证
            val_metrics = trainer._validate(val_loader)
            assert 0 <= val_metrics['rank1'] <= 1.0, f"{lt}: rank1 out of range"
            assert 0 <= val_metrics['mAP'] <= 1.0, f"{lt}: mAP out of range"
            assert val_metrics['valid_queries'] > 0, f"{lt}: no valid queries"

            print(f"  ✓ {lt}: loss={train_metrics['loss']:.4f}, "
                  f"rank1={val_metrics['rank1']:.4f}, mAP={val_metrics['mAP']:.4f}")

    print("✓ test_e2e_loss_switching_with_defaults PASSED")


def test_e2e_pk_sampler_with_dataloader():
    """端到端: PK sampler 与 DataLoader 正确集成"""
    from jaguar.sampler import PKSampler
    from torch.utils.data import DataLoader, TensorDataset

    # 5 类，每类 8 个样本
    all_labels = []
    for c in range(5):
        all_labels.extend([c] * 8)
    n = len(all_labels)

    images = torch.randn(n, 3, 32, 32)
    labels_tensor = torch.tensor(all_labels)
    dataset = TensorDataset(images, labels_tensor)

    sampler = PKSampler(all_labels, p=4, k=3, seed=42)
    batch_size = 4 * 3  # p * k

    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    num_batches = 0
    for batch_imgs, batch_labels in loader:
        assert batch_imgs.shape[0] == batch_size, \
            f"Batch size={batch_imgs.shape[0]}, expected {batch_size}"
        # 验证有 4 个不同类别
        unique_labels = batch_labels.unique()
        assert len(unique_labels) == 4, \
            f"Batch 应有 4 个不同类别，实际 {len(unique_labels)}: {unique_labels.tolist()}"
        num_batches += 1

    assert num_batches > 0, "DataLoader 应产生至少 1 个 batch"

    print("✓ test_e2e_pk_sampler_with_dataloader PASSED")


def test_e2e_aux_loss_with_defaults():
    """端到端: 辅助损失 (ArcFace + Triplet) 使用默认参数"""
    from jaguar.model import ReIDModel
    from jaguar.trainer import ReIDTrainer
    from jaguar.train import _create_loss

    device = torch.device('cpu')
    cfg = SimpleNamespace()

    model = ReIDModel('resnet18', embedding_dim=128, pretrained=False, use_gem=False)
    criterion = _create_loss(cfg, 'arcface', num_classes=5, embedding_dim=128)
    aux_criterion = _create_loss(cfg, 'triplet', num_classes=5, embedding_dim=128, prefix='aux_')

    # Triplet 应有默认 margin=0.3
    assert aux_criterion.margin == 0.3, f"Aux triplet margin={aux_criterion.margin}"

    all_params = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.Adam(all_params, lr=0.001)

    images = torch.randn(8, 3, 224, 224)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    with tempfile.TemporaryDirectory() as tmp:
        trainer = ReIDTrainer(
            model=model, criterion=criterion, loss_type='arcface',
            optimizer=optimizer, device=device, output_dir=tmp,
            aux_criterion=aux_criterion, aux_loss_weight=0.5
        )

        metrics = trainer._train_epoch(
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(images, labels), batch_size=8
            ),
            epoch=0, total_epochs=1
        )

        assert metrics['loss'] > 0
        assert 'aux_loss' in metrics
        assert metrics['aux_loss'] > 0

    print("✓ test_e2e_aux_loss_with_defaults PASSED")


# ================================================================
# 运行所有测试
# ================================================================

if __name__ == '__main__':
    tests = [
        # BUG-1
        test_bug1_imports_exist,
        test_bug1_inference_functions_accessible,
        # BUG-2
        test_bug2_loss_defaults_table,
        test_bug2_create_loss_uses_defaults,
        test_bug2_create_loss_override,
        test_bug2_aux_loss_prefix,
        test_bug2_no_cross_contamination,
        test_bug2_all_losses_create_successfully,
        # BUG-3
        test_bug3_pk_sampler_basic,
        test_bug3_pk_sampler_small_classes,
        test_bug3_pk_sampler_filters_singletons,
        test_bug3_pk_sampler_auto_reduce_p,
        test_bug3_pair_loss_uses_pk_sampler,
        # BUG-4
        test_bug4_gem_not_in_optimizer_for_convnext,
        test_bug4_gem_in_optimizer_for_resnet,
        # BUG-5
        test_bug5_validate_returns_coverage,
        # End-to-end
        test_e2e_loss_switching_with_defaults,
        test_e2e_pk_sampler_with_dataloader,
        test_e2e_aux_loss_with_defaults,
    ]

    passed = 0
    failed = 0
    errors = []

    print("=" * 60)
    print("Jaguar Re-ID BUG 修复验证测试")
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
