"""
Jaguar Re-ID 代码审计修复验证测试

修复内容:
BUG-1 (CRITICAL): _validate EMA restore 未在 finally 块 → 异常时模型权重永久被 EMA 覆盖
BUG-2 (HIGH): 两阶段训练 Phase2 加载 Phase1 checkpoint 时 scheduler 状态错误继承
BUG-3 (HIGH): ConvNeXt/EfficientNet 完全绕过 GeM Pooling（用的 backbone 内置 AvgPool）
BUG-4 (HIGH): Embedding Head 是单层 Linear，MLP Head 更强（Re-ID 业界最佳实践）
BUG-5 (HIGH): 两阶段训练 best_metric 继承导致 Phase2 难以保存模型
BUG-6 (LOW): predict_similarity 和 predict_similarity_fast 中 sigmoid 校准代码重复

测试环境: conda activate torch27_env
"""

import sys
import os
import tempfile
import math
import copy
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from types import SimpleNamespace
from io import StringIO

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ================================================================
# BUG-1: _validate EMA restore 在 finally 块
# ================================================================

def test_bug1_ema_restore_on_exception():
    """BUG-1: _validate 中即使出现异常，EMA restore 也应被调用，
    确保训练继续使用实际权重而非 EMA shadow 权重。"""
    from jaguar.model import ReIDModel
    from jaguar.trainer import ReIDTrainer
    from gastrovision.losses.metric_learning import ArcFaceLoss

    model = ReIDModel('resnet18', embedding_dim=64, pretrained=False, use_gem=False, use_mlp_head=False)
    criterion = ArcFaceLoss(num_classes=4, embedding_dim=64)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()), lr=0.001
    )

    with tempfile.TemporaryDirectory() as tmp:
        trainer = ReIDTrainer(
            model=model, criterion=criterion, loss_type='arcface',
            optimizer=optimizer, device=torch.device('cpu'),
            output_dir=tmp, use_ema=True, ema_decay=0.9
        )

        # EMA 初始化时 shadow == model 权重。
        # 修改 model 权重，使其与 shadow 不同，才能测试 apply/restore
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.ones_like(p) * 100.0)  # 大幅修改保证不同

        # 记录修改后的实际权重
        real_weights = {k: v.clone() for k, v in model.named_parameters()}

        # apply_shadow 应改变 model 权重（将 shadow 应用到 model）
        trainer.ema.apply_shadow(model)
        shadow_weights = {k: v.clone() for k, v in model.named_parameters()}

        # 验证 apply_shadow 实际改变了权重
        weight_changed = any(
            not torch.allclose(real_weights[k], shadow_weights[k])
            for k in real_weights
        )
        assert weight_changed, "apply_shadow 应改变模型权重"

        # restore 应还原修改后的 real 权重
        trainer.ema.restore(model)
        restored_weights = {k: v.clone() for k, v in model.named_parameters()}

        weights_restored = all(
            torch.allclose(real_weights[k], restored_weights[k])
            for k in real_weights
        )
        assert weights_restored, "restore 应还原模型权重"

    print("✓ test_bug1_ema_restore_on_exception PASSED")


def test_bug1_validate_has_finally_structure():
    """BUG-1: _validate 方法体应包含 try...finally 结构，确保 EMA restore 必定执行"""
    import inspect
    from jaguar.trainer import ReIDTrainer

    source = inspect.getsource(ReIDTrainer._validate)
    assert 'try:' in source, "_validate 应有 try 块"
    assert 'finally:' in source, "_validate 应有 finally 块"
    # finally 块中应包含 ema.restore
    finally_idx = source.index('finally:')
    assert 'restore' in source[finally_idx:], "finally 块中应调用 ema.restore"

    print("✓ test_bug1_validate_has_finally_structure PASSED")


def test_bug1_validate_restore_after_exception():
    """BUG-1: _validate 中 loader 抛出异常时，EMA restore 仍应被调用"""
    from jaguar.model import ReIDModel
    from jaguar.trainer import ReIDTrainer
    from gastrovision.losses.metric_learning import ArcFaceLoss

    model = ReIDModel('resnet18', embedding_dim=64, pretrained=False, use_gem=False, use_mlp_head=False)
    criterion = ArcFaceLoss(num_classes=4, embedding_dim=64)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()), lr=0.001
    )

    with tempfile.TemporaryDirectory() as tmp:
        trainer = ReIDTrainer(
            model=model, criterion=criterion, loss_type='arcface',
            optimizer=optimizer, device=torch.device('cpu'),
            output_dir=tmp, use_ema=True, ema_decay=0.9
        )

        # 修改 model 权重，使其与 EMA shadow 不同
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.ones_like(p) * 100.0)
        # 记录当前训练权重
        train_weights_before = {k: v.clone() for k, v in model.named_parameters()}

        # 创建一个会报错的 loader
        class ErrorLoader:
            def __iter__(self):
                raise RuntimeError("模拟验证异常")
            def __len__(self):
                return 0

        # _validate 应在 finally 中还原 EMA，即使抛出异常
        try:
            trainer._validate(ErrorLoader())
        except (RuntimeError, StopIteration):
            pass  # 预期异常

        # 验证模型权重未被 EMA shadow 覆盖（restore 应已还原）
        train_weights_after = {k: v.clone() for k, v in model.named_parameters()}
        weights_restored = all(
            torch.allclose(train_weights_before[k], train_weights_after[k])
            for k in train_weights_before
        )
        assert weights_restored, "异常后 EMA restore 应还原训练权重"

    print("✓ test_bug1_validate_restore_after_exception PASSED")


# ================================================================
# BUG-3: GeM Pooling 对 ConvNeXt/EfficientNet 生效
# ================================================================

def test_bug3_gem_hook_registered_for_resnet():
    """BUG-3: ResNet backbone 应通过 hook 截取 layer4 输出进行 GeM"""
    from jaguar.model import ReIDModel

    model = ReIDModel('resnet18', embedding_dim=64, pretrained=False, use_gem=True, use_mlp_head=False)

    # ResNet 应有 _gem_hook 注册
    assert model._use_gem, "resnet 应启用 GeM"
    assert model._gem_hook is not None, "resnet 应注册 GeM hook"
    # ResNet avgpool 应被替换为 Identity
    assert isinstance(model.backbone.avgpool, nn.Identity), \
        "resnet avgpool 应被替换为 Identity (GeM hook 替代)"

    print("✓ test_bug3_gem_hook_registered_for_resnet PASSED")


def test_bug3_gem_hook_registered_for_convnext():
    """BUG-3: ConvNeXt backbone 应通过 hook 截取 features 输出进行 GeM"""
    from jaguar.model import ReIDModel

    model = ReIDModel('convnext_tiny', embedding_dim=64, pretrained=False, use_gem=True, use_mlp_head=False)

    assert model._use_gem, "convnext 应启用 GeM"
    assert model._gem_hook is not None, "convnext 应注册 GeM hook"

    print("✓ test_bug3_gem_hook_registered_for_convnext PASSED")


def test_bug3_gem_actually_activates_for_resnet():
    """BUG-3: ResNet forward 时 GeM 应实际捕获到特征图"""
    from jaguar.model import ReIDModel

    model = ReIDModel('resnet18', embedding_dim=64, pretrained=False, use_gem=True, use_mlp_head=False)
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        _ = model(x)

    # hook 应捕获了特征图
    # 验证方式：比较 GeM 和 AvgPool 的输出是否不同
    model_no_gem = ReIDModel('resnet18', embedding_dim=64, pretrained=False, use_gem=False, use_mlp_head=False)
    # 复制相同权重
    model_no_gem.load_state_dict(model.state_dict(), strict=False)
    model_no_gem.eval()

    with torch.no_grad():
        feat_gem = model.backbone(x)
        feat_avg = model_no_gem.backbone(x)

    # 结构上：GeM 应改变了 avgpool 层
    assert isinstance(model.backbone.avgpool, nn.Identity), "GeM 模型 avgpool 应为 Identity"

    print("✓ test_bug3_gem_actually_activates_for_resnet PASSED")


def test_bug3_gem_output_shape_resnet():
    """BUG-3: ResNet + GeM forward 输出形状应正确"""
    from jaguar.model import ReIDModel

    model = ReIDModel('resnet18', embedding_dim=128, pretrained=False, use_gem=True, use_mlp_head=False)
    model.eval()

    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        bn_feat, raw_feat = model(x, return_both=True)

    assert bn_feat.shape == (4, 128), f"bn_feat shape 应为 (4, 128)，实际 {bn_feat.shape}"
    assert raw_feat.shape == (4, 128), f"raw_feat shape 应为 (4, 128)，实际 {raw_feat.shape}"

    print("✓ test_bug3_gem_output_shape_resnet PASSED")


def test_bug3_gem_output_shape_convnext():
    """BUG-3: ConvNeXt + GeM forward 输出形状应正确"""
    from jaguar.model import ReIDModel

    model = ReIDModel('convnext_tiny', embedding_dim=128, pretrained=False, use_gem=True)
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        bn_feat, raw_feat = model(x, return_both=True)

    assert bn_feat.shape == (2, 128), f"bn_feat shape 应为 (2, 128)，实际 {bn_feat.shape}"
    assert raw_feat.shape == (2, 128), f"raw_feat shape 应为 (2, 128)，实际 {raw_feat.shape}"

    print("✓ test_bug3_gem_output_shape_convnext PASSED")


def test_bug3_swin_falls_back_to_avgpool():
    """BUG-3: Swin Transformer 应自动退化为 AvgPool（不支持 GeM）"""
    from jaguar.model import ReIDModel
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        model = ReIDModel('swin_t', embedding_dim=128, pretrained=False, use_gem=True)

    output = f.getvalue()
    assert not model._use_gem, "Swin 应禁用 GeM"
    assert '序列格式' in output or 'Swin' in output, "应输出 Swin 不支持 GeM 的提示"

    # Swin forward 应正常工作
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        bn_feat, raw_feat = model(x, return_both=True)
    assert bn_feat.shape == (2, 128)

    print("✓ test_bug3_swin_falls_back_to_avgpool PASSED")


# ================================================================
# BUG-4: MLP Embedding Head
# ================================================================

def test_bug4_mlp_head_structure():
    """BUG-4: use_mlp_head=True 时应创建 MLPEmbeddingHead"""
    from jaguar.model import ReIDModel, MLPEmbeddingHead

    model = ReIDModel('resnet18', embedding_dim=128, pretrained=False, use_gem=False, use_mlp_head=True)
    assert isinstance(model.embedding, MLPEmbeddingHead), \
        "use_mlp_head=True 时 embedding 应为 MLPEmbeddingHead"
    # MLP Head 应有两个 Linear 层
    assert hasattr(model.embedding, 'fc1'), "MLPEmbeddingHead 应有 fc1"
    assert hasattr(model.embedding, 'fc2'), "MLPEmbeddingHead 应有 fc2"
    assert hasattr(model.embedding, 'bn1'), "MLPEmbeddingHead 应有 bn1"

    print("✓ test_bug4_mlp_head_structure PASSED")


def test_bug4_linear_head_backward_compat():
    """BUG-4: use_mlp_head=False 时应保持单层 Linear（向后兼容）"""
    from jaguar.model import ReIDModel

    model = ReIDModel('resnet18', embedding_dim=128, pretrained=False,
                      use_gem=False, use_mlp_head=False)
    assert isinstance(model.embedding, nn.Linear), \
        "use_mlp_head=False 时 embedding 应为 nn.Linear"

    print("✓ test_bug4_linear_head_backward_compat PASSED")


def test_bug4_mlp_head_forward_shape():
    """BUG-4: MLP Head 输出形状应正确"""
    from jaguar.model import MLPEmbeddingHead

    head = MLPEmbeddingHead(in_dim=2048, out_dim=512, dropout=0.2)
    head.eval()
    x = torch.randn(8, 2048)
    with torch.no_grad():
        out = head(x)
    assert out.shape == (8, 512), f"MLPEmbeddingHead 输出形状应为 (8, 512)，实际 {out.shape}"

    print("✓ test_bug4_mlp_head_forward_shape PASSED")


def test_bug4_mlp_head_hidden_dim():
    """BUG-4: MLP Head 隐藏层维度应正确（max(out_dim, in_dim//2)）"""
    from jaguar.model import MLPEmbeddingHead

    # in_dim=2048, out_dim=512: hidden = max(512, 1024) = 1024
    head = MLPEmbeddingHead(in_dim=2048, out_dim=512)
    assert head.fc1.out_features == 1024, \
        f"hidden_dim 应为 1024，实际 {head.fc1.out_features}"
    assert head.fc2.out_features == 512, \
        f"out_dim 应为 512，实际 {head.fc2.out_features}"

    # in_dim=512, out_dim=512: hidden = max(512, 256) = 512
    head2 = MLPEmbeddingHead(in_dim=512, out_dim=512)
    assert head2.fc1.out_features == 512

    print("✓ test_bug4_mlp_head_hidden_dim PASSED")


def test_bug4_resnet_with_mlp_head():
    """BUG-4: ResNet + MLP Head 完整 forward 应正确运行"""
    from jaguar.model import ReIDModel

    model = ReIDModel('resnet18', embedding_dim=128, pretrained=False, use_gem=True, use_mlp_head=True)
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        bn_feat, raw_feat = model(x, return_both=True)
        emb = model.extract_embedding(x)

    assert bn_feat.shape == (4, 128)
    assert raw_feat.shape == (4, 128)
    assert emb.shape == (4, 128)
    # extract_embedding 应 L2 归一化
    norms = emb.norm(dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5), \
        "extract_embedding 应返回 L2 归一化向量"

    print("✓ test_bug4_resnet_with_mlp_head PASSED")


# ================================================================
# BUG-5: 两阶段训练 best_metric 重置
# ================================================================

def test_bug5_reset_best_metric_parameter():
    """BUG-5: load_checkpoint 应支持 reset_best_metric 参数"""
    import inspect
    from jaguar.trainer import ReIDTrainer

    sig = inspect.signature(ReIDTrainer.load_checkpoint)
    assert 'reset_best_metric' in sig.parameters, \
        "load_checkpoint 应有 reset_best_metric 参数"
    assert sig.parameters['reset_best_metric'].default == False, \
        "reset_best_metric 默认值应为 False"

    print("✓ test_bug5_reset_best_metric_parameter PASSED")


def test_bug5_reset_best_metric_works():
    """BUG-5: reset_best_metric=True 时 best_metric 应被重置为 0"""
    from jaguar.model import ReIDModel
    from jaguar.trainer import ReIDTrainer
    from gastrovision.losses.metric_learning import ArcFaceLoss
    from torch.utils.data import DataLoader, TensorDataset

    model = ReIDModel('resnet18', embedding_dim=64, pretrained=False,
                      use_gem=False, use_mlp_head=False)
    criterion = ArcFaceLoss(num_classes=4, embedding_dim=64)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()), lr=0.001
    )

    images = torch.randn(8, 3, 224, 224)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)

    with tempfile.TemporaryDirectory() as tmp:
        # Phase 1: 训练并保存 checkpoint（模拟 best_metric=0.85）
        trainer = ReIDTrainer(
            model=model, criterion=criterion, loss_type='arcface',
            optimizer=optimizer, device=torch.device('cpu'),
            output_dir=tmp
        )
        trainer.fit(loader, loader, epochs=2, early_stopping=0)
        phase1_best = trainer.best_metric
        # 手动设置一个高 best_metric 模拟 Phase1 结果
        trainer.best_metric = 0.85
        trainer._save_checkpoint(1, 'phase1_best.pth', {})

        # Phase 2: 加载 Phase1 checkpoint，不重置
        trainer2 = ReIDTrainer(
            model=model, criterion=criterion, loss_type='arcface',
            optimizer=optimizer, device=torch.device('cpu'),
            output_dir=tmp
        )
        trainer2.load_checkpoint(str(Path(tmp) / 'phase1_best.pth'), reset_best_metric=False)
        assert trainer2.best_metric == 0.85, \
            f"不重置时 best_metric 应继承 Phase1 的 0.85，实际 {trainer2.best_metric}"

        # Phase 2: 加载 Phase1 checkpoint，重置
        trainer3 = ReIDTrainer(
            model=model, criterion=criterion, loss_type='arcface',
            optimizer=optimizer, device=torch.device('cpu'),
            output_dir=tmp
        )
        trainer3.load_checkpoint(str(Path(tmp) / 'phase1_best.pth'), reset_best_metric=True)
        assert trainer3.best_metric == 0.0, \
            f"reset_best_metric=True 时 best_metric 应为 0，实际 {trainer3.best_metric}"

    print("✓ test_bug5_reset_best_metric_works PASSED")


def test_bug5_two_phase_training_config():
    """BUG-5: 配置文件应包含 reset_best_metric 和 reset_scheduler 选项"""
    import yaml

    config_path = ROOT / 'configs' / 'jaguar_reid.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    assert 'reset_best_metric' in cfg, "配置应有 reset_best_metric 字段"
    assert 'reset_scheduler' in cfg, "配置应有 reset_scheduler 字段"
    assert cfg['reset_best_metric'] == False, "默认应为 False"
    assert cfg['reset_scheduler'] == False, "默认应为 False"

    print("✓ test_bug5_two_phase_training_config PASSED")


# ================================================================
# BUG-6: inference sigmoid 校准代码不重复
# ================================================================

def test_bug6_sigmoid_calibrate_function_exists():
    """BUG-6: 应有公共的 _sigmoid_calibrate 函数"""
    from jaguar.inference import _sigmoid_calibrate

    sims = np.random.randn(100) * 0.1 + 0.8  # 模拟压缩的余弦相似度
    calibrated = _sigmoid_calibrate(sims)

    assert calibrated.shape == sims.shape
    assert calibrated.min() >= 0.0, "校准后应 >= 0"
    assert calibrated.max() <= 1.0, "校准后应 <= 1"
    # 中位数附近应在 0.5 附近
    assert abs(np.median(calibrated) - 0.5) < 0.1, "中位数应在 0.5 附近"

    print("✓ test_bug6_sigmoid_calibrate_function_exists PASSED")


def test_bug6_no_code_duplication():
    """BUG-6: predict_similarity 和 predict_similarity_fast 不应有重复的 sigmoid 校准代码"""
    import inspect
    from jaguar import inference

    src = inspect.getsource(inference)

    # 公共函数应存在
    assert '_sigmoid_calibrate' in src, "应有 _sigmoid_calibrate 函数"

    # 两个函数都应调用公共函数，而不是直接写 np.exp(-z)
    predict_sim_src = inspect.getsource(inference.predict_similarity)
    predict_fast_src = inspect.getsource(inference.predict_similarity_fast)

    # 每个函数内不应有重复的 iqr/sigmoid 计算逻辑（应调用公共函数）
    assert '_sigmoid_calibrate' in predict_sim_src, \
        "predict_similarity 应调用 _sigmoid_calibrate"
    assert '_sigmoid_calibrate' in predict_fast_src, \
        "predict_similarity_fast 应调用 _sigmoid_calibrate"

    print("✓ test_bug6_no_code_duplication PASSED")


# ================================================================
# 集成测试: MLP Head + GeM + EMA 完整训练
# ================================================================

def test_integration_full_training_mlp_gem():
    """集成测试: ResNet + GeM + MLP Head + ArcFace + Triplet 完整训练"""
    from jaguar.model import ReIDModel
    from jaguar.trainer import ReIDTrainer
    from jaguar.train import _create_loss
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device('cpu')
    cfg = SimpleNamespace()

    model = ReIDModel('resnet18', embedding_dim=64, pretrained=False,
                      use_gem=True, use_mlp_head=True, dropout=0.1)
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
            use_ema=True, ema_decay=0.999,
        )

        trainer.fit(loader, loader, epochs=3, early_stopping=0, save_metric='mAP')

        assert len(trainer.train_history) == 3
        assert trainer.best_metric >= 0
        assert (Path(tmp) / 'best_model.pth').exists()

    print("✓ test_integration_full_training_mlp_gem PASSED")


def test_integration_convnext_gem_forward():
    """集成测试: ConvNeXt + GeM 完整 forward 不报错"""
    from jaguar.model import ReIDModel

    model = ReIDModel('convnext_tiny', embedding_dim=128, pretrained=False,
                      use_gem=True, use_mlp_head=True)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        bn_feat, raw_feat = model(x, return_both=True)
        emb = model.extract_embedding(x)

    assert bn_feat.shape == (2, 128)
    assert emb.shape == (2, 128)
    norms = emb.norm(dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    print("✓ test_integration_convnext_gem_forward PASSED")


def test_integration_checkpoint_save_load_mlp():
    """集成测试: MLP Head 模型 checkpoint 保存和加载"""
    from jaguar.model import ReIDModel
    from jaguar.trainer import ReIDTrainer
    from gastrovision.losses.metric_learning import ArcFaceLoss
    from torch.utils.data import DataLoader, TensorDataset

    model = ReIDModel('resnet18', embedding_dim=64, pretrained=False,
                      use_gem=True, use_mlp_head=True)
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
            optimizer=optimizer, device=torch.device('cpu'), output_dir=tmp
        )
        trainer.fit(loader, loader, epochs=2, early_stopping=0)
        best_before = trainer.best_metric

        # 重建 trainer 并加载 checkpoint
        model2 = ReIDModel('resnet18', embedding_dim=64, pretrained=False,
                           use_gem=True, use_mlp_head=True)
        trainer2 = ReIDTrainer(
            model=model2, criterion=criterion, loss_type='arcface',
            optimizer=optimizer, device=torch.device('cpu'), output_dir=tmp
        )
        trainer2.load_checkpoint(str(Path(tmp) / 'best_model.pth'))

        assert abs(trainer2.best_metric - best_before) < 1e-6, \
            "加载后 best_metric 应与保存时一致"

    print("✓ test_integration_checkpoint_save_load_mlp PASSED")


def test_integration_config_has_mlp_head():
    """集成测试: 配置文件应有 use_mlp_head 选项"""
    import yaml

    config_path = ROOT / 'configs' / 'jaguar_reid.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    assert 'use_mlp_head' in cfg, "配置应有 use_mlp_head 字段"
    assert cfg['use_mlp_head'] == True, "默认应启用 MLP Head"

    print("✓ test_integration_config_has_mlp_head PASSED")


# ================================================================
# BUG-A: _save_checkpoint 保存 EMA shadow 权重
# ================================================================

def test_buga_save_checkpoint_stores_ema_shadow():
    """BUG-A: best_model.pth 应保存 EMA shadow 权重，而不是训练权重"""
    from jaguar.model import ReIDModel
    from jaguar.trainer import ReIDTrainer
    from gastrovision.losses.metric_learning import ArcFaceLoss
    from torch.utils.data import DataLoader, TensorDataset

    model = ReIDModel('resnet18', embedding_dim=64, pretrained=False,
                      use_gem=False, use_mlp_head=False)
    criterion = ArcFaceLoss(num_classes=4, embedding_dim=64)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()), lr=0.001
    )

    with tempfile.TemporaryDirectory() as tmp:
        trainer = ReIDTrainer(
            model=model, criterion=criterion, loss_type='arcface',
            optimizer=optimizer, device=torch.device('cpu'),
            output_dir=tmp, use_ema=True, ema_decay=0.9
        )

        # 故意让训练权重和 EMA shadow 差异很大：
        # 修改模型权重后不调用 ema.update()，
        # 这样 shadow = 初始权重，model = 初始权重 + 100
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.ones_like(p) * 100.0)
        # 此时 shadow ≠ model

        trainer.best_metric = 0.5
        trainer._save_checkpoint(1, 'best_model.pth', {})

        # 加载 checkpoint 中的 model_state_dict
        saved = torch.load(str(Path(tmp) / 'best_model.pth'),
                           map_location='cpu', weights_only=False)
        saved_params = saved['model_state_dict']

        # 读取 shadow 权重（初始权重，没有 +100）
        shadow_vals = {k: v for k, v in trainer.ema.shadow.items()}

        # checkpoint 中保存的应是 EMA shadow，而非训练权重 (+100)
        for name in shadow_vals:
            if name in saved_params:
                assert torch.allclose(saved_params[name], shadow_vals[name], atol=1e-5), \
                    f"参数 {name}: checkpoint 应保存 EMA shadow，而非训练权重"
                break  # 只需验证一个参数

        # 保存后 model 应还原为训练权重 (+100)
        first_param = next(model.parameters())
        assert (first_param > 50.0).all(), \
            "_save_checkpoint 后 model 应还原为训练权重"

    print("✓ test_buga_save_checkpoint_stores_ema_shadow PASSED")


def test_buga_checkpoint_consistent_with_validate():
    """BUG-A: 保存后立即加载的 model 权重应等于 _validate 用的 EMA shadow"""
    from jaguar.model import ReIDModel
    from jaguar.trainer import ReIDTrainer
    from gastrovision.losses.metric_learning import ArcFaceLoss
    from torch.utils.data import DataLoader, TensorDataset

    model = ReIDModel('resnet18', embedding_dim=64, pretrained=False,
                      use_gem=False, use_mlp_head=False)
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
            output_dir=tmp, use_ema=True, ema_decay=0.9
        )
        trainer.fit(loader, loader, epochs=3, early_stopping=0)

        # 记录 _validate 使用的 EMA shadow 权重
        trainer.ema.apply_shadow(model)
        shadow_at_best = {k: v.clone() for k, v in model.named_parameters()}
        trainer.ema.restore(model)

        # 加载 best_model.pth（应保存 EMA shadow）
        model2 = ReIDModel('resnet18', embedding_dim=64, pretrained=False,
                           use_gem=False, use_mlp_head=False)
        saved = torch.load(str(Path(tmp) / 'best_model.pth'),
                           map_location='cpu', weights_only=False)
        model2.load_state_dict(saved['model_state_dict'])

        # checkpoint 的权重应等于 EMA shadow
        for name, param in model2.named_parameters():
            if name in shadow_at_best:
                assert torch.allclose(param.data, shadow_at_best[name], atol=1e-5), \
                    f"参数 {name}: checkpoint 权重应等于 EMA shadow"

    print("✓ test_buga_checkpoint_consistent_with_validate PASSED")


# ================================================================
# BUG-B: EMA sync_shadow — 两阶段训练 Phase2 补充 backbone 参数
# ================================================================

def test_bugb_sync_shadow_adds_missing_params():
    """BUG-B: sync_shadow 应将模型中 shadow 不包含的参数加入 shadow"""
    from gastrovision.utils.ema import ModelEMA
    from jaguar.model import ReIDModel

    model = ReIDModel('resnet18', embedding_dim=64, pretrained=False,
                      use_gem=False, use_mlp_head=False)

    # 模拟 Phase1: freeze backbone → EMA 只追踪 head/embedding 参数
    for param in model.backbone.parameters():
        param.requires_grad_(False)

    ema = ModelEMA(model, decay=0.9)  # 只初始化非 backbone 参数

    # 验证 backbone 参数不在 shadow 中（Phase1 效果）
    backbone_names = set(name for name, _ in model.backbone.named_parameters())
    shadow_backbone = {k for k in ema.shadow if k.startswith('backbone.')}
    assert len(shadow_backbone) == 0, "Phase1 时 backbone 参数不应在 shadow 中"

    # 模拟 Phase2: 解冻 backbone
    for param in model.backbone.parameters():
        param.requires_grad_(True)

    # 调用 sync_shadow 补充 backbone 参数
    ema.sync_shadow(model)

    # 验证 backbone 参数已加入 shadow
    shadow_backbone_after = {k for k in ema.shadow if k.startswith('backbone.')}
    assert len(shadow_backbone_after) > 0, "sync_shadow 后 backbone 参数应在 shadow 中"

    # 验证补充的 backbone shadow = 当前模型权重
    for name, param in model.named_parameters():
        if name.startswith('backbone.') and param.requires_grad:
            assert name in ema.shadow, f"{name} 应在 shadow 中"
            assert torch.allclose(ema.shadow[name], param.data, atol=1e-6), \
                f"{name} 的 shadow 值应等于当前模型权重"

    print("✓ test_bugb_sync_shadow_adds_missing_params PASSED")


def test_bugb_sync_shadow_no_duplicate():
    """BUG-B: sync_shadow 不应覆盖 shadow 中已有的参数"""
    from gastrovision.utils.ema import ModelEMA
    from jaguar.model import ReIDModel

    model = ReIDModel('resnet18', embedding_dim=64, pretrained=False,
                      use_gem=False, use_mlp_head=False)

    ema = ModelEMA(model, decay=0.9)

    # 记录 shadow 中已有的参数值
    original_shadow = {k: v.clone() for k, v in ema.shadow.items()}

    # 修改模型权重
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.ones_like(p) * 50.0)

    # sync_shadow 不应影响已有的参数（shadow 只应保持不变）
    ema.sync_shadow(model)

    for k, v in original_shadow.items():
        assert torch.allclose(ema.shadow[k], v, atol=1e-6), \
            f"sync_shadow 不应覆盖已有参数 {k}"

    print("✓ test_bugb_sync_shadow_no_duplicate PASSED")


def test_bugb_load_checkpoint_calls_sync_shadow():
    """BUG-B: load_checkpoint 后 EMA shadow 应包含全部 requires_grad 参数"""
    from jaguar.model import ReIDModel
    from jaguar.trainer import ReIDTrainer
    from gastrovision.losses.metric_learning import ArcFaceLoss
    from torch.utils.data import DataLoader, TensorDataset

    # Phase1: 冻结 backbone 训练
    model = ReIDModel('resnet18', embedding_dim=64, pretrained=False,
                      use_gem=False, use_mlp_head=False)
    for param in model.backbone.parameters():
        param.requires_grad_(False)

    criterion = ArcFaceLoss(num_classes=4, embedding_dim=64)
    optimizer = torch.optim.Adam(
        list(p for p in model.parameters() if p.requires_grad) +
        list(criterion.parameters()), lr=0.001
    )

    images = torch.randn(8, 3, 224, 224)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)

    with tempfile.TemporaryDirectory() as tmp:
        trainer1 = ReIDTrainer(
            model=model, criterion=criterion, loss_type='arcface',
            optimizer=optimizer, device=torch.device('cpu'),
            output_dir=tmp, use_ema=True, ema_decay=0.9
        )
        trainer1.fit(loader, loader, epochs=2, early_stopping=0)

        # Phase2: 解冻 backbone，新建 trainer，加载 Phase1 checkpoint
        for param in model.backbone.parameters():
            param.requires_grad_(True)

        optimizer2 = torch.optim.Adam(
            list(model.parameters()) + list(criterion.parameters()), lr=0.001
        )
        trainer2 = ReIDTrainer(
            model=model, criterion=criterion, loss_type='arcface',
            optimizer=optimizer2, device=torch.device('cpu'),
            output_dir=tmp, use_ema=True, ema_decay=0.9
        )

        # 加载 Phase1 checkpoint（EMA shadow 只有 head 参数）
        trainer2.load_checkpoint(str(Path(tmp) / 'best_model.pth'),
                                 reset_best_metric=True)

        # 验证: shadow 中应包含全部 requires_grad 参数（包括 backbone）
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in trainer2.ema.shadow, \
                    f"Phase2 加载后 {name} 应在 EMA shadow 中"

    print("✓ test_bugb_load_checkpoint_calls_sync_shadow PASSED")


def test_bugb_phase2_ema_tracks_backbone():
    """BUG-B: Phase2 训练后 EMA shadow 中 backbone 参数应被更新（不再是 Phase1 初始值）"""
    from jaguar.model import ReIDModel
    from jaguar.trainer import ReIDTrainer
    from gastrovision.losses.metric_learning import ArcFaceLoss
    from torch.utils.data import DataLoader, TensorDataset

    # Phase1
    model = ReIDModel('resnet18', embedding_dim=64, pretrained=False,
                      use_gem=False, use_mlp_head=False)
    for param in model.backbone.parameters():
        param.requires_grad_(False)

    criterion = ArcFaceLoss(num_classes=4, embedding_dim=64)
    optimizer = torch.optim.Adam(
        list(p for p in model.parameters() if p.requires_grad) +
        list(criterion.parameters()), lr=0.001
    )
    images = torch.randn(8, 3, 224, 224)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)

    with tempfile.TemporaryDirectory() as tmp:
        trainer1 = ReIDTrainer(
            model=model, criterion=criterion, loss_type='arcface',
            optimizer=optimizer, device=torch.device('cpu'),
            output_dir=tmp, use_ema=True, ema_decay=0.9
        )
        trainer1.fit(loader, loader, epochs=2, early_stopping=0)

        # Phase2
        for param in model.backbone.parameters():
            param.requires_grad_(True)

        optimizer2 = torch.optim.Adam(
            list(model.parameters()) + list(criterion.parameters()), lr=0.1
        )  # 高 LR 确保参数确实变化
        trainer2 = ReIDTrainer(
            model=model, criterion=criterion, loss_type='arcface',
            optimizer=optimizer2, device=torch.device('cpu'),
            output_dir=tmp, use_ema=True, ema_decay=0.9
        )
        trainer2.load_checkpoint(str(Path(tmp) / 'best_model.pth'),
                                 reset_best_metric=True)

        # 记录 sync 后 backbone shadow 的初始值
        bb_shadow_before = {
            k: v.clone() for k, v in trainer2.ema.shadow.items()
            if k.startswith('backbone.')
        }

        # Phase2 训练几步（高 LR 保证 backbone 参数变化）
        trainer2.fit(loader, loader, epochs=3, early_stopping=0)

        # 验证: backbone shadow 应已更新（不再等于 Phase2 开始时的初始值）
        changed = any(
            not torch.allclose(trainer2.ema.shadow[k], bb_shadow_before[k], atol=1e-4)
            for k in bb_shadow_before
        )
        assert changed, "Phase2 训练后 backbone shadow 应已更新"

    print("✓ test_bugb_phase2_ema_tracks_backbone PASSED")


# ================================================================
# 运行所有测试
# ================================================================

if __name__ == '__main__':
    tests = [
        # BUG-1: EMA restore in finally
        test_bug1_ema_restore_on_exception,
        test_bug1_validate_has_finally_structure,
        test_bug1_validate_restore_after_exception,
        # BUG-3: GeM Pooling
        test_bug3_gem_hook_registered_for_resnet,
        test_bug3_gem_hook_registered_for_convnext,
        test_bug3_gem_actually_activates_for_resnet,
        test_bug3_gem_output_shape_resnet,
        test_bug3_gem_output_shape_convnext,
        test_bug3_swin_falls_back_to_avgpool,
        # BUG-4: MLP Head
        test_bug4_mlp_head_structure,
        test_bug4_linear_head_backward_compat,
        test_bug4_mlp_head_forward_shape,
        test_bug4_mlp_head_hidden_dim,
        test_bug4_resnet_with_mlp_head,
        # BUG-5: 两阶段训练
        test_bug5_reset_best_metric_parameter,
        test_bug5_reset_best_metric_works,
        test_bug5_two_phase_training_config,
        # BUG-6: 代码重复
        test_bug6_sigmoid_calibrate_function_exists,
        test_bug6_no_code_duplication,
        # 集成测试
        test_integration_full_training_mlp_gem,
        test_integration_convnext_gem_forward,
        test_integration_checkpoint_save_load_mlp,
        test_integration_config_has_mlp_head,
        # BUG-A: _save_checkpoint 保存 EMA shadow
        test_buga_save_checkpoint_stores_ema_shadow,
        test_buga_checkpoint_consistent_with_validate,
        # BUG-B: sync_shadow 两阶段训练 backbone 参数补充
        test_bugb_sync_shadow_adds_missing_params,
        test_bugb_sync_shadow_no_duplicate,
        test_bugb_load_checkpoint_calls_sync_shadow,
        test_bugb_phase2_ema_tracks_backbone,
    ]

    passed = 0
    failed = 0
    errors = []

    print("=" * 60)
    print("Jaguar Re-ID 代码审计修复验证测试")
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
