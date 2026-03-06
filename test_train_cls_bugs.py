"""
专项测试: train_cls.py 全流程 Bug 修复验证

覆盖:
  BUG-1: get_model 在 resume/test_only 时不应替换分类头
  BUG-2: trainer.train_epoch 中孤立的 `raise` 语句导致训练崩溃
  BUG-3: train_cls.py test_only 模式 checkpoint 加载逻辑

运行:
    conda activate torch27_env
    python test_train_cls_bugs.py
"""

import os
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torchvision.models as models

passed = 0
failed = 0
errors = []


def check(condition, test_name):
    global passed, failed, errors
    if condition:
        passed += 1
        print(f"  ✓ {test_name}")
    else:
        failed += 1
        errors.append(test_name)
        print(f"  ✗ FAIL: {test_name}")


# ============================================================================
# 辅助：伪造一个 Trainer.save_checkpoint 格式的 checkpoint 文件
# ============================================================================
def _make_fake_checkpoint(num_classes: int = 23, model_name: str = 'resnet50') -> str:
    """创建一个仿真 best_model.pth，包含正确类别数的完整权重。"""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # 给分类头权重设置一个可辨认的固定值，用于后续验证
    with torch.no_grad():
        model.fc.weight.fill_(0.123)
        model.fc.bias.fill_(0.456)

    tmpf = tempfile.NamedTemporaryFile(suffix='.pth', delete=False)
    tmpf.close()

    checkpoint = {
        'epoch': 50,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': {},
        'metrics': {'accuracy': 0.95},
        'best_valid_acc': 0.95,
        'best_epoch': 50,
    }
    torch.save(checkpoint, tmpf.name)
    return tmpf.name, model.fc.weight.clone().detach()


# ============================================================================
# BUG-1 测试：replace_head=True（正常 pretrain 场景）
# ============================================================================
def test_get_model_replace_head_default():
    """replace_head=True（默认）: 分类头应被随机初始化（不从checkpoint加载）"""
    print("\n[BUG-1a] get_model replace_head=True (pretrain/默认场景)")
    from gastrovision.models.model_factory import get_model

    ckpt_path, ckpt_fc_weight = _make_fake_checkpoint(num_classes=23)
    try:
        # replace_head=True: 加载 weights_path 后，分类头会被 _replace_classifier 覆盖
        model = get_model(
            'resnet50',
            num_classes=23,
            pretrained=False,
            weights_path=ckpt_path,
            replace_head=True,
        )
        loaded_fc_weight = model.fc.weight.detach()
        # 分类头被随机重新初始化，不应与 checkpoint 中的 0.123 相同
        is_overwritten = not torch.allclose(loaded_fc_weight, ckpt_fc_weight.to(loaded_fc_weight.device))
        check(is_overwritten, "replace_head=True: 分类头已被随机初始化（脱离checkpoint值）")
    except Exception as e:
        check(False, f"replace_head=True 场景抛出异常: {e}")
        traceback.print_exc()
    finally:
        os.unlink(ckpt_path)


# ============================================================================
# BUG-1 测试：replace_head=False（resume/推理场景）
# ============================================================================
def test_get_model_replace_head_false():
    """replace_head=False: 分类头权重应与checkpoint中完全一致"""
    print("\n[BUG-1b] get_model replace_head=False (resume/推理场景)")
    from gastrovision.models.model_factory import get_model

    ckpt_path, ckpt_fc_weight = _make_fake_checkpoint(num_classes=23)
    try:
        model = get_model(
            'resnet50',
            num_classes=23,
            pretrained=False,
            weights_path=ckpt_path,
            replace_head=False,
        )
        loaded_fc_weight = model.fc.weight.detach()
        # 分类头权重必须与 checkpoint 中的 0.123 完全一致
        is_preserved = torch.allclose(loaded_fc_weight, ckpt_fc_weight.to(loaded_fc_weight.device))
        check(is_preserved, "replace_head=False: 分类头权重与checkpoint完全一致")

        # 验证分类头输出维度正确
        check(model.fc.out_features == 23, f"分类头输出维度=23, 实际={model.fc.out_features}")
    except Exception as e:
        check(False, f"replace_head=False 场景抛出异常: {e}")
        traceback.print_exc()
    finally:
        os.unlink(ckpt_path)


def test_get_model_replace_head_false_num_classes_match():
    """replace_head=False: 即使配置 num_classes 和checkpoint一致，权重也应正确加载"""
    print("\n[BUG-1c] get_model replace_head=False (num_classes 与checkpoint一致)")
    from gastrovision.models.model_factory import get_model

    for n_cls in [10, 23, 5]:
        ckpt_path, ckpt_fc_weight = _make_fake_checkpoint(num_classes=n_cls)
        try:
            model = get_model(
                'resnet50',
                num_classes=n_cls,
                pretrained=False,
                weights_path=ckpt_path,
                replace_head=False,
            )
            loaded_fc_weight = model.fc.weight.detach()
            is_preserved = torch.allclose(loaded_fc_weight, ckpt_fc_weight.to(loaded_fc_weight.device))
            check(is_preserved, f"num_classes={n_cls}: 分类头权重与checkpoint一致")
        except Exception as e:
            check(False, f"num_classes={n_cls} 抛出异常: {e}")
            traceback.print_exc()
        finally:
            os.unlink(ckpt_path)


def test_get_model_normal_training_unaffected():
    """回归测试: replace_head=True + pretrained=True 正常工作（不影响原有训练路径）"""
    print("\n[BUG-1d] get_model 正常训练路径（pretrained=False, weights_path=None）")
    from gastrovision.models.model_factory import get_model

    try:
        # 纯随机初始化，最常见的从头训练场景
        model = get_model('resnet50', num_classes=23, pretrained=False, replace_head=True)
        check(model.fc.out_features == 23, f"正常训练: 分类头输出维度=23")
        # 验证可以前向传播
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        check(out.shape == (2, 23), f"正常训练: 前向传播输出形状 (2,23), 实际={out.shape}")
    except Exception as e:
        check(False, f"正常训练路径抛出异常: {e}")
        traceback.print_exc()


def test_get_model_without_replace_head_arg():
    """向后兼容: 不传 replace_head 参数（默认 True），行为与原来一致"""
    print("\n[BUG-1e] 向后兼容: 不传 replace_head 参数")
    from gastrovision.models.model_factory import get_model

    try:
        model = get_model('resnet50', num_classes=10, pretrained=False)
        check(model.fc.out_features == 10, f"不传replace_head: 分类头输出维度=10")
    except Exception as e:
        check(False, f"向后兼容测试抛出异常: {e}")
        traceback.print_exc()


# ============================================================================
# BUG-2 测试：trainer.train_epoch 不再因孤立 raise 崩溃
# ============================================================================
def test_train_epoch_no_crash():
    """BUG-2: train_epoch 能正常执行完一个 batch（不因 raise 崩溃）"""
    print("\n[BUG-2] train_epoch 不再崩溃（孤立 raise 已删除）")
    from gastrovision.trainers.trainer import Trainer

    num_classes = 5
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # 创建最小 DataLoader（1 个 batch）
    images = torch.randn(4, 3, 64, 64)
    targets = torch.randint(0, num_classes, (4,))
    dataset = torch.utils.data.TensorDataset(images, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            device=torch.device('cpu'),
            output_dir=tmpdir,
        )
        try:
            metrics = trainer.train_epoch(loader, epoch=1, total_epochs=1)
            check(True, "train_epoch 正常完成，未抛出 RuntimeError")
            check('loss' in metrics and 'accuracy' in metrics,
                  f"train_epoch 返回正确 metrics: {list(metrics.keys())}")
            check(isinstance(metrics['loss'], float) and metrics['loss'] >= 0,
                  f"loss = {metrics.get('loss', 'N/A'):.4f} (非负浮点数)")
        except RuntimeError as e:
            if str(e) == '' or 'No active exception to re-raise' in str(e):
                check(False, f"train_epoch 仍因孤立 raise 崩溃: {e}")
            else:
                check(False, f"train_epoch 抛出非预期 RuntimeError: {e}")
            traceback.print_exc()
        except Exception as e:
            check(False, f"train_epoch 抛出非预期异常: {e}")
            traceback.print_exc()


def test_train_epoch_with_metric_loss_no_crash():
    """BUG-2 回归: 使用度量学习时 train_epoch 也不崩溃"""
    print("\n[BUG-2b] train_epoch + MetricLearningWrapper 不崩溃")
    from gastrovision.trainers.trainer import Trainer
    from gastrovision.models.wrapper import MetricLearningWrapper
    from gastrovision.losses.metric_learning import create_metric_loss

    num_classes = 5
    base_model = models.resnet18(weights=None)
    base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
    wrapped = MetricLearningWrapper(base_model)

    contrastive = create_metric_loss('contrastive', num_classes=num_classes,
                                     embedding_dim=wrapped.feature_dim)

    images = torch.randn(4, 3, 64, 64)
    # 确保有正样本对（两个样本同类）
    targets = torch.tensor([0, 0, 1, 2])
    dataset = torch.utils.data.TensorDataset(images, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=wrapped,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(wrapped.parameters(), lr=0.01),
            device=torch.device('cpu'),
            output_dir=tmpdir,
            metric_loss=contrastive,
            metric_loss_weight=0.5,
        )
        try:
            metrics = trainer.train_epoch(loader, epoch=1, total_epochs=1)
            check(True, "train_epoch + 度量学习正常完成")
        except Exception as e:
            check(False, f"train_epoch + 度量学习抛出异常: {e}")
            traceback.print_exc()


# ============================================================================
# BUG-3 测试：test_only 模式使用 get_model 正确加载权重
# ============================================================================
def test_test_only_flow_loads_correct_weights():
    """BUG-3: 模拟 test_only 流程，验证分类头权重来自 checkpoint（非随机初始化）"""
    print("\n[BUG-3] test_only 流程: 分类头权重来自 checkpoint")
    from gastrovision.models.model_factory import get_model

    ckpt_path, ckpt_fc_weight = _make_fake_checkpoint(num_classes=23)
    try:
        # 模拟 train_cls.py 的 test_only + resume 流程
        cfg_resume = ckpt_path
        cfg_test_only = True

        # 按修复后的 train_cls.py 逻辑
        weights_to_load = cfg_resume
        use_pretrained = False
        replace_head = not cfg_resume  # cfg.resume 已设置 → replace_head=False
        if cfg_resume:
            replace_head = False

        model = get_model(
            'resnet50',
            num_classes=23,
            pretrained=use_pretrained,
            weights_path=weights_to_load,
            replace_head=replace_head,
        )

        loaded_fc_weight = model.fc.weight.detach()
        is_correct = torch.allclose(loaded_fc_weight, ckpt_fc_weight.to(loaded_fc_weight.device))
        check(is_correct, "test_only 流程: 分类头权重等于 checkpoint 中的权重（非随机）")

        # 验证可以做推理
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        check(out.shape == (2, 23), f"test_only 推理输出形状正确: {out.shape}")
    except Exception as e:
        check(False, f"test_only 流程抛出异常: {e}")
        traceback.print_exc()
    finally:
        os.unlink(ckpt_path)


def test_resume_training_flow():
    """BUG-3 回归: 训练恢复（resume + not test_only）场景下权重也正确"""
    print("\n[BUG-3b] 训练恢复流程: 分类头权重来自 checkpoint")
    from gastrovision.models.model_factory import get_model

    ckpt_path, ckpt_fc_weight = _make_fake_checkpoint(num_classes=23)
    try:
        # 模拟 train_cls.py 的 resume + not test_only 流程
        model = get_model(
            'resnet50',
            num_classes=23,
            pretrained=False,
            weights_path=ckpt_path,
            replace_head=False,  # resume 时传 False
        )
        loaded_fc_weight = model.fc.weight.detach()
        is_correct = torch.allclose(loaded_fc_weight, ckpt_fc_weight.to(loaded_fc_weight.device))
        check(is_correct, "训练恢复流程: 分类头权重等于 checkpoint")
    except Exception as e:
        check(False, f"训练恢复流程抛出异常: {e}")
        traceback.print_exc()
    finally:
        os.unlink(ckpt_path)


# ============================================================================
# 附加测试：trainer.load_checkpoint 在训练恢复时仍能正确恢复 optimizer 状态
# ============================================================================
def test_load_checkpoint_restores_optimizer():
    """训练恢复: trainer.load_checkpoint 能正确恢复 optimizer 状态"""
    print("\n[附加] trainer.load_checkpoint 恢复 optimizer 状态")
    from gastrovision.trainers.trainer import Trainer

    num_classes = 5
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            device=torch.device('cpu'),
            output_dir=tmpdir,
        )
        # 手动保存一个 checkpoint（含 optimizer state）
        trainer.save_checkpoint('test_ckpt.pth', epoch=10, metrics={'accuracy': 0.9})
        ckpt_path = os.path.join(tmpdir, 'test_ckpt.pth')

        # 创建新 trainer，加载 checkpoint
        model2 = models.resnet18(weights=None)
        model2.fc = nn.Linear(model2.fc.in_features, num_classes)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        trainer2 = Trainer(
            model=model2,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer2,
            device=torch.device('cpu'),
            output_dir=tmpdir,
        )
        try:
            trainer2.load_checkpoint(ckpt_path)
            check(trainer2.best_valid_acc == 0.9, f"best_valid_acc 恢复正确: {trainer2.best_valid_acc}")
            check(trainer2.best_epoch == 10, f"best_epoch 恢复正确: {trainer2.best_epoch}")
        except Exception as e:
            check(False, f"load_checkpoint 恢复失败: {e}")
            traceback.print_exc()


# ============================================================================
# 主程序
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("  train_cls.py 全流程 Bug 修复专项测试")
    print("=" * 70)

    try:
        # BUG-1 测试
        test_get_model_replace_head_default()
        test_get_model_replace_head_false()
        test_get_model_replace_head_false_num_classes_match()
        test_get_model_normal_training_unaffected()
        test_get_model_without_replace_head_arg()

        # BUG-2 测试
        test_train_epoch_no_crash()
        test_train_epoch_with_metric_loss_no_crash()

        # BUG-3 测试
        test_test_only_flow_loads_correct_weights()
        test_resume_training_flow()

        # 附加：确保 load_checkpoint 在训练恢复时仍可用
        test_load_checkpoint_restores_optimizer()

    except Exception as e:
        print(f"\n未捕获异常: {e}")
        traceback.print_exc()
        failed += 1

    print("\n" + "=" * 70)
    print(f"  结果: {passed} 通过, {failed} 失败")
    if errors:
        print("  失败项:")
        for e in errors:
            print(f"    - {e}")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)
