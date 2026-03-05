"""
综合测试: BUG-12 修复 + Top-K/AUC 指标 + 可视化增强
测试 MetricLearningWrapper、所有度量学习损失函数、新指标输出、可视化函数

运行: python test_bugfixes_v3.py
"""

import sys
import os
import traceback
import tempfile
import json
import numpy as np

# 确保项目根目录在 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

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
# 1. MetricLearningWrapper 测试
# ============================================================================
def test_wrapper_resnet50():
    """测试 MetricLearningWrapper 对 torchvision ResNet50 的包装"""
    print("\n[1] MetricLearningWrapper - ResNet50")
    
    import torchvision.models as models
    from gastrovision.models.wrapper import MetricLearningWrapper
    
    # 创建 torchvision resnet50，替换 fc 为 23 类
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 23)
    
    # 包装
    wrapped = MetricLearningWrapper(model)
    
    # feature_dim 应该是 2048
    check(wrapped.feature_dim == 2048, "feature_dim == 2048 for ResNet50")
    
    # 前向传播
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        output = wrapped(x)
    
    check(isinstance(output, tuple), "output is tuple")
    check(len(output) == 2, "output has 2 elements")
    
    logits, features = output
    check(logits.shape == (4, 23), f"logits shape (4, 23), got {logits.shape}")
    check(features.shape == (4, 2048), f"features shape (4, 2048), got {features.shape}")
    
    # 梯度传播
    logits_grad, features_grad = wrapped(torch.randn(2, 3, 224, 224))
    loss = logits_grad.sum() + features_grad.sum()
    loss.backward()
    
    # 检查 fc 有梯度
    has_grad = wrapped.model.fc.weight.grad is not None
    check(has_grad, "gradients flow through wrapper")


def test_wrapper_convnext():
    """测试 MetricLearningWrapper 对 ConvNeXt Tiny 的包装"""
    print("\n[2] MetricLearningWrapper - ConvNeXt Tiny")
    
    try:
        import torchvision.models as models
        from gastrovision.models.wrapper import MetricLearningWrapper
        
        model = models.convnext_tiny(weights=None)
        in_feat = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feat, 10)
        
        wrapped = MetricLearningWrapper(model)
        check(wrapped.feature_dim == in_feat, f"feature_dim == {in_feat} for ConvNeXt")
        
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits, features = wrapped(x)
        
        check(logits.shape == (2, 10), f"logits shape (2, 10), got {logits.shape}")
        check(features.shape[0] == 2 and features.shape[1] == in_feat,
              f"features shape (2, {in_feat}), got {features.shape}")
    except Exception as e:
        check(False, f"ConvNeXt test failed: {e}")


def test_wrapper_swin():
    """测试 MetricLearningWrapper 对 Swin Transformer 的包装"""
    print("\n[3] MetricLearningWrapper - Swin Transformer")
    
    try:
        import torchvision.models as models
        from gastrovision.models.wrapper import MetricLearningWrapper
        
        model = models.swin_t(weights=None)
        in_feat = model.head.in_features
        model.head = nn.Linear(in_feat, 10)
        
        wrapped = MetricLearningWrapper(model)
        check(wrapped.feature_dim == in_feat, f"feature_dim == {in_feat} for Swin")
        
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits, features = wrapped(x)
        
        check(logits.shape == (2, 10), f"logits shape (2, 10), got {logits.shape}")
        check(features.shape[0] == 2, f"features batch dim correct")
    except Exception as e:
        check(False, f"Swin test failed: {e}")


# ============================================================================
# 2. 所有度量学习损失 + MetricLearningWrapper 端到端测试
# ============================================================================
def test_all_metric_losses_with_wrapper():
    """测试所有度量学习损失函数在 wrapper 提供的 features 上正常工作"""
    print("\n[4] 所有度量学习损失 + MetricLearningWrapper 端到端")
    
    import torchvision.models as models
    from gastrovision.models.wrapper import MetricLearningWrapper
    from gastrovision.losses.metric_learning import create_metric_loss
    
    num_classes = 10
    batch_size = 8
    
    # 创建包装模型
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    wrapped = MetricLearningWrapper(model)
    feature_dim = wrapped.feature_dim  # 2048
    
    x = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # 确保有足够标签多样性
    labels[0], labels[1] = 0, 0  # 至少一对相同标签
    labels[2], labels[3] = 1, 1
    
    loss_configs = [
        ('contrastive', {}),
        ('triplet', {}),
        ('lifted', {}),
        ('npair', {}),
        ('proxy_nca', {}),
        ('arcface', {}),
        ('cosface', {}),
        ('sphereface', {}),
        ('circle_cls', {}),
    ]
    
    for loss_type, kwargs in loss_configs:
        try:
            loss_fn = create_metric_loss(
                loss_type=loss_type,
                num_classes=num_classes,
                embedding_dim=feature_dim,
                **kwargs
            )
            
            with torch.no_grad():
                logits, features = wrapped(x)
            
            # 计算损失
            loss_val = loss_fn(features, labels)
            
            check(
                loss_val.dim() == 0 and not torch.isnan(loss_val) and loss_val.item() >= 0,
                f"{loss_type}: loss={loss_val.item():.4f} (features dim={features.shape[1]})"
            )
        except Exception as e:
            check(False, f"{loss_type} failed: {e}")
            traceback.print_exc()


def test_arcface_no_crash_with_wrapper():
    """BUG-12 核心验证: ArcFace 使用 wrapper features 不再崩溃"""
    print("\n[5] BUG-12 核心验证: ArcFace 不再崩溃")
    
    import torchvision.models as models
    from gastrovision.models.wrapper import MetricLearningWrapper
    from gastrovision.losses.metric_learning import create_metric_loss
    
    num_classes = 23
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    wrapped = MetricLearningWrapper(model)
    feature_dim = wrapped.feature_dim  # 2048
    
    # 创建 ArcFace，使用正确的 feature_dim（而非硬编码的 512）
    arcface = create_metric_loss(
        'arcface', num_classes=num_classes, embedding_dim=feature_dim
    )
    
    x = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, num_classes, (4,))
    
    try:
        with torch.no_grad():
            logits, features = wrapped(x)
        
        loss = arcface(features, labels)
        check(True, f"ArcFace with 2048-dim features: loss={loss.item():.4f}")
    except ValueError as e:
        check(False, f"ArcFace still crashes: {e}")


def test_proxy_nca_no_crash_with_wrapper():
    """BUG-12 核心验证: ProxyNCA 使用 wrapper features 不再崩溃"""
    print("\n[6] BUG-12 核心验证: ProxyNCA 不再崩溃")
    
    import torchvision.models as models
    from gastrovision.models.wrapper import MetricLearningWrapper
    from gastrovision.losses.metric_learning import create_metric_loss
    
    num_classes = 23
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    wrapped = MetricLearningWrapper(model)
    feature_dim = wrapped.feature_dim
    
    proxy_nca = create_metric_loss(
        'proxy_nca', num_classes=num_classes, embedding_dim=feature_dim
    )
    
    x = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, num_classes, (4,))
    
    try:
        with torch.no_grad():
            logits, features = wrapped(x)
        
        loss = proxy_nca(features, labels)
        check(True, f"ProxyNCA with 2048-dim features: loss={loss.item():.4f}")
    except ValueError as e:
        check(False, f"ProxyNCA still crashes: {e}")


# ============================================================================
# 3. _compute_metric_loss 与 wrapper 集成测试
# ============================================================================
def test_trainer_compute_metric_loss_with_wrapper():
    """测试 Trainer._compute_metric_loss 使用 wrapper features 正常工作"""
    print("\n[7] Trainer._compute_metric_loss + wrapper 集成")
    
    import torchvision.models as models
    from gastrovision.models.wrapper import MetricLearningWrapper
    from gastrovision.losses.metric_learning import create_metric_loss
    from gastrovision.trainers.trainer import Trainer
    
    num_classes = 10
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    wrapped = MetricLearningWrapper(model)
    feature_dim = wrapped.feature_dim
    
    # 创建各种需要 embedding_dim 的损失
    for loss_type in ['arcface', 'cosface', 'sphereface', 'proxy_nca', 'circle_cls']:
        loss_fn = create_metric_loss(
            loss_type, num_classes=num_classes, embedding_dim=feature_dim
        )
        
        trainer = Trainer(
            model=wrapped,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(wrapped.parameters(), lr=0.01),
            device='cpu',
            output_dir=tempfile.mkdtemp(),
            metric_loss=loss_fn,
            metric_loss_weight=0.5
        )
        
        x = torch.randn(4, 3, 224, 224)
        labels = torch.randint(0, num_classes, (4,))
        
        try:
            with torch.no_grad():
                logits, features = wrapped(x)
            
            ml_loss = trainer._compute_metric_loss(logits, features, labels)
            check(
                ml_loss.dim() == 0 and not torch.isnan(ml_loss),
                f"Trainer + {loss_type}: loss={ml_loss.item():.4f}"
            )
        except Exception as e:
            check(False, f"Trainer + {loss_type} failed: {e}")


# ============================================================================
# 4. Top-K 准确率和 AUC 测试
# ============================================================================
def test_top_k_accuracy():
    """测试 Top-1 到 Top-5 准确率计算"""
    print("\n[8] Top-K 准确率计算")
    
    # 模拟 10 个样本, 5 个类别
    num_samples = 100
    num_classes = 5
    
    np.random.seed(42)
    all_targets = np.random.randint(0, num_classes, num_samples)
    
    # 创建概率：正确类别有最高概率
    all_probs = np.random.rand(num_samples, num_classes) * 0.1
    for i in range(num_samples):
        all_probs[i, all_targets[i]] = 0.9
    # softmax-like normalization
    all_probs = all_probs / all_probs.sum(axis=1, keepdims=True)
    
    # Top-1 should be ~1.0 (we set highest prob for correct class)
    top1_preds = np.argsort(all_probs, axis=1)[:, -1:]
    top1_correct = np.array([t in p for t, p in zip(all_targets, top1_preds)])
    top1_acc = top1_correct.mean()
    check(top1_acc > 0.95, f"Top-1 accuracy = {top1_acc:.4f} (expected ~1.0)")
    
    # Top-5 should be 1.0 when num_classes == 5
    top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
    top5_correct = np.array([t in p for t, p in zip(all_targets, top5_preds)])
    top5_acc = top5_correct.mean()
    check(top5_acc == 1.0, f"Top-5 accuracy = {top5_acc:.4f} (expected 1.0 with 5 classes)")
    
    # Top-k monotonically increasing
    accuracies = []
    for k in range(1, 6):
        topk_preds = np.argsort(all_probs, axis=1)[:, -k:]
        topk_correct = np.array([t in p for t, p in zip(all_targets, topk_preds)])
        accuracies.append(topk_correct.mean())
    
    is_monotonic = all(a <= b for a, b in zip(accuracies, accuracies[1:]))
    check(is_monotonic, f"Top-K monotonically increasing: {[f'{a:.3f}' for a in accuracies]}")


def test_per_class_auc():
    """测试每类 AUC 计算"""
    print("\n[9] 每类 AUC 计算")
    
    from sklearn.metrics import roc_auc_score
    
    num_samples = 200
    num_classes = 5
    
    np.random.seed(42)
    all_targets = np.random.randint(0, num_classes, num_samples)
    
    # 生成有区分度但不完美的概率：约 20% 样本故意给错误类最高概率
    all_probs = np.random.rand(num_samples, num_classes) * 0.2
    for i in range(num_samples):
        if np.random.rand() < 0.8:
            # 80%: 正确类概率最高
            all_probs[i, all_targets[i]] += 0.6
        else:
            # 20%: 随机错误类概率最高（模拟误分类）
            wrong_class = (all_targets[i] + np.random.randint(1, num_classes)) % num_classes
            all_probs[i, wrong_class] += 0.6
    all_probs = all_probs / all_probs.sum(axis=1, keepdims=True)
    
    # one-hot
    y_true_onehot = np.zeros((num_samples, num_classes))
    for i, t in enumerate(all_targets):
        y_true_onehot[i, t] = 1
    
    per_class_auc = {}
    for c in range(num_classes):
        if y_true_onehot[:, c].sum() > 0 and y_true_onehot[:, c].sum() < num_samples:
            auc_val = roc_auc_score(y_true_onehot[:, c], all_probs[:, c])
            per_class_auc[c] = auc_val
    
    valid_aucs = list(per_class_auc.values())
    macro_auc = np.mean(valid_aucs)
    
    check(len(per_class_auc) == num_classes, f"All {num_classes} classes have AUC values")
    check(all(0.5 < v < 1.0 for v in valid_aucs), 
          f"All AUCs in (0.5, 1.0): {[f'{v:.3f}' for v in valid_aucs]}")
    check(0.5 < macro_auc < 1.0, f"Macro AUC = {macro_auc:.4f}")


# ============================================================================
# 5. 可视化函数测试
# ============================================================================
def test_confusion_matrix_with_counts():
    """测试混淆矩阵显示计数"""
    print("\n[10] 混淆矩阵显示计数")
    
    from gastrovision.utils.visualization import plot_confusion_matrix
    
    cm = np.array([
        [50, 3, 0],
        [2, 45, 1],
        [0, 1, 48]
    ])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, 'cm_test.png')
        
        try:
            plot_confusion_matrix(
                cm, 
                class_names=['A', 'B', 'C'],
                output_path=output_path,
                normalize=True
            )
            check(os.path.exists(output_path), "Normalized CM with counts saved")
        except Exception as e:
            check(False, f"CM plot failed: {e}")
            traceback.print_exc()
        
        output_path2 = os.path.join(tmpdir, 'cm_test_raw.png')
        try:
            plot_confusion_matrix(
                cm,
                class_names=['A', 'B', 'C'],
                output_path=output_path2,
                normalize=False
            )
            check(os.path.exists(output_path2), "Raw count CM saved")
        except Exception as e:
            check(False, f"CM raw plot failed: {e}")


def test_per_class_auc_curves_plot():
    """测试每类 AUC 曲线 MxN 子图"""
    print("\n[11] 每类 AUC 曲线 MxN 子图")
    
    from gastrovision.utils.visualization import plot_per_class_auc_curves
    
    num_samples = 200
    num_classes = 8
    
    np.random.seed(42)
    all_targets = np.random.randint(0, num_classes, num_samples).tolist()
    
    all_probs_raw = np.random.rand(num_samples, num_classes) * 0.2
    for i in range(num_samples):
        all_probs_raw[i, all_targets[i]] = 0.8 + np.random.rand() * 0.2
    all_probs_raw = all_probs_raw / all_probs_raw.sum(axis=1, keepdims=True)
    all_probs = all_probs_raw.tolist()
    
    class_names = [f'class_{i}' for i in range(num_classes)]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, 'auc_curves.png')
        try:
            plot_per_class_auc_curves(
                all_targets=all_targets,
                all_probs=all_probs,
                class_names=class_names,
                output_path=output_path
            )
            check(os.path.exists(output_path), f"AUC curves plot saved ({num_classes} classes)")
        except Exception as e:
            check(False, f"AUC curves plot failed: {e}")
            traceback.print_exc()


def test_create_test_results_summary():
    """测试完整的测试结果汇总生成"""
    print("\n[12] 测试结果汇总 (含新指标)")
    
    from gastrovision.utils.visualization import create_test_results_summary
    
    num_samples = 100
    num_classes = 5
    
    np.random.seed(42)
    all_targets = np.random.randint(0, num_classes, num_samples).tolist()
    all_probs_raw = np.random.rand(num_samples, num_classes)
    for i in range(num_samples):
        all_probs_raw[i, all_targets[i]] += 2.0
    all_probs_raw = all_probs_raw / all_probs_raw.sum(axis=1, keepdims=True)
    
    # 模拟 test_results.json 结构
    results = {
        'accuracy': 0.88,
        'top1_accuracy': 0.88,
        'top2_accuracy': 0.94,
        'top3_accuracy': 0.97,
        'top4_accuracy': 0.99,
        'top5_accuracy': 1.0,
        'macro_auc': 0.95,
        'per_class_auc': {f'class_{i}': 0.9 + i * 0.02 for i in range(num_classes)},
        'precision_macro': 0.85,
        'recall_macro': 0.84,
        'f1_macro': 0.84,
        'precision_weighted': 0.87,
        'recall_weighted': 0.88,
        'f1_weighted': 0.87,
        'confusion_matrix': np.eye(num_classes, dtype=int).tolist(),
        'classification_report': {
            f'class_{i}': {
                'precision': 0.8 + i * 0.04,
                'recall': 0.85,
                'f1-score': 0.82 + i * 0.02,
                'support': 20
            } for i in range(num_classes)
        },
        'num_samples': num_samples,
        'num_classes': num_classes,
        'all_probs': all_probs_raw.tolist(),
        'all_targets': all_targets
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        results_path = os.path.join(tmpdir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f)
        
        class_names = [f'class_{i}' for i in range(num_classes)]
        
        try:
            create_test_results_summary(results_path, tmpdir, class_names)
            
            # 检查生成的文件
            expected_files = [
                'confusion_matrix_normalized.png',
                'per_class_metrics.png',
                'per_class_auc_curves.png',
                'test_summary.png'
            ]
            for fname in expected_files:
                fpath = os.path.join(tmpdir, fname)
                check(os.path.exists(fpath), f"Generated: {fname}")
        except Exception as e:
            check(False, f"Summary generation failed: {e}")
            traceback.print_exc()


# ============================================================================
# 6. print_test_results 测试
# ============================================================================
def test_print_test_results():
    """测试 print_test_results 输出新指标"""
    print("\n[13] print_test_results 输出新指标")
    
    from gastrovision.trainers.trainer import print_test_results
    import io
    from contextlib import redirect_stdout
    
    results = {
        'accuracy': 0.88,
        'top1_accuracy': 0.88,
        'top2_accuracy': 0.94,
        'top3_accuracy': 0.97,
        'top4_accuracy': 0.99,
        'top5_accuracy': 1.0,
        'macro_auc': 0.95,
        'per_class_auc': {'class_0': 0.92, 'class_1': 0.95, 'class_2': None},
        'precision_macro': 0.85,
        'recall_macro': 0.84,
        'f1_macro': 0.84,
        'precision_weighted': 0.87,
        'recall_weighted': 0.88,
        'f1_weighted': 0.87,
        'num_samples': 100,
        'num_classes': 3
    }
    
    captured = io.StringIO()
    with redirect_stdout(captured):
        print_test_results(results)
    
    output = captured.getvalue()
    
    check('Top-1 Accuracy' in output, "Output contains Top-1")
    check('Top-2 Accuracy' in output, "Output contains Top-2")
    check('Top-3 Accuracy' in output, "Output contains Top-3")
    check('Top-4 Accuracy' in output, "Output contains Top-4")
    check('Top-5 Accuracy' in output, "Output contains Top-5")
    check('Macro AUC' in output, "Output contains Macro AUC")
    check('每类 AUC' in output, "Output contains per-class AUC")
    check('N/A' in output, "Output contains N/A for class with no AUC")


# ============================================================================
# 7. create_metric_loss_function 集成测试
# ============================================================================
def test_create_metric_loss_function_integration():
    """测试 main.py 中的 create_metric_loss_function 正确包装模型"""
    print("\n[14] create_metric_loss_function 集成测试")
    
    import torchvision.models as models
    from gastrovision.models.wrapper import MetricLearningWrapper
    
    # 模拟 args
    class Args:
        metric_loss = 'arcface'
        metric_loss_margin = 0.0
        metric_loss_scale = 0.0
        metric_loss_weight = 0.5
        embedding_dim = 512  # 用户默认值，应被自动覆盖为 2048
    
    args = Args()
    num_classes = 23
    device = 'cpu'
    
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # 导入并调用
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from main import create_metric_loss_function
    
    metric_criterion, wrapped_model = create_metric_loss_function(args, num_classes, device, model)
    
    check(metric_criterion is not None, "metric_criterion created")
    check(isinstance(wrapped_model, MetricLearningWrapper), "model is wrapped")
    
    # 验证 ArcFace weight 维度与 feature_dim 匹配
    arcface_weight_dim = metric_criterion.weight.shape[1]
    check(
        arcface_weight_dim == 2048,
        f"ArcFace weight dim = {arcface_weight_dim} (should be 2048, not 512)"
    )
    
    # 端到端前向传播
    x = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, num_classes, (4,))
    
    with torch.no_grad():
        logits, features = wrapped_model(x)
    
    loss = metric_criterion(features, labels)
    check(
        loss.dim() == 0 and not torch.isnan(loss),
        f"End-to-end ArcFace loss = {loss.item():.4f}"
    )


def test_no_wrapper_when_metric_loss_none():
    """测试 metric_loss=none 时不包装模型"""
    print("\n[15] metric_loss=none 时不包装模型")
    
    import torchvision.models as models
    from gastrovision.models.wrapper import MetricLearningWrapper
    from main import create_metric_loss_function
    
    class Args:
        metric_loss = 'none'
        metric_loss_margin = 0.0
        metric_loss_scale = 0.0
        metric_loss_weight = 0.5
        embedding_dim = 512
    
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    metric_criterion, returned_model = create_metric_loss_function(Args(), 10, 'cpu', model)
    
    check(metric_criterion is None, "No metric criterion when loss=none")
    check(not isinstance(returned_model, MetricLearningWrapper), "Model not wrapped when loss=none")


# ============================================================================
# 主程序
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("  综合测试: BUG-12 修复 + Top-K/AUC 指标 + 可视化增强")
    print("=" * 70)
    
    try:
        test_wrapper_resnet50()
        test_wrapper_convnext()
        test_wrapper_swin()
        test_all_metric_losses_with_wrapper()
        test_arcface_no_crash_with_wrapper()
        test_proxy_nca_no_crash_with_wrapper()
        test_trainer_compute_metric_loss_with_wrapper()
        test_top_k_accuracy()
        test_per_class_auc()
        test_confusion_matrix_with_counts()
        test_per_class_auc_curves_plot()
        test_create_test_results_summary()
        test_print_test_results()
        test_create_metric_loss_function_integration()
        test_no_wrapper_when_metric_loss_none()
    except Exception as e:
        print(f"\n未捕获异常: {e}")
        traceback.print_exc()
        failed += 1
    
    print("\n" + "=" * 70)
    print(f"  结果: {passed} 通过, {failed} 失败")
    if errors:
        print(f"  失败项:")
        for e in errors:
            print(f"    - {e}")
    print("=" * 70)
    
    sys.exit(0 if failed == 0 else 1)
