"""
度量学习损失函数测试脚本

测试环境: conda activate torch27_env
运行: python test_metric_losses.py
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import math

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
# 工具函数测试
# =============================================================================

def test_utility_functions():
    """测试工具函数"""
    print("\n[工具函数] l2_normalize, pairwise_euclidean_distance, pairwise_cosine_similarity")
    from gastrovision.losses.metric_learning import (
        l2_normalize, pairwise_euclidean_distance, pairwise_cosine_similarity, get_pairwise_labels
    )
    
    # L2 归一化
    x = torch.randn(4, 128)
    x_norm = l2_normalize(x)
    norms = x_norm.norm(dim=1)
    check("L2 归一化后范数为 1", torch.allclose(norms, torch.ones(4), atol=1e-5),
          f"norms={norms.tolist()}")
    
    # 欧氏距离矩阵
    dist = pairwise_euclidean_distance(x_norm)
    check("距离矩阵形状正确", dist.shape == (4, 4))
    check("对角线距离近似为 0", torch.allclose(dist.diag(), torch.zeros(4), atol=1e-3),
          f"diag={dist.diag().tolist()}")
    check("距离矩阵对称", torch.allclose(dist, dist.t(), atol=1e-5))
    check("距离非负", (dist >= -1e-5).all().item())
    
    # 余弦相似度
    sim = pairwise_cosine_similarity(x)
    check("相似度矩阵形状正确", sim.shape == (4, 4))
    check("对角线相似度为 1", torch.allclose(sim.diag(), torch.ones(4), atol=1e-5),
          f"diag={sim.diag().tolist()}")
    check("相似度范围 [-1, 1]", (sim >= -1.01).all().item() and (sim <= 1.01).all().item())
    
    # 标签掩码
    labels = torch.tensor([0, 0, 1, 1])
    pos_mask, neg_mask = get_pairwise_labels(labels)
    expected_pos = torch.tensor([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=torch.float)
    check("正样本掩码正确", torch.equal(pos_mask, expected_pos))
    check("负样本掩码 = 1 - match", (pos_mask + neg_mask + torch.eye(4)).sum().item() == 16)


# =============================================================================
# 1. ContrastiveLoss 测试
# =============================================================================

def test_contrastive_loss():
    """测试对比损失"""
    print("\n[1] ContrastiveLoss")
    from gastrovision.losses.metric_learning import ContrastiveLoss
    
    loss_fn = ContrastiveLoss(margin=1.0, normalize=True)
    
    # 基本前向传播
    features = torch.randn(8, 128, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = loss_fn(features, labels)
    check("前向传播成功", loss.item() >= 0, f"loss={loss.item():.4f}")
    check("损失可微", loss.requires_grad)
    
    # 梯度回传
    loss.backward()
    check("梯度回传成功", features.grad is not None)
    check("梯度数值有效", torch.isfinite(features.grad).all().item())
    
    # 完美聚类: 同类特征相同，异类特征远离
    # 预期损失接近 0
    perfect_features = torch.zeros(4, 2)
    perfect_features[0] = torch.tensor([10.0, 0.0])
    perfect_features[1] = torch.tensor([10.0, 0.0])
    perfect_features[2] = torch.tensor([-10.0, 0.0])
    perfect_features[3] = torch.tensor([-10.0, 0.0])
    labels_2 = torch.tensor([0, 0, 1, 1])
    
    loss_fn_no_norm = ContrastiveLoss(margin=1.0, normalize=False)
    loss_perfect = loss_fn_no_norm(perfect_features, labels_2)
    check("完美聚类损失极小", loss_perfect.item() < 0.1,
          f"loss={loss_perfect.item():.4f}")


# =============================================================================
# 2. TripletMarginLoss 测试
# =============================================================================

def test_triplet_margin_loss():
    """测试三元组损失"""
    print("\n[2] TripletMarginLoss")
    from gastrovision.losses.metric_learning import TripletMarginLoss
    
    # Hard margin
    loss_fn = TripletMarginLoss(margin=0.3, normalize=True, soft=False)
    features = torch.randn(12, 128, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    loss = loss_fn(features, labels)
    check("Hard margin 前向传播", loss.item() >= 0)
    loss.backward()
    check("Hard margin 梯度回传", features.grad is not None)
    
    # Soft margin
    features2 = torch.randn(12, 128, requires_grad=True)
    loss_fn_soft = TripletMarginLoss(margin=0.3, normalize=True, soft=True)
    loss_soft = loss_fn_soft(features2, labels)
    check("Soft margin 前向传播", loss_soft.item() >= 0)
    loss_soft.backward()
    check("Soft margin 梯度回传", features2.grad is not None)
    
    # 单类别场景 (无正/负样本对) — 不应崩溃
    single_labels = torch.tensor([0, 0, 0, 0])
    features3 = torch.randn(4, 128)
    loss_single = loss_fn(features3, single_labels)
    check("单类别不崩溃", True)  # 如果到这里就没崩溃


# =============================================================================
# 3. LiftedStructureLoss 测试
# =============================================================================

def test_lifted_structure_loss():
    """测试提升结构损失"""
    print("\n[3] LiftedStructureLoss")
    from gastrovision.losses.metric_learning import LiftedStructureLoss
    
    loss_fn = LiftedStructureLoss(margin=1.0, normalize=True)
    features = torch.randn(8, 128, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    
    loss = loss_fn(features, labels)
    check("前向传播成功", loss.item() >= 0, f"loss={loss.item():.4f}")
    loss.backward()
    check("梯度回传成功", features.grad is not None)
    
    # 无正样本对的极端情况
    all_diff_labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    features2 = torch.randn(8, 128)
    loss_no_pos = loss_fn(features2, all_diff_labels)
    check("无正样本对返回 0", loss_no_pos.item() == 0.0)


# =============================================================================
# 4. ProxyNCALoss 测试
# =============================================================================

def test_proxy_nca_loss():
    """测试代理 NCA 损失"""
    print("\n[4] ProxyNCALoss")
    from gastrovision.losses.metric_learning import ProxyNCALoss
    
    num_classes = 10
    embed_dim = 128
    loss_fn = ProxyNCALoss(num_classes=num_classes, embedding_dim=embed_dim, scale=8.0)
    
    # 检查可学习参数
    check("含可学习 proxies", hasattr(loss_fn, 'proxies'))
    check("proxies 形状正确", loss_fn.proxies.shape == (num_classes, embed_dim))
    check("proxies 需要梯度", loss_fn.proxies.requires_grad)
    
    features = torch.randn(16, embed_dim, requires_grad=True)
    labels = torch.randint(0, num_classes, (16,))
    loss = loss_fn(features, labels)
    check("前向传播成功", loss.item() > 0, f"loss={loss.item():.4f}")
    
    loss.backward()
    check("特征梯度回传", features.grad is not None)
    check("proxy 梯度回传", loss_fn.proxies.grad is not None)
    check("proxy 梯度非零", loss_fn.proxies.grad.abs().sum().item() > 0)


# =============================================================================
# 5. NPairLoss 测试
# =============================================================================

def test_npair_loss():
    """测试 N-pair 损失"""
    print("\n[5] NPairLoss")
    from gastrovision.losses.metric_learning import NPairLoss
    
    loss_fn = NPairLoss(normalize=True, l2_reg=0.02)
    features = torch.randn(12, 128, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    
    loss = loss_fn(features, labels)
    check("前向传播成功", loss.item() > 0, f"loss={loss.item():.4f}")
    loss.backward()
    check("梯度回传成功", features.grad is not None)
    
    # 不使用 L2 正则
    loss_fn_no_reg = NPairLoss(normalize=True, l2_reg=0.0)
    features2 = torch.randn(12, 128, requires_grad=True)
    loss_no_reg = loss_fn_no_reg(features2, labels)
    check("无 L2 正则前向传播", loss_no_reg.item() > 0)
    
    # 所有不同类别（无正样本对）
    all_diff = torch.arange(12)
    features3 = torch.randn(12, 128)
    loss_no_pos = loss_fn(features3, all_diff)
    check("无正样本对返回 0", loss_no_pos.item() < 1e-5)


# =============================================================================
# 6. ArcFaceLoss 测试 (包含 CosFace, SphereFace)
# =============================================================================

def test_arcface_loss():
    """测试 ArcFace/CosFace/SphereFace 损失"""
    print("\n[6] ArcFaceLoss / CosFace / SphereFace")
    from gastrovision.losses.metric_learning import ArcFaceLoss
    
    num_classes = 10
    embed_dim = 128
    batch_size = 16
    
    # ArcFace
    arcface = ArcFaceLoss(num_classes=num_classes, embedding_dim=embed_dim,
                          scale=30.0, margin=0.5, margin_type='arcface')
    features = torch.randn(batch_size, embed_dim, requires_grad=True)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    loss = arcface(features, labels)
    check("ArcFace 前向传播", loss.item() > 0, f"loss={loss.item():.4f}")
    loss.backward()
    check("ArcFace 梯度回传", features.grad is not None)
    check("ArcFace 权重梯度", arcface.weight.grad is not None)
    
    # CosFace
    cosface = ArcFaceLoss(num_classes=num_classes, embedding_dim=embed_dim,
                          scale=30.0, margin=0.35, margin_type='cosface', m3=0.35)
    features2 = torch.randn(batch_size, embed_dim, requires_grad=True)
    loss_cos = cosface(features2, labels)
    check("CosFace 前向传播", loss_cos.item() > 0, f"loss={loss_cos.item():.4f}")
    loss_cos.backward()
    check("CosFace 梯度回传", features2.grad is not None)
    
    # SphereFace
    sphereface = ArcFaceLoss(num_classes=num_classes, embedding_dim=embed_dim,
                             scale=30.0, margin=0.0, margin_type='sphereface', m1=4.0)
    features3 = torch.randn(batch_size, embed_dim, requires_grad=True)
    loss_sphere = sphereface(features3, labels)
    check("SphereFace 前向传播", loss_sphere.item() >= 0, f"loss={loss_sphere.item():.4f}")
    loss_sphere.backward()
    check("SphereFace 梯度回传", features3.grad is not None)
    
    # Combined
    combined = ArcFaceLoss(num_classes=num_classes, embedding_dim=embed_dim,
                           scale=30.0, margin=0.3, margin_type='combined', m1=1.0, m3=0.1)
    features4 = torch.randn(batch_size, embed_dim, requires_grad=True)
    loss_combined = combined(features4, labels)
    check("Combined 前向传播", loss_combined.item() > 0, f"loss={loss_combined.item():.4f}")
    
    # Easy margin
    arcface_easy = ArcFaceLoss(num_classes=num_classes, embedding_dim=embed_dim,
                               scale=30.0, margin=0.5, easy_margin=True)
    features5 = torch.randn(batch_size, embed_dim, requires_grad=True)
    loss_easy = arcface_easy(features5, labels)
    check("ArcFace easy_margin 前向传播", loss_easy.item() > 0)


# =============================================================================
# 7. CircleLoss 测试
# =============================================================================

def test_circle_loss():
    """测试圆损失"""
    print("\n[7] CircleLoss (pair-level)")
    from gastrovision.losses.metric_learning import CircleLoss
    
    loss_fn = CircleLoss(margin=0.25, scale=256.0, normalize=True)
    features = torch.randn(12, 128, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    
    loss = loss_fn(features, labels)
    check("前向传播成功", loss.item() > 0, f"loss={loss.item():.4f}")
    loss.backward()
    check("梯度回传成功", features.grad is not None)
    
    # 无正样本对
    all_diff = torch.arange(12)
    features2 = torch.randn(12, 128)
    loss_no_pos = loss_fn(features2, all_diff)
    check("无正样本对返回 0", loss_no_pos.item() < 1e-5)


def test_circle_loss_class_level():
    """测试类级别圆损失"""
    print("\n[7b] CircleLossClassLevel")
    from gastrovision.losses.metric_learning import CircleLossClassLevel
    
    num_classes = 10
    embed_dim = 128
    loss_fn = CircleLossClassLevel(num_classes=num_classes, embedding_dim=embed_dim,
                                    margin=0.25, scale=256.0)
    
    check("含可学习权重", hasattr(loss_fn, 'weight'))
    check("权重形状正确", loss_fn.weight.shape == (num_classes, embed_dim))
    
    features = torch.randn(16, embed_dim, requires_grad=True)
    labels = torch.randint(0, num_classes, (16,))
    loss = loss_fn(features, labels)
    check("前向传播成功", loss.item() > 0, f"loss={loss.item():.4f}")
    loss.backward()
    check("特征梯度回传", features.grad is not None)
    check("权重梯度回传", loss_fn.weight.grad is not None)


# =============================================================================
# 8. 工厂函数测试
# =============================================================================

def test_factory_function():
    """测试 create_metric_loss 工厂函数"""
    print("\n[8] create_metric_loss 工厂函数")
    from gastrovision.losses.metric_learning import create_metric_loss
    
    num_classes = 10
    embed_dim = 128
    
    # 所有类型都能创建
    pair_losses = ['contrastive', 'triplet', 'lifted', 'npair', 'circle']
    for name in pair_losses:
        loss_fn = create_metric_loss(name)
        features = torch.randn(8, embed_dim)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = loss_fn(features, labels)
        check(f"工厂创建 {name}", loss.item() >= 0, f"loss={loss.item():.4f}")
    
    class_losses = ['proxy_nca', 'arcface', 'cosface', 'sphereface', 'circle_cls']
    for name in class_losses:
        loss_fn = create_metric_loss(name, num_classes=num_classes, embedding_dim=embed_dim)
        features = torch.randn(8, embed_dim)
        labels = torch.randint(0, num_classes, (8,))
        loss = loss_fn(features, labels)
        check(f"工厂创建 {name}", loss.item() >= 0, f"loss={loss.item():.4f}")
    
    # 无效类型应报错
    try:
        create_metric_loss('invalid_loss')
        check("无效类型报错", False, "未抛出 ValueError")
    except ValueError:
        check("无效类型报错", True)


# =============================================================================
# 9. __init__.py 导出测试
# =============================================================================

def test_init_exports():
    """测试 __init__.py 正确导出所有新损失"""
    print("\n[9] __init__.py 导出检查")
    from gastrovision.losses import (
        ContrastiveLoss, TripletMarginLoss, LiftedStructureLoss,
        ProxyNCALoss, NPairLoss, ArcFaceLoss,
        CircleLoss, CircleLossClassLevel, create_metric_loss
    )
    
    check("ContrastiveLoss 可导入", ContrastiveLoss is not None)
    check("TripletMarginLoss 可导入", TripletMarginLoss is not None)
    check("LiftedStructureLoss 可导入", LiftedStructureLoss is not None)
    check("ProxyNCALoss 可导入", ProxyNCALoss is not None)
    check("NPairLoss 可导入", NPairLoss is not None)
    check("ArcFaceLoss 可导入", ArcFaceLoss is not None)
    check("CircleLoss 可导入", CircleLoss is not None)
    check("CircleLossClassLevel 可导入", CircleLossClassLevel is not None)
    check("create_metric_loss 可导入", create_metric_loss is not None)


# =============================================================================
# 10. main.py 集成测试
# =============================================================================

def test_main_integration():
    """测试 main.py 的度量学习损失集成"""
    print("\n[10] main.py 集成检查")
    import re
    
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查 import
    check("main.py 导入 create_metric_loss",
          "from gastrovision.losses.metric_learning import create_metric_loss" in content)
    
    # 检查 argparse 参数
    check("--metric_loss 参数存在", "'--metric_loss'" in content)
    check("--metric_loss_weight 参数存在", "'--metric_loss_weight'" in content)
    check("--metric_loss_margin 参数存在", "'--metric_loss_margin'" in content)
    check("--metric_loss_scale 参数存在", "'--metric_loss_scale'" in content)
    check("--embedding_dim 参数存在", "'--embedding_dim'" in content)
    
    # 检查 create_metric_loss_function 辅助函数
    check("create_metric_loss_function 函数存在",
          "def create_metric_loss_function" in content)
    
    # 检查两条训练路径都集成了
    check("multilabel 路径有 metric_criterion",
          content.count("metric_criterion") >= 4)
    check("Trainer 支持 metric_loss 参数",
          "metric_loss=metric_criterion" in content)


# =============================================================================
# 11. Trainer 扩展测试
# =============================================================================

def test_trainer_metric_loss():
    """测试 Trainer 和 MultilabelTrainer 的度量学习损失支持"""
    print("\n[11] Trainer/MultilabelTrainer 度量学习扩展")
    
    # 检查 Trainer 源码
    with open('gastrovision/trainers/trainer.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    check("Trainer.__init__ 有 metric_loss 参数",
          "metric_loss:" in content or "metric_loss =" in content)
    check("Trainer.train_epoch 有度量学习损失计算",
          "self.metric_loss" in content)
    check("Trainer.validate 处理 tuple 输出",
          "isinstance(outputs, tuple)" in content)
    
    # 检查 MultilabelTrainer 源码
    with open('gastrovision/trainers/multilabel.py', 'r', encoding='utf-8') as f:
        ml_content = f.read()
    
    check("MultilabelTrainer.__init__ 有 metric_loss 参数",
          "metric_loss:" in ml_content or "self.metric_loss = metric_loss" in ml_content)
    check("MultilabelTrainer.train_epoch 有度量学习损失计算",
          "self.metric_loss_weight * ml_loss" in ml_content)
    check("MultilabelTrainer 打印 metric 信息",
          "metric_info" in ml_content)


# =============================================================================
# 12. GPU 兼容性测试
# =============================================================================

def test_gpu_compatibility():
    """测试 GPU 兼容性（如果可用）"""
    print("\n[12] GPU 兼容性")
    
    if not torch.cuda.is_available():
        print("  ⚠ GPU 不可用，跳过")
        return
    
    from gastrovision.losses.metric_learning import (
        ContrastiveLoss, TripletMarginLoss, ArcFaceLoss, CircleLoss, ProxyNCALoss
    )
    
    device = torch.device('cuda')
    
    # 测试 pair-level losses on GPU
    for LossClass, kwargs in [
        (ContrastiveLoss, {}),
        (TripletMarginLoss, {}),
        (CircleLoss, {}),
    ]:
        loss_fn = LossClass(**kwargs).to(device)
        features = torch.randn(8, 128, device=device, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=device)
        loss = loss_fn(features, labels)
        loss.backward()
        check(f"{LossClass.__name__} GPU 兼容", features.grad is not None)
    
    # 测试 class-level losses on GPU
    for LossClass, kwargs in [
        (ArcFaceLoss, {'num_classes': 10, 'embedding_dim': 128}),
        (ProxyNCALoss, {'num_classes': 10, 'embedding_dim': 128}),
    ]:
        loss_fn = LossClass(**kwargs).to(device)
        features = torch.randn(8, 128, device=device, requires_grad=True)
        labels = torch.randint(0, 10, (8,), device=device)
        loss = loss_fn(features, labels)
        loss.backward()
        check(f"{LossClass.__name__} GPU 兼容", features.grad is not None)


# =============================================================================
# 13. 数值稳定性测试
# =============================================================================

def test_numerical_stability():
    """测试数值稳定性"""
    print("\n[13] 数值稳定性")
    from gastrovision.losses.metric_learning import (
        ContrastiveLoss, ArcFaceLoss, CircleLoss, LiftedStructureLoss
    )
    
    # 极端特征值
    large_features = torch.randn(8, 128) * 100
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    
    for LossClass, name in [
        (ContrastiveLoss, "ContrastiveLoss"),
        (CircleLoss, "CircleLoss"),
        (LiftedStructureLoss, "LiftedStructureLoss"),
    ]:
        loss_fn = LossClass(normalize=True)
        loss = loss_fn(large_features, labels)
        check(f"{name} 大特征值无 nan/inf",
              not torch.isnan(loss).any() and not torch.isinf(loss).any(),
              f"loss={loss.item()}")
    
    # ArcFace 极端情况
    arcface = ArcFaceLoss(num_classes=4, embedding_dim=128, scale=30.0, margin=0.5)
    loss = arcface(large_features, labels)
    check("ArcFace 大特征值无 nan/inf",
          not torch.isnan(loss).any() and not torch.isinf(loss).any(),
          f"loss={loss.item()}")
    
    # 小 batch (B=2)
    small_features = torch.randn(2, 128)
    small_labels = torch.tensor([0, 1])
    contrastive = ContrastiveLoss()
    loss = contrastive(small_features, small_labels)
    check("ContrastiveLoss 小 batch 无崩溃", not torch.isnan(loss).any())


# =============================================================================
# 运行所有测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("度量学习损失函数测试")
    print("=" * 60)
    
    test_utility_functions()
    test_contrastive_loss()
    test_triplet_margin_loss()
    test_lifted_structure_loss()
    test_proxy_nca_loss()
    test_npair_loss()
    test_arcface_loss()
    test_circle_loss()
    test_circle_loss_class_level()
    test_factory_function()
    test_init_exports()
    test_main_integration()
    test_trainer_metric_loss()
    test_gpu_compatibility()
    test_numerical_stability()
    
    print("\n" + "=" * 60)
    print(f"结果: {PASSED} 通过, {FAILED} 失败")
    if FAILED_CASES:
        print("失败用例:")
        for idx, (name, detail) in enumerate(FAILED_CASES, 1):
            print(f"  {idx}. {name} :: {detail}")
    print("=" * 60)
    
    sys.exit(0 if FAILED == 0 else 1)
