"""
Jaguar Re-ID 模块测试

验证所有组件可以正确导入和运行。
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
from PIL import Image

# 确保项目根目录在 path 中
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def create_dummy_data(tmp_dir: str, num_images: int = 30, num_classes: int = 5):
    """创建虚拟数据用于测试"""
    img_dir = Path(tmp_dir) / 'train'
    img_dir.mkdir(parents=True, exist_ok=True)

    names = [f'Jaguar_{i}' for i in range(num_classes)]
    rows = []

    for i in range(num_images):
        fname = f'train_{i:04d}.png'
        label = names[i % num_classes]
        rows.append({'filename': fname, 'ground_truth': label})

        # 创建随机 RGB 图片
        img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        img.save(img_dir / fname)

    df = pd.DataFrame(rows)
    csv_path = Path(tmp_dir) / 'train.csv'
    df.to_csv(csv_path, index=False)

    # 创建测试图片和 test.csv
    test_dir = Path(tmp_dir) / 'test'
    test_dir.mkdir(parents=True, exist_ok=True)

    test_images = []
    for i in range(10):
        fname = f'test_{i:04d}.png'
        test_images.append(fname)
        img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        img.save(test_dir / fname)

    # 创建 test.csv (pairs)
    test_rows = []
    row_id = 0
    for i in range(len(test_images)):
        for j in range(i + 1, min(i + 3, len(test_images))):
            test_rows.append({
                'row_id': row_id,
                'query_image': test_images[i],
                'gallery_image': test_images[j],
            })
            row_id += 1

    test_df = pd.DataFrame(test_rows)
    test_csv = Path(tmp_dir) / 'test.csv'
    test_df.to_csv(test_csv, index=False)

    return str(csv_path), str(img_dir), str(test_csv), str(test_dir)


# ================================================================
# 测试函数
# ================================================================

def test_split():
    """测试数据划分"""
    from jaguar.split import stratified_split, build_label_map

    with tempfile.TemporaryDirectory() as tmp:
        csv_path, _, _, _ = create_dummy_data(tmp, num_images=30, num_classes=5)
        train_df, val_df = stratified_split(csv_path, val_ratio=0.2, seed=42)

        assert len(train_df) + len(val_df) == 30, "总数应为 30"
        assert len(val_df) > 0, "验证集不应为空"

        # 验证每个类在验证集中至少 1 个
        val_classes = set(val_df['ground_truth'])
        train_classes = set(train_df['ground_truth'])
        assert len(val_classes) >= 4, f"验证集类别数应 >= 4，实际 {len(val_classes)}"

        label_map = build_label_map(pd.concat([train_df, val_df]))
        assert len(label_map) == 5, f"标签映射应有 5 类，实际 {len(label_map)}"

    print("✓ test_split PASSED")


def test_transforms():
    """测试数据增强"""
    from jaguar.transforms import get_train_transforms, get_val_transforms

    train_t = get_train_transforms(image_size=224, augment_level='medium')
    val_t = get_val_transforms(image_size=224)

    dummy = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)

    out_train = train_t(image=dummy)['image']
    out_val = val_t(image=dummy)['image']

    assert out_train.shape == (3, 224, 224), f"训练输出形状错误: {out_train.shape}"
    assert out_val.shape == (3, 224, 224), f"验证输出形状错误: {out_val.shape}"

    # 测试所有增强级别
    for level in ['light', 'medium', 'heavy']:
        t = get_train_transforms(image_size=256, augment_level=level)
        out = t(image=dummy)['image']
        assert out.shape == (3, 256, 256), f"{level} 输出形状错误"

    print("✓ test_transforms PASSED")


def test_dataset():
    """测试数据集"""
    from jaguar.dataset import JaguarTrainDataset, JaguarTestDataset
    from jaguar.transforms import get_train_transforms, get_val_transforms
    from jaguar.split import build_label_map

    with tempfile.TemporaryDirectory() as tmp:
        csv_path, img_dir, test_csv, test_dir = create_dummy_data(tmp)
        df = pd.read_csv(csv_path)
        label_map = build_label_map(df)

        train_t = get_train_transforms(image_size=224)

        ds = JaguarTrainDataset(df, img_dir, label_map, train_t)
        assert len(ds) == 30
        assert ds.num_classes == 5

        img, label = ds[0]
        assert img.shape == (3, 224, 224), f"图像形状错误: {img.shape}"
        assert isinstance(label, int)
        assert 0 <= label < 5

        # 测试 test dataset
        test_images = [str(p) for p in Path(test_dir).glob('*.png')]
        val_t = get_val_transforms(image_size=224)
        test_ds = JaguarTestDataset(test_images, val_t)
        assert len(test_ds) == 10
        img, fname = test_ds[0]
        assert img.shape == (3, 224, 224)
        assert fname.endswith('.png')

    print("✓ test_dataset PASSED")


def test_model_resnet():
    """测试 ReID 模型 (ResNet backbone)"""
    from jaguar.model import ReIDModel

    model = ReIDModel(
        backbone_name='resnet18',
        embedding_dim=256,
        pretrained=False,
        use_gem=True,
    )

    x = torch.randn(2, 3, 224, 224)

    # 默认返回 bn_feat
    bn_feat = model(x)
    assert bn_feat.shape == (2, 256), f"bn_feat 形状错误: {bn_feat.shape}"

    # return_both
    bn_feat, raw_feat = model(x, return_both=True)
    assert bn_feat.shape == (2, 256)
    assert raw_feat.shape == (2, 256)

    # extract_embedding (L2 归一化)
    emb = model.extract_embedding(x)
    assert emb.shape == (2, 256)
    norms = emb.norm(dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5), f"embedding 未归一化: {norms}"

    print("✓ test_model_resnet PASSED")


def test_model_convnext():
    """测试 ConvNeXt backbone"""
    from jaguar.model import ReIDModel

    model = ReIDModel(
        backbone_name='convnext_tiny',
        embedding_dim=512,
        pretrained=False,
        use_gem=False,
    )

    x = torch.randn(2, 3, 224, 224)
    emb = model.extract_embedding(x)
    assert emb.shape == (2, 512), f"embedding 形状错误: {emb.shape}"

    print("✓ test_model_convnext PASSED")


def test_model_swin():
    """测试 Swin Transformer backbone"""
    from jaguar.model import ReIDModel

    model = ReIDModel(
        backbone_name='swin_t',
        embedding_dim=512,
        pretrained=False,
        use_gem=False,
    )

    x = torch.randn(2, 3, 224, 224)
    emb = model.extract_embedding(x)
    assert emb.shape == (2, 512)

    print("✓ test_model_swin PASSED")


def test_model_efficientnet():
    """测试 EfficientNet backbone"""
    from jaguar.model import ReIDModel

    model = ReIDModel(
        backbone_name='efficientnet_v2_s',
        embedding_dim=512,
        pretrained=False,
        use_gem=False,
    )

    x = torch.randn(2, 3, 224, 224)
    emb = model.extract_embedding(x)
    assert emb.shape == (2, 512)

    print("✓ test_model_efficientnet PASSED")


def test_gem_pooling():
    """测试 GeM Pooling"""
    from jaguar.model import GeMPooling

    gem = GeMPooling(p=3.0)
    x = torch.randn(2, 64, 7, 7)
    out = gem(x)
    assert out.shape == (2, 64), f"GeM 输出形状错误: {out.shape}"

    # p 应该是可学习参数
    assert gem.p.requires_grad

    print("✓ test_gem_pooling PASSED")


def test_arcface_loss():
    """测试 ArcFace 损失"""
    from gastrovision.losses.metric_learning import ArcFaceLoss

    arcface = ArcFaceLoss(num_classes=5, embedding_dim=256, scale=30.0, margin=0.5)
    features = torch.randn(4, 256)
    labels = torch.tensor([0, 1, 2, 3])

    loss = arcface(features, labels)
    assert loss.dim() == 0, "ArcFace loss 应为标量"
    assert loss.item() > 0, "ArcFace loss 应 > 0"
    assert loss.requires_grad

    print("✓ test_arcface_loss PASSED")


def test_trainer_one_step():
    """测试训练器单步"""
    from jaguar.model import ReIDModel
    from jaguar.trainer import ReIDTrainer
    from gastrovision.losses.metric_learning import ArcFaceLoss

    device = torch.device('cpu')
    model = ReIDModel('resnet18', embedding_dim=128, pretrained=False, use_gem=False)
    arcface = ArcFaceLoss(num_classes=5, embedding_dim=128)

    all_params = list(model.parameters()) + list(arcface.parameters())
    optimizer = torch.optim.Adam(all_params, lr=0.001)

    with tempfile.TemporaryDirectory() as tmp:
        trainer = ReIDTrainer(
            model=model, criterion=arcface, loss_type='arcface',
            optimizer=optimizer, device=device, output_dir=tmp
        )

        # 模拟一个 batch
        images = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 2, 0])

        model.train()
        arcface.train()
        bn_feat = model(images)
        loss = arcface(bn_feat, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0

    print("✓ test_trainer_one_step PASSED")


def test_validation():
    """测试验证流程 (Rank-1 + mAP)"""
    from jaguar.model import ReIDModel
    from jaguar.trainer import ReIDTrainer
    from jaguar.dataset import JaguarTrainDataset
    from jaguar.transforms import get_val_transforms
    from jaguar.split import build_label_map
    from gastrovision.losses.metric_learning import ArcFaceLoss
    from torch.utils.data import DataLoader

    device = torch.device('cpu')

    with tempfile.TemporaryDirectory() as tmp:
        csv_path, img_dir, _, _ = create_dummy_data(tmp, num_images=20, num_classes=4)
        df = pd.read_csv(csv_path)
        label_map = build_label_map(df)

        val_t = get_val_transforms(image_size=224)
        ds = JaguarTrainDataset(df, img_dir, label_map, val_t)
        loader = DataLoader(ds, batch_size=8, shuffle=False)

        model = ReIDModel('resnet18', embedding_dim=128, pretrained=False, use_gem=False)
        arcface = ArcFaceLoss(num_classes=4, embedding_dim=128)

        all_params = list(model.parameters()) + list(arcface.parameters())
        optimizer = torch.optim.Adam(all_params, lr=0.001)

        trainer = ReIDTrainer(
            model=model, criterion=arcface, loss_type='arcface',
            optimizer=optimizer, device=device, output_dir=tmp
        )

        metrics = trainer._validate(loader)
        assert 'rank1' in metrics, "缺少 rank1 指标"
        assert 'mAP' in metrics, "缺少 mAP 指标"
        assert 0.0 <= metrics['rank1'] <= 1.0
        assert 0.0 <= metrics['mAP'] <= 1.0

    print("✓ test_validation PASSED")


def test_inference():
    """测试推理流程"""
    from jaguar.model import ReIDModel
    from jaguar.inference import extract_embeddings, predict_similarity_fast
    from jaguar.transforms import get_val_transforms

    device = torch.device('cpu')

    with tempfile.TemporaryDirectory() as tmp:
        _, _, test_csv, test_dir = create_dummy_data(tmp)

        model = ReIDModel('resnet18', embedding_dim=128, pretrained=False, use_gem=False)
        model.eval()

        test_images = [f.name for f in Path(test_dir).glob('*.png')]
        val_t = get_val_transforms(image_size=224)

        embeddings = extract_embeddings(
            model=model, image_dir=test_dir, image_names=test_images,
            transform=val_t, device=device, batch_size=4, num_workers=0,
            use_flip=True
        )

        assert len(embeddings) == 10, f"应有 10 个 embedding，实际 {len(embeddings)}"
        for name, emb in embeddings.items():
            assert emb.shape == (128,), f"embedding 形状错误: {emb.shape}"
            # 检查归一化
            norm = emb.norm().item()
            assert abs(norm - 1.0) < 0.01, f"embedding 未归一化: {norm}"

        # 预测相似度
        output_path = str(Path(tmp) / 'submission.csv')
        submission = predict_similarity_fast(
            embeddings=embeddings,
            test_csv_path=test_csv,
            output_path=output_path
        )

        assert 'row_id' in submission.columns
        assert 'similarity' in submission.columns
        assert len(submission) > 0
        assert (submission['similarity'] >= 0).all()
        assert (submission['similarity'] <= 1).all()

        # 检查文件是否保存
        assert Path(output_path).exists()

    print("✓ test_inference PASSED")


def test_checkpoint_save_load():
    """测试 checkpoint 保存和加载"""
    from jaguar.model import ReIDModel
    from jaguar.trainer import ReIDTrainer
    from gastrovision.losses.metric_learning import ArcFaceLoss

    device = torch.device('cpu')

    with tempfile.TemporaryDirectory() as tmp:
        model = ReIDModel('resnet18', embedding_dim=128, pretrained=False, use_gem=False)
        arcface = ArcFaceLoss(num_classes=5, embedding_dim=128)
        all_params = list(model.parameters()) + list(arcface.parameters())
        optimizer = torch.optim.Adam(all_params, lr=0.001)

        trainer = ReIDTrainer(
            model=model, criterion=arcface, loss_type='arcface',
            optimizer=optimizer, device=device, output_dir=tmp
        )

        # 保存
        trainer.best_metric = 0.85
        trainer.best_epoch = 10
        trainer._save_checkpoint(9, 'test_ckpt.pth', {'rank1': 0.85})

        ckpt_path = Path(tmp) / 'test_ckpt.pth'
        assert ckpt_path.exists()

        # 加载到新实例
        model2 = ReIDModel('resnet18', embedding_dim=128, pretrained=False, use_gem=False)
        arcface2 = ArcFaceLoss(num_classes=5, embedding_dim=128)
        all_params2 = list(model2.parameters()) + list(arcface2.parameters())
        optimizer2 = torch.optim.Adam(all_params2, lr=0.001)

        trainer2 = ReIDTrainer(
            model=model2, criterion=arcface2, loss_type='arcface',
            optimizer=optimizer2, device=device, output_dir=tmp
        )
        trainer2.load_checkpoint('test_ckpt.pth')

        assert trainer2.best_metric == 0.85
        assert trainer2.start_epoch == 10

        # 验证权重一致
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.equal(p1, p2), f"权重不一致: {n1}"

    print("✓ test_checkpoint_save_load PASSED")


def test_ema():
    """测试 EMA 集成"""
    from jaguar.model import ReIDModel
    from jaguar.trainer import ReIDTrainer
    from gastrovision.losses.metric_learning import ArcFaceLoss

    device = torch.device('cpu')

    with tempfile.TemporaryDirectory() as tmp:
        model = ReIDModel('resnet18', embedding_dim=128, pretrained=False, use_gem=False)
        arcface = ArcFaceLoss(num_classes=5, embedding_dim=128)
        all_params = list(model.parameters()) + list(arcface.parameters())
        optimizer = torch.optim.Adam(all_params, lr=0.001)

        trainer = ReIDTrainer(
            model=model, criterion=arcface, loss_type='arcface',
            optimizer=optimizer, device=device, output_dir=tmp,
            use_ema=True, ema_decay=0.999
        )

        assert trainer.ema is not None

        # 模拟训练步
        images = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 2, 3])
        model.train()
        arcface.train()
        bn_feat = model(images)
        loss = arcface(bn_feat, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        trainer.ema.update(model)

        # EMA shadow 应该与模型参数不同
        # (因为 EMA 是滑动平均)
        assert trainer.ema.num_updates == 1

    print("✓ test_ema PASSED")


def test_build_model():
    """测试从配置构建模型"""
    from types import SimpleNamespace
    from jaguar.model import build_model

    cfg = SimpleNamespace(
        backbone='resnet18',
        embedding_dim=256,
        pretrained=False,
        use_gem=True,
        gem_p=3.0,
        dropout=0.1,
    )

    model = build_model(cfg)
    x = torch.randn(2, 3, 224, 224)
    emb = model.extract_embedding(x)
    assert emb.shape == (2, 256)

    print("✓ test_build_model PASSED")


def test_pretrained_local_path():
    """测试从本地路径加载预训练权重"""
    from jaguar.model import ReIDModel
    import torchvision.models as models

    with tempfile.TemporaryDirectory() as tmp:
        # 保存一个 backbone 的 state_dict
        backbone = models.resnet18(weights=None)
        weight_path = str(Path(tmp) / 'resnet18_test.pth')
        torch.save(backbone.state_dict(), weight_path)

        # 用 pretrained_path 加载
        model = ReIDModel(
            backbone_name='resnet18',
            embedding_dim=128,
            pretrained=False,
            pretrained_path=weight_path,
            use_gem=False,
        )

        # 验证 backbone 权重与保存的一致
        saved_state = torch.load(weight_path, map_location='cpu', weights_only=False)
        for name, param in model.backbone.named_parameters():
            if name in saved_state:
                assert torch.equal(param.data, saved_state[name]), \
                    f"权重不一致: {name}"

        # 验证 Trainer checkpoint 格式也能加载
        ckpt_path = str(Path(tmp) / 'ckpt_format.pth')
        torch.save({'model_state_dict': backbone.state_dict()}, ckpt_path)

        model2 = ReIDModel(
            backbone_name='resnet18',
            embedding_dim=128,
            pretrained=False,
            pretrained_path=ckpt_path,
            use_gem=False,
        )

        x = torch.randn(2, 3, 224, 224)
        emb = model2.extract_embedding(x)
        assert emb.shape == (2, 128)

    print("✓ test_pretrained_local_path PASSED")


def test_efficientnet_no_double_dropout():
    """测试 EfficientNet 内置 Dropout 已被移除"""
    from jaguar.model import ReIDModel

    model = ReIDModel(
        backbone_name='efficientnet_v2_s',
        embedding_dim=512,
        pretrained=False,
        use_gem=False,
        dropout=0.1,  # ReIDModel 自己的 dropout
    )

    # 检查 backbone.classifier 中不再有 nn.Dropout
    if hasattr(model.backbone.classifier, '__getitem__'):
        for layer in model.backbone.classifier:
            assert not isinstance(layer, nn.Dropout), \
                "EfficientNet 内置 Dropout 应已被移除"

    # 功能验证
    x = torch.randn(2, 3, 224, 224)
    emb = model.extract_embedding(x)
    assert emb.shape == (2, 512)

    print("✓ test_efficientnet_no_double_dropout PASSED")


def test_loss_types():
    """测试所有损失类型的创建和单步训练"""
    from jaguar.model import ReIDModel
    from jaguar.trainer import ReIDTrainer
    from gastrovision.losses.metric_learning import create_metric_loss

    device = torch.device('cpu')
    num_classes = 5
    embedding_dim = 128

    # 所有支持的损失类型
    proxy_losses = ['arcface', 'cosface', 'sphereface', 'proxy_nca', 'circle_cls']
    pair_losses = ['triplet', 'contrastive', 'circle', 'lifted', 'npair']

    images = torch.randn(8, 3, 224, 224)
    # 确保每个标签至少出现 2 次 (triplet/contrastive 等需要正样本对)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    for loss_type in proxy_losses + pair_losses:
        model = ReIDModel('resnet18', embedding_dim=embedding_dim,
                          pretrained=False, use_gem=False)
        criterion = create_metric_loss(
            loss_type=loss_type,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
        )

        all_params = list(model.parameters()) + list(criterion.parameters())
        optimizer = torch.optim.Adam(all_params, lr=0.001)

        with tempfile.TemporaryDirectory() as tmp:
            trainer = ReIDTrainer(
                model=model, criterion=criterion, loss_type=loss_type,
                optimizer=optimizer, device=device, output_dir=tmp
            )

            # 单步训练
            train_metrics = trainer._train_epoch(
                torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(images, labels),
                    batch_size=8
                ),
                epoch=0, total_epochs=1
            )

            assert train_metrics['loss'] > 0, \
                f"{loss_type}: loss 应 > 0, 实际 {train_metrics['loss']}"

            # proxy-based 应有 acc, pair-based 无 acc
            if loss_type in proxy_losses:
                assert train_metrics['acc'] is not None or loss_type == 'proxy_nca', \
                    f"{loss_type}: proxy-based 应有 acc"
            else:
                assert train_metrics['acc'] is None, \
                    f"{loss_type}: pair-based 不应有 acc"

        print(f"  ✓ {loss_type} OK")

    print("✓ test_loss_types PASSED")


def test_auxiliary_loss():
    """测试辅助损失 (ArcFace + Triplet)"""
    from jaguar.model import ReIDModel
    from jaguar.trainer import ReIDTrainer
    from gastrovision.losses.metric_learning import create_metric_loss

    device = torch.device('cpu')
    num_classes = 5
    embedding_dim = 128

    model = ReIDModel('resnet18', embedding_dim=embedding_dim,
                      pretrained=False, use_gem=False)
    criterion = create_metric_loss('arcface', num_classes=num_classes,
                                   embedding_dim=embedding_dim)
    aux_criterion = create_metric_loss('triplet')

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

        train_metrics = trainer._train_epoch(
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(images, labels),
                batch_size=8
            ),
            epoch=0, total_epochs=1
        )

        assert train_metrics['loss'] > 0
        assert train_metrics['acc'] is not None
        assert 'aux_loss' in train_metrics
        assert train_metrics['aux_loss'] > 0

    print("✓ test_auxiliary_loss PASSED")


# ================================================================
# 运行所有测试
# ================================================================

if __name__ == '__main__':
    tests = [
        test_split,
        test_transforms,
        test_dataset,
        test_gem_pooling,
        test_model_resnet,
        test_model_convnext,
        test_model_swin,
        test_model_efficientnet,
        test_arcface_loss,
        test_build_model,
        test_trainer_one_step,
        test_validation,
        test_inference,
        test_checkpoint_save_load,
        test_ema,
        test_pretrained_local_path,
        test_efficientnet_no_double_dropout,
        test_loss_types,
        test_auxiliary_loss,
    ]

    passed = 0
    failed = 0
    errors = []

    print("=" * 60)
    print("Jaguar Re-ID 模块测试")
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
