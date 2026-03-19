"""
Jaguar Re-ID 训练入口

用法:
    python -m jaguar.train configs/jaguar_reid.yaml

纯 YAML 配置驱动。
"""

import warnings
warnings.filterwarnings('ignore', message='.*pynvml.*deprecated.*')

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader

# 确保项目根目录在 path 中
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gastrovision.utils.config import load_config, save_config
from gastrovision.utils.scheduler import get_scheduler
from gastrovision.losses.metric_learning import ArcFaceLoss

from jaguar.split import stratified_split, build_label_map
from jaguar.dataset import JaguarTrainDataset
from jaguar.transforms import get_train_transforms, get_val_transforms
from jaguar.model import build_model
from jaguar.trainer import ReIDTrainer
from jaguar.inference import extract_embeddings, predict_similarity_fast


def main():
    # ---- 加载配置 ----
    if len(sys.argv) < 2:
        print("用法: python -m jaguar.train <config.yaml>")
        sys.exit(1)

    cfg = load_config(sys.argv[1])

    # ---- 随机种子 ----
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # ---- 设备 ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("Jaguar Re-Identification Training")
    print("=" * 60)
    print(f"时间:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备:   {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"配置:   {cfg._config_path}")
    print()

    # ---- 输出目录 ----
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, str(output_dir / 'config.txt'))

    # ---- 数据划分 ----
    print("=" * 40)
    print("数据划分")
    print("=" * 40)
    train_csv = cfg.train_csv
    train_df, val_df = stratified_split(
        train_csv,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        output_dir=str(output_dir)
    )

    # 构建标签映射（从完整训练集）
    full_df = pd.concat([train_df, val_df])
    label_map = build_label_map(full_df)
    num_classes = len(label_map)
    print(f"\n类别数: {num_classes}")

    # 保存标签映射
    with open(output_dir / 'label_map.txt', 'w', encoding='utf-8') as f:
        for name, idx in sorted(label_map.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{name}\n")

    # ---- 数据增强 ----
    train_transform = get_train_transforms(
        image_size=cfg.image_size,
        augment_level=cfg.augment_level,
    )
    val_transform = get_val_transforms(image_size=cfg.image_size)

    # ---- 数据集 ----
    train_dataset = JaguarTrainDataset(
        train_df, cfg.train_image_dir, label_map, train_transform
    )
    val_dataset = JaguarTrainDataset(
        val_df, cfg.train_image_dir, label_map, val_transform
    )

    print(f"\n训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")

    # ---- DataLoader ----
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # ---- 模型 ----
    print(f"\n创建模型: {cfg.backbone}")
    model = build_model(cfg)
    model = model.to(device)

    # ---- ArcFace Head ----
    arcface_head = ArcFaceLoss(
        num_classes=num_classes,
        embedding_dim=cfg.embedding_dim,
        scale=cfg.arcface_scale,
        margin=cfg.arcface_margin,
    )
    arcface_head = arcface_head.to(device)
    print(f"ArcFace: scale={cfg.arcface_scale}, margin={cfg.arcface_margin}")

    # ---- 优化器 ----
    # 分组学习率：backbone 较低, embedding + arcface 较高
    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.embedding.parameters()) +
        list(model.bn_neck.parameters()) +
        list(arcface_head.parameters())
    )
    if hasattr(model, 'pool') and hasattr(model.pool, 'parameters'):
        head_params += list(model.pool.parameters())

    param_groups = [
        {'params': backbone_params, 'lr': cfg.lr * cfg.backbone_lr_mult},
        {'params': head_params, 'lr': cfg.lr},
    ]

    if cfg.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.Adam(param_groups, weight_decay=cfg.weight_decay)

    print(f"优化器: {cfg.optimizer} (backbone_lr={cfg.lr * cfg.backbone_lr_mult:.6f}, head_lr={cfg.lr:.6f})")

    # ---- 调度器 ----
    scheduler = get_scheduler(
        optimizer, cfg.scheduler, cfg.epochs,
        steps_per_epoch=len(train_loader),
        warmup_epochs=cfg.warmup_epochs
    )
    if scheduler:
        print(f"调度器: {cfg.scheduler} (warmup={cfg.warmup_epochs})")

    # ---- 训练器 ----
    trainer = ReIDTrainer(
        model=model,
        arcface_head=arcface_head,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        output_dir=str(output_dir),
        use_ema=cfg.ema,
        ema_decay=cfg.ema_decay,
    )

    # ---- 恢复训练 ----
    if cfg.resume:
        trainer.load_checkpoint(cfg.resume)

    # ---- 测试模式 ----
    if cfg.test_only:
        if not cfg.resume:
            # 尝试加载 best_model.pth
            trainer.load_checkpoint('best_model.pth')
        _run_inference(cfg, model, device, output_dir)
        return

    # ---- 训练 ----
    print(f"\n{'=' * 40}")
    print("开始训练")
    print(f"{'=' * 40}\n")

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.epochs,
        early_stopping=cfg.early_stopping,
    )

    # ---- 推理 ----
    print(f"\n{'=' * 40}")
    print("加载最佳模型进行推理")
    print(f"{'=' * 40}")

    trainer.load_checkpoint('best_model.pth')
    _run_inference(cfg, model, device, output_dir)

    print("\n完成!")


def _run_inference(cfg, model, device, output_dir):
    """运行推理并生成提交文件"""
    test_csv = getattr(cfg, 'test_csv', '')
    test_image_dir = getattr(cfg, 'test_image_dir', '')

    if not test_csv or not test_image_dir:
        print("[跳过推理] 未配置 test_csv 或 test_image_dir")
        return

    if not Path(test_csv).exists():
        print(f"[跳过推理] test_csv 不存在: {test_csv}")
        return

    # 获取所有需要的测试图片
    test_df = pd.read_csv(test_csv)
    all_test_images = sorted(set(
        test_df['query_image'].tolist() + test_df['gallery_image'].tolist()
    ))
    print(f"测试图片数: {len(all_test_images)}")

    val_transform = get_val_transforms(image_size=cfg.image_size)
    use_flip_tta = getattr(cfg, 'use_flip_tta', True)

    embeddings = extract_embeddings(
        model=model,
        image_dir=test_image_dir,
        image_names=all_test_images,
        transform=val_transform,
        device=device,
        batch_size=cfg.batch_size * 2,
        num_workers=cfg.num_workers,
        use_flip=use_flip_tta,
    )

    submission_path = str(output_dir / 'submission.csv')
    predict_similarity_fast(
        embeddings=embeddings,
        test_csv_path=test_csv,
        output_path=submission_path,
    )


if __name__ == '__main__':
    main()
