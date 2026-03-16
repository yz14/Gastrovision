"""
鲸鱼分类训练入口脚本

用法:
    python train_whale.py configs/train_whale.yaml

YAML 驱动, 所有参数通过配置文件控制。
"""

import os
import sys
import yaml
import argparse
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gastrovision.models.whale_model import get_whale_model
from gastrovision.losses.whale_losses import WhaleCompositeLoss
from gastrovision.losses.loss_factory import create_metric_loss_function
from gastrovision.data.whale_dataset import (
    WhaleDataset,
    WhaleRandomIdentitySampler
)
from gastrovision.trainers.whale_trainer import WhaleTrainer


def load_config(yaml_path: str) -> SimpleNamespace:
    """加载 YAML 配置文件"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)

    cfg = SimpleNamespace(**cfg_dict)

    # 设置默认值
    defaults = {
        # 数据
        'whale_data_dir': 'D:/codes/data/humpback-whale-identification',
        'list_dir': 'D:/codes/work-projects/colonnav_ssl/image_list',
        'bbox_dir': 'D:/codes/work-projects/colonnav_ssl/bbox_model',
        'fold_index': 0,
        'image_h': 256,
        'image_w': 512,
        'whale_id_num': 5004,
        'is_pseudo': False,
        'NW_ratio': 0.25,

        # 模型
        'model': 'resnet101',
        's1': 64.,
        'm1': 0.5,
        's2': 16.,
        'pretrained': True,
        'freeze_layers': ['layer0', 'layer1'],

        # 原始损失
        'focal_w': 1.0,
        'softmax_w': 0.1,
        'triplet_w': 1.0,
        'triplet_margin': 0.3,

        # Gastrovision 度量学习 (可替换 triplet)
        'metric_loss': 'none',
        'metric_loss_weight': 0.5,
        'metric_loss_margin': 0.5,
        'embedding_dim': 2048,

        # 训练
        'epochs': 100,
        'batch_size': 32,
        'num_instances': 4,
        'lr': 0.001,
        'weight_decay': 1e-4,
        'scheduler': 'cosine',
        'warmup_epochs': 5,
        'early_stopping': 0,

        # OHEM
        'initial_hard_ratio': 1.0,
        'min_hard_ratio': 0.2,
        'hard_ratio_step': 0.1,

        # 运行
        'mode': 'train',
        'pretrained_model': '',
        'output_dir': './output/whale',
        'num_workers': 4,
        'seed': 42,
    }

    for key, default_value in defaults.items():
        if not hasattr(cfg, key):
            setattr(cfg, key, default_value)

    return cfg


def set_seed(seed: int):
    """设置随机种子"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_optimizer(model, composite_loss, cfg):
    """创建优化器"""
    # 模型参数
    params = [
        {'params': model.parameters(), 'lr': cfg.lr}
    ]

    # 可学习损失参数 (如 Gastrovision 度量学习损失的权重)
    loss_params = composite_loss.parameters()
    if loss_params:
        params.append({
            'params': loss_params,
            'lr': cfg.lr * 0.1  # 损失参数用更小的学习率
        })

    optimizer = torch.optim.Adam(
        params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    return optimizer


def create_scheduler(optimizer, cfg, steps_per_epoch):
    """创建学习率调度器"""
    scheduler_type = getattr(cfg, 'scheduler', 'cosine')

    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs,
            eta_min=cfg.lr * 0.01
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, cfg.epochs // 3),
            gamma=0.1
        )
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=10,
            factor=0.5
        )
    elif scheduler_type == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            steps_per_epoch=steps_per_epoch,
            epochs=cfg.epochs,
            pct_start=0.1
        )
    else:
        scheduler = None
        print(f"  未知调度器: {scheduler_type}, 不使用学习率调度")

    return scheduler


def main():
    parser = argparse.ArgumentParser(description='鲸鱼分类训练')
    parser.add_argument('config', type=str, help='YAML 配置文件路径')
    args = parser.parse_args()

    # 加载配置
    cfg = load_config(args.config)

    print("=" * 60)
    print("鲸鱼分类训练")
    print("=" * 60)
    print(f"配置文件: {args.config}")
    print(f"模式: {cfg.mode}")
    print()

    # 设置随机种子
    set_seed(cfg.seed)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # ---- 创建模型 ----
    print("\n创建模型...")
    model = get_whale_model(cfg)
    model = model.to(device)

    # 加载预训练权重 (如果指定)
    if cfg.pretrained_model and os.path.isfile(cfg.pretrained_model):
        print(f"加载预训练模型: {cfg.pretrained_model}")
        state_dict = torch.load(
            cfg.pretrained_model, map_location='cpu', weights_only=False)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)

    # ---- 创建损失 ----
    print("\n创建损失函数...")

    # Gastrovision 度量学习损失 (可选, 替代 triplet)
    metric_loss = None
    if cfg.metric_loss and cfg.metric_loss.lower() != 'none':
        metric_loss = create_metric_loss_function(
            metric_loss_type=cfg.metric_loss,
            num_classes=cfg.whale_id_num * 2,
            embedding_dim=cfg.embedding_dim,
            margin=cfg.metric_loss_margin,
            device=device
        )
        print(f"  使用 Gastrovision 度量学习: {cfg.metric_loss}")

    composite_loss = WhaleCompositeLoss(
        focal_w=cfg.focal_w,
        softmax_w=cfg.softmax_w,
        triplet_w=cfg.triplet_w,
        triplet_margin=cfg.triplet_margin,
        metric_loss=metric_loss,
        metric_loss_weight=cfg.metric_loss_weight
    )

    # ---- 创建数据集 ----
    print("\n加载数据...")
    image_size = (cfg.image_h, cfg.image_w)

    train_dataset = WhaleDataset(
        mode='train',
        data_dir=cfg.whale_data_dir,
        list_dir=cfg.list_dir,
        bbox_dir=cfg.bbox_dir,
        fold_index=cfg.fold_index,
        image_size=image_size,
        is_pseudo=cfg.is_pseudo
    )

    val_dataset = WhaleDataset(
        mode='val',
        data_dir=cfg.whale_data_dir,
        list_dir=cfg.list_dir,
        bbox_dir=cfg.bbox_dir,
        fold_index=cfg.fold_index,
        image_size=image_size
    )

    # ---- 创建采样器和 DataLoader ----
    print("\n创建数据加载器...")

    # 采样器需要 train_list 模式
    sampler_dataset = WhaleDataset(
        mode='train_list',
        data_dir=cfg.whale_data_dir,
        list_dir=cfg.list_dir,
        fold_index=cfg.fold_index,
        image_size=image_size,
        is_pseudo=cfg.is_pseudo
    )

    sampler = WhaleRandomIdentitySampler(
        data_source=sampler_dataset,
        batch_size=cfg.batch_size,
        num_instances=cfg.num_instances,
        NW_ratio=cfg.NW_ratio
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0
    )

    # ---- 创建优化器和调度器 ----
    print("\n创建优化器...")
    optimizer = create_optimizer(model, composite_loss, cfg)
    scheduler = create_scheduler(optimizer, cfg, len(train_loader))

    print(f"  优化器: Adam (lr={cfg.lr}, weight_decay={cfg.weight_decay})")
    if scheduler:
        print(f"  调度器: {cfg.scheduler}")

    # ---- 训练 ----
    if cfg.mode == 'train':
        trainer = WhaleTrainer(
            model=model,
            composite_loss=composite_loss,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            output_dir=cfg.output_dir,
            whale_id_num=cfg.whale_id_num
        )

        history = trainer.fit(
            train_loader=train_loader,
            val_dataset=val_dataset,
            epochs=cfg.epochs,
            val_batch_size=cfg.batch_size,
            val_num_workers=cfg.num_workers,
            initial_hard_ratio=cfg.initial_hard_ratio,
            min_hard_ratio=cfg.min_hard_ratio,
            hard_ratio_step=cfg.hard_ratio_step,
            early_stopping=cfg.early_stopping
        )

        print("\n✓ 训练完成!")

    elif cfg.mode == 'test':
        print("\n推理模式 (下一轮实现)")
        print("请使用 mode: train 开始训练")

    else:
        print(f"未知模式: {cfg.mode}")


if __name__ == '__main__':
    main()
