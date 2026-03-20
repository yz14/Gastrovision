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
from gastrovision.losses.metric_learning import create_metric_loss

from jaguar.split import stratified_split, build_label_map
from jaguar.dataset import JaguarTrainDataset
from jaguar.transforms import get_train_transforms, get_val_transforms
from jaguar.model import build_model
from jaguar.trainer import ReIDTrainer, is_proxy_loss
from jaguar.sampler import PKSampler
from jaguar.inference import extract_embeddings, predict_similarity_fast


# 每种损失函数的合理默认参数
# 用户如果在 YAML 中显式设置了 loss_scale/loss_margin，则覆盖这些默认值
LOSS_DEFAULTS = {
    'arcface':     {'scale': 30.0,  'margin': 0.5},
    'cosface':     {'scale': 30.0,  'margin': 0.35},
    'sphereface':  {'scale': 30.0,  'margin': 1.5},
    'proxy_nca':   {'scale': 8.0,   'margin': 0.0},
    'circle_cls':  {'scale': 256.0, 'margin': 0.25},
    'triplet':     {'scale': 1.0,   'margin': 0.3},
    'contrastive': {'scale': 1.0,   'margin': 1.0},
    'circle':      {'scale': 256.0, 'margin': 0.25},
    'lifted':      {'scale': 1.0,   'margin': 1.0},
    'npair':       {'scale': 1.0,   'margin': 0.0},
}


def _create_loss(cfg, loss_type: str, num_classes: int, embedding_dim: int, prefix: str = ''):
    """
    根据配置创建损失函数

    优先级：
      1. 用户在 YAML 中的 loss-type-specific 配置 (如 arcface_scale)
      2. LOSS_DEFAULTS 中该损失的合理默认值
      3. create_metric_loss 工厂函数的内置默认值

    不再使用全局 loss_scale / loss_margin，因为不同损失的参数含义和范围不同。

    Args:
        cfg: 配置对象
        loss_type: 损失类型名称
        num_classes: 类别数
        embedding_dim: embedding 维度
        prefix: 配置参数前缀 (用于辅助损失: 'aux_')

    Returns:
        nn.Module 损失函数
    """
    lt = loss_type.lower()
    defaults = LOSS_DEFAULTS.get(lt, {'scale': 30.0, 'margin': 0.5})

    # 优先读取 loss-type-specific 配置 (如 arcface_scale, triplet_margin)
    # 回退到 LOSS_DEFAULTS 中的合理默认值
    scale = getattr(cfg, f'{prefix}{lt}_scale', defaults['scale'])
    margin = getattr(cfg, f'{prefix}{lt}_margin', defaults['margin'])
    label_smoothing = getattr(cfg, 'label_smoothing', 0.1)

    print(f"  [{prefix or 'primary'}] {loss_type}: scale={scale}, margin={margin}")

    # 创建损失
    loss = create_metric_loss(
        loss_type=loss_type,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        scale=scale,
        margin=margin,
        label_smoothing=label_smoothing,
    )

    return loss


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
    # Pair-based 损失需要 PK sampler 确保每个 batch 有足够正样本对
    # 如果主损失或辅助损失中有 pair-based，都需要 PK sampler
    loss_type = getattr(cfg, 'loss_type', 'arcface')
    aux_loss_type = getattr(cfg, 'aux_loss_type', 'none')
    _has_pair_loss = (
        not is_proxy_loss(loss_type)
        or (aux_loss_type and aux_loss_type != 'none' and not is_proxy_loss(aux_loss_type))
    )
    use_pk_sampler = _has_pair_loss

    if use_pk_sampler:
        pk_p = getattr(cfg, 'pk_p', 8)   # 每 batch 的类别数
        pk_k = getattr(cfg, 'pk_k', 4)   # 每类别的实例数
        train_sampler = PKSampler(
            labels=train_dataset.get_labels(),
            p=pk_p,
            k=pk_k,
            seed=cfg.seed,
        )
        effective_batch_size = pk_p * pk_k
        print(f"  PK Sampler: P={pk_p}, K={pk_k} (batch_size={effective_batch_size})")
        train_loader = DataLoader(
            train_dataset,
            batch_size=effective_batch_size,
            sampler=train_sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    else:
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

    # ---- 损失函数 ----
    print(f"\n主损失: {loss_type}")

    criterion = _create_loss(cfg, loss_type, num_classes, cfg.embedding_dim)
    criterion = criterion.to(device)

    # 辅助损失 (可选)
    aux_loss_type = getattr(cfg, 'aux_loss_type', 'none')
    aux_criterion = None
    aux_loss_weight = getattr(cfg, 'aux_loss_weight', 0.5)

    if aux_loss_type and aux_loss_type != 'none':
        print(f"辅助损失: {aux_loss_type} (weight={aux_loss_weight})")
        aux_criterion = _create_loss(
            cfg, aux_loss_type, num_classes, cfg.embedding_dim, prefix='aux_'
        )
        aux_criterion = aux_criterion.to(device)

    # ---- 优化器 ----
    # 分组学习率：backbone 较低, embedding + 损失头 较高
    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.embedding.parameters()) +
        list(model.bn_neck.parameters()) +
        list(criterion.parameters())  # proxy-based 损失的可学习权重
    )
    if aux_criterion is not None:
        head_params += list(aux_criterion.parameters())
    if model._needs_pool and hasattr(model, 'pool') and hasattr(model.pool, 'parameters'):
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

    # ---- EMA decay 自动计算 ----
    ema_decay = cfg.ema_decay
    if cfg.ema and ema_decay >= 0.9999:
        # 自动计算: 目标是训练结束时初始权重保留率 < 5%
        # decay^(steps_per_epoch * epochs) < 0.05
        # decay < 0.05^(1/(steps*epochs))
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * cfg.epochs
        if total_steps > 0:
            # 使用半衰期为 total_steps/3 的 decay
            auto_decay = 1.0 - 3.0 / total_steps
            auto_decay = max(auto_decay, 0.99)  # 不低于 0.99
            auto_decay = min(auto_decay, 0.9999)  # 不高于 0.9999
            print(f"\n[AUTO] EMA decay 自动调整: {cfg.ema_decay} -> {auto_decay:.6f} "
                  f"(steps/epoch={steps_per_epoch}, total={total_steps})")
            ema_decay = auto_decay

    # ---- 训练器 ----
    trainer = ReIDTrainer(
        model=model,
        criterion=criterion,
        loss_type=loss_type,
        optimizer=optimizer,
        device=device,
        aux_criterion=aux_criterion,
        aux_loss_weight=aux_loss_weight,
        scheduler=scheduler,
        output_dir=str(output_dir),
        use_ema=cfg.ema,
        ema_decay=ema_decay,
    )

    # ---- 恢复训练 ----
    if cfg.resume:
        trainer.load_checkpoint(cfg.resume)

    # ---- 测试模式 ----
    if cfg.test_only:
        if not cfg.resume:
            # 尝试加载 best_model.pth
            trainer.load_checkpoint('best_model.pth')
        if trainer.ema is not None:
            trainer.ema.apply_shadow(model)
            print("  已应用 EMA 权重")
        _run_inference(cfg, model, device, output_dir)
        return

    # ---- 训练 ----
    print(f"\n{'=' * 40}")
    print("开始训练")
    print(f"{'=' * 40}\n")

    save_metric = getattr(cfg, 'save_metric', 'mAP')
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.epochs,
        early_stopping=cfg.early_stopping,
        save_metric=save_metric,
    )

    # ---- 推理 ----
    print(f"\n{'=' * 40}")
    print("加载最佳模型进行推理")
    print(f"{'=' * 40}")

    trainer.load_checkpoint('best_model.pth')
    # 推理时应用 EMA 权重 (验证选择最佳模型时使用的就是 EMA 权重)
    if trainer.ema is not None:
        trainer.ema.apply_shadow(model)
        print("  已应用 EMA 权重")
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
