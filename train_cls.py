"""
内镜图像单标签分类 — 训练脚本

纯 YAML 配置驱动，所有参数在 configs/train_cls.yaml 中定义。

用法:
    python train_cls.py configs/train_cls.yaml
"""

import warnings
warnings.filterwarnings('ignore', message='.*pynvml.*deprecated.*')

from pathlib import Path
from datetime import datetime

import torch

from gastrovision.data import create_dataloaders
from gastrovision.models.model_factory import get_model
from gastrovision.trainers import Trainer, print_test_results
from gastrovision.losses.loss_factory import create_loss_function, create_metric_loss_function
from gastrovision.utils.optimizer import get_optimizer_from_params
from gastrovision.utils.scheduler import get_scheduler
from gastrovision.utils.config import get_config_from_cli, save_config
from gastrovision.utils.tta import TTAPredictor, TTA_LIGHT, TTA_MEDIUM, TTA_HEAVY


def load_class_names(data_dir: str) -> list:
    """加载类别名称"""
    class_names_file = Path(data_dir) / 'class_names.txt'
    if class_names_file.exists():
        with open(class_names_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    return None


def main():
    # ---- 加载配置 ----
    cfg = get_config_from_cli()

    # ---- 随机种子 ----
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # ---- 设备 ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("内镜图像单标签分类训练")
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

    # ---- 类别信息 ----
    class_names = load_class_names(cfg.data_dir)
    num_classes = len(class_names) if class_names else 23
    print(f"类别数: {num_classes}")

    # ---- 模型 ----
    print(f"\n创建模型: {cfg.model}")
    weights_to_load = cfg.weights_path if cfg.weights_path else None
    use_pretrained  = cfg.pretrained

    if cfg.resume:
        weights_to_load = cfg.resume
        use_pretrained  = False
        print(f"  从 checkpoint 加载权重: {cfg.resume}")

    classifier_head    = getattr(cfg, 'classifier_head', 'linear')
    classifier_dropout = getattr(cfg, 'classifier_dropout', 0.2)

    model = get_model(
        cfg.model,
        num_classes=num_classes,
        pretrained=use_pretrained,
        weights_path=weights_to_load,
        freeze_backbone=cfg.freeze_backbone,
        classifier_head=classifier_head,
        classifier_dropout=classifier_dropout)
    model = model.to(device)

    # ---- 数据 ----
    print("\n加载数据集...")
    train_loader, valid_loader, test_loader, _ = create_dataloaders(
        cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        image_size=cfg.image_size,
        use_albumentations=cfg.use_albumentations,
        augment_level=cfg.augment_level)
    print()

    # ---- 损失函数 ----
    criterion = create_loss_function(cfg, device)
    print(f"损失函数: {type(criterion).__name__}")

    # ---- 度量学习 ----
    metric_criterion, model = create_metric_loss_function(
        cfg, num_classes, device, model)

    # ---- 优化器 ----
    all_params = list(model.parameters())
    if metric_criterion is not None:
        metric_params = list(metric_criterion.parameters())
        if metric_params:
            all_params += metric_params
    optimizer = get_optimizer_from_params(
        all_params, cfg.optimizer, cfg.lr, cfg.weight_decay)
    print(f"优化器: {type(optimizer).__name__} (lr={cfg.lr})")

    # ---- 调度器 ----
    scheduler = get_scheduler(
        optimizer, cfg.scheduler, cfg.epochs,
        steps_per_epoch=len(train_loader),
        warmup_epochs=cfg.warmup_epochs)
    if scheduler:
        print(f"调度器: {cfg.scheduler}")
        if cfg.scheduler == 'warmup_cosine':
            print(f"  预热轮数: {cfg.warmup_epochs}")
    print()

    # ---- 训练器 ----
    use_ema   = getattr(cfg, 'ema', False)
    ema_decay = getattr(cfg, 'ema_decay', 0.9999)
    metric_debug = getattr(cfg, 'metric_debug', False)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        output_dir=str(output_dir),
        class_names=class_names,
        metric_loss=metric_criterion,
        metric_loss_weight=cfg.metric_loss_weight,
        mixup_alpha=cfg.mixup,
        cutmix_alpha=cfg.cutmix,
        metric_debug=metric_debug,
        use_ema=use_ema,
        ema_decay=ema_decay)

    # ---- 恢复训练 ----
    if cfg.resume and not cfg.test_only:
        print(f"从 checkpoint 恢复: {cfg.resume}")
        trainer.load_checkpoint(cfg.resume)

    # ---- 测试模式 ----
    if cfg.test_only:
        print("运行测试...")
        results = trainer.test(test_loader)
        print_test_results(results)
        return

    # ---- 训练 ----
    trainer.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=cfg.epochs,
        early_stopping=cfg.early_stopping)

    # ---- 最终测试 ----
    print("\n加载最佳模型进行测试...")
    trainer.load_checkpoint('best_model.pth')

    use_tta   = getattr(cfg, 'tta', False)
    tta_level = getattr(cfg, 'tta_level', 'medium')

    if use_tta:
        tta_transforms = {'light': TTA_LIGHT, 'medium': TTA_MEDIUM, 'heavy': TTA_HEAVY}[tta_level]
        tta_predictor  = TTAPredictor(model, device, num_classes=num_classes, transforms=tta_transforms)
        print(f"\n使用 TTA ({tta_level}, {len(tta_transforms)} views) 测试...")
        tta_predictor.evaluate(test_loader, class_names=class_names)
    else:
        results = trainer.test(test_loader)
        print_test_results(results)

    print("\n完成!")


if __name__ == "__main__":
    main()
