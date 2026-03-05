"""
Gastrovision 多标签分类训练脚本

用法:
    python train_multilabel.py --config configs/train_multilabel.yaml
    python train_multilabel.py --config configs/train_multilabel.yaml --lr 0.0005

完整参数:
    python train_multilabel.py --help
"""

import os
import sys
import warnings

# 关闭 pynvml 警告
warnings.filterwarnings('ignore', message='.*pynvml.*deprecated.*')

import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn

# gastrovision 包
from gastrovision.data import create_multilabel_dataloaders
from gastrovision.models.model_factory import get_model
from gastrovision.models.wrapper import MetricLearningWrapper
from gastrovision.trainers.multilabel import MultilabelTrainer, print_multilabel_test_results
from gastrovision.losses import CombinedMultilabelLoss, TripletLoss
from gastrovision.losses.loss_factory import create_metric_loss_function
from gastrovision.utils.optimizer import get_optimizer_from_params
from gastrovision.utils.scheduler import get_scheduler
from gastrovision.utils.config import merge_args_with_config


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="Gastrovision 多标签分类训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 配置文件
    parser.add_argument('--config', type=str, default=None,
                        help='YAML 配置文件路径')

    # 数据参数
    parser.add_argument('--data_dir', type=str,
                        default='D:/codes/data/hyper_kvasir/labeled-images',
                        help='数据目录')
    parser.add_argument('--output_dir', type=str,
                        default='D:/codes/work-projects/Gastrovision_results/multilabel_default',
                        help='输出目录')

    # 模型参数
    parser.add_argument('--model', type=str, default='resnet50', help='模型名称')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='使用 ImageNet 预训练权重')
    parser.add_argument('--weights_path', type=str, default='',
                        help='预训练权重路径')
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                        help='冻结 backbone 只训练分类头')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'], help='优化器')
    parser.add_argument('--scheduler', type=str, default='warmup_cosine',
                        choices=['step', 'cosine', 'plateau', 'onecycle', 'warmup_cosine', 'none'],
                        help='学习率调度器')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='学习率预热轮数')
    parser.add_argument('--early_stopping', type=int, default=15,
                        help='早停耐心值（0 表示不使用）')

    # 数据加载参数
    parser.add_argument('--image_size', type=int, default=224, help='输入图像尺寸')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader 工作进程数')

    # 多标签参数
    parser.add_argument('--num_classes', type=int, default=16, help='多标签类别数量')
    parser.add_argument('--multilabel_threshold', type=float, default=0.5,
                        help='多标签分类阈值')
    parser.add_argument('--multilabel_loss', type=str, default='focal_ohem',
                        choices=['bce', 'focal', 'focal_ohem', 'asymmetric', 'asl',
                                 'label_smoothing', 'poly', 'dice', 'softmax'],
                        help='多标签损失函数类型')
    parser.add_argument('--multilabel_focal_gamma', type=float, default=2.0,
                        help='多标签 Focal Loss gamma')
    parser.add_argument('--multilabel_ohem_ratio', type=float, default=0.4,
                        help='OHEM 困难样本比例')
    parser.add_argument('--asl_gamma_neg', type=float, default=4.0, help='ASL 负样本 gamma')
    parser.add_argument('--asl_gamma_pos', type=float, default=1.0, help='ASL 正样本 gamma')
    parser.add_argument('--asl_clip', type=float, default=0.05, help='ASL 概率裁剪值')
    parser.add_argument('--label_smoothing_factor', type=float, default=0.1,
                        help='标签平滑系数')
    parser.add_argument('--poly_epsilon', type=float, default=1.0, help='Poly Loss epsilon')

    # Triplet Loss 相关参数
    parser.add_argument('--use_triplet', action='store_true', default=False,
                        help='启用 Triplet Loss 训练')
    parser.add_argument('--triplet_weight', type=float, default=1.0, help='Triplet Loss 权重')
    parser.add_argument('--triplet_margin', type=float, default=0.3, help='Triplet Loss margin')
    parser.add_argument('--triplet_num_instances', type=int, default=4,
                        help='每个身份的实例数')

    # 度量学习
    parser.add_argument('--metric_loss', type=str, default='none',
                        choices=['none', 'contrastive', 'triplet', 'lifted', 'proxy_nca',
                                 'npair', 'arcface', 'cosface', 'sphereface', 'circle', 'circle_cls'],
                        help='度量学习损失类型 (none=禁用)')
    parser.add_argument('--metric_loss_weight', type=float, default=0.5,
                        help='度量学习损失权重')
    parser.add_argument('--metric_loss_margin', type=float, default=0.0,
                        help='度量学习损失 margin')
    parser.add_argument('--metric_loss_scale', type=float, default=0.0,
                        help='度量学习损失 scale')
    parser.add_argument('--embedding_dim', type=int, default=512,
                        help='度量学习嵌入维度')

    # 数据增强策略
    parser.add_argument('--mixup', type=float, default=0.0, help='Mixup alpha')
    parser.add_argument('--cutmix', type=float, default=0.0, help='CutMix alpha')

    # 其他
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--test_only', action='store_true', help='只进行测试')
    parser.add_argument('--resume', type=str, default='',
                        help='从 checkpoint 恢复训练（空=不恢复）')
    parser.add_argument('--metric_debug', action='store_true', default=False,
                        help='启用度量学习诊断信息')

    return parser


def main():
    parser = build_parser()
    args = merge_args_with_config(parser)

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("Gastrovision 多标签分类训练")
    print("=" * 60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    if args.config:
        print(f"配置: {args.config}")
    print()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    config_path = output_dir / 'config.txt'
    with open(config_path, 'w', encoding='utf-8') as f:
        for key, value in sorted(vars(args).items()):
            f.write(f"{key}: {value}\n")

    # 加载多标签数据
    print("\n加载多标签数据集...")
    train_loader, valid_loader, test_loader, num_classes, class_names = \
        create_multilabel_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            num_classes=args.num_classes,
            use_triplet_sampler=args.use_triplet,
            triplet_num_instances=args.triplet_num_instances)
    print()

    # 创建模型
    print(f"创建模型: {args.model}")

    weights_to_load = args.weights_path if args.weights_path else None
    use_pretrained = args.pretrained

    if args.resume:
        weights_to_load = args.resume
        use_pretrained = False

    model = get_model(
        args.model,
        num_classes=num_classes,
        pretrained=use_pretrained,
        weights_path=weights_to_load,
        freeze_backbone=args.freeze_backbone
    )
    model = model.to(device)

    # 多标签损失函数
    if args.multilabel_loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
        print(f"损失函数: BCEWithLogitsLoss")
    else:
        criterion = CombinedMultilabelLoss(
            loss_type=args.multilabel_loss,
            focal_gamma=args.multilabel_focal_gamma,
            ohem_ratio=args.multilabel_ohem_ratio,
            asl_gamma_neg=args.asl_gamma_neg,
            asl_gamma_pos=args.asl_gamma_pos,
            asl_clip=args.asl_clip,
            label_smoothing=args.label_smoothing_factor,
            poly_epsilon=args.poly_epsilon)
        print(f"损失函数: {args.multilabel_loss.upper()}")
        if 'focal' in args.multilabel_loss:
            print(f"  - gamma: {args.multilabel_focal_gamma}")
        if 'ohem' in args.multilabel_loss:
            print(f"  - ohem_ratio: {args.multilabel_ohem_ratio}")
        if args.multilabel_loss in ['asymmetric', 'asl']:
            print(f"  - gamma_neg: {args.asl_gamma_neg}, gamma_pos: {args.asl_gamma_pos}, clip: {args.asl_clip}")

    # 度量学习损失
    metric_criterion, model = create_metric_loss_function(
        args, num_classes, device, model)

    # 优化器
    all_params = list(model.parameters())
    if metric_criterion is not None:
        metric_params = list(metric_criterion.parameters())
        if metric_params:
            all_params += metric_params
    optimizer = get_optimizer_from_params(
        all_params, args.optimizer, args.lr, args.weight_decay)
    print(f"优化器: {type(optimizer).__name__} (lr={args.lr})")

    # 学习率调度器
    scheduler = get_scheduler(
        optimizer, args.scheduler, args.epochs,
        steps_per_epoch=len(train_loader),
        warmup_epochs=args.warmup_epochs)
    if scheduler:
        print(f"调度器: {args.scheduler}")
    print()

    # Triplet Loss
    triplet_criterion = None
    if args.use_triplet:
        triplet_criterion = TripletLoss(margin=args.triplet_margin)
        print(f"Triplet Loss: margin={args.triplet_margin}, weight={args.triplet_weight}")

    # 创建多标签训练器
    trainer = MultilabelTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        output_dir=str(output_dir),
        class_names=class_names,
        threshold=args.multilabel_threshold,
        triplet_loss=triplet_criterion,
        triplet_weight=args.triplet_weight,
        metric_loss=metric_criterion,
        metric_loss_weight=args.metric_loss_weight,
        mixup_alpha=args.mixup,
        cutmix_alpha=args.cutmix
    )

    # 从 checkpoint 恢复
    if args.resume and not args.test_only:
        print(f"从 checkpoint 恢复: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # 只测试模式
    if args.test_only:
        print("运行测试...")
        results = trainer.test(test_loader)
        print_multilabel_test_results(results)
        return

    # 训练
    trainer.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=args.epochs,
        early_stopping=args.early_stopping
    )

    # 加载最佳模型进行测试
    print("\n加载最佳模型进行测试...")
    trainer.load_checkpoint('best_model.pth')

    # 优化每个类别的阈值
    trainer.optimize_thresholds(valid_loader)

    # 测试
    results = trainer.test(test_loader)
    print_multilabel_test_results(results)

    print("\n多标签训练完成!")


if __name__ == "__main__":
    main()
