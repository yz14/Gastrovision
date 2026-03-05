"""
Gastrovision 模型训练入口（兼容旧接口）

推荐使用独立脚本:
    python train_cls.py --config configs/train_cls.yaml           # 单标签分类
    python train_multilabel.py --config configs/train_multilabel.yaml  # 多标签分类

兼容旧用法 (会自动转发到对应脚本):
    python main.py --model resnet50 --epochs 50 --batch_size 32
    python main.py --multilabel --model resnet50
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
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import (StepLR, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR)

# gastrovision 包
from gastrovision.data import (
    create_dataloaders, create_multilabel_dataloaders, get_class_weights,
    MultilabelIdentitySampler, GastrovisionMultilabelDataset
)
from gastrovision.data.augmentation import (
    mixup_data, mixup_criterion, cutmix_data, WarmupCosineScheduler
)
from gastrovision.trainers import Trainer, print_test_results
from gastrovision.trainers.multilabel import MultilabelTrainer, print_multilabel_test_results
from gastrovision.losses import (
    LabelSmoothingCrossEntropy, FocalLoss, ClassBalancedLoss,
    CombinedMultilabelLoss, FocalLossMultilabel, FocalOHEMLoss, AsymmetricLoss, TripletLoss
)
from gastrovision.losses.metric_learning import create_metric_loss
from gastrovision.losses.loss_factory import (
    create_loss_function, create_metric_loss_function, get_samples_per_class
)
from gastrovision.models.model_factory import get_model
from gastrovision.models.wrapper import MetricLearningWrapper
from gastrovision.utils.optimizer import get_optimizer, get_optimizer_from_params
from gastrovision.utils.scheduler import get_scheduler


# SSL 模块 (懒加载)
# 注意: SSL 模块位于 experiments/ssl_methods/，需要先添加路径或安装
def _get_ssl_model(args, backbone):
    """根据参数获取 SSL 模型"""
    # 尝试从 experiments 目录加载 SSL 方法
    ssl_path = Path(__file__).parent / 'experiments' / 'ssl_methods'
    if ssl_path.exists() and str(ssl_path) not in sys.path:
        sys.path.insert(0, str(ssl_path))

    try:
        from ssl_methods import (
            SimSiam, MoCo, SimCLR, BYOL, BarlowTwins,
            SwAV, DINO, MAE, InstDisc, RotationPrediction,
            SiameseNetwork, TripletNetwork)
    except ImportError:
        raise ImportError(
            "SSL 方法模块未找到。请确保 experiments/ssl_methods/ 目录包含所需的 SSL 实现。\n"
            "如果不需要 SSL 预训练，请使用 --mode train")

    method = args.ssl_method.lower()

    ssl_models = {
        'simsiam': lambda: SimSiam(backbone, proj_dim=args.projector_dim, pred_dim=args.predictor_dim),
        'moco': lambda: MoCo(backbone, dim=128, K=4096, m=args.momentum, T=args.temperature, mlp=True),
        'simclr': lambda: SimCLR(backbone, proj_dim=128, temperature=args.temperature),
        'byol': lambda: BYOL(backbone, proj_dim=256, hidden_dim=4096, momentum=args.momentum),
        'barlow_twins': lambda: BarlowTwins(backbone, projector_sizes=[args.projector_dim]*3),
        'swav': lambda: SwAV(backbone, proj_dim=128, num_prototypes=3000),
        'dino': lambda: DINO(backbone, out_dim=65536, momentum=args.momentum),
        'mae': lambda: MAE(backbone, mask_ratio=0.75),
        'instdisc': lambda: InstDisc(backbone, num_samples=50000, feature_dim=128),
        'rotation': lambda: RotationPrediction(backbone),
        'siamese': lambda: SiameseNetwork(backbone, proj_dim=128),
        'triplet': lambda: TripletNetwork(backbone, proj_dim=128),
    }

    if method not in ssl_models:
        raise ValueError(f"Unknown SSL method: {method}")

    return ssl_models[method]()


def run_ssl_pretrain(args):
    """运行 SSL 预训练"""
    from gastrovision.trainers.ssl import SSLTrainer, SSLDataset
    from gastrovision.data.ssl_augmentations import TwoCropsTransform, get_ssl_augmentation
    from torchvision import datasets
    from torch.utils.data import DataLoader

    print("="*60)
    print(f"SSL 预训练 - 方法: {args.ssl_method.upper()}")
    print("="*60)

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    # 创建 backbone
    print(f"\n创建 backbone: {args.model}")
    print(f"  使用 ImageNet 预训练: {args.pretrained}")
    backbone = get_model(
        args.model,
        num_classes=1000,  # 占位，会被 SSL 方法移除
        pretrained=args.pretrained,
        weights_path=args.weights_path if args.weights_path else None)

    # 创建 SSL 模型
    ssl_model = _get_ssl_model(args, backbone)
    print(f"SSL 模型: {args.model} + {args.ssl_method}")

    # 打印模型参数量
    total_params = sum(p.numel() for p in ssl_model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in ssl_model.parameters() if p.requires_grad) / 1e6
    print(f"  总参数: {total_params:.2f}M, 可训练: {trainable_params:.2f}M")

    # 数据增强
    ssl_transform = TwoCropsTransform(get_ssl_augmentation(args.ssl_method, args.image_size))

    # 加载数据集
    train_dir = Path(args.data_dir) / 'train'
    if train_dir.exists():
        train_dataset = datasets.ImageFolder(str(train_dir), transform=ssl_transform)
    else:
        from gastrovision.data.dataset import GastrovisionDataset
        base_dataset = GastrovisionDataset(
            Path(args.data_dir) / 'train.txt',
            transform=None)
        train_dataset = SSLDataset(base_dataset, transform=ssl_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True)
    print(f"训练样本数: {len(train_dataset)}")

    # 优化器
    optimizer = torch.optim.SGD(
        ssl_model.parameters(),
        lr=0.03 * args.batch_size / 256,
        momentum=0.9,
        weight_decay=1e-4)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.ssl_epochs)

    # 创建 SSL Trainer
    trainer = SSLTrainer(
        model=ssl_model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=args.ssl_output_dir,
        method_name=args.ssl_method)

    # 训练
    trainer.fit(
        train_loader=train_loader,
        epochs=args.ssl_epochs,
        save_freq=10,
        resume_from=args.resume if args.resume else None)

    # 保存编码器
    trainer.save_encoder(f'{args.ssl_method}_encoder.pth')

    print(f"\nSSL 预训练完成!")
    print(f"编码器已保存到: {args.ssl_output_dir}/{args.ssl_method}_encoder.pth")
    print(f"\n下一步 - 使用 SSL 权重进行分类训练:")
    print(f"  python train_cls.py --resume {args.ssl_output_dir}/{args.ssl_method}_encoder.pth")


def load_class_names(data_dir: str) -> list:
    """加载类别名称"""
    class_names_file = Path(data_dir) / 'class_names.txt'
    if class_names_file.exists():
        with open(class_names_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    return None


def run_multilabel_training(args, device, output_dir):
    """运行多标签训练"""
    print("=" * 60)
    print("多标签分类训练模式")
    print("=" * 60)

    # 加载多标签数据
    print("\n加载多标签数据集...")
    train_loader, valid_loader, test_loader, num_classes, class_names = create_multilabel_dataloaders(
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
    if args.mode == 'finetune' and args.resume:
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

    # 度量学习损失
    metric_criterion, model = create_metric_loss_function(args, num_classes, device, model)

    # 优化器
    all_params = list(model.parameters())
    if metric_criterion is not None:
        metric_params = list(metric_criterion.parameters())
        if metric_params:
            all_params += metric_params
    optimizer = get_optimizer_from_params(all_params, args.optimizer, args.lr, args.weight_decay)
    print(f"优化器: {type(optimizer).__name__} (lr={args.lr})")

    # 学习率调度器
    scheduler = get_scheduler(
        optimizer, args.scheduler, args.epochs,
        steps_per_epoch=len(train_loader),
        warmup_epochs=args.warmup_epochs
    )
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
    if args.resume and args.mode == 'train':
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


def main():
    parser = argparse.ArgumentParser(description="Gastrovision 模型训练", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 模式参数
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'ssl_pretrain', 'finetune', 'test'], help='运行模式: train(分类训练), ssl_pretrain(SSL预训练), finetune(微调), test(测试)')

    # SSL 预训练参数
    parser.add_argument('--ssl_method', type=str, default='simsiam', choices=['simsiam', 'moco', 'simclr', 'byol', 'barlow_twins', 'swav', 'dino', 'mae', 'instdisc', 'rotation', 'siamese', 'triplet'], help='SSL 方法')
    parser.add_argument('--ssl_epochs', type=int, default=10, help='SSL 预训练轮数')
    parser.add_argument('--ssl_output_dir', type=str, default='D:/codes/work-projects/Gastrovision_results/simsiam', help='SSL checkpoint 输出目录')
    parser.add_argument('--projector_dim', type=int, default=2048, help='投影头输出维度')
    parser.add_argument('--predictor_dim', type=int, default=512, help='预测头隐藏层维度 (SimSiam/BYOL)')
    parser.add_argument('--temperature', type=float, default=0.07, help='对比学习温度参数')
    parser.add_argument('--momentum', type=float, default=0.996, help='动量参数 (MoCo/BYOL/DINO)')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='D:/codes/data/hyper_kvasir/labeled-images', help='数据目录（包含 train.txt, valid.txt 等）')
    parser.add_argument('--output_dir', type=str, default='D:/codes/work-projects/Gastrovision_results/res50_kvasir_arcface', help='输出目录')

    # 模型参数
    parser.add_argument('--model', type=str, default='resnet50', help='模型名称')
    parser.add_argument('--pretrained', action='store_true', default=True, help='使用 ImageNet 预训练权重')
    parser.add_argument('--weights_path', type=str, default='', help='预训练权重路径')
    parser.add_argument('--freeze_backbone', action='store_true', default=False, help='冻结 backbone 只训练分类头')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='优化器')
    parser.add_argument('--scheduler', type=str, default='warmup_cosine', choices=['step', 'cosine', 'plateau', 'onecycle', 'warmup_cosine', 'none'], help='学习率调度器')
    parser.add_argument('--early_stopping', type=int, default=15, help='早停耐心值')

    # 数据加载参数
    parser.add_argument('--image_size', type=int, default=224, help='输入图像尺寸')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader 工作进程数')
    parser.add_argument('--use_albumentations', action='store_true', default=False, help='使用 Albumentations')
    parser.add_argument('--augment_level', type=str, default='medium', choices=['light', 'medium', 'heavy'], help='增强级别')

    # 精度提升策略
    parser.add_argument('--mixup', type=float, default=0.0, help='Mixup alpha')
    parser.add_argument('--cutmix', type=float, default=0.0, help='CutMix alpha')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='标签平滑系数')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='预热轮数')
    parser.add_argument('--loss_type', type=str, default='focal', choices=['ce', 'focal', 'cb_focal', 'cb_softmax'], help='损失函数类型')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal Loss gamma')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--use_class_weights', action='store_true', help='使用类别权重')
    parser.add_argument('--test_only', action='store_true', help='只进行测试')
    parser.add_argument('--resume', type=str, default='', help='从 checkpoint 恢复训练')
    parser.add_argument('--visualize', action='store_true', default=True, help='生成可视化图表')

    # 多标签参数
    parser.add_argument('--multilabel', action='store_true', default=False, help='使用多标签训练模式')
    parser.add_argument('--multilabel_threshold', type=float, default=0.5, help='多标签分类阈值')
    parser.add_argument('--num_classes', type=int, default=16, help='多标签类别数量')
    parser.add_argument('--multilabel_loss', type=str, default='focal_ohem',
                        choices=['bce', 'focal', 'focal_ohem', 'asymmetric', 'asl',
                                 'label_smoothing', 'poly', 'dice', 'softmax'],
                        help='多标签损失函数类型')
    parser.add_argument('--multilabel_focal_gamma', type=float, default=2.0, help='多标签 Focal gamma')
    parser.add_argument('--multilabel_ohem_ratio', type=float, default=0.4, help='OHEM 比例')
    parser.add_argument('--asl_gamma_neg', type=float, default=4.0, help='ASL 负样本 gamma')
    parser.add_argument('--asl_gamma_pos', type=float, default=1.0, help='ASL 正样本 gamma')
    parser.add_argument('--asl_clip', type=float, default=0.05, help='ASL 裁剪值')
    parser.add_argument('--label_smoothing_factor', type=float, default=0.1, help='标签平滑系数')
    parser.add_argument('--poly_epsilon', type=float, default=1.0, help='Poly Loss epsilon')

    # Triplet Loss
    parser.add_argument('--use_triplet', action='store_true', default=False, help='启用 Triplet Loss')
    parser.add_argument('--triplet_weight', type=float, default=1.0, help='Triplet Loss 权重')
    parser.add_argument('--triplet_margin', type=float, default=0.3, help='Triplet Loss margin')
    parser.add_argument('--triplet_num_instances', type=int, default=4, help='每个身份的实例数')

    # 度量学习
    parser.add_argument('--metric_loss', type=str, default='arcface',
                        choices=['none', 'contrastive', 'triplet', 'lifted', 'proxy_nca',
                                 'npair', 'arcface', 'cosface', 'sphereface', 'circle', 'circle_cls'],
                        help='度量学习损失类型')
    parser.add_argument('--metric_loss_weight', type=float, default=0.5, help='度量学习损失权重')
    parser.add_argument('--metric_loss_margin', type=float, default=0.0, help='度量学习损失 margin')
    parser.add_argument('--metric_loss_scale', type=float, default=0.0, help='度量学习损失 scale')
    parser.add_argument('--embedding_dim', type=int, default=512, help='度量学习嵌入维度')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # SSL 预训练模式
    if args.mode == 'ssl_pretrain':
        run_ssl_pretrain(args)
        return

    # 测试模式
    if args.mode == 'test':
        args.test_only = True

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("Gastrovision 模型训练")
    print("=" * 60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    config_path = output_dir / 'config.txt'
    with open(config_path, 'w', encoding='utf-8') as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    # 多标签模式分支
    if args.multilabel:
        run_multilabel_training(args, device, output_dir)
        return

    # ---- 单标签分类训练 ----

    # 加载类别名称
    class_names = load_class_names(args.data_dir)
    num_classes = len(class_names) if class_names else 22
    print(f"类别数: {num_classes}")

    # 创建模型
    print(f"\n创建模型: {args.model}")

    weights_to_load = args.weights_path if args.weights_path else None
    use_pretrained = args.pretrained
    if args.mode == 'finetune' and args.resume:
        weights_to_load = args.resume
        print(f"  微调模式: 加载 SSL 编码器权重 {args.resume}")
        use_pretrained = False

    model = get_model(
        args.model,
        num_classes=num_classes,
        pretrained=use_pretrained,
        weights_path=weights_to_load,
        freeze_backbone=args.freeze_backbone)
    model = model.to(device)

    # 创建数据加载器
    print("\n加载数据集...")
    train_loader, valid_loader, test_loader, _ = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        use_albumentations=args.use_albumentations,
        augment_level=args.augment_level)
    print()

    # 损失函数
    criterion = create_loss_function(args, device)
    print(f"损失函数: {type(criterion).__name__}")

    # 度量学习损失
    metric_criterion, model = create_metric_loss_function(args, num_classes, device, model)

    # 优化器
    all_params = list(model.parameters())
    if metric_criterion is not None:
        metric_params = list(metric_criterion.parameters())
        if metric_params:
            all_params += metric_params
    optimizer = get_optimizer_from_params(all_params, args.optimizer, args.lr, args.weight_decay)
    print(f"优化器: {type(optimizer).__name__} (lr={args.lr})")

    # 学习率调度器
    scheduler = get_scheduler(
        optimizer, args.scheduler, args.epochs,
        steps_per_epoch=len(train_loader),
        warmup_epochs=args.warmup_epochs)
    if scheduler:
        print(f"调度器: {args.scheduler}")
        if args.scheduler == 'warmup_cosine':
            print(f"  预热轮数: {args.warmup_epochs}")
    print()

    # 创建训练器
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        output_dir=str(output_dir),
        class_names=class_names,
        metric_loss=metric_criterion,
        metric_loss_weight=args.metric_loss_weight,
        mixup_alpha=args.mixup,
        cutmix_alpha=args.cutmix)

    # 从 checkpoint 恢复
    if args.resume and args.mode == 'train':
        print(f"从 checkpoint 恢复: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # 只测试模式
    if args.test_only:
        print("运行测试...")
        results = trainer.test(test_loader)
        print_test_results(results)
        return

    # 训练
    trainer.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=args.epochs,
        early_stopping=args.early_stopping)

    # 加载最佳模型进行测试
    print("\n加载最佳模型进行测试...")
    trainer.load_checkpoint('best_model.pth')
    results = trainer.test(test_loader)
    print_test_results(results)

    print("\n完成!")


if __name__ == "__main__":
    main()
