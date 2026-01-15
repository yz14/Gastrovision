"""
Gastrovision 模型训练入口

用法:
    python main.py --model resnet50 --epochs 50 --batch_size 32

完整参数:
    python main.py --help
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

# 本地模块
from dataset import create_dataloaders, get_class_weights
from trainer import Trainer, print_test_results
from augmentation import (mixup_data, mixup_criterion, cutmix_data, WarmupCosineScheduler)
from losses import LabelSmoothingCrossEntropy, FocalLoss, ClassBalancedLoss
import resnet

# SSL 模块 (懒加载以避免不需要时的开销)
def _get_ssl_model(args, backbone):
    """根据参数获取 SSL 模型"""
    from ssl_methods import (
        SimSiam, MoCo, SimCLR, BYOL, BarlowTwins,
        SwAV, DINO, MAE, InstDisc, RotationPrediction,
        SiameseNetwork, TripletNetwork)
    
    method = args.ssl_method.lower()
    
    if method == 'simsiam':
        return SimSiam(backbone, proj_dim=args.projector_dim, pred_dim=args.predictor_dim)
    elif method == 'moco':
        return MoCo(backbone, dim=128, K=4096, m=args.momentum, T=args.temperature, mlp=True)
    elif method == 'simclr':
        return SimCLR(backbone, proj_dim=128, temperature=args.temperature)
    elif method == 'byol':
        return BYOL(backbone, proj_dim=256, hidden_dim=4096, momentum=args.momentum)
    elif method == 'barlow_twins':
        return BarlowTwins(backbone, projector_sizes=[args.projector_dim]*3)
    elif method == 'swav':
        return SwAV(backbone, proj_dim=128, num_prototypes=3000)
    elif method == 'dino':
        return DINO(backbone, out_dim=65536, momentum=args.momentum)
    elif method == 'mae':
        return MAE(backbone, mask_ratio=0.75)
    elif method == 'instdisc':
        return InstDisc(backbone, num_samples=50000, feature_dim=128)
    elif method == 'rotation':
        return RotationPrediction(backbone)
    elif method == 'siamese':
        return SiameseNetwork(backbone, proj_dim=128)
    elif method == 'triplet':
        return TripletNetwork(backbone, proj_dim=128)
    else:
        raise ValueError(f"Unknown SSL method: {method}")


def run_ssl_pretrain(args):
    """运行 SSL 预训练"""
    from ssl_trainer import SSLTrainer, SSLDataset
    from ssl_augmentations import TwoCropsTransform, get_ssl_augmentation
    from torchvision import datasets
    from torch.utils.data import DataLoader
    
    print("="*60)
    print(f"SSL 预训练 - 方法: {args.ssl_method.upper()}")
    print("="*60)
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # 创建 backbone (注意：pretrained=True 会在 get_model 中加载 ImageNet 权重)
    print(f"\n创建 backbone: {args.model}")
    print(f"  使用 ImageNet 预训练: {args.pretrained}")
    backbone = get_model(
        args.model, 
        num_classes=1000,  # 占位，会被 SSL 方法移除
        pretrained=args.pretrained,
        weights_path=args.weights_path)
    
    # 创建 SSL 模型
    ssl_model = _get_ssl_model(args, backbone)
    print(f"SSL 模型: {args.model} + {args.ssl_method}")
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in ssl_model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in ssl_model.parameters() if p.requires_grad) / 1e6
    print(f"  总参数: {total_params:.2f}M, 可训练: {trainable_params:.2f}M")
    
    # 数据增强
    ssl_transform = TwoCropsTransform(get_ssl_augmentation(args.ssl_method, args.image_size))
    
    # 加载数据集 (使用 ImageFolder 格式或从 train.txt 读取)
    train_dir = Path(args.data_dir) / 'train'
    if train_dir.exists():
        # ImageFolder 格式
        train_dataset = datasets.ImageFolder(str(train_dir), transform=ssl_transform)
    else:
        # 从 train.txt 读取
        from dataset import GastrovisionDataset
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
    
    # 优化器 (SSL 通常使用 SGD with momentum 或 LARS)
    optimizer = torch.optim.SGD(
        ssl_model.parameters(),
        lr=0.03 * args.batch_size / 256,  # 线性缩放
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
    
    # 训练 (支持 resume)
    trainer.fit(
        train_loader=train_loader,
        epochs=args.ssl_epochs,
        save_freq=10,
        resume_from=args.resume)
    
    # 保存编码器供微调使用
    trainer.save_encoder(f'{args.ssl_method}_encoder.pth')
    
    print(f"\nSSL 预训练完成!")
    print(f"编码器已保存到: {args.ssl_output_dir}/{args.ssl_method}_encoder.pth")
    print(f"\n下一步 - 使用 SSL 权重进行分类训练:")
    print(f"  python main.py --mode finetune --model {args.model} --resume {args.ssl_output_dir}/{args.ssl_method}_encoder.pth")


def get_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = False,
    weights_path: str = None,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    获取模型
    
    Args:
        model_name: 模型名称，支持：
            - ResNet 系列: resnet18/34/50/101/152, resnext50/101, wide_resnet50/101
            - ConvNeXt 系列: convnext_tiny/small/base/large
            - EfficientNet V2 系列: efficientnet_v2_s/m/l
            - Swin Transformer 系列: swin_t/s/b
            - GastroNet: gastronet_resnet50_dino, gastronet_vit_small
        num_classes: 分类类别数
        pretrained: 是否使用 ImageNet 预训练权重
        weights_path: 预训练权重路径
        freeze_backbone: 是否冻结backbone只训练分类头
        
    Returns:
        PyTorch 模型
    """
    import torchvision.models as models
    
    # GastroNet 预训练模型
    if model_name.startswith('gastronet_'):
        from GastroNet_5m import get_gastronet_model
        model = get_gastronet_model(
            model_name, 
            num_classes=num_classes,
            weights_dir=weights_path,
            freeze_backbone=freeze_backbone)
        return model
    
    # 模型配置
    model_configs = {
        # ResNet 系列
        'resnet18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 'fc'),
        'resnet34': (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, 'fc'),
        'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, 'fc'),
        'resnet101': (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2, 'fc'),
        'resnet152': (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V2, 'fc'),
        'resnext50': (models.resnext50_32x4d, models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2, 'fc'),
        'resnext101': (models.resnext101_32x8d, models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2, 'fc'),
        'wide_resnet50': (models.wide_resnet50_2, models.Wide_ResNet50_2_Weights.IMAGENET1K_V2, 'fc'),
        'wide_resnet101': (models.wide_resnet101_2, models.Wide_ResNet101_2_Weights.IMAGENET1K_V2, 'fc'),
        # ConvNeXt 系列
        'convnext_tiny': (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, 'classifier'),
        'convnext_small': (models.convnext_small, models.ConvNeXt_Small_Weights.IMAGENET1K_V1, 'classifier'),
        'convnext_base': (models.convnext_base, models.ConvNeXt_Base_Weights.IMAGENET1K_V1, 'classifier'),
        'convnext_large': (models.convnext_large, models.ConvNeXt_Large_Weights.IMAGENET1K_V1, 'classifier'),
        # EfficientNet V2 系列
        'efficientnet_v2_s': (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.IMAGENET1K_V1, 'classifier'),
        'efficientnet_v2_m': (models.efficientnet_v2_m, models.EfficientNet_V2_M_Weights.IMAGENET1K_V1, 'classifier'),
        'efficientnet_v2_l': (models.efficientnet_v2_l, models.EfficientNet_V2_L_Weights.IMAGENET1K_V1, 'classifier'),
        # Swin Transformer 系列
        'swin_t': (models.swin_t, models.Swin_T_Weights.IMAGENET1K_V1, 'head'),
        'swin_s': (models.swin_s, models.Swin_S_Weights.IMAGENET1K_V1, 'head'),
        'swin_b': (models.swin_b, models.Swin_B_Weights.IMAGENET1K_V1, 'head'),
    }
    
    if model_name not in model_configs:
        raise ValueError(f"不支持的模型: {model_name}. 支持: {list(model_configs.keys())}")
    
    model_fn, weights_enum, head_type = model_configs[model_name]
    
    # 创建模型
    # 权重加载优先级: weights_path > torch cache > pretrained (网络下载)
    if weights_path and os.path.isfile(weights_path):
        # 使用本地权重文件
        model = model_fn(weights=None)
        print(f"  加载本地权重: {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict, strict=False)
    elif weights_path:
        # weights_path 指定但文件不存在，检查 torch cache
        print(f"  警告: 指定的权重文件不存在: {weights_path}")
        cache_dir = os.path.expanduser('~/.cache/torch/hub/checkpoints')
        # 尝试从文件名推断 cache 文件
        expected_cache = {
            'resnet50': 'resnet50-11ad3fa6.pth',
            'resnet18': 'resnet18-f37072fd.pth',
            'convnext_tiny': 'convnext_tiny-983f1562.pth',
            'convnext_base': 'convnext_base-6075fbad.pth',
        }
        cache_file = expected_cache.get(model_name)
        if cache_file:
            cache_path = os.path.join(cache_dir, cache_file)
            if os.path.isfile(cache_path):
                print(f"  从 torch cache 加载: {cache_path}")
                model = model_fn(weights=None)
                state_dict = torch.load(cache_path, map_location='cpu', weights_only=False)
                model.load_state_dict(state_dict, strict=False)
            elif pretrained:
                print(f"  从网络下载预训练权重...")
                model = model_fn(weights=weights_enum)
            else:
                model = model_fn(weights=None)
                print(f"  使用随机初始化权重")
        elif pretrained:
            print(f"  从网络下载预训练权重...")
            model = model_fn(weights=weights_enum)
        else:
            model = model_fn(weights=None)
            print(f"  使用随机初始化权重")
    elif pretrained:
        # 从网上下载 ImageNet 预训练权重
        model = model_fn(weights=weights_enum)
        print(f"  加载 ImageNet 预训练权重: {weights_enum}")
    else:
        model = model_fn(weights=None)
        print(f"  使用随机初始化权重")
    
    # 替换分类头
    in_features = _get_classifier_in_features(model, head_type)
    _replace_classifier(model, head_type, in_features, num_classes)
    
    # 冻结 backbone
    if freeze_backbone:
        _freeze_backbone(model, head_type)
    
    _print_model_params(model)
    return model


def _get_classifier_in_features(model: nn.Module, head_type: str) -> int:
    """获取分类头的输入特征维度"""
    if head_type == 'fc':
        return model.fc.in_features
    elif head_type == 'classifier':
        if hasattr(model.classifier, '__getitem__'):
            return model.classifier[-1].in_features
        return model.classifier.in_features
    elif head_type == 'head':
        return model.head.in_features
    else:
        raise ValueError(f"Unknown head type: {head_type}")


def _replace_classifier(model: nn.Module, head_type: str, in_features: int, num_classes: int):
    """替换分类头"""
    if head_type == 'fc':
        model.fc = nn.Linear(in_features, num_classes)
    elif head_type == 'classifier':
        if hasattr(model.classifier, '__getitem__'):
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            model.classifier = nn.Linear(in_features, num_classes)
    elif head_type == 'head':
        model.head = nn.Linear(in_features, num_classes)


def _freeze_backbone(model: nn.Module, head_type: str):
    """冻结 backbone，只训练分类头"""
    for name, param in model.named_parameters():
        if head_type == 'fc' and 'fc' not in name:
            param.requires_grad = False
        elif head_type == 'classifier' and 'classifier' not in name:
            param.requires_grad = False
        elif head_type == 'head' and 'head' not in name:
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  冻结 backbone: 可训练参数 {trainable/1e6:.2f}M / 总参数 {total/1e6:.2f}M")


def _print_model_params(model: nn.Module):
    """打印模型参数量"""
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  模型参数: {total:.2f}M, 可训练: {trainable:.2f}M")


def get_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
    weight_decay: float
) -> torch.optim.Optimizer:
    """获取优化器"""
    if optimizer_name == 'adam':
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    steps_per_epoch: int = None,
    warmup_epochs: int = 5
):
    """获取学习率调度器"""
    if scheduler_name == 'step':
        return StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    elif scheduler_name == 'onecycle':
        return OneCycleLR(optimizer, max_lr=optimizer.param_groups[0]['lr'],
                         epochs=epochs, steps_per_epoch=steps_per_epoch)
    elif scheduler_name == 'warmup_cosine':
        return WarmupCosineScheduler(optimizer, warmup_epochs=warmup_epochs, 
                                     total_epochs=epochs)
    elif scheduler_name == 'none':
        return None
    else:
        raise ValueError(f"不支持的调度器: {scheduler_name}")


def get_samples_per_class(data_dir: str) -> list:
    """获取每个类别的样本数"""
    train_file = Path(data_dir) / 'train.txt'
    if not train_file.exists():
        return None
    
    class_counts = {}
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                label = int(parts[-1])
                class_counts[label] = class_counts.get(label, 0) + 1
    
    num_classes = max(class_counts.keys()) + 1
    samples_per_class = [class_counts.get(i, 0) for i in range(num_classes)]
    return samples_per_class


def create_loss_function(args, device):
    """创建损失函数"""
    if args.loss_type == 'ce':
        if args.label_smoothing > 0:
            return LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        return nn.CrossEntropyLoss()
    
    elif args.loss_type == 'focal':
        if args.label_smoothing > 0:
            return FocalLoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
        return FocalLoss(gamma=args.focal_gamma)
    
    elif args.loss_type in ['cb_focal', 'cb_softmax']:
        samples_per_class = get_samples_per_class(args.data_dir)
        if samples_per_class is None:
            print("警告: 无法获取类别样本数，使用普通 Focal Loss")
            return FocalLoss(gamma=args.focal_gamma)
        
        loss_type = 'focal' if args.loss_type == 'cb_focal' else 'softmax'
        return ClassBalancedLoss(
            samples_per_class=samples_per_class,
            loss_type=loss_type,
            beta=0.9999,
            gamma=args.focal_gamma,
            device=device)
    
    else:
        raise ValueError(f"不支持的损失函数: {args.loss_type}")


def load_class_names(data_dir: str) -> list:
    """加载类别名称"""
    class_names_file = Path(data_dir) / 'class_names.txt'
    if class_names_file.exists():
        with open(class_names_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    return None


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
    parser.add_argument('--data_dir', type=str, default='D:/codes/work-projects/Gastrovision_models', help='数据目录（包含 train.txt, valid.txt 等）')
    parser.add_argument('--output_dir', type=str, default='D:/codes/work-projects/Gastrovision_results/convnext_tiny', help='输出目录')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='convnext_tiny', help='模型名称 (resnet18/34/50/101/152, resnext50/101, wide_resnet50/101, gastronet_resnet50_dino, gastronet_vit_small 等)')
    parser.add_argument('--pretrained', action='store_true', default=True, help='使用 ImageNet 预训练权重')
    parser.add_argument('--weights_path', type=str, default='D:/codes/work-projects/Gastrovision_results/pretrained/convnext_tiny.pth', help='预训练权重路径/目录（ImageNet 权重或 GastroNet 权重目录）')
    parser.add_argument('--freeze_backbone', action='store_true', default=False, help='冻结 backbone 只训练分类头')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=40, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='优化器')
    parser.add_argument('--scheduler', type=str, default='warmup_cosine', choices=['step', 'cosine', 'plateau', 'onecycle', 'warmup_cosine', 'none'], help='学习率调度器')
    parser.add_argument('--early_stopping', type=int, default=15, help='早停耐心值（0表示不使用）')
    
    # 数据加载参数
    parser.add_argument('--image_size', type=int, default=224, help='输入图像尺寸')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader 工作进程数')
    parser.add_argument('--use_albumentations', action='store_true', default=False, help='使用 Albumentations 数据增强（需安装 albumentations）')
    parser.add_argument('--augment_level', type=str, default='medium', choices=['light', 'medium', 'heavy'], help='Albumentations 增强级别')
    
    # 精度提升策略
    parser.add_argument('--mixup', type=float, default=0.0, help='Mixup alpha 参数（0=禁用，推荐0.2-0.4）')
    parser.add_argument('--cutmix', type=float, default=0.0, help='CutMix alpha 参数（0=禁用，推荐1.0）')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='标签平滑系数（0=禁用，推荐0.1）')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='学习率预热轮数')
    parser.add_argument('--loss_type', type=str, default='focal', choices=['ce', 'focal', 'cb_focal', 'cb_softmax'], help='损失函数类型（ce/focal/cb_focal/cb_softmax）')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal Loss 的 gamma 参数')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--use_class_weights', action='store_true', help='使用类别权重处理不平衡')
    parser.add_argument('--test_only', action='store_true', help='只进行测试（需要提供 checkpoint）')
    parser.add_argument('--resume', type=str, default=None, help='从 checkpoint 恢复训练')
    parser.add_argument('--visualize', action='store_true', default=True, help='训练后生成可视化图表')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # SSL 预训练模式
    if args.mode == 'ssl_pretrain':
        run_ssl_pretrain(args)
        return
    
    # 测试模式（使用 test_only 参数兼容）
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
    
    # 加载类别名称
    class_names = load_class_names(args.data_dir)
    num_classes = len(class_names) if class_names else 22  # 默认 22 类
    print(f"类别数: {num_classes}")
    
    # 创建模型
    print(f"\n创建模型: {args.model}")
    
    # 确定要加载的权重
    if args.mode == 'finetune' and args.resume:
        # finetune 模式：从 SSL 编码器权重加载
        weights_to_load = args.resume
        print(f"  微调模式: 加载 SSL 编码器权重 {args.resume}")
        use_pretrained = False
    else:
        # train 模式：使用 weights_path
        weights_to_load = args.weights_path
        use_pretrained = args.pretrained
    
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
    
    # 加载类别名称
    class_names = load_class_names(args.data_dir)
    
    # 损失函数
    criterion = create_loss_function(args, device)
    print(f"损失函数: {type(criterion).__name__}")
    
    # 优化器
    optimizer = get_optimizer(model, args.optimizer, args.lr, args.weight_decay)
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
        class_names=class_names)
    
    # 从 checkpoint 恢复（分类训练）
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
