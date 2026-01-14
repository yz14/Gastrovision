"""
GastroNet-5M 预训练模型集成模块 (修复版)

GastroNet-5M 是目前最大的公开胃肠内镜图像数据集（482万张图像），
预训练模型可显著提升下游任务性能。

权重来源: https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights

可用的预训练模型:
- ResNet50 + DINO (推荐)
- ResNet50 + SIMCLRv2
- ResNet50 + MOCOv2
- ResNet50 + Billion-Scale + DINO
- ViT-small + DINO

下载权重:
    pip install huggingface_hub
    python GastroNet_5m.py --download gastronet_resnet50_dino

引用:
    Jong, M. R., et al. (2025). GastroNet-5M: A Multicenter Dataset for Developing Foundation Models... Gastroenterology.
    Boers, T. G. W., et al. (2024). Foundation models in gastrointestinal endoscopic AI... Medical Image Analysis.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    import torchvision.models as tv_models
except ImportError:
    tv_models = None

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


# ========== 修复：使用正确的 Hugging Face 文件名 ==========
GASTRONET_WEIGHTS = {
    # ResNet50 变体
    'gastronet_resnet50_dino': {
        'filename': 'RN50_GastroNet-5M_DINOv1.pth',  # 修复文件名
        'arch': 'resnet50',
        'description': 'ResNet50 + DINO on GastroNet-5M (推荐)',
    },
    'gastronet_resnet50_simclr': {
        'filename': 'RN50_GastroNet-5M_SIMCLRv2.pth',  # 修复文件名
        'arch': 'resnet50',
        'description': 'ResNet50 + SIMCLRv2 on GastroNet-5M',
    },
    'gastronet_resnet50_moco': {
        'filename': 'RN50_GastroNet-5M_MOCOv2.pth',  # 修复文件名
        'arch': 'resnet50',
        'description': 'ResNet50 + MOCOv2 on GastroNet-5M',
    },
    'gastronet_resnet50_billion_dino': {
        'filename': 'RN50_Billion-Scale-SWSL+GastroNet-5M_DINOv1.pth',  # 修复文件名
        'arch': 'resnet50',
        'description': 'ResNet50 (Billion-Scale init) + DINO on GastroNet-5M',
    },
    # 不同数据量的变体
    'gastronet_resnet50_dino_1m': {
        'filename': 'RN50_GastroNet-1M_DINOv1.pth',  # 修复文件名
        'arch': 'resnet50',
        'description': 'ResNet50 + DINO on GastroNet-1M',
    },
    'gastronet_resnet50_dino_200k': {
        'filename': 'RN50_GastroNet-200K_DINOv1.pth',  # 修复文件名
        'arch': 'resnet50',
        'description': 'ResNet50 + DINO on GastroNet-200K',
    },
    # ViT 变体（注意：文件名未验证，可能不可用）
    # 'gastronet_vit_small': {
    #     'filename': 'ViT-S_GastroNet-5M_DINOv1.pth',  # 文件名需验证
    #     'arch': 'vit_small',
    #     'description': 'ViT-small + DINO on GastroNet-5M (暂不可用)',
    # },
}


def list_available_models() -> Dict[str, str]:
    """列出所有可用的 GastroNet-5M 模型"""
    return {name: info['description'] for name, info in GASTRONET_WEIGHTS.items()}


def download_weights(
    model_name: str,
    save_dir: str = './pretrained',
    force: bool = False
) -> str:
    """
    从 Hugging Face 下载 GastroNet-5M 预训练权重
    
    Args:
        model_name: 模型名称（如 'gastronet_resnet50_dino'）
        save_dir: 保存目录
        force: 是否强制重新下载
        
    Returns:
        权重文件路径
    """
    if model_name not in GASTRONET_WEIGHTS:
        raise ValueError(f"未知模型: {model_name}。可用模型: {list(GASTRONET_WEIGHTS.keys())}")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filename = GASTRONET_WEIGHTS[model_name]['filename']
    local_path = save_dir / filename
    
    if local_path.exists() and not force:
        print(f"✓ 权重文件已存在: {local_path}")
        return str(local_path)
    
    try:
        from huggingface_hub import hf_hub_download
        
        print(f"正在从 Hugging Face 下载 {filename}...")
        print(f"提示: 该仓库需要接受使用条款才能访问")
        print(f"请访问: https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights")
        print(f"并登录 Hugging Face 账号接受条款后再下载\n")
        
        downloaded_path = hf_hub_download(
            repo_id='tgwboers/GastroNet-5M_Pretrained_Weights',
            filename=filename,
            local_dir=str(save_dir),
            cache_dir=None,  # 避免使用缓存
            force_download=force
        )
        print(f"✓ 下载完成: {downloaded_path}")
        return downloaded_path
        
    except ImportError:
        raise ImportError(
            "需要安装 huggingface_hub: pip install huggingface_hub\n"
            "或者手动下载: https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights"
        )
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print(f"\n可能的原因:")
        print(f"1. 需要登录 Hugging Face 并接受使用条款")
        print(f"   运行: huggingface-cli login")
        print(f"   然后访问: https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights")
        print(f"2. 文件名可能不存在（ViT 模型文件名需要验证）")
        print(f"3. 网络连接问题\n")
        raise


def create_resnet50_backbone() -> nn.Module:
    """创建 ResNet50 backbone（不含分类头）"""
    if tv_models is None:
        raise ImportError("需要安装 torchvision")
    
    # 创建标准 ResNet50
    model = tv_models.resnet50(weights=None)
    return model


def create_vit_small_backbone() -> nn.Module:
    """创建 ViT-small backbone"""
    if not HAS_TIMM:
        raise ImportError("ViT 模型需要安装 timm: pip install timm")
    
    # 使用 timm 创建 ViT-small（保留分类头以获取 num_features）
    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=1000)
    return model


def load_gastronet_weights(
    model: nn.Module,
    weights_path: str,
    strict: bool = False
) -> nn.Module:
    """
    加载 GastroNet-5M 预训练权重
    
    Args:
        model: PyTorch 模型
        weights_path: 权重文件路径
        strict: 是否严格匹配
        
    Returns:
        加载权重后的模型
    """
    print(f"正在加载权重: {weights_path}")
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    
    # 处理不同的 checkpoint 格式
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 移除可能的前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        # 移除 DDP 前缀
        if k.startswith('module.'):
            k = k[7:]
        # 移除 backbone 前缀
        if k.startswith('backbone.'):
            k = k[9:]
        new_state_dict[k] = v
    
    # 加载权重
    missing, unexpected = model.load_state_dict(new_state_dict, strict=strict)
    
    if missing:
        print(f"⚠ 缺失的键: {len(missing)} 个")
        if len(missing) <= 5:
            print(f"   {missing}")
    if unexpected:
        print(f"⚠ 未预期的键: {len(unexpected)} 个")
        if len(unexpected) <= 5:
            print(f"   {unexpected}")
    
    return model


def get_gastronet_model(
    model_name: str,
    num_classes: int,
    weights_dir: str = './pretrained',
    freeze_backbone: bool = False,
    download: bool = True
) -> nn.Module:
    """
    获取 GastroNet-5M 预训练模型并适配到指定类别数
    
    Args:
        model_name: 模型名称（如 'gastronet_resnet50_dino'）
        num_classes: 目标类别数量
        weights_dir: 权重保存目录
        freeze_backbone: 是否冻结 backbone
        download: 是否自动下载权重
        
    Returns:
        适配后的模型
    """
    if model_name not in GASTRONET_WEIGHTS:
        raise ValueError(f"未知模型: {model_name}。可用模型: {list(GASTRONET_WEIGHTS.keys())}")
    
    info = GASTRONET_WEIGHTS[model_name]
    arch = info['arch']
    
    # 创建 backbone
    if arch == 'resnet50':
        model = create_resnet50_backbone()
        in_features = model.fc.in_features  # 2048
    elif arch == 'vit_small':
        model = create_vit_small_backbone()
        in_features = model.head.in_features  # 384
    else:
        raise ValueError(f"不支持的架构: {arch}")
    
    # 下载或定位权重
    weights_path = Path(weights_dir) / info['filename']
    if download and not weights_path.exists():
        weights_path = download_weights(model_name, weights_dir)
    
    if not Path(weights_path).exists():
        raise FileNotFoundError(
            f"权重文件不存在: {weights_path}\n"
            f"请先下载: python GastroNet_5m.py --download {model_name}"
        )
    
    # 加载预训练权重
    print(f"正在加载 GastroNet-5M 预训练权重: {info['description']}")
    model = load_gastronet_weights(model, str(weights_path), strict=False)
    
    # 替换分类头
    if arch == 'resnet50':
        model.fc = nn.Linear(in_features, num_classes)
    elif arch == 'vit_small':
        model.head = nn.Linear(in_features, num_classes)
    
    print(f"✓ 替换分类头: {in_features} -> {num_classes}")
    
    # 冻结 backbone（如果需要）
    if freeze_backbone:
        freeze_backbone_layers(model, arch)
    
    return model


def freeze_backbone_layers(model: nn.Module, arch: str) -> None:
    """冻结 backbone 层，只训练分类头"""
    if arch == 'resnet50':
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    elif arch == 'vit_small':
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"✓ 冻结 backbone: 可训练参数 {trainable:,} / 总参数 {total:,}")


def unfreeze_backbone_layers(model: nn.Module) -> None:
    """解冻所有层"""
    for param in model.parameters():
        param.requires_grad = True
    
    total = sum(p.numel() for p in model.parameters())
    print(f"✓ 解冻所有层: 可训练参数 {total:,}")


# 命令行入口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GastroNet-5M 预训练模型工具")
    parser.add_argument('--list', action='store_true', help='列出可用模型')
    parser.add_argument('--download', type=str, help='下载指定模型权重')
    parser.add_argument('--download_all', action='store_true', help='下载所有模型权重')
    parser.add_argument('--save_dir', type=str, default='./pretrained', help='保存目录')
    parser.add_argument('--login_info', action='store_true', help='显示 Hugging Face 登录说明')
    
    args = parser.parse_args()
    
    if args.login_info:
        print("\n" + "="*60)
        print("Hugging Face 登录说明")
        print("="*60)
        print("\n该仓库需要接受使用条款才能访问，请按以下步骤操作:\n")
        print("1. 安装 Hugging Face CLI:")
        print("   pip install huggingface_hub\n")
        print("2. 登录 Hugging Face:")
        print("   huggingface-cli login")
        print("   (输入你的 Hugging Face token)\n")
        print("3. 访问模型页面并接受使用条款:")
        print("   https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights\n")
        print("4. 然后重新运行下载命令")
        print("="*60 + "\n")
        
    elif args.list:
        print("\n可用的 GastroNet-5M 预训练模型:")
        print("=" * 60)
        for name, desc in list_available_models().items():
            info = GASTRONET_WEIGHTS[name]
            print(f"\n  {name}:")
            print(f"    描述: {desc}")
            print(f"    文件: {info['filename']}")
        print("\n" + "=" * 60 + "\n")
        
    elif args.download:
        try:
            download_weights(args.download, args.save_dir)
        except Exception as e:
            print(f"\n提示: 如需帮助，运行 python GastroNet_5m.py --login_info")
        
    elif args.download_all:
        print("下载所有 GastroNet-5M 预训练权重...")
        for name in GASTRONET_WEIGHTS.keys():
            try:
                download_weights(name, args.save_dir)
            except Exception as e:
                print(f"✗ 下载 {name} 失败: {e}")
    else:
        parser.print_help()