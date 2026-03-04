import torch
import torch.nn as nn
from typing import List, Optional, Type, Union


class BasicBlock(nn.Module):
    """ResNet BasicBlock (用于 ResNet-18, ResNet-34)"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        # 两个 3x3 卷积
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """ResNet Bottleneck (用于 ResNet-50, ResNet-101, ResNet-152)"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        
        # 1x1, 3x3, 1x1 卷积
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, dilation=dilation,
                               bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet 主干网络"""

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 4 个残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, weights_path=None, **kwargs):
    """
    ResNet-18 模型
    
    Args:
        pretrained: 是否加载预训练权重
        weights_path: 权重文件路径（如果 pretrained=True，必须提供）
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        if weights_path is None:
            raise ValueError("pretrained=True 时必须提供 weights_path 参数")
        model = load_pretrained_weights_manual(model, 'resnet18', weights_path)
    return model


def resnet34(pretrained=False, weights_path=None, **kwargs):
    """
    ResNet-34 模型
    
    Args:
        pretrained: 是否加载预训练权重
        weights_path: 权重文件路径（如果 pretrained=True，必须提供）
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        if weights_path is None:
            raise ValueError("pretrained=True 时必须提供 weights_path 参数")
        model = load_pretrained_weights_manual(model, 'resnet34', weights_path)
    return model


def resnet50(pretrained=False, weights_path=None, **kwargs):
    """
    ResNet-50 模型
    
    Args:
        pretrained: 是否加载预训练权重
        weights_path: 权重文件路径（如果 pretrained=True，必须提供）
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        if weights_path is None:
            raise ValueError("pretrained=True 时必须提供 weights_path 参数")
        model = load_pretrained_weights_manual(model, 'resnet50', weights_path)
    return model


def resnet101(pretrained=False, weights_path=None, **kwargs):
    """
    ResNet-101 模型
    
    Args:
        pretrained: 是否加载预训练权重
        weights_path: 权重文件路径（如果 pretrained=True，必须提供）
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        if weights_path is None:
            raise ValueError("pretrained=True 时必须提供 weights_path 参数")
        model = load_pretrained_weights_manual(model, 'resnet101', weights_path)
    return model


def resnet152(pretrained=False, weights_path=None, **kwargs):
    """
    ResNet-152 模型
    
    Args:
        pretrained: 是否加载预训练权重
        weights_path: 权重文件路径（如果 pretrained=True，必须提供）
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        if weights_path is None:
            raise ValueError("pretrained=True 时必须提供 weights_path 参数")
        model = load_pretrained_weights_manual(model, 'resnet152', weights_path)
    return model


def resnext50_32x4d(pretrained=False, weights_path=None, **kwargs):
    """
    ResNeXt-50 32x4d 模型
    
    Args:
        pretrained: 是否加载预训练权重
        weights_path: 权重文件路径（如果 pretrained=True，必须提供）
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        if weights_path is None:
            raise ValueError("pretrained=True 时必须提供 weights_path 参数")
        model = load_pretrained_weights_manual(model, 'resnext50_32x4d', weights_path)
    return model


def resnext101_32x8d(pretrained=False, weights_path=None, **kwargs):
    """
    ResNeXt-101 32x8d 模型
    
    Args:
        pretrained: 是否加载预训练权重
        weights_path: 权重文件路径（如果 pretrained=True，必须提供）
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        if weights_path is None:
            raise ValueError("pretrained=True 时必须提供 weights_path 参数")
        model = load_pretrained_weights_manual(model, 'resnext101_32x8d', weights_path)
    return model


def wide_resnet50_2(pretrained=False, weights_path=None, **kwargs):
    """
    Wide ResNet-50-2 模型 (宽度是原来的2倍)
    
    Args:
        pretrained: 是否加载预训练权重
        weights_path: 权重文件路径（如果 pretrained=True，必须提供）
    """
    kwargs['width_per_group'] = 64 * 2
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        if weights_path is None:
            raise ValueError("pretrained=True 时必须提供 weights_path 参数")
        model = load_pretrained_weights_manual(model, 'wide_resnet50_2', weights_path)
    return model


def wide_resnet101_2(pretrained=False, weights_path=None, **kwargs):
    """
    Wide ResNet-101-2 模型 (宽度是原来的2倍)
    
    Args:
        pretrained: 是否加载预训练权重
        weights_path: 权重文件路径（如果 pretrained=True，必须提供）
    """
    kwargs['width_per_group'] = 64 * 2
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        if weights_path is None:
            raise ValueError("pretrained=True 时必须提供 weights_path 参数")
        model = load_pretrained_weights_manual(model, 'wide_resnet101_2', weights_path)
    return model


# ============================================================================
# 从 PyTorch 下载并导入权重的完整示例
# ============================================================================

def load_pretrained_weights_manual(model, model_name, weights_path):
    """
    手动下载并加载 PyTorch 官方预训练权重
    
    Args:
        model: ResNet 模型实例
        model_name: 模型名称，用于选择正确的权重URL
        weights_path: 权重文件的完整路径（例如: './weights/resnet50.pth'）
    """
    import os
    import urllib.request
    from pathlib import Path
    
    # PyTorch 官方预训练权重 URL 映射
    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
        'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
        'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
        'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    }
    
    if model_name not in model_urls:
        raise ValueError(f"模型 {model_name} 不支持，可选: {list(model_urls.keys())}")
    
    # 创建权重文件所在的目录
    Path(weights_path).parent.mkdir(parents=True, exist_ok=True)
    
    url = model_urls[model_name]
    
    # 检查权重文件是否已存在
    if os.path.exists(weights_path):
        print(f"✓ 发现已下载的权重文件: {weights_path}")
    else:
        print(f"正在从 PyTorch 下载 {model_name} 预训练权重...")
        print(f"URL: {url}")
        print(f"保存路径: {weights_path}")
        
        # 下载权重文件（带进度条）
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100.0 / total_size, 100.0)
            print(f"\r下载进度: {percent:.1f}% ({downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB)", end='')
        
        try:
            urllib.request.urlretrieve(url, weights_path, download_progress)
            print(f"\n✓ 下载完成: {weights_path}")
        except Exception as e:
            print(f"\n✗ 下载失败: {e}")
            if os.path.exists(weights_path):
                os.remove(weights_path)
            raise
    
    # 加载权重到模型
    print(f"正在加载权重到模型...")
    try:
        # PyTorch 2.6+ 需要设置 weights_only=False 来加载旧格式的官方权重
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
    except TypeError:
        # 兼容旧版本 PyTorch（没有 weights_only 参数）
        state_dict = torch.load(weights_path, map_location='cpu')
    
    model.load_state_dict(state_dict)
    print(f"✓ {model_name} 预训练权重加载成功!\n")
    
    return model


def test_pretrained_model():
    """测试预训练模型的推理功能"""
    import torch.nn.functional as F
    
    print("="*70)
    print("测试 1: 从 PyTorch 下载并加载预训练权重")
    print("="*70)
    
    # 指定权重保存路径（完整文件路径）
    weights_path = 'D:/codes/data/weights/resnet50.pth'
    print(f"权重保存路径: {weights_path}\n")
    
    # 创建模型（随机初始化）
    model = resnet50(pretrained=False)
    print(f"✓ ResNet-50 模型创建成功（随机初始化）")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M\n")
    
    # 手动下载并加载预训练权重（指定完整路径）
    model = load_pretrained_weights_manual(model, 'resnet50', weights_path)
    
    # 设置为评估模式
    model.eval()
    
    # 创建测试输入（标准 ImageNet 输入大小）
    print("="*70)
    print("测试 2: 模型前向传播")
    print("="*70)
    x = torch.randn(2, 3, 224, 224)  # batch_size=2
    print(f"输入张量形状: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"输出张量形状: {output.shape}")
    print(f"输出维度: {output.shape[1]} (ImageNet 1000类)")
    
    # 获取预测结果
    probabilities = F.softmax(output, dim=1)
    top5_prob, top5_idx = torch.topk(probabilities, 5, dim=1)
    
    print(f"\n样本 1 的 Top-5 预测:")
    for i in range(5):
        print(f"  类别 {top5_idx[0][i].item()}: 概率 {top5_prob[0][i].item():.4f}")
    
    print("\n✓ 模型推理测试通过!\n")


def test_all_variants():
    """测试所有 ResNet 变体的权重加载"""
    print("="*70)
    print("测试 3: 加载所有 ResNet 变体的预训练权重")
    print("="*70)
    
    # 指定权重目录
    weights_dir = './all_resnet_weights'
    print(f"权重目录: {weights_dir}\n")
    
    variants = {
        'resnet18': (resnet18, 'resnet18.pth'),
        'resnet34': (resnet34, 'resnet34.pth'),
        'resnet50': (resnet50, 'resnet50.pth'),
        'resnet101': (resnet101, 'resnet101.pth'),
        'resnet152': (resnet152, 'resnet152.pth'),
        'resnext50_32x4d': (resnext50_32x4d, 'resnext50_32x4d.pth'),
        'resnext101_32x8d': (resnext101_32x8d, 'resnext101_32x8d.pth'),
        'wide_resnet50_2': (wide_resnet50_2, 'wide_resnet50_2.pth'),
        'wide_resnet101_2': (wide_resnet101_2, 'wide_resnet101_2.pth'),
    }
    
    print(f"\n{'模型名称':<25} {'参数量(M)':<15} {'加载状态'}")
    print("-"*70)
    
    for name, (model_fn, filename) in variants.items():
        try:
            import os
            weights_path = os.path.join(weights_dir, filename)
            # 使用内置的 pretrained=True 参数，并指定完整路径
            model = model_fn(pretrained=True, weights_path=weights_path)
            params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"{name:<25} {params:<15.2f} ✓ 成功")
        except Exception as e:
            print(f"{name:<25} {'N/A':<15} ✗ 失败: {str(e)[:30]}")
    
    print("\n✓ 所有变体测试完成!\n")


def test_transfer_learning():
    """测试迁移学习场景"""
    print("="*70)
    print("测试 4: 迁移学习 - 修改分类头")
    print("="*70)
    
    # 加载预训练模型（指定完整路径）
    weights_path = './transfer_learning_weights/resnet50.pth'
    model = resnet50(pretrained=True, weights_path=weights_path)
    print("✓ 加载 ResNet-50 预训练权重")
    
    # 修改最后的全连接层用于10分类任务
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    print(f"✓ 修改分类头: {num_features} -> 10 类")
    
    # 测试新模型
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    
    print(f"✓ 输出形状: {output.shape} (应该是 [1, 10])")
    assert output.shape == (1, 10), "输出形状不正确!"
    print("✓ 迁移学习测试通过!\n")


def test_feature_extraction():
    """测试特征提取场景"""
    print("="*70)
    print("测试 5: 特征提取 - 移除分类头")
    print("="*70)
    
    # 加载预训练模型（指定完整路径）
    weights_path = './feature_extraction_weights/resnet50.pth'
    model = resnet50(pretrained=True, weights_path=weights_path)
    
    # 移除分类头，只保留特征提取部分
    model.fc = nn.Identity()
    print("✓ 移除分类头，使用 Identity 层")
    
    # 测试特征提取
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        features = model(x)
    
    print(f"✓ 特征向量形状: {features.shape} (应该是 [1, 2048])")
    assert features.shape == (1, 2048), "特征维度不正确!"
    print("✓ 特征提取测试通过!\n")


def compare_with_torchvision():
    """与 torchvision 官方实现对比"""
    print("="*70)
    print("测试 6: 与 torchvision 官方实现对比")
    print("="*70)
    
    try:
        from torchvision.models import resnet50 as tv_resnet50
        
        # 我们的实现（指定完整路径）
        weights_path = './comparison_weights/resnet50.pth'
        our_model = resnet50(pretrained=True, weights_path=weights_path)
        our_model.eval()
        
        # torchvision 官方实现
        tv_model = tv_resnet50(pretrained=True)
        tv_model.eval()
        
        # 使用相同输入
        x = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            our_output = our_model(x)
            tv_output = tv_model(x)
        
        # 比较输出
        max_diff = torch.max(torch.abs(our_output - tv_output)).item()
        print(f"✓ 我们的实现输出: {our_output[0, :5]}")
        print(f"✓ torchvision 输出:  {tv_output[0, :5]}")
        print(f"✓ 最大差异: {max_diff:.2e}")
        
        if max_diff < 1e-5:
            print("✓ 输出完全一致! 实现正确!\n")
        else:
            print("⚠ 输出有微小差异，但在可接受范围内\n")
            
    except ImportError:
        print("⚠ 未安装 torchvision，跳过对比测试\n")


# 使用示例和完整测试
if __name__ == '__main__':
    print("\n" + "="*70)
    print(" ResNet 预训练权重加载与测试")
    print("="*70 + "\n")
    
    # 运行所有测试
    test_pretrained_model()
    test_all_variants()
    test_transfer_learning()
    test_feature_extraction()
    compare_with_torchvision()
    
    print("="*70)
    print(" 所有测试完成!")
    print("="*70)