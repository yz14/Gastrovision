"""
鲸鱼分类双头模型

保留原始鲸鱼分类的完整架构:
- Backbone: torchvision ResNet 系列
- BinaryHead: L2 归一化 + 线性缩放 (用于 focal_OHEM 损失)
- MarginHead: ArcFace 角度间隔分类头 (用于 softmax 损失)
- 输出: (logit_binary, logit_margin, features)

原始来源: colonnav_ssl/net/model_resnet101.py + MagrinLinear.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ============================================================
# 基础组件
# ============================================================

def l2_norm(x, axis=1):
    """L2 归一化"""
    norm = torch.norm(x, 2, axis, True)
    return torch.div(x, norm)


class MarginLinear(nn.Module):
    """
    ArcFace 角度间隔线性层

    实现 Additive Angular Margin Loss (ArcFace):
    https://arxiv.org/abs/1801.05599

    将余弦相似度转换为角度空间, 在目标类的角度上增加 margin m,
    再乘以缩放因子 s, 使 softmax 能正常工作。

    Args:
        embedding_size: 输入特征维度
        classnum: 类别数
        s: 缩放因子 (默认 64)
        m: 角度间隔 (默认 0.5, 约 28.6°)
    """

    def __init__(self, embedding_size=512, classnum=10008, s=64., m=0.5):
        super().__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, classnum))
        # 均匀初始化 + L2 归一化
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # sin(m) * m
        self.threshold = math.cos(math.pi - m)

    def forward(self, embeddings, label=None, is_infer=False):
        """
        Args:
            embeddings: L2 归一化后的特征 [B, D]
            label: 类别标签 [B] (训练时需要, 推理时为 None)
            is_infer: 是否为推理模式 (True 时不加 margin)

        Returns:
            缩放后的 logits [B, classnum]
        """
        # 权重 L2 归一化
        kernel_norm = l2_norm(self.kernel, axis=0)

        # 计算余弦相似度: cos(theta)
        cos_theta = torch.mm(embeddings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # 数值稳定性

        # 推理模式: 直接返回缩放后的余弦相似度
        output = cos_theta * 1.0  # 避免 in-place 操作

        if not is_infer and label is not None:
            # 训练模式: 在目标类添加角度 margin
            # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
            cos_theta_2 = torch.pow(cos_theta, 2)
            sin_theta_2 = 1 - cos_theta_2
            sin_theta = torch.sqrt(sin_theta_2.clamp(min=1e-12))
            cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

            # theta + m 超出 [0, pi] 范围时, 使用 CosFace 替代
            cond_v = cos_theta - self.threshold
            cond_mask = cond_v <= 0
            keep_val = cos_theta - self.mm
            cos_theta_m[cond_mask] = keep_val[cond_mask]

            # 只在目标类位置应用 margin
            nB = len(embeddings)
            idx_ = torch.arange(0, nB, dtype=torch.long, device=embeddings.device)
            output[idx_, label] = cos_theta_m[idx_, label]

        output *= self.s  # 缩放
        return output


# ============================================================
# 分类头
# ============================================================

class BinaryHead(nn.Module):
    """
    Binary 分类头

    对特征做 L2 归一化后通过线性层, 再乘以缩放因子。
    用于 focal_OHEM 损失。

    Args:
        num_class: 类别数
        emb_size: 输入特征维度
        s: 缩放因子
    """

    def __init__(self, num_class=10008, emb_size=2048, s=16.0):
        super().__init__()
        self.s = s
        self.fc = nn.Linear(emb_size, num_class)

    def forward(self, fea):
        fea = l2_norm(fea)
        logit = self.fc(fea) * self.s
        return logit


class MarginHead(nn.Module):
    """
    ArcFace Margin 分类头

    对特征做 L2 归一化后通过 MarginLinear 层。
    用于 softmax 损失。

    Args:
        num_class: 类别数
        emb_size: 输入特征维度
        s: ArcFace 缩放因子
        m: ArcFace 角度间隔
    """

    def __init__(self, num_class=10008, emb_size=2048, s=64., m=0.5):
        super().__init__()
        self.fc = MarginLinear(
            embedding_size=emb_size,
            classnum=num_class,
            s=s,
            m=m
        )

    def forward(self, fea, label=None, is_infer=False):
        fea = l2_norm(fea)
        logit = self.fc(fea, label, is_infer)
        return logit


# ============================================================
# Backbone 配置表
# ============================================================

# (工厂函数, 特征维度, 是否有 layer0 属性)
WHALE_BACKBONE_CONFIGS = {
    'resnet50':  (tvm.resnet50,  2048),
    'resnet101': (tvm.resnet101, 2048),
    'resnet152': (tvm.resnet152, 2048),
}


# ============================================================
# 完整模型
# ============================================================

class WhaleNet(nn.Module):
    """
    鲸鱼分类双头模型

    架构:
        Input → Backbone → AdaptiveAvgPool → BN → (BinaryHead, MarginHead)

    输出:
        (logit_binary, logit_margin, features)

    Args:
        backbone_name: backbone 名称 (见 WHALE_BACKBONE_CONFIGS)
        num_class: 类别数 (默认 10008 = 5004 * 2, 含翻转)
        s1: MarginHead 的缩放因子 (默认 64)
        m1: MarginHead 的角度间隔 (默认 0.5)
        s2: BinaryHead 的缩放因子 (默认 16)
        pretrained: 是否使用 ImageNet 预训练权重
        freeze_layers: 冻结的层列表 (如 ['layer0', 'layer1'])
    """

    def __init__(
        self,
        backbone_name='resnet101',
        num_class=10008,
        s1=64.,
        m1=0.5,
        s2=16.,
        pretrained=True,
        freeze_layers=None
    ):
        super().__init__()

        if backbone_name not in WHALE_BACKBONE_CONFIGS:
            raise ValueError(
                f"不支持的 backbone: {backbone_name}. "
                f"支持: {list(WHALE_BACKBONE_CONFIGS.keys())}"
            )

        model_fn, emb_size = WHALE_BACKBONE_CONFIGS[backbone_name]

        # 构建 backbone
        self.basemodel = model_fn(pretrained=pretrained)
        self.basemodel.avgpool = nn.AdaptiveAvgPool2d(1)
        self.basemodel.fc = nn.Sequential()  # 移除原始分类头

        # 构建 layer0 (conv1 + bn1 + relu + maxpool)
        self.basemodel.layer0 = nn.Sequential(
            self.basemodel.conv1,
            self.basemodel.bn1,
            self.basemodel.relu,
            self.basemodel.maxpool
        )

        # 特征 BN
        self.fea_bn = nn.BatchNorm1d(emb_size)
        self.fea_bn.bias.requires_grad_(False)

        # 双头
        self.margin_head = MarginHead(num_class, emb_size=emb_size, s=s1, m=m1)
        self.binary_head = BinaryHead(num_class, emb_size=emb_size, s=s2)

        # 冻结指定层
        if freeze_layers:
            self._freeze_layers(freeze_layers)

        # 打印参数量
        total = sum(p.numel() for p in self.parameters()) / 1e6
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        print(f"  WhaleNet ({backbone_name}): {total:.2f}M 参数, {trainable:.2f}M 可训练")

    def _freeze_layers(self, layer_names):
        """冻结指定层的参数"""
        for name in layer_names:
            layer = getattr(self.basemodel, name, None)
            if layer is not None:
                for p in layer.parameters():
                    p.requires_grad = False
                print(f"  冻结: {name}")

    def forward(self, x, label=None, is_infer=False):
        """
        前向传播

        Args:
            x: 输入图像 [B, 3, H, W], 值域 [0, 1]
            label: 类别标签 [B] (训练时需要, 用于 ArcFace margin)
            is_infer: 是否为推理模式

        Returns:
            logit_binary: Binary head 输出 [B, num_class]
            logit_margin: Margin head 输出 [B, num_class]
            fea: 特征向量 [B, emb_size]
        """
        # ImageNet 归一化 (在模型内部完成, 与原始代码一致)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        x = torch.cat([
            (x[:, [0]] - mean[0]) / std[0],
            (x[:, [1]] - mean[1]) / std[1],
            (x[:, [2]] - mean[2]) / std[2],
        ], 1)

        # Backbone 前向
        x = self.basemodel.layer0(x)
        x = self.basemodel.layer1(x)
        x = self.basemodel.layer2(x)
        x = self.basemodel.layer3(x)
        x = self.basemodel.layer4(x)

        # 全局平均池化 + 展平
        x = F.adaptive_avg_pool2d(x, 1)
        fea = x.view(x.size(0), -1)

        # 特征 BN
        fea = self.fea_bn(fea)

        # 双头输出
        logit_binary = self.binary_head(fea)
        logit_margin = self.margin_head(fea, label=label, is_infer=is_infer)

        return logit_binary, logit_margin, fea


def get_whale_model(cfg):
    """
    根据 YAML 配置创建鲸鱼模型

    Args:
        cfg: 配置对象, 需要包含:
            - model: backbone 名称
            - s1, m1, s2: ArcFace 参数
            - pretrained: 是否预训练
            - freeze_layers: 冻结层列表 (可选)

    Returns:
        WhaleNet 实例
    """
    # 计算类别数
    whale_id_num = getattr(cfg, 'whale_id_num', 5004)
    num_class = whale_id_num * 2  # 翻转后类别翻倍

    freeze_layers = getattr(cfg, 'freeze_layers', ['layer0', 'layer1'])

    model = WhaleNet(
        backbone_name=cfg.model,
        num_class=num_class,
        s1=cfg.s1,
        m1=cfg.m1,
        s2=cfg.s2,
        pretrained=cfg.pretrained,
        freeze_layers=freeze_layers
    )

    return model
