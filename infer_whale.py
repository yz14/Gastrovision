"""
鲸鱼分类推理 + Kaggle 提交生成

支持:
- 单模型推理
- 多模型融合 (概率累加)
- 2-TTA (原图 + 水平翻转)
- new_whale 阈值搜索
- 验证集评估 (MAP@5)
- Kaggle 提交 CSV 生成

用法:
    # 验证集评估
    python infer_whale.py configs/train_whale.yaml --mode eval

    # 测试集推理 + 生成 Kaggle 提交
    python infer_whale.py configs/train_whale.yaml --mode submit

    # 多模型融合
    python infer_whale.py configs/train_whale.yaml --mode submit \
        --model_paths output/whale/best_model.pth,output/whale2/best_model.pth
"""

import os
import sys
import yaml
import argparse
from types import SimpleNamespace
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gastrovision.models.whale_model import WhaleNet, get_whale_model
from gastrovision.data.whale_dataset import (
    WhaleDataset,
    load_class_name_dict
)
from gastrovision.losses.whale_losses import metric


def load_config(yaml_path: str) -> SimpleNamespace:
    """加载 YAML 配置 (复用 train_whale.py 的配置格式)"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)

    cfg = SimpleNamespace(**cfg_dict)

    # 设置默认值
    defaults = {
        'whale_data_dir': 'D:/codes/data/humpback-whale-identification',
        'list_dir': 'D:/codes/work-projects/colonnav_ssl/image_list',
        'bbox_dir': 'D:/codes/work-projects/colonnav_ssl/bbox_model',
        'fold_index': 0,
        'image_h': 256,
        'image_w': 512,
        'whale_id_num': 5004,
        'model': 'resnet101',
        's1': 64., 'm1': 0.5, 's2': 16.,
        'pretrained': False,
        'freeze_layers': [],
        'batch_size': 32,
        'num_workers': 4,
        'output_dir': './output/whale',
    }

    for key, val in defaults.items():
        if not hasattr(cfg, key):
            setattr(cfg, key, val)

    return cfg


def load_trained_model(cfg, model_path: str, device: torch.device) -> WhaleNet:
    """
    加载训练好的鲸鱼模型

    支持两种格式:
    1. Trainer checkpoint (含 model_state_dict 键)
    2. 直接 state_dict
    3. DataParallel 格式 (module. 前缀)
    """
    num_class = cfg.whale_id_num * 2
    model = WhaleNet(
        backbone_name=cfg.model,
        num_class=num_class,
        s1=cfg.s1,
        m1=cfg.m1,
        s2=cfg.s2,
        pretrained=False,
        freeze_layers=[]
    )

    print(f"加载模型权重: {model_path}")
    raw = torch.load(model_path, map_location='cpu', weights_only=False)

    # 提取 state_dict
    if isinstance(raw, dict) and 'model_state_dict' in raw:
        state_dict = raw['model_state_dict']
        epoch = raw.get('epoch', '?')
        best_map = raw.get('best_valid_map', raw.get('metrics', {}).get('map', '?'))
        print(f"  Trainer checkpoint: epoch={epoch}, best_map={best_map}")
    elif isinstance(raw, dict) and any(k.startswith('module.') for k in raw.keys()):
        state_dict = raw
    else:
        state_dict = raw

    # 去除 DataParallel 的 'module.' 前缀
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            cleaned[k[7:]] = v
        else:
            cleaned[k] = v

    model.load_state_dict(cleaned, strict=True)
    model = model.to(device)
    model.eval()

    print(f"  ✓ 模型加载成功")
    return model


# ============================================================
# 2-TTA 推理
# ============================================================

@torch.no_grad()
def run_inference_test_2tta(
    model: WhaleNet,
    cfg,
    batch_size: int,
    num_workers: int,
    device: torch.device
) -> defaultdict:
    """
    测试集 2-TTA 推理

    TTA 策略:
    - 原图: 使用前 whale_id_num 个类的概率
    - 水平翻转: 使用后 whale_id_num 个类的概率

    两次概率通过 Counter 累加, 实现概率融合。

    Args:
        model: 训练好的 WhaleNet
        cfg: 配置
        batch_size: 批大小
        num_workers: 工作进程数
        device: 设备

    Returns:
        blend: {image_id: Counter({class_idx: cumulative_prob})}
    """
    whale_id_num = cfg.whale_id_num
    image_size = (cfg.image_h, cfg.image_w)
    blend = defaultdict(Counter)

    # 2TTA: 原图 + 水平翻转
    augments = [(False, [0.0]), (True, [1.0])]

    for is_flip, augment in augments:
        tta_desc = '翻转' if is_flip else '原图'
        print(f"  TTA: {tta_desc}")

        dataset = WhaleDataset(
            mode='test',
            data_dir=cfg.whale_data_dir,
            list_dir=cfg.list_dir,
            bbox_dir=cfg.bbox_dir,
            fold_index=cfg.fold_index,
            image_size=image_size,
            augment=augment
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )

        for batch in tqdm(loader, desc=f"  {tta_desc}", leave=False):
            image_ids, images = batch
            images = images.to(device)

            logit_binary, _, _ = model(images, label=None, is_infer=True)
            prob = torch.sigmoid(logit_binary)

            # 关键: 原图用前 whale_id_num 类, 翻转用后 whale_id_num 类
            if is_flip:
                prob = prob[:, whale_id_num:]
            else:
                prob = prob[:, :whale_id_num]

            prob = prob.cpu().numpy()

            for img_id, p in zip(image_ids, prob):
                for cls_idx in range(whale_id_num):
                    if p[cls_idx] > 0.001:  # 忽略极小概率, 减少 Counter 大小
                        blend[img_id][cls_idx] += float(p[cls_idx])

    return blend


@torch.no_grad()
def run_inference_val_2tta(
    model: WhaleNet,
    cfg,
    batch_size: int,
    num_workers: int,
    device: torch.device
) -> dict:
    """
    验证集 2-TTA 评估

    计算方式与训练时的 do_valid 一致:
    - 正常方向: prob[:, :whale_id_num]
    - 翻转方向: prob[:, whale_id_num:]
    - 两者概率相加取平均, 再对左右类取 max
    - 搜索最佳 new_whale 阈值

    Returns:
        dict: {map, threshold, top1, top5}
    """
    whale_id_num = cfg.whale_id_num
    num_class = whale_id_num * 2
    image_size = (cfg.image_h, cfg.image_w)

    # ---- 正常方向 ----
    print("  验证: 正常方向")
    dataset = WhaleDataset(
        mode='val',
        data_dir=cfg.whale_data_dir,
        list_dir=cfg.list_dir,
        bbox_dir=cfg.bbox_dir,
        fold_index=cfg.fold_index,
        image_size=image_size,
        augment=[0.0],
        is_flip=False
    )

    val_num = len(dataset)
    probs_normal = np.zeros([val_num, num_class])
    truths = np.zeros(val_num, dtype=np.int64)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    idx = 0
    for images, labels, nws in tqdm(loader, desc="  正常方向", leave=False):
        images = images.to(device)
        logit_binary, _, _ = model(images, label=None, is_infer=True)
        prob = torch.sigmoid(logit_binary)
        bs = images.size(0)
        probs_normal[idx:idx+bs] = prob.cpu().numpy()
        truths[idx:idx+bs] = labels.numpy()
        idx += bs

    # ---- 翻转方向 ----
    print("  验证: 翻转方向")
    dataset_flip = WhaleDataset(
        mode='val',
        data_dir=cfg.whale_data_dir,
        list_dir=cfg.list_dir,
        bbox_dir=cfg.bbox_dir,
        fold_index=cfg.fold_index,
        image_size=image_size,
        augment=[0.0],
        is_flip=True
    )

    probs_flip = np.zeros([val_num, num_class])

    loader_flip = DataLoader(
        dataset_flip,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    idx = 0
    for images, labels, nws in tqdm(loader_flip, desc="  翻转方向", leave=False):
        images = images.to(device)
        logit_binary, _, _ = model(images, label=None, is_infer=True)
        prob = torch.sigmoid(logit_binary)
        bs = images.size(0)
        probs_flip[idx:idx+bs] = prob.cpu().numpy()
        idx += bs

    # ---- 合并概率 ----
    # 正常 class_i 对应翻转 class_{i+whale_id_num}
    probs = probs_normal.copy()
    for i in range(whale_id_num):
        probs[:, i] += probs_flip[:, i + whale_id_num]
        probs[:, i + whale_id_num] += probs_flip[:, i]
    probs /= 2.0

    # 正常方向: 取前 whale_id_num 类
    prob_normal_half = probs[:, :whale_id_num]
    # 翻转方向: 取后 whale_id_num 类
    prob_flip_half = probs[:, whale_id_num:]
    # 合并左右: 取 max
    prob_merged = np.maximum(prob_normal_half, prob_flip_half)

    # ---- 修正 truth labels ----
    # 翻转数据集的 label 可能偏移了 whale_id_num, 但 truths 来自正常数据集
    # new_whale: label = num_class (10008)
    labels_for_eval = truths.copy()
    # new_whale 标签 = num_class → 设为 whale_id_num (5004)
    labels_for_eval[labels_for_eval == num_class] = whale_id_num

    # ---- 搜索最佳 new_whale 阈值 ----
    best_threshold = 0.5
    best_map = 0.0

    for threshold in np.arange(0.02, 0.98, 0.02):
        map_score, top5 = metric(prob_merged, labels_for_eval, thres=threshold)
        if map_score > best_map:
            best_map = map_score
            best_threshold = threshold

    _, best_top5 = metric(prob_merged, labels_for_eval, thres=best_threshold)

    results = {
        'map': best_map,
        'threshold': best_threshold,
        'top1': best_top5[0] if len(best_top5) > 0 else 0,
        'top5': best_top5,
    }

    return results


# ============================================================
# Kaggle 提交生成
# ============================================================

def generate_submission(
    blend: defaultdict,
    label_dict: dict,
    new_whale_threshold: float,
    output_path: str,
    num_tta: int = 2,
    num_models: int = 1
) -> str:
    """
    生成 Kaggle 提交 CSV

    Args:
        blend: {image_id: Counter({class_idx: cumulative_prob})}
        label_dict: {integer_label: whale_id_string}
        new_whale_threshold: new_whale 阈值
        output_path: 输出 CSV 路径
        num_tta: TTA 次数
        num_models: 模型数量

    Returns:
        输出文件路径
    """
    # new_whale (index=5004) 的伪概率 = threshold * TTA数 * 模型数
    new_whale_prob = new_whale_threshold * num_tta * num_models

    rows = []
    new_whale_count = 0

    for img_id in sorted(blend.keys()):
        counter = blend[img_id].copy()
        # 添加 new_whale 的伪概率
        counter[5004] = new_whale_prob

        # 取 top-5
        top5 = counter.most_common(5)

        # 将整数索引转换为鲸鱼 ID 字符串
        top5_names = []
        for idx, _ in top5:
            if idx in label_dict:
                top5_names.append(label_dict[idx])
            elif idx == 5004:
                top5_names.append('new_whale')
            else:
                top5_names.append(f'unknown_{idx}')

        if top5_names[0] == 'new_whale':
            new_whale_count += 1

        rows.append({
            'Image': img_id,
            'Id': ' '.join(top5_names)
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"\n提交文件: {output_path}")
    print(f"  总样本: {len(rows)}")
    print(f"  new_whale 预测数: {new_whale_count} ({100.0*new_whale_count/max(len(rows), 1):.1f}%)")

    # 显示前 5 行示例
    print(f"\n  前 5 行:")
    for _, row in df.head(5).iterrows():
        print(f"    {row['Image']}: {row['Id']}")

    return output_path


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='鲸鱼分类推理')
    parser.add_argument('config', type=str, help='YAML 配置文件路径')
    parser.add_argument('--mode', type=str, default='eval',
                        choices=['eval', 'submit'],
                        help='eval=验证集评估, submit=测试集推理+提交')
    parser.add_argument('--model_paths', type=str, default='',
                        help='模型路径 (逗号分隔多个模型做融合). '
                             '默认使用 output_dir/best_model.pth')
    parser.add_argument('--new_whale_threshold', type=float, default=0.3,
                        help='new_whale 阈值 (默认 0.3)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批大小 (默认使用 YAML 配置)')
    parser.add_argument('--output', type=str, default=None,
                        help='提交文件输出路径')

    args = parser.parse_args()

    # 加载配置
    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size or cfg.batch_size

    print("=" * 60)
    print(f"鲸鱼分类推理 ({args.mode})")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"配置: {args.config}")
    print()

    # 确定模型路径
    if args.model_paths:
        model_paths = [p.strip() for p in args.model_paths.split(',')]
    else:
        default_path = os.path.join(cfg.output_dir, 'best_model.pth')
        model_paths = [default_path]

    print(f"模型数: {len(model_paths)}")
    for p in model_paths:
        print(f"  - {p}")
    print()

    # ---- 验证集评估 ----
    if args.mode == 'eval':
        for model_path in model_paths:
            print(f"\n{'─' * 40}")
            model = load_trained_model(cfg, model_path, device)

            results = run_inference_val_2tta(
                model, cfg, batch_size, cfg.num_workers, device)

            print(f"\n验证结果:")
            print(f"  MAP@5:     {results['map']:.4f}")
            print(f"  Threshold: {results['threshold']:.2f}")
            print(f"  Top-1:     {results['top1']:.4f}")
            if results['top5']:
                print(f"  Top-5:     {[f'{t:.4f}' for t in results['top5']]}")

            del model
            torch.cuda.empty_cache()

    # ---- 测试集推理 + Kaggle 提交 ----
    elif args.mode == 'submit':
        # 加载标签字典 (index → whale_id)
        label_dict, _ = load_class_name_dict(
            os.path.join(cfg.list_dir, 'label_list.txt'))

        final_blend = defaultdict(Counter)

        for model_path in model_paths:
            print(f"\n{'─' * 40}")
            model = load_trained_model(cfg, model_path, device)

            blend = run_inference_test_2tta(
                model, cfg, batch_size, cfg.num_workers, device)

            # 累加到最终 blend
            for img_id, counter in blend.items():
                for cls_idx, score in counter.items():
                    final_blend[img_id][cls_idx] += score

            del model
            torch.cuda.empty_cache()

        # 生成提交文件
        if args.output:
            output_path = args.output
        else:
            output_path = os.path.join(
                cfg.output_dir,
                f"submission_{len(model_paths)}models_th{args.new_whale_threshold}.csv"
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        generate_submission(
            final_blend,
            label_dict,
            args.new_whale_threshold,
            output_path,
            num_tta=2,
            num_models=len(model_paths)
        )

    print("\n✓ 推理完成!")


if __name__ == '__main__':
    main()
