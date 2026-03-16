"""
鲸鱼数据准备与校验脚本

校验原始数据和预处理文件完整性, 打印统计信息。

用法:
    python scripts/prepare_whale_data.py [--data_dir DATA_DIR] [--list_dir LIST_DIR]
"""

import os
import argparse
from collections import Counter


def check_data_integrity(data_dir, list_dir, bbox_dir=None, fold_index=0):
    """校验数据完整性并打印统计信息"""

    print("=" * 60)
    print("鲸鱼数据完整性校验")
    print("=" * 60)

    # ---- 1. 原始数据 ----
    print("\n1. 原始数据目录:", data_dir)
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    train_csv = os.path.join(data_dir, 'train.csv')

    for path, name in [(train_dir, 'train/'), (test_dir, 'test/'), (train_csv, 'train.csv')]:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        if exists and os.path.isdir(path):
            count = len([f for f in os.listdir(path) if f.endswith('.jpg')])
            print(f"  {status} {name}: {count} 张图像")
        elif exists:
            print(f"  {status} {name}: 存在")
        else:
            print(f"  {status} {name}: 缺失!")

    # ---- 2. 预处理列表 ----
    print(f"\n2. 预处理列表目录:", list_dir)

    files_to_check = [
        'label_list.txt',
        'train_image_list.txt',
        'pseudo_list.txt',
    ] + [f'val{i}.txt' for i in range(5)]

    for fname in files_to_check:
        fpath = os.path.join(list_dir, fname)
        exists = os.path.exists(fpath)
        status = "✓" if exists else "✗"
        if exists:
            with open(fpath, 'r') as f:
                lines = f.readlines()
            print(f"  {status} {fname}: {len(lines)} 行")
        else:
            print(f"  {status} {fname}: 缺失")

    # ---- 3. Bbox 文件 ----
    if bbox_dir:
        print(f"\n3. Bbox 目录:", bbox_dir)
        for fname in ['se50_bbox.csv', 'se101_bbox.csv']:
            fpath = os.path.join(bbox_dir, fname)
            exists = os.path.exists(fpath)
            status = "✓" if exists else "✗"
            print(f"  {status} {fname}: {'存在' if exists else '缺失'}")

    # ---- 4. 训练数据分析 ----
    train_list_path = os.path.join(list_dir, 'train_image_list.txt')
    if os.path.exists(train_list_path):
        print(f"\n4. 训练数据统计:")
        labels = []
        image_names = []
        with open(train_list_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) >= 2:
                    image_names.append(parts[0])
                    labels.append(int(parts[1]))

        label_counts = Counter(labels)
        total = len(labels)
        new_whale_count = label_counts.get(-1, 0)
        known_count = total - new_whale_count

        print(f"  总样本数: {total}")
        print(f"  已识别 (known): {known_count}")
        print(f"  新鲸鱼 (new_whale): {new_whale_count}")
        print(f"  unique 类别数: {len(label_counts) - (1 if -1 in label_counts else 0)}")

        # 每类样本数分布
        known_counts = [v for k, v in label_counts.items() if k != -1]
        if known_counts:
            print(f"  每类样本数: min={min(known_counts)}, "
                  f"max={max(known_counts)}, "
                  f"avg={sum(known_counts)/len(known_counts):.1f}")

        # 图像是否存在
        if os.path.isdir(train_dir):
            missing = 0
            for img_name in image_names[:100]:  # 抽查前 100 个
                if not os.path.exists(os.path.join(train_dir, img_name)):
                    missing += 1
            if missing > 0:
                print(f"  ⚠ 抽查 100 张: {missing} 张图像缺失!")
            else:
                print(f"  ✓ 抽查 100 张: 全部存在")

    # ---- 5. 验证集分析 ----
    val_path = os.path.join(list_dir, f'val{fold_index}.txt')
    if os.path.exists(val_path):
        print(f"\n5. 验证集 fold{fold_index} 统计:")
        val_labels = []
        with open(val_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) >= 2:
                    val_labels.append(int(parts[1]))

        val_counts = Counter(val_labels)
        val_total = len(val_labels)
        val_nw = val_counts.get(-1, 0)

        print(f"  总样本数: {val_total}")
        print(f"  已识别: {val_total - val_nw}")
        print(f"  new_whale: {val_nw}")

    print(f"\n{'=' * 60}")
    print("校验完成")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='鲸鱼数据准备与校验')
    parser.add_argument('--data_dir', type=str,
                        default='D:/codes/data/humpback-whale-identification',
                        help='鲸鱼原始数据目录')
    parser.add_argument('--list_dir', type=str,
                        default='D:/codes/work-projects/colonnav_ssl/image_list',
                        help='预处理列表目录')
    parser.add_argument('--bbox_dir', type=str,
                        default='D:/codes/work-projects/colonnav_ssl/bbox_model',
                        help='Bbox 目录')
    parser.add_argument('--fold', type=int, default=0,
                        help='验证集 fold 索引')

    args = parser.parse_args()
    check_data_integrity(args.data_dir, args.list_dir, args.bbox_dir, args.fold)
