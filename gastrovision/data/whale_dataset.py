"""
鲸鱼分类数据集模块

移植自 colonnav_ssl/process/ 目录:
- data.py: WhaleDataset
- data_helper.py: 标签、bbox、训练列表加载
- augmentation.py: 数据增强、bbox 裁剪
- triplet_sampler.py: WhaleRandomIdentitySampler

鲸鱼数据关键特性:
1. 图像需要 bbox 裁剪 (如果 bbox CSV 存在)
2. 训练时通过水平翻转将数据量翻倍 (label 偏移 whale_id_num)
3. new_whale (label=-1) 需要特殊处理
4. Triplet 采样器保证每个 batch 有足够的同类样本对
"""

import os
import copy
import random
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

try:
    import albumentations as A
    HAS_ALBUM = True
except ImportError:
    HAS_ALBUM = False


# ============================================================
# 数据辅助函数 (来自 data_helper.py)
# ============================================================

def load_label_dict(label_list_path):
    """
    加载标签字典: whale_id -> 整数标签

    格式: 每行 "whale_id 整数标签"
    new_whale 的标签为 -1
    """
    label_dict = {}
    with open(label_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ')
            whale_id = parts[0]
            index = int(parts[1])
            label_dict[whale_id] = index
    return label_dict


def load_train_list(train_image_list_path):
    """
    加载训练图像列表

    格式: 每行 "图像文件名 整数标签 whale_id"
    返回: [(filename, label), ...]
    """
    samples = []
    with open(train_image_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ')
            img_name = parts[0]
            index = int(parts[1])
            samples.append([img_name, index])
    return samples


def read_val_txt(txt_path):
    """
    读取验证集文件

    格式: 每行 "图像文件名 整数标签"
    返回: [(filename, label), ...]
    """
    samples = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ')
            samples.append([parts[0], int(parts[1])])
    return samples


def image_list2dict(image_list):
    """将图像列表转换为 {label: [images]} 字典"""
    d = {}
    id_list = []
    for image, label in image_list:
        if label in d:
            d[label].append(image)
        else:
            d[label] = [image]
            id_list.append(label)
    return d, id_list


def load_pseudo_list(path):
    """加载伪标签列表 (可选)"""
    if not os.path.exists(path):
        print(f"伪标签文件不存在, 跳过: {path}")
        return []

    samples = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ')
            samples.append([parts[0], int(parts[1])])
    return samples


def load_class_name_dict(label_list_path):
    """加载类别名称字典: 整数标签 -> whale_id"""
    label_dict = {}
    id_dict = {}
    with open(label_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ')
            whale_id = parts[0]
            index = int(parts[1])
            if index == -1:
                index = 5004  # new_whale 特殊处理
            label_dict[index] = whale_id
            id_dict[whale_id] = index
    return label_dict, id_dict


# ============================================================
# BBox 加载 (来自 data_helper.py)
# ============================================================

class DefaultBboxDict(dict):
    """当 key 不存在时返回 None, 让 get_cropped_img 使用全图"""
    def __missing__(self, key):
        return None


def _get_bbox_list_from_csv(csv_path):
    """从 CSV 读取 bbox 列表"""
    bbox_data = pd.read_csv(csv_path)
    label_name = bbox_data['Image'].tolist()
    x0 = bbox_data['x0'].tolist()
    y0 = bbox_data['y0'].tolist()
    x1 = bbox_data['x1'].tolist()
    y1 = bbox_data['y1'].tolist()
    return label_name, x0, y0, x1, y1


def load_bbox_dict(bbox_dir):
    """
    加载 bbox 字典

    尝试加载 se50_bbox.csv 和 se101_bbox.csv, 取两者的平均 bbox。
    如果不存在, 返回空的 DefaultBboxDict (所有图像使用全图)。

    Args:
        bbox_dir: 包含 bbox CSV 文件的目录

    Returns:
        dict: {image_filename: [x0, y0, x1, y1]} 或 DefaultBboxDict
    """
    bbox_dict = DefaultBboxDict()

    if bbox_dir is None or not os.path.isdir(bbox_dir):
        print("未指定 bbox 目录, 使用全图训练模式")
        return bbox_dict

    se50_path = os.path.join(bbox_dir, 'se50_bbox.csv')
    se101_path = os.path.join(bbox_dir, 'se101_bbox.csv')

    if os.path.exists(se50_path) and os.path.exists(se101_path):
        print(f"加载 bbox: {se50_path}")
        _, x0_50, y0_50, x1_50, y1_50 = _get_bbox_list_from_csv(se50_path)
        print(f"加载 bbox: {se101_path}")
        names, x0_101, y0_101, x1_101, y1_101 = _get_bbox_list_from_csv(se101_path)

        for (name, a0, b0, a1, b1, c0, d0, c1, d1) in zip(
            names,
            x0_50, y0_50, x1_50, y1_50,
            x0_101, y0_101, x1_101, y1_101
        ):
            bbox_dict[name] = [
                (a0 + c0) // 2,
                (b0 + d0) // 2,
                (a1 + c1) // 2,
                (b1 + d1) // 2
            ]
        print(f"  加载了 {len(bbox_dict)} 个 bbox")
    else:
        print("未找到 bbox 文件, 使用全图训练模式")

    return bbox_dict


# ============================================================
# 数据增强 (来自 augmentation.py)
# ============================================================

def get_cropped_img(image, bbox):
    """
    根据 bbox 裁剪图像, 带 10% 的 margin

    Args:
        image: BGR 图像 [H, W, 3]
        bbox: [x0, y0, x1, y1] 或 None (返回全图)
    """
    if bbox is None:
        return image

    crop_margin = 0.1
    size_x = image.shape[1]
    size_y = image.shape[0]

    x0, y0, x1, y1 = bbox
    dx = x1 - x0
    dy = y1 - y0

    x0 = max(0, x0 - dx * crop_margin)
    x1 = min(size_x, x1 + dx * crop_margin + 1)
    y0 = max(0, y0 - dy * crop_margin)
    y1 = min(size_y, y1 + dy * crop_margin + 1)

    crop = image[int(y0):int(y1), int(x0):int(x1), :]
    return crop


def _perspective_aug(img, threshold1=0.25, threshold2=0.75):
    """透视变换增强"""
    rows, cols, ch = img.shape
    x0 = random.randint(0, int(cols * threshold1))
    y0 = random.randint(0, int(rows * threshold1))
    x1 = random.randint(int(cols * threshold2), cols - 1)
    y1 = random.randint(0, int(rows * threshold1))
    x2 = random.randint(int(cols * threshold2), cols - 1)
    y2 = random.randint(int(rows * threshold2), rows - 1)
    x3 = random.randint(0, int(cols * threshold1))
    y3 = random.randint(int(rows * threshold2), rows - 1)

    pts = np.float32([(x0, y0), (x1, y1), (x2, y2), (x3, y3)])

    # 四点透视变换
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    original = np.array([
        [0, 0], [cols - 1, 0],
        [cols - 1, rows - 1], [0, rows - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(original, rect)
    warped = cv2.warpPerspective(img, M, (cols, rows))

    x_ = np.asarray([x0, x1, x2, x3])
    y_ = np.asarray([y0, y1, y2, y3])
    warped = warped[np.min(y_):np.max(y_), np.min(x_):np.max(x_), :]

    return warped


def aug_image(image, is_infer=False, augment=None):
    """
    数据增强

    Args:
        image: BGR 图像
        is_infer: 是否为推理模式
        augment: 增强参数列表 (推理时 [0.0]=不翻转, [1.0]=水平翻转)
    """
    if is_infer:
        if augment is None:
            return image
        flip_code = augment[0] if augment else 0
        if flip_code == 1:
            image = cv2.flip(image, 1)
        elif flip_code == 2:
            image = cv2.flip(image, 0)
        elif flip_code == 3:
            image = cv2.flip(image, -1)
        return image

    if HAS_ALBUM:
        transform = A.Compose([
            A.Rotate(limit=15, border_mode=cv2.BORDER_REPLICATE, p=0.5),
            A.Affine(shear=(-15, 15), border_mode=cv2.BORDER_REPLICATE, p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.GaussNoise(std_range=(0.1, 0.2), p=1.0),
                A.HueSaturationValue(
                    hue_shift_limit=5, sat_shift_limit=5,
                    val_shift_limit=5, p=1.0),
            ], p=0.5),
        ])
        result = transform(image=image)
        image = result['image']
    else:
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta = random.randint(-20, 20)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return image


# ============================================================
# 鲸鱼数据集
# ============================================================

class WhaleDataset(Dataset):
    """
    鲸鱼分类数据集

    关键行为:
    - train 模式: 数据量翻倍 (index >= len(train_list) 时水平翻转, label += whale_id_num)
    - train_list 模式: 仅返回标签, 供 WhaleRandomIdentitySampler 使用
    - val 模式: 标准验证, is_flip=True 时对图像做水平翻转 + label 偏移
    - test 模式: 返回 (image_id, image), 无标签

    数据流:
        原图 → bbox 裁剪 → resize → 增强 → (可能翻转) → 归一化 → tensor

    Args:
        mode: 'train', 'train_list', 'val', 'test'
        data_dir: 鲸鱼原始数据目录 (包含 train/, test/)
        list_dir: image_list 目录 (包含 train_image_list.txt, val0.txt 等)
        bbox_dir: bbox CSV 目录 (可选)
        fold_index: 验证集 fold 索引
        image_size: (height, width) 输出尺寸
        augment: 推理增强参数 (如 [0.0] 或 [1.0])
        is_pseudo: 是否加载伪标签
        is_flip: 验证时是否翻转 (TTA)
    """

    def __init__(
        self,
        mode,
        data_dir,
        list_dir,
        bbox_dir=None,
        fold_index=0,
        image_size=(256, 512),
        augment=None,
        is_pseudo=False,
        is_flip=False
    ):
        super().__init__()
        self.mode = mode
        self.data_dir = data_dir
        self.augment = augment
        self.is_pseudo = is_pseudo
        self.is_flip = is_flip
        self.image_size = image_size

        self.train_image_path = os.path.join(data_dir, 'train') + '/'
        self.test_image_path = os.path.join(data_dir, 'test') + '/'

        # 加载标签字典和 bbox
        label_list_path = os.path.join(list_dir, 'label_list.txt')
        self.label_dict = load_label_dict(label_list_path)
        self.class_num = len(self.label_dict)
        self.whale_id_num = self.class_num  # 不含 new_whale 的类别数 (5004)

        self.bbox_dict = load_bbox_dict(bbox_dir)

        # 伪标签
        self.pseudo_list = []
        if self.is_pseudo:
            pseudo_path = os.path.join(list_dir, 'pseudo_list.txt')
            self.pseudo_list = load_pseudo_list(pseudo_path)

        # 列表目录
        self.list_dir = list_dir

        # 数据列表 (由 set_mode 设置)
        self.train_list = []
        self.val_list = []
        self.test_list = []
        self.train_dict = {}
        self.id_list = []
        self.num_data = 0
        self.fold_index = None

        self.set_mode(mode, fold_index)

    def set_mode(self, mode, fold_index):
        """设置数据集模式"""
        self.mode = mode
        self.fold_index = fold_index
        print(f'fold index set: {fold_index}')

        if mode in ('train', 'train_list'):
            self.train_list = load_train_list(
                os.path.join(self.list_dir, 'train_image_list.txt'))

            # 排除验证集
            val_path = os.path.join(self.list_dir, f'val{fold_index}.txt')
            val_list = read_val_txt(val_path)
            print(f'验证集文件: {val_path}')
            val_set = set(name for name, _ in val_list)
            self.train_list = [
                item for item in self.train_list
                if item[0] not in val_set
            ]

            # 加入伪标签
            self.train_list += self.pseudo_list

            # 构建 identity 字典 (供采样器使用)
            self.train_dict, self.id_list = image_list2dict(self.train_list)

            # 数据量翻倍 (水平翻转)
            self.num_data = len(self.train_list) * 2
            print(f'set dataset mode: {mode}, 原始样本: {len(self.train_list)}, '
                  f'翻倍后: {self.num_data}')

        elif mode == 'val':
            val_path = os.path.join(self.list_dir, f'val{fold_index}.txt')
            self.val_list = read_val_txt(val_path)
            print(f'验证集文件: {val_path}')
            self.num_data = len(self.val_list)
            print(f'set dataset mode: val, samples: {self.num_data}')

        elif mode == 'test':
            self.test_list = [
                f for f in os.listdir(self.test_image_path)
                if f.endswith('.jpg')
            ]
            self.num_data = len(self.test_list)
            print(f'set dataset mode: test, samples: {self.num_data}')

        print(f'data num: {self.num_data}')

    def _load_image(self, image_name):
        """加载图像, 自动在 train/ 和 test/ 中查找"""
        image_path = os.path.join(self.train_image_path, image_name)
        if not os.path.exists(image_path):
            image_path = os.path.join(self.test_image_path, image_name)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"警告: 无法读取图像 {image_path}")
            # 返回全黑图像作为 fallback
            return np.zeros(
                (self.image_size[0], self.image_size[1], 3), dtype=np.uint8)

        return image

    def __getitem__(self, index):
        if self.fold_index is None:
            raise RuntimeError('fold_index 未设置! 请先调用 set_mode()')

        # ---- train_list 模式: 仅返回标签 (供采样器使用) ----
        if self.mode == 'train_list':
            if index >= len(self.train_list):
                image_index = index - len(self.train_list)
            else:
                image_index = index
            _, label = self.train_list[image_index]

            if label == -1:
                return None, label, None

            # 翻转部分的标签偏移
            if index >= len(self.train_list):
                label += self.whale_id_num

            return None, label, None

        # ---- train 模式 ----
        if self.mode == 'train':
            if index >= len(self.train_list):
                image_index = index - len(self.train_list)
            else:
                image_index = index

            image_name, label = self.train_list[image_index]
            image = self._load_image(image_name)

            # bbox 裁剪
            image = get_cropped_img(image, self.bbox_dict.get(image_name))

            # resize
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))

            # 随机灰度化 (50% 概率)
            if random.randint(0, 1) == 0:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # 随机透视变换 (50% 概率)
            if random.randint(0, 1) == 0:
                image = _perspective_aug(image)
                image = cv2.resize(image, (self.image_size[1], self.image_size[0]))

            # 常规增强
            image = aug_image(image)

            # new_whale 处理: 随机翻转
            if label == -1:
                if random.random() > 0.5:
                    image = cv2.flip(image, 1)
            # 翻转数据: 确定性翻转 + 标签偏移
            elif index >= len(self.train_list):
                image = cv2.flip(image, 1)
                label += self.whale_id_num

        # ---- val 模式 ----
        elif self.mode == 'val':
            image_name, label = self.val_list[index]
            image = self._load_image(image_name)

            # bbox 裁剪
            image = get_cropped_img(image, self.bbox_dict.get(image_name))

            # resize
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))

            # 推理增强
            image = aug_image(image, is_infer=True, augment=self.augment)

            # 验证 TTA: 翻转
            if self.is_flip:
                image = cv2.flip(image, 1)
                if label != -1:
                    label += self.whale_id_num

        # ---- test 模式 ----
        elif self.mode == 'test':
            image_name = self.test_list[index]
            image_path = os.path.join(self.test_image_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if image is None:
                print(f"警告: 无法读取测试图像 {image_path}")
                image = np.zeros(
                    (self.image_size[0], self.image_size[1], 3), dtype=np.uint8)

            # bbox 裁剪
            image = get_cropped_img(image, self.bbox_dict.get(image_name))

            # resize + 推理增强
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            image = aug_image(image, is_infer=True, augment=self.augment)

            # 转 tensor
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32) / 255.0
            image = image.reshape([3, self.image_size[0], self.image_size[1]])

            return image_name, torch.FloatTensor(image)

        # ---- 转 tensor (train / val) ----
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = image.reshape([-1, self.image_size[0], self.image_size[1]])
        image = image / 255.0

        # new_whale 标记
        NW = 0
        if label == -1:
            label = self.whale_id_num * 2  # class_num (10008)
            NW = 1

        return torch.FloatTensor(image), label, NW

    def __len__(self):
        return self.num_data


# ============================================================
# Triplet 身份采样器 (来自 triplet_sampler.py)
# ============================================================

class WhaleRandomIdentitySampler(Sampler):
    """
    随机身份采样器

    保证每个 batch 包含 N 个身份 (鲸鱼个体),
    每个身份 K 个样本, 因此 batch_size = N * K。

    特殊处理: new_whale 按固定比例 (NW_ratio) 加入每个 batch。

    Args:
        data_source: WhaleDataset (train_list 模式)
        batch_size: 批次大小
        num_instances: 每个身份的样本数 (默认 4)
        is_newwhale: 是否包含 new_whale 特殊处理
        NW_ratio: new_whale 在每个 batch 中的比例
    """

    def __init__(
        self,
        data_source,
        batch_size,
        num_instances,
        is_newwhale=True,
        NW_ratio=0.25
    ):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances

        self.is_NW = is_newwhale
        self.NW_ratio = NW_ratio
        self.NW_id_num = 0

        if self.is_NW:
            self.NW_id_num = int(self.num_pids_per_batch * self.NW_ratio)

        print(f'NW ratio: {NW_ratio}')
        print(f'NW id num: {self.NW_id_num}')

        # 构建 {label: [indices]} 索引
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)

        self.pids = list(self.index_dic.keys())

        # 估算 epoch 长度
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

        print(f'采样器估算样本数: {self.length}')

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])

            if len(idxs) < self.num_instances:
                idxs = list(np.random.choice(
                    idxs, size=self.num_instances, replace=True))

            random.shuffle(idxs)
            batch_idxs = []

            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        if self.is_NW and -1 in batch_idxs_dict:
            # 有 new_whale: 按比例混合
            avai_pids_noNW = copy.deepcopy(self.pids)
            if -1 in avai_pids_noNW:
                avai_pids_noNW.remove(-1)
            final_idxs = []

            needed = self.num_pids_per_batch - self.NW_id_num
            while len(avai_pids_noNW) >= needed:
                selected_pids = random.sample(avai_pids_noNW, needed)

                for pid in selected_pids:
                    batch_idxs = batch_idxs_dict[pid].pop(0)
                    final_idxs.extend(batch_idxs)
                    if len(batch_idxs_dict[pid]) == 0:
                        avai_pids_noNW.remove(pid)

                # 加入 new_whale 样本
                for _ in range(self.NW_id_num):
                    nw_batches = batch_idxs_dict.get(-1, [])
                    if nw_batches:
                        random_pos = random.randint(0, len(nw_batches) - 1)
                        batch_idxs = nw_batches[random_pos]
                        final_idxs.extend(batch_idxs)

            return iter(final_idxs)
        else:
            # 无 new_whale: 标准采样
            avai_pids = copy.deepcopy(self.pids)
            final_idxs = []

            while len(avai_pids) >= self.num_pids_per_batch:
                selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
                for pid in selected_pids:
                    batch_idxs = batch_idxs_dict[pid].pop(0)
                    final_idxs.extend(batch_idxs)
                    if len(batch_idxs_dict[pid]) == 0:
                        avai_pids.remove(pid)

            return iter(final_idxs)

    def __len__(self):
        return self.length
