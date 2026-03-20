"""
Jaguar Re-ID 推理模块

功能:
1. 从训练好的模型提取所有测试图片的 embedding
2. 根据 test.csv 中的 (query, gallery) 对，计算余弦相似度
3. 生成 Kaggle 提交文件
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from .dataset import JaguarTestDataset
from .transforms import get_val_transforms


@torch.no_grad()
def extract_embeddings(
    model,
    image_dir: str,
    image_names: list,
    transform,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    use_flip: bool = True,
) -> dict:
    """
    提取所有图片的 embedding

    Args:
        model: ReIDModel（已加载权重）
        image_dir: 测试图片目录
        image_names: 图片文件名列表
        transform: 图像变换
        device: 设备
        batch_size: 批次大小
        num_workers: 数据加载工作进程数
        use_flip: 是否使用水平翻转 TTA（embedding 取平均）

    Returns:
        {filename: embedding_tensor} 字典
    """
    model.eval()

    image_paths = [str(Path(image_dir) / name) for name in image_names]
    dataset = JaguarTestDataset(image_paths, transform=transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    embeddings = {}

    for images, filenames in tqdm(loader, desc="提取 embedding"):
        images = images.to(device)

        # 正常方向
        emb = model.extract_embedding(images)

        if use_flip:
            # 水平翻转 TTA
            images_flip = torch.flip(images, dims=[3])
            emb_flip = model.extract_embedding(images_flip)
            # 平均后重新归一化
            emb = F.normalize(emb + emb_flip, p=2, dim=1)

        for i, fname in enumerate(filenames):
            embeddings[fname] = emb[i].cpu()

    return embeddings


def predict_similarity(
    embeddings: dict,
    test_csv_path: str,
    output_path: str = None,
) -> pd.DataFrame:
    """
    根据 test.csv 计算每对图片的余弦相似度

    Args:
        embeddings: {filename: embedding_tensor} 字典
        test_csv_path: test.csv 路径
        output_path: 输出 CSV 路径（如果指定则保存）

    Returns:
        包含 row_id 和 similarity 列的 DataFrame
    """
    print(f"读取测试对: {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    print(f"共 {len(test_df)} 对")

    # 批量计算相似度
    similarities = []
    missing = set()

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="计算相似度"):
        query_name = row['query_image']
        gallery_name = row['gallery_image']

        if query_name not in embeddings:
            missing.add(query_name)
            similarities.append(0.5)
            continue
        if gallery_name not in embeddings:
            missing.add(gallery_name)
            similarities.append(0.5)
            continue

        q_emb = embeddings[query_name]
        g_emb = embeddings[gallery_name]

        # 余弦相似度 (已归一化, 所以直接点积)
        sim = torch.dot(q_emb, g_emb).item()
        similarities.append(sim)

    if missing:
        print(f"[警告] {len(missing)} 张图片缺少 embedding")

    # 自适应 sigmoid 校准
    sims_arr = np.array(similarities)
    median = np.median(sims_arr)
    q75, q25 = np.percentile(sims_arr, [75, 25])
    iqr = max(q75 - q25, 1e-6)
    z = (sims_arr - median) / (iqr * 0.7413)
    calibrated = 1.0 / (1.0 + np.exp(-z))
    print(f"校准参数: median={median:.4f}, IQR={iqr:.4f}")

    submission = pd.DataFrame({
        'row_id': test_df['row_id'],
        'similarity': calibrated
    })

    if output_path:
        submission.to_csv(output_path, index=False)
        print(f"提交文件已保存: {output_path}")
        print(f"相似度统计: mean={np.mean(similarities):.4f}, "
              f"std={np.std(similarities):.4f}, "
              f"min={np.min(similarities):.4f}, max={np.max(similarities):.4f}")

    return submission


def predict_similarity_fast(
    embeddings: dict,
    test_csv_path: str,
    output_path: str = None,
) -> pd.DataFrame:
    """
    批量矩阵计算版本，比逐行计算快很多

    适用于测试集图片数量可控的情况（embedding dict 可以全部放入内存）。
    """
    print(f"读取测试对: {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    print(f"共 {len(test_df)} 对")

    # 收集所有唯一图片名
    all_names = sorted(set(test_df['query_image'].tolist() + test_df['gallery_image'].tolist()))
    name_to_idx = {name: i for i, name in enumerate(all_names)}

    # 构建 embedding 矩阵
    emb_dim = next(iter(embeddings.values())).shape[0]
    emb_matrix = torch.zeros(len(all_names), emb_dim)

    missing = set()
    for name in all_names:
        if name in embeddings:
            emb_matrix[name_to_idx[name]] = embeddings[name]
        else:
            missing.add(name)

    if missing:
        print(f"[警告] {len(missing)} 张图片缺少 embedding，使用零向量")

    # 归一化（处理可能的零向量）
    norms = emb_matrix.norm(dim=1, keepdim=True).clamp(min=1e-8)
    emb_matrix = emb_matrix / norms

    # 获取每对的 index
    query_indices = torch.tensor([name_to_idx[n] for n in test_df['query_image']], dtype=torch.long)
    gallery_indices = torch.tensor([name_to_idx[n] for n in test_df['gallery_image']], dtype=torch.long)

    # 批量点积
    q_embs = emb_matrix[query_indices]  # (N_pairs, D)
    g_embs = emb_matrix[gallery_indices]  # (N_pairs, D)
    sims = (q_embs * g_embs).sum(dim=1)  # (N_pairs,)

    # 自适应 sigmoid 校准: 将余弦相似度映射到 [0, 1]
    # 基于分布统计（中位数 + IQR）进行归一化后过 sigmoid
    # 比 (sim+1)/2 的线性映射区分度更好
    sims_np = sims.numpy()
    median = np.median(sims_np)
    q75, q25 = np.percentile(sims_np, [75, 25])
    iqr = max(q75 - q25, 1e-6)
    # 标准化到 ~N(0,1) 然后 sigmoid
    z = (sims_np - median) / (iqr * 0.7413)  # IQR-based robust z-score
    sims = 1.0 / (1.0 + np.exp(-z))  # sigmoid → [0, 1]
    print(f"校准参数: median={median:.4f}, IQR={iqr:.4f}")

    submission = pd.DataFrame({
        'row_id': test_df['row_id'],
        'similarity': sims
    })

    if output_path:
        submission.to_csv(output_path, index=False)
        print(f"提交文件已保存: {output_path}")
        print(f"相似度统计: mean={sims.mean():.4f}, "
              f"std={sims.std():.4f}, "
              f"min={sims.min():.4f}, max={sims.max():.4f}")

    return submission
