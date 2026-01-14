"""
SSL 预训练器模块

用于自监督学习预训练的训练器，支持多种 SSL 方法。
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from ssl_methods.base import SSLMethod


class SSLTrainer:
    """
    SSL 预训练器
    
    用于训练自监督学习模型的通用训练器。
    
    Args:
        model: SSL 模型 (继承自 SSLMethod)
        optimizer: 优化器
        scheduler: 学习率调度器 (可选)
        device: 训练设备
        output_dir: 输出目录
        
    Example:
        >>> model = SimSiam(backbone)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
        >>> trainer = SSLTrainer(model, optimizer, device='cuda')
        >>> trainer.fit(train_loader, epochs=200)
    """
    
    def __init__(
        self,
        model: SSLMethod,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        device: str = 'cuda',
        output_dir: str = './ssl_checkpoints',
        method_name: str = 'ssl'
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.method_name = method_name
        
        # 训练历史
        self.history: Dict[str, List] = {
            'train_loss': [],
            'learning_rate': []
        }
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
    
    def fit(
        self,
        train_loader: DataLoader,
        epochs: int,
        save_freq: int = 10,
        log_freq: int = 100,
        resume_from: str = None
    ) -> Dict[str, List]:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器 (返回 (view1, view2) 或 ((view1, view2), _))
            epochs: 训练轮数
            save_freq: 保存 checkpoint 的频率 (epochs)
            log_freq: 打印日志的频率 (steps)
            resume_from: 恢复训练的 checkpoint 路径
            
        Returns:
            训练历史
        """
        if resume_from:
            self._load_checkpoint(resume_from)
        
        print("=" * 60)
        print(f"开始 {self.method_name} SSL 预训练")
        print("=" * 60)
        print(f"设备: {self.device}")
        print(f"训练样本: {len(train_loader.dataset)}")
        print(f"总 Epochs: {epochs}")
        print(f"批次大小: {train_loader.batch_size}")
        print()
        
        total_start_time = time.time()
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch + 1
            epoch_loss = self._train_epoch(train_loader, log_freq)
            
            # 记录历史
            self.history['train_loss'].append(epoch_loss)
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            # 打印 epoch 信息
            print(f"Epoch [{self.current_epoch}/{epochs}] "
                  f"Loss: {epoch_loss:.4f} "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存 checkpoint
            if (self.current_epoch) % save_freq == 0:
                self._save_checkpoint(f'checkpoint_epoch_{self.current_epoch}.pth')
        
        total_time = time.time() - total_start_time
        
        print("=" * 60)
        print("SSL 预训练完成")
        print("=" * 60)
        print(f"总耗时: {total_time / 60:.1f} 分钟")
        print(f"最终 Loss: {self.history['train_loss'][-1]:.4f}")
        
        # 保存最终模型
        self._save_checkpoint('final_model.pth')
        self._save_training_log()
        
        return self.history
    
    def _train_epoch(self, train_loader: DataLoader, log_freq: int) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # 处理不同的数据格式
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    # (views, labels) 或 (view1, view2)
                    views, labels_or_view2 = batch
                    if isinstance(views, (list, tuple)):
                        # ((view1, view2), labels)
                        view1, view2 = views
                    else:
                        # (view1, view2) - 无标签
                        view1 = views
                        view2 = labels_or_view2
                else:
                    raise ValueError(f"不支持的 batch 格式: {len(batch)} 个元素")
            else:
                raise ValueError(f"不支持的 batch 类型: {type(batch)}")
            
            view1 = view1.to(self.device)
            view2 = view2.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(view1, view2)
            loss = output['loss']
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def _save_checkpoint(self, filename: str):
        """保存 checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'method_name': self.method_name
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = self.output_dir / filename
        torch.save(checkpoint, path)
        print(f"✓ 保存 checkpoint: {path}")
    
    def _load_checkpoint(self, path: str):
        """加载 checkpoint"""
        print(f"加载 checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.history = checkpoint.get('history', self.history)
        
        print(f"✓ 从 epoch {self.current_epoch} 恢复训练")
    
    def _save_training_log(self):
        """保存训练日志"""
        log = {
            'method': self.method_name,
            'history': self.history,
            'epochs': self.current_epoch,
            'timestamp': datetime.now().isoformat()
        }
        
        log_path = self.output_dir / 'training_log.json'
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 保存训练日志: {log_path}")
    
    def get_encoder(self) -> nn.Module:
        """获取预训练的编码器 (用于下游任务)"""
        return self.model.get_encoder()
    
    def save_encoder(self, filename: str = 'pretrained_encoder.pth'):
        """
        仅保存编码器权重 (用于下游任务微调)
        """
        encoder = self.get_encoder()
        path = self.output_dir / filename
        torch.save(encoder.state_dict(), path)
        print(f"✓ 保存编码器: {path}")


class SSLDataset(Dataset):
    """
    SSL 数据集包装器
    
    将普通数据集转换为返回双视图的 SSL 数据集。
    """
    
    def __init__(
        self,
        dataset: Dataset,
        transform = None
    ):
        """
        Args:
            dataset: 基础数据集
            transform: SSL 变换 (应该返回两个视图)
        """
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # 获取原始图像
        item = self.dataset[idx]
        
        if isinstance(item, tuple):
            img = item[0]  # (img, label)
        else:
            img = item
        
        # 应用变换生成两个视图
        if self.transform:
            view1, view2 = self.transform(img)
            return view1, view2
        else:
            return img, img


# ============ 测试代码 ============
if __name__ == '__main__':
    print("测试 SSLTrainer...")
    
    import torch
    from torchvision.models import resnet18
    from torch.utils.data import TensorDataset, DataLoader
    
    # 添加路径
    import sys
    sys.path.insert(0, '.')
    
    from ssl_methods.simsiam import SimSiam
    
    # 创建模型
    backbone = resnet18(weights=None)
    model = SimSiam(backbone, proj_dim=2048, pred_dim=512)
    
    # 创建优化器
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.03,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # 创建 Trainer
    trainer = SSLTrainer(
        model=model,
        optimizer=optimizer,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        output_dir='./test_ssl_checkpoints',
        method_name='simsiam'
    )
    
    # 创建假数据
    fake_data = torch.randn(100, 3, 224, 224)
    fake_labels = torch.zeros(100)
    
    # 创建双视图数据集
    class FakeTwoCropDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            # 返回两个视图 (简单起见使用相同数据加噪声)
            x = self.data[idx]
            view1 = x + torch.randn_like(x) * 0.1
            view2 = x + torch.randn_like(x) * 0.1
            return view1, view2
    
    dataset = FakeTwoCropDataset(fake_data)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 训练 2 个 epoch
    print("\n开始测试训练...")
    history = trainer.fit(loader, epochs=2, save_freq=1)
    
    print(f"\n训练历史: {history}")
    print("\nSSLTrainer 测试通过！")
