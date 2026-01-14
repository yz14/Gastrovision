"""
SSL Methods Package

自监督学习方法模块，包含各种对比学习算法的实现。

算法时间线:
- 1993: Siamese Network
- 2015: Triplet Loss
- 2018: Rotation, InstDisc
- 2019: MoCo v1
- 2020: SimCLR, MoCo v2, BYOL, SwAV
- 2021: SimSiam, Barlow Twins, DINO
- 2022: MAE

使用方法:
    from ssl_methods import SimSiam, MoCo, SimCLR, BYOL
    
    model = SimSiam(backbone=resnet50, dim=2048, pred_dim=512)
    loss = model(x1, x2)
"""

from .base import SSLMethod, ProjectionHead, PredictionHead
from .simsiam import SimSiam
from .moco import MoCo
from .simclr import SimCLR
from .byol import BYOL
from .barlow_twins import BarlowTwins
from .rotation import RotationPrediction
from .siamese import SiameseNetwork, TripletNetwork
from .instdisc import InstDisc
from .swav import SwAV
from .dino import DINO
from .mae import MAE

__all__ = [
    'SSLMethod',
    'ProjectionHead', 
    'PredictionHead',
    # Modern SSL methods (2019-2022)
    'SimSiam',
    'MoCo',
    'SimCLR',
    'BYOL',
    'BarlowTwins',
    'SwAV',
    'DINO',
    'MAE',
    'InstDisc',
    # Early methods (1993-2018)
    'RotationPrediction',
    'SiameseNetwork',
    'TripletNetwork',
]
