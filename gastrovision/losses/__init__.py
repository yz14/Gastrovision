"""损失函数模块"""
from .classification import (
    LabelSmoothingCrossEntropy,
    FocalLoss,
    ClassBalancedLoss
)
from .multilabel import (
    FocalLossMultilabel,
    FocalOHEMLoss,
    TripletLoss,
    AsymmetricLoss,
    LabelSmoothingBCE,
    WeightedBCELoss,
    PolyLoss,
    DiceLoss,
    SoftmaxLossMultilabel,
    CombinedMultilabelLoss
)
from .metric_learning import (
    ContrastiveLoss,
    TripletMarginLoss,
    LiftedStructureLoss,
    ProxyNCALoss,
    NPairLoss,
    ArcFaceLoss,
    CircleLoss,
    CircleLossClassLevel,
    create_metric_loss
)
from .loss_factory import (
    create_loss_function,
    create_metric_loss_function,
    get_samples_per_class
)
from .whale_losses import WhaleCompositeLoss, TripletLoss as WhaleTripletLoss
