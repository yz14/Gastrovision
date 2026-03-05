"""验证重构后所有模块可以正确导入"""
import sys

PASSED = 0
FAILED = 0

def check(name, fn):
    global PASSED, FAILED
    try:
        fn()
        PASSED += 1
        print(f"  [OK] {name}")
    except Exception as e:
        FAILED += 1
        print(f"  [FAIL] {name}: {e}")

# 基础数据模块
check("dataset.create_dataloaders",
      lambda: __import__('gastrovision.data.dataset', fromlist=['create_dataloaders']))
check("augmentation.mixup_data",
      lambda: __import__('gastrovision.data.augmentation', fromlist=['mixup_data']))
check("transforms.get_wrapped_transforms",
      lambda: __import__('gastrovision.data.transforms', fromlist=['get_wrapped_transforms']))

# 新提取的工厂模块
check("model_factory.get_model",
      lambda: __import__('gastrovision.models.model_factory', fromlist=['get_model']))
check("optimizer.get_optimizer",
      lambda: __import__('gastrovision.utils.optimizer', fromlist=['get_optimizer']))
check("scheduler.get_scheduler",
      lambda: __import__('gastrovision.utils.scheduler', fromlist=['get_scheduler']))
check("loss_factory.create_loss_function",
      lambda: __import__('gastrovision.losses.loss_factory', fromlist=['create_loss_function']))
check("metrics.AverageMeter",
      lambda: __import__('gastrovision.utils.metrics', fromlist=['AverageMeter']))
check("config.merge_args_with_config",
      lambda: __import__('gastrovision.utils.config', fromlist=['merge_args_with_config']))

# Trainers
check("trainers.Trainer",
      lambda: __import__('gastrovision.trainers', fromlist=['Trainer']))
check("trainers.MultilabelTrainer",
      lambda: __import__('gastrovision.trainers.multilabel', fromlist=['MultilabelTrainer']))

# losses 包
check("losses.__init__ (full)",
      lambda: __import__('gastrovision.losses', fromlist=['FocalLoss', 'ArcFaceLoss', 'create_loss_function']))

# models 包
check("models.__init__ (full)",
      lambda: __import__('gastrovision.models', fromlist=['MetricLearningWrapper', 'get_model']))

# SSL 模块
check("trainers.ssl.SSLTrainer",
      lambda: __import__('gastrovision.trainers.ssl', fromlist=['SSLTrainer']))
check("data.ssl_augmentations",
      lambda: __import__('gastrovision.data.ssl_augmentations', fromlist=['TwoCropsTransform']))

print(f"\n{'='*40}")
print(f"Results: {PASSED} passed, {FAILED} failed")
print(f"{'='*40}")
sys.exit(0 if FAILED == 0 else 1)
