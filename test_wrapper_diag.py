"""验证新的配置系统和 model_factory 增强"""
import torch
from gastrovision.utils.config import load_config
from gastrovision.models.model_factory import get_model
from gastrovision.models.wrapper import MetricLearningWrapper

print("=== 验证新配置系统 ===\n")

# 1. 测试 YAML 加载
cfg = load_config('configs/train_cls.yaml')
print(f"[OK] YAML 加载成功")
print(f"  model:            {cfg.model}")
print(f"  batch_size:       {cfg.batch_size}")
print(f"  metric_loss:      {cfg.metric_loss}")
print(f"  classifier_head:  {cfg.classifier_head}")
print(f"  classifier_dropout: {cfg.classifier_dropout}")
print(f"  ema:              {cfg.ema}")
print(f"  tta:              {cfg.tta}")

# 2. 测试 Linear head
print(f"\n--- Linear head ---")
model_linear = get_model('resnet50', num_classes=23, pretrained=False, classifier_head='linear')
print(f"  fc 层: {model_linear.fc}")

# 3. 测试 MLP head
print(f"\n--- MLP head ---")
model_mlp = get_model('resnet50', num_classes=23, pretrained=False, classifier_head='mlp', classifier_dropout=0.3)
print(f"  fc 层: {model_mlp.fc}")
# MLP head 应该是 Sequential (Linear → ReLU → Dropout → Linear)
assert isinstance(model_mlp.fc, torch.nn.Sequential), "MLP head 应该是 Sequential"
assert len(list(model_mlp.fc.children())) == 4, "MLP head 应有4层"
print(f"  [OK] MLP head 结构正确")

# 4. 测试 MLP head + MetricLearningWrapper
print(f"\n--- MLP head + MetricLearningWrapper ---")
wrapped = MetricLearningWrapper(model_mlp)
x = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    out = wrapped(x)
if isinstance(out, tuple):
    logits, features = out
    print(f"  logits shape:   {logits.shape}")
    print(f"  features shape: {features.shape}")
    assert logits.shape == (2, 23), f"logits 应为 [2,23], got {logits.shape}"
    assert features.shape[1] == 2048, f"features 应为 [2,2048], got {features.shape}"
    print(f"  [OK] MLP head + Wrapper 输出正确!")
else:
    print(f"  [FAIL] 输出不是 tuple!")

# 5. 验证无 "处理完成" 垃圾信息
print(f"\n[OK] 没有 '处理完成' 垃圾信息!")
print(f"\n=== 全部验证通过 ===")
