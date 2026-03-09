"""
Quick ASCII-only test for the MLP head fix (wrapper.py feature_dim).
"""
import os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torchvision.models as tvm

passed = 0
failed = 0

def check(cond, name):
    global passed, failed
    if cond: passed += 1; print(f"  PASS: {name}")
    else:    failed += 1; print(f"  FAIL: {name}")

# ---- feature_dim with Linear head (regression) ----
print("\n[1] Linear head - feature_dim")
from gastrovision.models.wrapper import MetricLearningWrapper
m = tvm.resnet50(weights=None)
m.fc = nn.Linear(m.fc.in_features, 23)
w = MetricLearningWrapper(m)
check(w.feature_dim == 2048, f"Linear head feature_dim=2048 (got {w.feature_dim})")

# ---- feature_dim with MLP head (the bug) ----
print("\n[2] MLP head - feature_dim must be backbone dim (2048), not hidden (1024)")
m2 = tvm.resnet50(weights=None)
in_f = m2.fc.in_features  # 2048
m2.fc = nn.Sequential(
    nn.Linear(in_f, in_f // 2),   # 2048 -> 1024
    nn.ReLU(inplace=True),
    nn.Dropout(0.2),
    nn.Linear(in_f // 2, 23)       # 1024 -> 23
)
try:
    w2 = MetricLearningWrapper(m2)
    dim = w2.feature_dim
    check(dim == 2048, f"MLP head feature_dim=2048 (got {dim})")
    check(dim != 1024, "MLP head feature_dim != 1024 (intermediate dim)")
    check(dim != 23,   "MLP head feature_dim != 23 (num_classes)")
except Exception as e:
    check(False, f"MLP feature_dim raised: {e}")
    traceback.print_exc()

# ---- Forward pass: hook captures avgpool output (2048), not mlp output ----
print("\n[3] MLP forward: hook captures avgpool (2048-dim), not head output")
x = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    logits, feats = w2(x)
check(logits.shape == (2, 23),   f"logits shape (2,23) got {logits.shape}")
check(feats.shape  == (2, 2048), f"features shape (2,2048) got {feats.shape}")

# ---- feature_dim with ConvNeXt (classifier Sequential - regression) ----
print("\n[4] ConvNeXt tiny - feature_dim (regression)")
try:
    m3 = tvm.convnext_tiny(weights=None)
    w3 = MetricLearningWrapper(m3)
    dim3 = w3.feature_dim
    check(dim3 > 0, f"ConvNeXt feature_dim={dim3} > 0")
    x3 = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits3, feats3 = w3(x3)
    check(feats3.shape[1] == dim3, f"hook feats match feature_dim={dim3}")
except Exception as e:
    check(False, f"ConvNeXt raised: {e}")

# ---- get_model mlp + wrapper e2e via loss_factory ----
print("\n[5] get_model (mlp head) + create_metric_loss_function - no crash")
from gastrovision.models.model_factory import get_model
from gastrovision.losses.loss_factory import create_metric_loss_function

class FakeCfg:
    metric_loss = 'contrastive'
    metric_loss_margin = 0.0
    metric_loss_scale = 0.0
    metric_loss_weight = 0.5
    embedding_dim = 512

try:
    model = get_model('resnet50', num_classes=23, pretrained=False,
                      classifier_head='mlp', classifier_dropout=0.2)
    mc, wrapped = create_metric_loss_function(FakeCfg(), 23, 'cpu', model)
    check(mc is not None, "metric_criterion created (no crash)")
    check(wrapped.feature_dim == 2048, f"wrapped.feature_dim=2048 (got {wrapped.feature_dim})")
    x = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0, 0, 1, 2])
    with torch.no_grad():
        logits, feats = wrapped(x)
    loss = mc(feats, labels)
    check(not torch.isnan(loss), f"contrastive loss={loss.item():.4f} (not NaN)")
    check(logits.shape == (4, 23), f"logits (4,23) got {logits.shape}")
    check(feats.shape  == (4, 2048), f"features (4,2048) got {feats.shape}")
except Exception as e:
    check(False, f"end-to-end MLP+metric raised: {e}")
    traceback.print_exc()

print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed")
print('='*50)
sys.exit(0 if failed == 0 else 1)
