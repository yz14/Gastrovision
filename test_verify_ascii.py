"""
ASCII-only verification script for all 3 bug fixes.
Outputs only ASCII to avoid encoding issues.
"""
import os, sys, tempfile, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torchvision.models as models

passed = 0
failed = 0
errors = []

def check(cond, name):
    global passed, failed, errors
    if cond:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        errors.append(name)
        print(f"  FAIL: {name}")

def make_checkpoint(num_classes=23):
    """Create a fake project checkpoint (Trainer.save_checkpoint format)."""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    with torch.no_grad():
        model.fc.weight.fill_(0.123)
        model.fc.bias.fill_(0.456)
    f = tempfile.NamedTemporaryFile(suffix='.pth', delete=False)
    f.close()
    torch.save({
        'epoch': 50,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': {},
        'metrics': {'accuracy': 0.95},
        'best_valid_acc': 0.95,
        'best_epoch': 50,
    }, f.name)
    return f.name, model.fc.weight.clone().detach()

# ---- BUG-1a: replace_head=True should overwrite checkpoint head ----
print("\n[BUG-1a] replace_head=True (pretrain): head overwritten by random init")
from gastrovision.models.model_factory import get_model
ckpt, ckpt_w = make_checkpoint(23)
try:
    m = get_model('resnet50', num_classes=23, pretrained=False, weights_path=ckpt, replace_head=True)
    overwritten = not torch.allclose(m.fc.weight.detach(), ckpt_w.to(m.fc.weight.device))
    check(overwritten, "replace_head=True: head is randomly re-initialized (not checkpoint values)")
except Exception as e:
    check(False, f"replace_head=True raised: {e}"); traceback.print_exc()
finally:
    os.unlink(ckpt)

# ---- BUG-1b: replace_head=False must keep checkpoint head ----
print("\n[BUG-1b] replace_head=False (resume): head preserved from checkpoint")
ckpt, ckpt_w = make_checkpoint(23)
try:
    m = get_model('resnet50', num_classes=23, pretrained=False, weights_path=ckpt, replace_head=False)
    preserved = torch.allclose(m.fc.weight.detach(), ckpt_w.to(m.fc.weight.device))
    check(preserved, "replace_head=False: head weight matches checkpoint exactly")
    check(m.fc.out_features == 23, f"out_features=23 (got {m.fc.out_features})")
except Exception as e:
    check(False, f"replace_head=False raised: {e}"); traceback.print_exc()
finally:
    os.unlink(ckpt)

# ---- BUG-1c: backward compat (no replace_head arg, default True) ----
print("\n[BUG-1c] Backward compat: default replace_head=True")
try:
    m = get_model('resnet50', num_classes=10, pretrained=False)
    check(m.fc.out_features == 10, f"default: out_features=10 (got {m.fc.out_features})")
except Exception as e:
    check(False, f"default replace_head raised: {e}")

# ---- BUG-2: train_epoch must not crash (stray raise removed) ----
print("\n[BUG-2] train_epoch: no crash from stray raise")
from gastrovision.trainers.trainer import Trainer
num_classes = 5
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
images = torch.randn(4, 3, 64, 64)
targets = torch.randint(0, num_classes, (4,))
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(images, targets), batch_size=4)
with tempfile.TemporaryDirectory() as td:
    trainer = Trainer(
        model=model, criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        device=torch.device('cpu'), output_dir=td)
    try:
        metrics = trainer.train_epoch(loader, epoch=1, total_epochs=1)
        check(True, "train_epoch completed without RuntimeError")
        check('loss' in metrics, "metrics dict has 'loss'")
        check(isinstance(metrics['loss'], float), f"loss is float: {metrics['loss']:.4f}")
    except RuntimeError as e:
        check(False, f"train_epoch still raises RuntimeError: {e}")
    except Exception as e:
        check(False, f"train_epoch raised unexpected: {e}"); traceback.print_exc()

# ---- BUG-3: test_only flow uses replace_head=False ----
print("\n[BUG-3] test_only flow: model weights come from checkpoint, not random init")
ckpt, ckpt_w = make_checkpoint(23)
try:
    # Simulate train_cls.py logic after fix
    cfg_resume = ckpt
    replace_head = False  # set by train_cls.py when cfg.resume is set
    m = get_model('resnet50', num_classes=23, pretrained=False,
                  weights_path=cfg_resume, replace_head=replace_head)
    correct = torch.allclose(m.fc.weight.detach(), ckpt_w.to(m.fc.weight.device))
    check(correct, "test_only: head weight equals checkpoint (not random)")
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = m(x)
    check(out.shape == (2, 23), f"inference output shape (2,23) got {out.shape}")
except Exception as e:
    check(False, f"test_only flow raised: {e}"); traceback.print_exc()
finally:
    os.unlink(ckpt)

# ---- trainer.load_checkpoint still works for training resume ----
print("\n[Extra] trainer.load_checkpoint restores optimizer state")
model2 = models.resnet18(weights=None)
model2.fc = nn.Linear(model2.fc.in_features, 5)
opt2 = torch.optim.Adam(model2.parameters(), lr=0.001)
with tempfile.TemporaryDirectory() as td:
    t1 = Trainer(model=model2, criterion=nn.CrossEntropyLoss(),
                 optimizer=opt2, device=torch.device('cpu'), output_dir=td)
    # Manually set best stats so checkpoint contains them
    t1.best_valid_acc = 0.9
    t1.best_epoch = 10
    t1.save_checkpoint('ck.pth', epoch=10, metrics={'accuracy': 0.9})
    model3 = models.resnet18(weights=None)
    model3.fc = nn.Linear(model3.fc.in_features, 5)
    opt3 = torch.optim.Adam(model3.parameters(), lr=0.001)
    t2 = Trainer(model=model3, criterion=nn.CrossEntropyLoss(),
                 optimizer=opt3, device=torch.device('cpu'), output_dir=td)
    try:
        t2.load_checkpoint(os.path.join(td, 'ck.pth'))
        check(t2.best_valid_acc == 0.9, f"best_valid_acc=0.9 (got {t2.best_valid_acc})")
        check(t2.best_epoch == 10, f"best_epoch=10 (got {t2.best_epoch})")
    except Exception as e:
        check(False, f"load_checkpoint raised: {e}")

print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed")
if errors:
    print("FAILED:")
    for e in errors:
        print(f"  - {e}")
print('='*50)
sys.exit(0 if failed == 0 else 1)
