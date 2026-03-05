"""验证度量学习诊断器功能"""
import sys
import torch
import torch.nn as nn
from pathlib import Path

PASSED = 0
FAILED = 0

def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [OK] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name} — {detail}")


def test_debugger_disabled():
    """当 enabled=False 时，所有方法应该是 no-op"""
    print("\n[Test 1] MetricLearningDebugger disabled mode")
    from gastrovision.utils.metric_debug import MetricLearningDebugger
    
    debugger = MetricLearningDebugger('/tmp/test_debug', enabled=False)
    # 这些调用不应抛出异常
    debugger.on_batch_end(1, 0, torch.randn(4, 128), torch.tensor([0,1,2,3]), 1.0, 0.5)
    debugger.on_epoch_end(1, nn.Linear(10, 10))
    debugger.save_report()
    check("disabled mode no errors", True)


def test_debugger_enabled():
    """enabled=True时，应该正常计算诊断统计"""
    print("\n[Test 2] MetricLearningDebugger enabled mode")
    from gastrovision.utils.metric_debug import MetricLearningDebugger
    
    import tempfile, os
    tmpdir = tempfile.mkdtemp()
    debugger = MetricLearningDebugger(tmpdir, log_interval=1, enabled=True)
    
    # 模拟一批有 4 个类别的特征
    features = torch.randn(16, 128)
    labels = torch.tensor([0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3])
    
    debugger.on_batch_end(1, 0, features, labels, cls_loss=2.0, metric_loss=0.5)
    check("on_batch_end no errors", True)
    
    # 检查 epoch stats 累积
    check("epoch stats accumulated", len(debugger._epoch_stats) > 0,
          f"got {len(debugger._epoch_stats)} stats")
    
    # 检查 sim_gap 在有同类样本时为正值
    if 'sim_gap' in debugger._epoch_stats:
        gap = debugger._epoch_stats['sim_gap'][0]
        # 随机特征的同类/异类距离差距接近 0，但不一定为正
        check("sim_gap computed", isinstance(gap, float), f"gap={gap}")
    
    # epoch end 
    model = nn.Linear(128, 10)
    debugger.on_epoch_end(1, model)
    check("on_epoch_end no errors", True)
    check("epoch_summaries recorded", len(debugger.history['epoch_summaries']) == 1)
    
    # 保存报告
    debugger.save_report()
    report_path = Path(tmpdir) / 'metric_learning_diagnostics.json'
    check("report file created", report_path.exists())
    
    # 清理
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_debugger_with_arcface():
    """测试带 ArcFace 参数的诊断"""
    print("\n[Test 3] MetricLearningDebugger with ArcFace proxy weights")
    from gastrovision.utils.metric_debug import MetricLearningDebugger
    from gastrovision.losses.metric_learning import ArcFaceLoss
    
    import tempfile
    tmpdir = tempfile.mkdtemp()
    debugger = MetricLearningDebugger(tmpdir, log_interval=1, enabled=True)
    
    arcface = ArcFaceLoss(num_classes=23, embedding_dim=2048, scale=30.0, margin=0.5)
    features = torch.randn(8, 2048)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    
    debugger.on_batch_end(1, 0, features, labels, cls_loss=2.0, metric_loss=3.5,
                         metric_module=arcface)
    
    # 检查 proxy 统计
    stats = debugger._epoch_stats
    check("proxy_inter_sim tracked", 'proxy_inter_sim_mean' in stats,
          f"keys: {list(stats.keys())}")
    check("proxy_norm tracked", 'proxy_norm_mean' in stats)
    
    # Few positive pairs warning (8 samples, 8 classes → 0 pos pairs)
    check("num_pos_pairs correct (should be 0)", 
          stats.get('num_pos_pairs', [None])[0] == 0,
          f"got {stats.get('num_pos_pairs', [None])[0]}")
    
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_debugger_pair_level_warning():
    """测试正样本对不足时的警告"""
    print("\n[Test 4] Positive pair count analysis")
    from gastrovision.utils.metric_debug import MetricLearningDebugger
    
    import tempfile
    tmpdir = tempfile.mkdtemp()
    debugger = MetricLearningDebugger(tmpdir, log_interval=999, enabled=True)
    
    # 32 samples, 23 classes → avg ~1.4 samples/class → very few pos pairs
    features = torch.randn(32, 2048)
    labels = torch.tensor([i % 23 for i in range(32)])
    
    debugger.on_batch_end(1, 0, features, labels, cls_loss=2.0, metric_loss=0.5)
    
    pos_pairs = debugger._epoch_stats['num_pos_pairs'][0]
    unique_classes = debugger._epoch_stats['unique_classes_in_batch'][0]
    spc = debugger._epoch_stats['samples_per_class_avg'][0]
    
    check(f"unique classes = 23", unique_classes == 23)
    check(f"samples_per_class ≈ 1.4", abs(spc - 32/23) < 0.1, f"got {spc}")
    check(f"positive pairs very few", pos_pairs < 20,
          f"got {pos_pairs} (23 classes in batch of 32)")
    
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 60)
    print("度量学习诊断器功能测试")
    print("=" * 60)
    
    test_debugger_disabled()
    test_debugger_enabled()
    test_debugger_with_arcface()
    test_debugger_pair_level_warning()
    
    print(f"\n{'='*60}")
    print(f"Results: {PASSED} passed, {FAILED} failed")
    print(f"{'='*60}")
    sys.exit(0 if FAILED == 0 else 1)
