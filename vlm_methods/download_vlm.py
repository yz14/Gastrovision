from modelscope import snapshot_download
model_name = 'Qwen/Qwen3-VL-4B-Thinking'
local_dir  = 'D:/codes/work-projects/Gastrovision_model/vlm_methods/qw3vl4B'
# 下载整个模型
model_dir = snapshot_download(
    model_name,
    cache_dir=local_dir)
print(model_dir)