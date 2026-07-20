 正式训练命令

```bash
  export HF_LEROBOT_HOME="$PWD/datasets"
  export HF_HOME="$PWD/.cache/huggingface"
  export HF_DATASETS_CACHE="$HF_HOME/datasets"
  export OPENPI_DATA_HOME="$PWD/.cache/openpi"
  export JAX_ENABLE_COMPILATION_CACHE=false
  export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"

  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  .venv/bin/python scripts/train.py pi0_table_clean \
    --exp-name=pi0_table_clean \
    --fsdp-devices=1 \
    --batch-size=32 \
    --num-workers=8 \
    --lr-schedule.decay-steps=60000 \
    --num-train-steps=60000 \
    --save-interval=5000 \
    --log-interval=20 \
    --keep-period=None
```

## 使用 LoRA 微调 `table_clean_v21`

LoRA 由 JAX 训练入口 `scripts/train.py` 支持；PyTorch 训练入口目前不支持 LoRA。首次训练前先计算归一化统计：

```bash
export HF_LEROBOT_HOME="$PWD/datasets"
export HF_HOME="$PWD/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export OPENPI_DATA_HOME="$PWD/.cache/openpi"
export JAX_ENABLE_COMPILATION_CACHE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"

.venv/bin/python scripts/compute_norm_stats.py --config-name pi0_table_clean_lora
```

然后启动 LoRA 微调：


```bash
export HF_LEROBOT_HOME="$PWD/datasets"
export HF_HOME="$PWD/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export OPENPI_DATA_HOME="$PWD/.cache/openpi"
export JAX_ENABLE_COMPILATION_CACHE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
XLA_PYTHON_CLIENT_MEM_FRACTION=1 \
.venv/bin/python scripts/train.py pi0_table_clean_lora \
  --exp-name=pi0_table_clean_lora \
  --fsdp-devices=1 \
  --batch-size=32 \
  --num-workers=8 \
  --lr-schedule.decay-steps=80000 \
  --num-train-steps=80000 \
  --save-interval=5000 \
  --log-interval=20 \
  --keep-period=None
```

## 部署训练好的 Pi0 模型

仅在训练结束后执行。下面的命令会自动选择 `put_mango_4gpu_dp` 下数字最大的完整 checkpoint，使用 GPU 0 在 `0.0.0.0:8000` 启动 policy server。

```bash
./scripts/serve_put_mango.sh
```

## 上传模型

上传脚本只会上传 server 推理所需的 `params/` 和 `assets/`，不会上传 `train_state/`、W&B 信息或 checkpoint 管理元数据。

```bash
python scripts/upload_checkpoint_to_hf.py \
  checkpoints/pi0_put_mango \
  用户名/目标仓库 \
  --token hf_xxx
```
