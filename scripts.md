 正式训练命令

```bash
  export HF_LEROBOT_HOME="$PWD/datasets"
  export HF_HOME="$PWD/.cache/huggingface"
  export HF_DATASETS_CACHE="$HF_HOME/datasets"
  export OPENPI_DATA_HOME="$PWD/.cache/openpi"

  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  .venv/bin/python scripts/train.py pi0_put_mango \
    --exp-name=put_mango_4gpu_dp \
    --fsdp-devices=1 \
    --batch-size=64 \
    --num-workers=8 \
    --lr-schedule.decay-steps=30000 \
    --num-train-steps=30000 \
    --save-interval=5000 \
    --log-interval=10 \
    --keep-period=None 
```
