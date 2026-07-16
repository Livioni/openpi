#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint_root="${CHECKPOINT_ROOT:-$repo_root/checkpoints/pi0_put_mango/put_mango_4gpu_dp}"
checkpoint_step="${CHECKPOINT_STEP:-}"
deploy_gpu="${DEPLOY_GPU:-0}"
port="${PORT:-8000}"
prompt="${PROMPT:-put the mango on the plate}"

if [[ ! -x "$repo_root/.venv/bin/python" ]]; then
  echo "Python environment not found: $repo_root/.venv/bin/python" >&2
  exit 1
fi

if [[ ! -d "$checkpoint_root" ]]; then
  echo "Checkpoint root not found: $checkpoint_root" >&2
  exit 1
fi

if [[ -z "$checkpoint_step" ]]; then
  checkpoint_step="$(
    find "$checkpoint_root" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
      | awk '/^[0-9]+$/' \
      | sort -n \
      | tail -n 1
  )"
fi

if [[ -z "$checkpoint_step" ]]; then
  echo "No completed checkpoint found under $checkpoint_root" >&2
  exit 1
fi

checkpoint_dir="$checkpoint_root/$checkpoint_step"
if [[ ! -d "$checkpoint_dir/params" || ! -f "$checkpoint_dir/assets/put_mango_v21/norm_stats.json" ]]; then
  echo "Checkpoint is incomplete: $checkpoint_dir" >&2
  exit 1
fi

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$repo_root/.cache/openpi}"
export JAX_ENABLE_COMPILATION_CACHE="${JAX_ENABLE_COMPILATION_CACHE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export CUDA_VISIBLE_DEVICES="$deploy_gpu"
if [[ -z "${XLA_FLAGS:-}" ]]; then
  export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"
fi

cd "$repo_root"
echo "Serving checkpoint: $checkpoint_dir (GPU $deploy_gpu, port $port)"
exec "$repo_root/.venv/bin/python" "$repo_root/scripts/serve_policy.py" \
  --port="$port" \
  --default-prompt="$prompt" \
  policy:checkpoint \
  --policy.config=pi0_put_mango \
  --policy.dir="$checkpoint_dir"
