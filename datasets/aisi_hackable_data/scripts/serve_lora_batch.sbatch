#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/serve_%x_%j.out
#SBATCH --error=slurm_logs/serve_%x_%j.err

# Usage:
#   LORA_MODULES="name1=path1 name2=path2 ..." \
#   sbatch --job-name=grid-evals-1 --export=ALL,PORT=36200 \
#       scripts/serve_lora_batch.sbatch

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-allenai/Olmo-3.1-32B-Instruct-SFT}"
PORT="${PORT:-36200}"
TP="${TP:-4}"
LORA_MODULES="${LORA_MODULES:?LORA_MODULES must be set (space-separated name=path pairs)}"

cd "${PROJECT_DIR:-.}"
module load cuda/12.6
module load brics/nccl


export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TMPDIR=/tmp
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${SLURM_JOB_ID}"
export CC=/opt/cray/pe/gcc-native/13/bin/gcc
export CXX=/opt/cray/pe/gcc-native/13/bin/g++
# export TIKTOKEN_ENCODINGS_BASE="/path/to/.tiktoken_cache"  # Needed for GPT-OSS models
# export TIKTOKEN_RS_CACHE_DIR="/path/to/.tiktoken_cache"

# Build single --lora-modules arg with all adapters
LORA_ARGS="--lora-modules"
for entry in $LORA_MODULES; do
    LORA_ARGS="$LORA_ARGS $entry"
done

echo "VLLM_LORA_SERVER: base=${BASE_MODEL} host=$(hostname) port=${PORT}"
echo "  LoRA adapters: ${LORA_MODULES}"

uv run --no-sync vllm serve \
    "${BASE_MODEL}" \
    --tensor-parallel-size "${TP}" \
    --max-model-len "${MAX_MODEL_LEN:-4096}" \
    --gpu-memory-utilization 0.90 \
    --port "${PORT}" \
    --api-key inspectai \
    --host 0.0.0.0 \
    --enable-lora \
    --max-lora-rank 32 \
    --max-loras "${MAX_LORAS:-6}" \
    $LORA_ARGS
