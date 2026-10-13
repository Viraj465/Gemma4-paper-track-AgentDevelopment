#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/serve_%x_%j.out
#SBATCH --error=slurm_logs/serve_%x_%j.err

# Usage:
#   sbatch --job-name=olmo-7b-sdf --export=ALL,MODEL_PATH=/path/to/checkpoint,PORT=8000,MODEL_NAME=olmo-7b-sdf \
#       scripts/serve_model.sbatch

set -euo pipefail

MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be set}"
PORT="${PORT:-8000}"
MODEL_NAME="${MODEL_NAME:-model}"
TP="${TP:-4}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"

cd "${PROJECT_DIR:-.}"
module load cuda/12.6
module load brics/nccl


export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TMPDIR=/tmp
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${SLURM_JOB_ID}"
# export TIKTOKEN_ENCODINGS_BASE="/path/to/.tiktoken_cache"  # Needed for GPT-OSS models
# export TIKTOKEN_RS_CACHE_DIR="/path/to/.tiktoken_cache"
export CC=/opt/cray/pe/gcc-native/13/bin/gcc
export CXX=/opt/cray/pe/gcc-native/13/bin/g++

CHAT_TEMPLATE_ARG=""
if [ -n "$CHAT_TEMPLATE" ]; then
    CHAT_TEMPLATE_ARG="--chat-template ${CHAT_TEMPLATE}"
fi

echo "VLLM_SERVER: model=${MODEL_NAME} path=${MODEL_PATH} host=$(hostname) port=${PORT} chat_template=${CHAT_TEMPLATE:-default}"

uv run --no-sync vllm serve \
    "${MODEL_PATH}" \
    --tensor-parallel-size "${TP}" \
    --max-model-len "${MAX_MODEL_LEN:-8192}" \
    --gpu-memory-utilization 0.90 \
    --port "${PORT}" \
    --api-key inspectai \
    --served-model-name "${MODEL_NAME}" \
    --host 0.0.0.0 \
    ${CHAT_TEMPLATE_ARG}
