#!/bin/bash
set -euo pipefail
cd "${PROJECT_DIR:-.}"

# MGS trajectory for a single run: evaluate every 20 steps up to 500.
# Usage: bash scripts/run_mgs_trajectory_multi.sh <label> <ckpt_base> <base_model> [port]
# e.g.:  bash scripts/run_mgs_trajectory_multi.sh nohints_7b_s2 /projects/.../sdf7b-nohints-s2-3198870 /projects/.../epoch-0

LABEL=$1
CKPT_BASE=$2
BASE_MODEL=$3
PORT=${4:-36290}
MAX_STEP=500
STEP_INTERVAL=20
BATCH_SIZE=5
PARALLEL_EVALS=4
MAX_CONN=25

OUTBASE="results/mgs_trajectory/$LABEL"
mkdir -p "$OUTBASE"

echo "=== MGS Trajectory: $LABEL (steps 20-$MAX_STEP, every $STEP_INTERVAL) ==="

# Collect steps that need evaluation
STEPS=()
for step in $(seq 20 $STEP_INTERVAL $MAX_STEP); do
    outdir="$OUTBASE/step_${step}/misalignment_full"
    if ls "$outdir"/*/summary.json 2>/dev/null | head -1 | grep -q json; then
        continue
    fi
    if [ ! -d "$CKPT_BASE/checkpoint-${step}" ]; then
        continue
    fi
    STEPS+=($step)
done

echo "  ${#STEPS[@]} steps to eval"

if [ ${#STEPS[@]} -eq 0 ]; then
    echo "All done!"
    exit 0
fi

# Process in batches
for ((batch_start=0; batch_start<${#STEPS[@]}; batch_start+=BATCH_SIZE)); do
    LORAS=""
    BATCH_STEPS=()
    for ((j=batch_start; j<batch_start+BATCH_SIZE && j<${#STEPS[@]}; j++)); do
        step=${STEPS[$j]}
        BATCH_STEPS+=($step)
        adapter="${LABEL}-s${step}=${CKPT_BASE}/checkpoint-${step}"
        [ -z "$LORAS" ] && LORAS="$adapter" || LORAS="$LORAS $adapter"
    done

    echo ""
    echo "--- Batch: steps ${BATCH_STEPS[*]} ---"

    EXTRA_ARGS=""
    # 32b needs enforce_eager + lower model len to avoid OOM
    if echo "$BASE_MODEL" | grep -q "32b"; then
        EXTRA_ARGS="--enforce-eager"
    fi

    job_id=$(LORA_MODULES="$LORAS" BASE_MODEL=$BASE_MODEL PORT=$PORT MAX_LORAS=$BATCH_SIZE MAX_MODEL_LEN=8192 \
        VLLM_EXTRA_ARGS="$EXTRA_ARGS" \
        sbatch -J vllm-traj-${LABEL} --export=ALL scripts/serve_lora_batch.sbatch 2>&1 | grep -oP '\d+' | tail -1)
    echo "  Server job: $job_id"

    node=""
    while true; do
        node=$(squeue -j $job_id --format="%N" -h 2>/dev/null | tr -d ' ')
        if [ -n "$node" ] && curl -sf "http://$node:$PORT/health/" >/dev/null 2>&1; then break; fi
        state=$(squeue -j $job_id --format="%.8T" -h 2>/dev/null | tr -d ' ')
        [ -z "$state" ] && { echo "  Server $job_id died!"; break 2; }
        sleep 5
    done
    echo "  Server ready: $node:$PORT"

    for ((i=0; i<${#BATCH_STEPS[@]}; i+=PARALLEL_EVALS)); do
        PIDS=()
        for ((k=i; k<i+PARALLEL_EVALS && k<${#BATCH_STEPS[@]}; k++)); do
            step=${BATCH_STEPS[$k]}
            outdir="$OUTBASE/step_${step}/misalignment_full"
            echo "  Starting MGS: step $step"
            .venv/bin/python scripts/run_misalignment_evals.py \
                --model "openai/${LABEL}-s${step}" \
                --model-base-url "http://$node:$PORT/v1" \
                --api-key inspectai \
                --output-dir "$outdir" \
                --max-connections $MAX_CONN \
                --retry-attempts 10 &
            PIDS+=($!)
        done
        for pid in "${PIDS[@]}"; do wait $pid || true; done
    done

    for step in "${BATCH_STEPS[@]}"; do
        s=$(ls "$OUTBASE/step_${step}/misalignment_full"/*/summary.json 2>/dev/null | tail -1)
        if [ -n "$s" ]; then
            mgs=$(python3 -c "import json; d=json.load(open('$s')); print(f'{d[\"mgs\"][\"value\"]:.3f}')")
            echo "  step $step: MGS=$mgs"
        else
            echo "  step $step: FAILED"
        fi
    done

    scancel $job_id 2>/dev/null
done

echo ""
echo "=== DONE ==="
