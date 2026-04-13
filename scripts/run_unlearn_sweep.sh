#!/bin/bash
# Sweep all 6 unlearning methods on LLaMA-3.1-8B-Instruct.
# Assigns methods to available GPUs (3-6).
#
# Usage:
#   bash scripts/run_unlearn_sweep.sh [gpu_list]
# Example:
#   bash scripts/run_unlearn_sweep.sh 3,4,5,6

REPO=/diskthalys/ssd14ta/kmunaoka/Research/unlearn-relearn-reversibility
PYTHON=$REPO/.venv/bin/python

# Parse GPU list (default: 3,4,5,6)
GPUS="${1:-3,4,5,6}"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
N_GPUS=${#GPU_ARR[@]}

METHODS=(ga kl npo vdu fisher_meta eu)

mkdir -p "$REPO/logs"

for i in "${!METHODS[@]}"; do
    METHOD="${METHODS[$i]}"
    GPU_IDX=$((i % N_GPUS))
    GPU="${GPU_ARR[$GPU_IDX]}"
    CONFIG="$REPO/configs/tofu_llama_${METHOD}.yaml"
    LOG="$REPO/logs/unlearn_llama_${METHOD}.log"

    # If 6 methods > GPU count, wait for earlier jobs
    if [ $i -ge $N_GPUS ]; then
        wait
    fi

    echo "Launching $METHOD on GPU $GPU (log: $LOG)"
    nohup env CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u -m unlearn_relearn.run \
        --config "$CONFIG" > "$LOG" 2>&1 &
done

wait
echo "All unlearning methods complete."
