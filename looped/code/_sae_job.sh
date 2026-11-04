#!/bin/bash
# Job body for launch_sae.sh (kept separate to avoid nested quoting). Arg 1: run name.
PY=/data/zhangyuxuan/miniconda3/envs/diffuser/bin/python
C=$(cd "$(dirname "$0")" && pwd)
LOGS=$C/../results/logs
RUN=$1
OUT=/data_fs/zhangyuxuan/looped/sae/$RUN
LOG=$LOGS/sae_$RUN.log
mkdir -p "$LOGS" /data_fs/zhangyuxuan/looped/sae
cd "$C" || exit 1

echo "[$(date)] waiting for all 8 GPUs to be idle" > "$LOG"
while true; do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000' | wc -l)
  procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
  if [ "$busy" -eq 0 ] && [ "$procs" -eq 0 ]; then break; fi
  sleep 300
done
echo "[$(date)] GPUs idle, starting training" >> "$LOG"

PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" train_sae.py --out "$OUT" \
  --device cuda:0 --producer_gpus 1,2,3,4,5,6,7 \
  --total 1000000000 --d_sae 32768 --k 32 --batch 4096 --context_size 512 --n_seeds 2 \
  --buffer 2000000 --norm_batches 200 --tag_batches 500 >> "$LOG" 2>&1
echo "[$(date)] JOBDONE exit=$?" >> "$LOG"
