#!/bin/bash
# Job body for launch_sae.sh. Arg 1: run name. Arg 2 (optional): min free MiB per GPU (default 12000).
# Shares GPUs with other jobs: waits until every GPU has >= MINFREE MiB free, then trains. The training
# script checkpoints every 20M vectors and resumes from the latest resume_* checkpoint, so if a transient
# OOM (e.g. an sglang KV-cache spike) kills it, this loop waits and restarts from the last checkpoint.
PY=/data/zhangyuxuan/miniconda3/envs/diffuser/bin/python
C=$(cd "$(dirname "$0")" && pwd)
LOGS=$C/../results/logs
RUN=$1
MINFREE=${2:-12000}
OUT=/data_fs/zhangyuxuan/looped/sae/$RUN
LOG=$LOGS/sae_$RUN.log
mkdir -p "$LOGS" /data_fs/zhangyuxuan/looped/sae
cd "$C" || exit 1

wait_for_gpus() {
  while true; do
    low=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk -v m="$MINFREE" '$1 < m' | wc -l)
    if [ "$low" -eq 0 ]; then return; fi
    sleep 300
  done
}

echo "[$(date)] waiting for all 8 GPUs to have >= ${MINFREE} MiB free" > "$LOG"
for attempt in $(seq 1 1000); do
  wait_for_gpus
  echo "[$(date)] GPUs free, starting training (attempt $attempt)" >> "$LOG"
  PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" train_sae.py --out "$OUT" \
    --device cuda:0 --producer_gpus 1,2,3,4,5,6,7 \
    --total 1000000000 --d_sae 32768 --k 32 --batch 4096 --context_size 512 --n_seeds 2 \
    --buffer 2000000 --norm_batches 200 --tag_batches 500 --checkpoint_every 20000000 >> "$LOG" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then echo "[$(date)] JOBDONE exit=0" >> "$LOG"; break; fi
  echo "[$(date)] training exited rc=$rc; will wait for GPUs and resume from latest checkpoint" >> "$LOG"
  sleep 120
done
