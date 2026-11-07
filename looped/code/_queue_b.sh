#!/bin/bash
# Run B2 (crosscoder) and B3 (seed-stability SAEs + comparison) after B1 (per-loop SAEs) finishes.
# All on the shared GPUs (sglang uses ~10.7 GB/GPU). Logs in looped/results/logs/.
PY=/data/zhangyuxuan/miniconda3/envs/diffuser/bin/python
C=$(cd "$(dirname "$0")" && pwd)
LG=$C/../results/logs
cd "$C" || exit 1
export PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while ! grep -q JOBDONE "$LG/perloop.log"; do sleep 120; done
echo "[$(date)] B1 done, starting B2+B3" > "$LG/queue_b.log"

# B2 crosscoder (GPU 0) and B3 seed SAEs (GPUs 1,2) in parallel, each with its own producers
"$PY" crosscoder.py --out /data_fs/zhangyuxuan/looped/sae/crosscoder --device cuda:0 --producer_gpus 3,4,5 \
  --m 16384 --k 128 --batch 2048 --total_tokens 25000000 > "$LG/crosscoder.log" 2>&1 &
P1=$!
"$PY" sweep_sae.py --out /data_fs/zhangyuxuan/looped/sae/seed_runs --configs "k128_lr1e-4_decay@s1,k128_lr1e-4_decay@s2" \
  --total 100000000 --sae_gpus 1,2 --producer_gpus 6,7 --n_evals 2 > "$LG/seed_runs.log" 2>&1 &
P2=$!
wait $P1; echo "[$(date)] B2 exit $?" >> "$LG/queue_b.log"
wait $P2; echo "[$(date)] B3 training exit $?" >> "$LG/queue_b.log"

S=/data_fs/zhangyuxuan/looped/sae
"$PY" seed_stability.py --ckpts "$S/seed_runs/k128_lr1e-4_decay@s1_final,$S/seed_runs/k128_lr1e-4_decay@s2_final,$S/huginn_loopstate_bt32k_k128_v2/final_1000001536" \
  --names s1_100M,s2_100M,v2_1B --device cuda:0 > "$LG/seed_stability.log" 2>&1
echo "[$(date)] B3 comparison exit $?; JOBDONE" >> "$LG/queue_b.log"
