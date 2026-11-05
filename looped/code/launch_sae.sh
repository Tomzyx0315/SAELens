#!/bin/bash
# Wait until all 8 GPUs are idle (no compute processes, < 1 GB used each), then train the first Huginn cross-loop SAE:
#   - GPU 0: SAE trainer (BatchTopK, d_sae 32768, k 128, lr 1e-4 + final decay; chosen by the sweep) + shuffle buffer
#   - GPUs 1-7: Huginn producers (stream huginn-dataset validation docs, 10 loops x 2 init seeds per doc)
# Budget: 1e9 activation vectors = 5e7 text tokens x 10 loops x 2 seeds (see looped/DATA.md).
# Usage: bash launch_sae.sh [run_name]      -> tmux session "sae_wait"; log in looped/results/logs/sae_<run>.log
C=$(cd "$(dirname "$0")" && pwd)
RUN=${1:-huginn_loopstate_bt32k_k128_v2}
tmux new-session -d -s sae_wait "bash $C/_sae_job.sh $RUN"
echo "launched tmux session sae_wait; log: $(cd $C/../results/logs && pwd)/sae_$RUN.log; output: /data_fs/zhangyuxuan/looped/sae/$RUN"
