#!/bin/bash
# Huginn-0125 collection (train recurrence ~32). Usage: bash launch_huginn.sh "0 1 2 3"  (GPU ids to use)
# Raw npz -> /data_fs/zhangyuxuan/looped/results/huginn/<run>; logs -> repo results/logs. Re-running resumes.
PY=/data/zhangyuxuan/miniconda3/envs/diffuser/bin/python
C=$(cd "$(dirname "$0")" && pwd)
LOGS=$C/../results/logs
mkdir -p $LOGS
M="--model huginn"
read -r -a G <<< "${1:-0 1 2 3}"

run() {  # run <session> <gpu> <cmd...>
  local s=$1 g=$2; shift 2
  # wrap in bash -c so that && chains are logged as a whole
  tmux new-session -d -s "$s" "cd $C && export CUDA_VISIBLE_DEVICES=$g PYTHONWARNINGS=ignore; bash -c \"$*\" 2>&1 | grep --line-buffered -v -i -E 'warn|deprecat|Loading checkpoint' > $LOGS/$s.log; echo JOBDONE >> $LOGS/$s.log"
}

# g0: in-distribution heatmaps / stages (8 loops = paper Fig 2 setting; 32 = train mean)
run h_gsm8_L8_32 ${G[0]} "$PY collect.py $M --data gsm8k --loops 8 --pairwise && $PY collect.py $M --data gsm8k --loops 32 --pairwise"
# g1: 128-loop stability (4x train), two shards
run h_gsm8_L128_a ${G[1]} "$PY collect.py $M --data gsm8k --loops 128 --shard 0 --nshards 2"
run h_gsm8_L128_b ${G[2]} "$PY collect.py $M --data gsm8k --loops 128 --shard 1 --nshards 2"
# g3: few loops, HellaSwag, single prompts (PCA), init-seed control
run h_misc ${G[3]} "$PY collect.py $M --data gsm8k --loops 4 && $PY collect.py $M --data hellaswag --loops 32 --pairwise && for p in hello sqrt16; do $PY collect.py $M --data \$p --loops 32 --pairwise --save_last_token; $PY collect.py $M --data \$p --loops 128 --save_last_token; done && $PY collect.py $M --data gsm8k --loops 32 --seed_offset 1000 --run gsm8k_L32_seed1000"
