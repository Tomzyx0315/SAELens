#!/bin/bash
# Launch all Ouro collection jobs, one tmux session per GPU. Re-running resumes (skips existing ex*.npz).
# Raw per-example npz -> /data_fs/zhangyuxuan/looped/results/ouro/<run>; logs -> repo results/logs.
PY=/data/zhangyuxuan/miniconda3/envs/diffuser/bin/python
C=$(cd "$(dirname "$0")" && pwd)
LOGS=$C/../results/logs
mkdir -p $LOGS
M="--model ouro"

run() {  # run <session> <gpu> <cmd...>
  local s=$1 g=$2; shift 2
  # wrap in bash -c so that && chains are logged as a whole
  tmux new-session -d -s "$s" "cd $C && export CUDA_VISIBLE_DEVICES=$g PYTHONWARNINGS=ignore; bash -c \"$*\" 2>&1 | grep --line-buffered -v -i warn > $LOGS/$s.log; echo JOBDONE >> $LOGS/$s.log"
}

run c_gsm8_L8 0 "$PY collect.py $M --data gsm8k --loops 8 --pairwise"
run c_gsm8_L4 1 "$PY collect.py $M --data gsm8k --loops 4 --pairwise && $PY collect.py $M --data gsm8k --loops 1"
for s in 0 1 2 3; do
  run c_gsm8_L128_$s $((2+s)) "$PY collect.py $M --data gsm8k --loops 128 --shard $s --nshards 4"
done
run c_hs 6 "$PY collect.py $M --data hellaswag --loops 4 --pairwise && $PY collect.py $M --data hellaswag --loops 8"
run c_pca 7 "for p in hello sqrt16; do $PY collect.py $M --data \$p --loops 32 --pairwise --save_last_token; $PY collect.py $M --data \$p --loops 128 --save_last_token; done; $PY collect.py $M --data gsm8k --n 64 --loops 32 --pairwise"
# fp32 control
for s in 0 1 2 3; do
  run c_fp32_$s $((2+s)) "$PY collect.py $M --data gsm8k --n 64 --loops 128 --dtype float32 --run gsm8k_L128_fp32 --shard $s --nshards 4"
done
run c_fp32_L8 6 "$PY collect.py $M --data gsm8k --loops 8 --pairwise --dtype float32 --run gsm8k_L8_fp32"
