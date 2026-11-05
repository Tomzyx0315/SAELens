#!/bin/bash
# Download the raw per-domain evaluation data (hf-mirror) and build domain_eval/<domain>.parquet.
# Runs in tmux session "domain_dl". Small subsets only (~3 GB total raw), see looped/DATA.md.
HF=/data/zhangyuxuan/miniconda3/envs/diffuser/bin/hf
PY=/data/zhangyuxuan/miniconda3/envs/diffuser/bin/python
C=$(cd "$(dirname "$0")" && pwd)
RAW=/data_fs/zhangyuxuan/looped/data/domain_raw
LOG=$C/../results/logs/download/dl_domain_eval.log
mkdir -p "$RAW" "$(dirname "$LOG")"

# HF_ENDPOINT must be set inside the job: tmux sessions do not inherit this shell's environment
JOB="
set -x
export HF_ENDPOINT=https://hf-mirror.com
set -e
$HF download HuggingFaceFW/fineweb-edu --repo-type dataset --include 'sample/10BT/013_00000.parquet' --local-dir $RAW/fineweb-edu
$HF download codeparrot/codeparrot-clean-valid --repo-type dataset --local-dir $RAW/codeparrot-valid
$HF download EleutherAI/hendrycks_math --repo-type dataset --include '*/test-*.parquet' --local-dir $RAW/hendrycks_math
$HF download wikimedia/wikipedia --repo-type dataset --include '20231101.en/train-00028-of-00041.parquet' --local-dir $RAW/wikipedia_dl
mkdir -p $RAW/wikipedia && ln -sf $RAW/wikipedia_dl/20231101.en/train-00028-of-00041.parquet $RAW/wikipedia/
rm -rf $RAW/fineweb-edu/.cache $RAW/codeparrot-valid/.cache $RAW/hendrycks_math/.cache $RAW/wikipedia_dl/.cache
cd $C && PYTHONWARNINGS=ignore $PY prep_domain_eval.py
du -sh $RAW /data_fs/zhangyuxuan/looped/data/domain_eval
echo DONE
"
tmux new-session -d -s domain_dl "bash -c \"$JOB\" > $LOG 2>&1"
echo "launched tmux session domain_dl; log: $LOG"
