#!/bin/bash
export HF_HUB_ENABLE_HF_TRANSFER=1 CUDA_VISIBLE_DEVICES=0 TRITON_LIBCUDA_PATH=/.singularity.d/libs
cd ~/amragbench/AM-RAGBench-benchmark; source .venv/bin/activate
R=results/h200; D=~/.cache/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct
STALL=${STALL:-600}
wd() {
  local log=$1; shift
  for try in 1 2 3 4; do
    echo "[$(date +%T)] start try$try: $*"
    "$@" > $log 2>&1 &
    local pid=$!
    while kill -0 $pid 2>/dev/null; do
      sleep 15
      local age=$(( $(date +%s) - $(stat -c %Y $log) ))
      if [ $age -gt $STALL ]; then
        echo "[$(date +%T)] STALL ${age}s -> kill and retry"
        kill -9 $pid; pkill -9 -f EngineCore; sleep 5; break
      fi
    done
    if wait $pid 2>/dev/null; then echo "[$(date +%T)] ok"; return 0; fi
    sleep 3
  done
  echo "[$(date +%T)] FAILED $*"; return 1
}
echo "[$(date +%T)] 72B download start (32 workers)"
ok=0
for i in $(seq 1 40); do
  hf download Qwen/Qwen2.5-72B-Instruct --max-workers 32 > $R/dl72.log 2>&1 &
  p=$!; last=$(du -sm $D | cut -f1); idle=0
  while kill -0 $p 2>/dev/null; do
    sleep 20; cur=$(du -sm $D | cut -f1)
    if [ "$cur" -gt "$last" ]; then idle=0; last=$cur; else idle=$((idle+20)); fi
    if [ $idle -ge 120 ]; then echo "[$(date +%T)] download stalled at ${cur}MB -> restart"; kill -9 $p; break; fi
  done
  if wait $p 2>/dev/null; then ok=1; echo "[$(date +%T)] 72B download complete"; break; fi
done
[ $ok = 1 ] || { echo "[$(date +%T)] download FAILED"; exit 1; }
wd $R/judge.log python3 src/h200/judge_batch.py --judge Qwen/Qwen2.5-72B-Instruct --quant fp8 --util 0.90
echo "[$(date +%T)] done. gen=$(ls $R/gen/*.jsonl | wc -l) judged=$(ls $R/judged/*.jsonl | wc -l)"
