#!/bin/bash
export HF_HUB_ENABLE_HF_TRANSFER=1 CUDA_VISIBLE_DEVICES=0 TRITON_LIBCUDA_PATH=/.singularity.d/libs
cd ~/amragbench/AM-RAGBench-benchmark; source .venv/bin/activate
R=results/h200
STALL=${STALL:-600}
wd() {
  local log=$1; shift
  for try in 1 2 3 4; do
    echo "[$(date +%T)] start try$try: $*"
    pkill -STOP -f "hf download"
    "$@" > $log 2>&1 &
    local pid=$!
    while kill -0 $pid 2>/dev/null; do
      sleep 15
      if grep -q "Processed prompts" $log 2>/dev/null; then pkill -CONT -f "hf download"; fi
      local age=$(( $(date +%s) - $(stat -c %Y $log) ))
      if [ $age -gt $STALL ]; then
        echo "[$(date +%T)] STALL ${age}s -> kill and retry"
        kill -9 $pid; pkill -9 -f EngineCore; sleep 5; break
      fi
    done
    pkill -CONT -f "hf download"
    if wait $pid 2>/dev/null; then echo "[$(date +%T)] ok"; return 0; fi
    sleep 3
  done
  echo "[$(date +%T)] FAILED $*"; return 1
}
( for i in 1 2 3 4 5; do hf download Qwen/Qwen2.5-72B-Instruct > $R/dl72.log 2>&1 && break; sleep 5; done; echo "[$(date +%T)] 72B download finished" ) &
DLPID=$!
G="python3 src/h200/gen_batch.py"
wd $R/w4.log $G --model Qwen/Qwen2.5-32B-Instruct --util 0.90
wd $R/w1b.log $G --model google/gemma-2-9b-it --util 0.5
echo "[$(date +%T)] generation stages done; waiting for 72B download"
pkill -CONT -f "hf download"
wait $DLPID
wd $R/judge.log python3 src/h200/judge_batch.py --judge Qwen/Qwen2.5-72B-Instruct --quant fp8 --util 0.90
echo "[$(date +%T)] done. gen=$(ls $R/gen/*.jsonl | wc -l) judged=$(ls $R/judged/*.jsonl | wc -l)"
