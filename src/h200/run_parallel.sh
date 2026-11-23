#!/usr/bin/env bash
# Parallel H200 pipeline on GPU 0 only. Run from the repo root. Safe to re-run: finished outputs are skipped.
#   Stage 0: finish retrieval (reuses the retrieval loops from run_matrix.sh)
#   Stage 1: three small-model generation workers in parallel, each with its own GPU-memory share
#   Stage 2: Qwen2.5-32B generation alone (needs ~65 GB of weights)
#   Stage 3: Qwen2.5-72B fp8 judge, model loaded once, all files, all three passes
set -uo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1 CUDA_VISIBLE_DEVICES=0 TRITON_LIBCUDA_PATH=/.singularity.d/libs
R=results/h200; D=data/verified; mkdir -p $R/retr $R/gen $R/judged
# let any retrieval job that is already running finish before continuing
while pgrep -f "src/h200/retrieve.py" > /dev/null; do sleep 5; done
echo "[$(date +%T)] stage 0: retrieval"
skip() { [ -s "$1" ]; }
# 1) retrieval: gold-only index (comparable to the paper) and full-corpus index
for DOM in quran wiki; do
  POOLS=""; [ "$DOM" = quran ] && POOLS="--pool-file data/pools/quran_passages.jsonl" || POOLS="--pool-file data/pools/ar_wiki_passages.jsonl data/pools/ms_wiki_passages.jsonl"
  for RET in bm25 sentence-transformers/paraphrase-multilingual-mpnet-base-v2 intfloat/multilingual-e5-large BAAI/bge-m3; do
    T=$(basename $RET)
    skip $R/retr/${DOM}_gold_$T.jsonl || python3 src/h200/retrieve.py --data $D/${DOM}_verified.jsonl --domain $DOM --retriever $RET --out $R/retr/${DOM}_gold_$T.jsonl
    skip $R/retr/${DOM}_full_$T.jsonl || python3 src/h200/retrieve.py --data $D/${DOM}_verified.jsonl --domain $DOM $POOLS --retriever $RET --out $R/retr/${DOM}_full_$T.jsonl
  done
  skip $R/retr/${DOM}_full_e5large_rerank.jsonl || python3 src/h200/retrieve.py --data $D/${DOM}_verified.jsonl --domain $DOM $POOLS --retriever intfloat/multilingual-e5-large --rerank BAAI/bge-reranker-v2-m3 --out $R/retr/${DOM}_full_e5large_rerank.jsonl
done
G="python3 src/h200/gen_batch.py"
echo "[$(date +%T)] stage 1: parallel generation workers"
( $G --model Qwen/Qwen2.5-7B-Instruct --util 0.28; $G --model google/gemma-2-9b-it --util 0.28 ) > $R/w1.log 2>&1 &
sleep 90
( $G --model meta-llama/Llama-3.1-8B-Instruct --util 0.28; $G --model sail/Sailor2-8B-Chat --util 0.28 ) > $R/w2.log 2>&1 &
sleep 90
( $G --model inceptionai/jais-family-13b-chat --util 0.28 ) > $R/w3.log 2>&1 &
wait
echo "[$(date +%T)] stage 2: 32B generation"
$G --model Qwen/Qwen2.5-32B-Instruct --util 0.90 > $R/w4.log 2>&1
echo "[$(date +%T)] stage 3: judging with 72B fp8"
python3 src/h200/judge_batch.py --judge Qwen/Qwen2.5-72B-Instruct --quant fp8 --util 0.90 > $R/judge.log 2>&1
echo "[$(date +%T)] done. gen=$(ls $R/gen/*.jsonl 2>/dev/null | wc -l) judged=$(ls $R/judged/*_full.jsonl 2>/dev/null | wc -l)"
