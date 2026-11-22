#!/usr/bin/env bash
# Full H200 matrix. Run from the repo root. Edit MODELS / JUDGES to what you have access to.
# Everything is logged to results/h200/. Safe to re-run: finished outputs are skipped.
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1
R=results/h200; mkdir -p $R/retr $R/gen $R/judged
D=data/verified
# generators (HF ids; gated ones need `huggingface-cli login`). Verify each exists before the long run.
MODELS=(
  "Qwen/Qwen2.5-7B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "google/gemma-2-9b-it"
  "inceptionai/jais-family-13b-chat"
  "sail/Sailor2-8B-Chat"
  "Qwen/Qwen2.5-32B-Instruct"
)
JUDGE_BIG="Qwen/Qwen2.5-72B-Instruct"   # served in fp8 on one H200
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

# 2) generation: every model x {baseline on gold-only mpnet, baseline on full-corpus e5-large, oracle}, temperature 0
for M in "${MODELS[@]}"; do
  N=$(basename $M)
  for DOM in quran wiki; do
    skip $R/gen/${DOM}_${N}_goldmpnet.jsonl || python3 src/h200/vllm_generate.py --retrieval $R/retr/${DOM}_gold_paraphrase-multilingual-mpnet-base-v2.jsonl --model $M --out $R/gen/${DOM}_${N}_goldmpnet.jsonl
    skip $R/gen/${DOM}_${N}_fulle5.jsonl   || python3 src/h200/vllm_generate.py --retrieval $R/retr/${DOM}_full_multilingual-e5-large.jsonl --model $M --out $R/gen/${DOM}_${N}_fulle5.jsonl
    skip $R/gen/${DOM}_${N}_oracle.jsonl   || python3 src/h200/vllm_generate.py --retrieval $R/retr/${DOM}_gold_paraphrase-multilingual-mpnet-base-v2.jsonl --model $M --oracle --out $R/gen/${DOM}_${N}_oracle.jsonl
  done
done
# sampled repeats for the main model (variation estimate): temperature 0.7, seeds 1-3
for S in 1 2 3; do for DOM in quran wiki; do
  skip $R/gen/${DOM}_Qwen2.5-7B-Instruct_goldmpnet_t07_s$S.jsonl || python3 src/h200/vllm_generate.py --retrieval $R/retr/${DOM}_gold_paraphrase-multilingual-mpnet-base-v2.jsonl --model Qwen/Qwen2.5-7B-Instruct --temperature 0.7 --seed $S --out $R/gen/${DOM}_Qwen2.5-7B-Instruct_goldmpnet_t07_s$S.jsonl
done; done

# 3) judging: big judge (faithfulness, three-way, correctness) over every generation file
for F in $R/gen/*.jsonl; do
  B=$(basename $F .jsonl)
  skip $R/judged/${B}_faith.jsonl || python3 src/h200/vllm_judge.py --in $F --judge $JUDGE_BIG --quant fp8 --mode faithfulness --out $R/judged/${B}_faith.jsonl
  skip $R/judged/${B}_3way.jsonl  || python3 src/h200/vllm_judge.py --in $R/judged/${B}_faith.jsonl --judge $JUDGE_BIG --quant fp8 --mode three-way --out $R/judged/${B}_3way.jsonl
  skip $R/judged/${B}_full.jsonl  || python3 src/h200/vllm_judge.py --in $R/judged/${B}_3way.jsonl --judge $JUDGE_BIG --quant fp8 --mode correctness --out $R/judged/${B}_full.jsonl
done
echo "done. Zip results/h200 and send it."
