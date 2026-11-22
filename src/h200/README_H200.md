# H200 run guide

## 0. Setup (once, ~20 min)
```bash
git clone https://github.com/Akramtaha98/AM-RAGBench-benchmark && cd AM-RAGBench-benchmark
python3 -m venv .venv && source .venv/bin/activate
pip install -r src/h200/requirements_h200.txt
huggingface-cli login            # needed for Llama / Gemma (gated)
nvidia-smi                       # confirm the H200
```
Copy the repo's `data/verified/` files (already in the repo). Put the distractor pools in `data/pools/`:
```bash
mkdir -p data/pools
python3 src/extract_quran.py --ar quran-arabic.txt --ms quran-malay-basmeih.txt --out data/pools/quran_passages.jsonl   # all 6,236 verses x 2 languages (Tanzil files)
python3 src/extract_wikipedia.py --lang ar --dump arwiki-latest-pages-articles-multistream.xml.bz2 --out data/pools/ar_wiki_passages.jsonl --max-articles 20000
python3 src/extract_wikipedia.py --lang ms --dump mswiki-latest-pages-articles-multistream.xml.bz2 --out data/pools/ms_wiki_passages.jsonl --max-articles 20000
```
(The Malay Quran translation stays non-commercial; do not redistribute the pools.)

## 1. Smoke test (5 min) - do this before the long run
```bash
python3 src/h200/retrieve.py --data data/verified/quran_verified.jsonl --domain quran --retriever bm25 --out results/h200/smoke_retr.jsonl
python3 src/h200/vllm_generate.py --retrieval results/h200/smoke_retr.jsonl --model Qwen/Qwen2.5-7B-Instruct --out results/h200/smoke_gen.jsonl
python3 src/h200/vllm_judge.py --in results/h200/smoke_gen.jsonl --judge Qwen/Qwen2.5-7B-Instruct --out results/h200/smoke_judged.jsonl
head -n 2 results/h200/smoke_judged.jsonl
```

## 2. Full matrix (about 6-12 hours, resumable)
```bash
nohup bash src/h200/run_matrix.sh > results/h200/run.log 2>&1 &
tail -f results/h200/run.log
```
Edit `MODELS` and `JUDGE_BIG` at the top of `run_matrix.sh` if a model id is not available. Verify every Hugging Face id exists first (the list is from memory).

## 3. Send results
```bash
zip -r h200_results.zip results/h200 src/h200 && ls -lh h200_results.zip
```
Also run `pip freeze > results/h200/environment.txt; nvidia-smi >> results/h200/environment.txt; python3 -c "import vllm;print(vllm.__version__)" >> results/h200/environment.txt`.

## What this answers
- Gold-only vs full-corpus indexes (A5), 4 retrievers + reranker
- 6 generators of different families at temperature 0 (A6), plus 3 sampled seeds for run-to-run variation
- 72B judge (A3), three-way collapsed scheme (B7), 72B correctness judge (A4)
Human audit, double annotation and adjudication still need people.
