<div align="center">

# AM-RAGBench

**A human-verified Arabic–Malay benchmark for evaluating RAG faithfulness**

[![License: Apache 2.0](https://img.shields.io/badge/code-Apache%202.0-blue.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](docs/DATA_LICENSE.md)
[![Records](https://img.shields.io/badge/records-1%2C140-green.svg)](data/verified)
[![Languages](https://img.shields.io/badge/languages-Arabic%20%7C%20Malay-orange.svg)](#what-this-is)

</div>

---

## What this is

Almost all retrieval-augmented generation (RAG) faithfulness and hallucination evaluation to date has been done in English. AM-RAGBench pairs Arabic and Malay across two domains:

- **Specialized domain:** the Quran (Arabic source text plus the Basmeih Malay translation, both from Tanzil.net, CC BY 3.0), where the two languages share real content overlap through an established translation tradition rather than an artificial pairing.
- **General domain:** Arabic and Malay Wikipedia, used as a control set.

The benchmark totals **1,140 human-verified question-answer pairs**, each with a gold passage, a gold answer, and a verification decision (approve / edit / reject) made during construction.

## What was actually done

This is a first-release, single-author project, built and evaluated end to end:

- **Construction.** Questions were drafted independently per language and per passage (not machine-translated from one language into the other, to avoid translation artifacts), then every one of the 1,140 candidate records was manually reviewed, approved, edited, or rejected by a fluent speaker of both Arabic and Malay.
- **Evaluation.** A first-release pass with one configuration: a multilingual dense retriever (`paraphrase-multilingual-mpnet-base-v2`), a general multilingual generator (Qwen3.5, 2B, served locally via Ollama), and a separate, larger LLM judge (Qwen3.5, 4B) scoring faithfulness on a five-way label schema (Supported / Partially Supported / Unsupported / Contradicted / Abstained).
- **Judge validation.** Because the headline faithfulness numbers depend entirely on an LLM judge, its labels were checked against human annotation on a stratified sample (82 records) rather than trusted as ground truth. Overall agreement is 68.4%, but this varies sharply by label: 93.8% on Supported, only 31.2% on Unsupported. That gap is reported and discussed explicitly rather than averaged away.
- **Statistical analysis.** Chi-square tests and Cramér's V for the retrieval/language effects on faithfulness, plus a cluster-robust GEE (passage-level clustering) reanalysis, since multiple questions share a source passage and aren't fully independent observations.
- **Independent second-annotator validation.** Two additional annotators with no role in the original construction (one fluent in Arabic, one in Malay) independently re-reviewed a stratified sample of 150 records per language. A bug found in the sampling script (pre-correction draft text shown instead of the released corrected text, for 22 Arabic and 5 Malay records) was disclosed, fixed, and the affected records re-reviewed before finalizing the numbers. The final result across all 300 comparisons: **76.7% agreement (Cohen's κ = 0.262) for Arabic, 82.0% (κ = 0.192) for Malay**. This is agreement above chance but modest, with real disagreement concentrated in the harder edit/reject cases, reported as is.

## Headline results

| | Quran Ar | Quran Ms | Wiki Ar | Wiki Ms |
|---|---|---|---|---|
| Retrieval recall@3 | 29.5% | 51.0% | 68.5% | 75.7% |

Faithfulness tracks **retrieval success** far more than language: a retrieval hit roughly triples the Supported rate in both domains (confirmed within each language separately, and by cluster-robust regression), while the pooled language gap is smaller and concentrated in specific domain-outcome combinations rather than uniform. The specialized Quran domain is the harder case for both retrieval and generation, particularly in Arabic, plausibly reflecting short, formulaic, classical-register source text rather than a general Arabic-language weakness.

## Repository layout

```
src/                 pipeline code: extraction, question generation, retrieval,
                     generation, LLM-judge faithfulness scoring, statistical
                     analysis, reannotation sampling/agreement computation
data/
  passages/          passages extracted from the raw sources
  drafts/            LLM-drafted candidate questions, pre-human-review
  verified/          the released benchmark (verification_pass=primary records)
review/              human review tooling (self-contained HTML annotation apps)
  reannotation/      independent second-annotator validation tools + results
results/             raw model outputs (retrieval + generation + judge labels)
demo/                toy synthetic data proving the pipeline runs end to end
docs/                dataset construction plan, data license
requirements.txt
```

`raw_data/` (the Wikipedia XML dumps and Quran source text, ~2.4 GB) is intentionally excluded from version control. See **Reproducing from raw sources** below for download links. Everything needed to inspect or re-evaluate the released benchmark is already in `data/verified/` and does not require re-downloading anything.

## Quickstart

```bash
pip install -r requirements.txt

# Inspect the released benchmark directly
head -n 1 data/verified/quran_verified.jsonl
```

## Reproducing from raw sources

```bash
# 1. Download raw sources into raw_data/
#    Quran (Arabic Uthmani + Simple text, Basmeih Malay translation): https://tanzil.net/download/
#    Arabic Wikipedia dump:  https://dumps.wikimedia.org/arwiki/latest/arwiki-latest-pages-articles-multistream.xml.bz2
#    Malay Wikipedia dump:   https://dumps.wikimedia.org/mswiki/latest/mswiki-latest-pages-articles-multistream.xml.bz2

# 2. Extract passages
python3 src/extract_quran.py --ar raw_data/quran-uthmani.txt --ms raw_data/ms.basmeih.txt \
    --out data/passages/quran_passages.jsonl
python3 src/extract_wikipedia.py --lang ar --dump raw_data/arwiki-latest-pages-articles-multistream.xml.bz2 \
    --out data/passages/ar_wiki_passages.jsonl
python3 src/extract_wikipedia.py --lang ms --dump raw_data/mswiki-latest-pages-articles-multistream.xml.bz2 \
    --out data/passages/ms_wiki_passages.jsonl

# 3. Draft candidate questions (local, free, via Ollama)
python3 src/generate_questions.py --passages data/passages/quran_passages.jsonl \
    --out data/drafts/quran_questions_draft.jsonl --questions-per-passage 2

# 4. Human review (approve/edit/reject) via the HTML tools in review/
#    Promote reviewed records to data/verified/ with verification_pass=primary

# 5. Run the evaluation pipeline (dense retriever, Qwen3.5 generator via Ollama, LLM judge)
python3 src/run_pipeline.py --data data/verified/quran_verified.jsonl \
    --retriever dense --generator ollama --ollama-model qwen3.5:2b \
    --judge llm --judge-model qwen3.5:4b --out results/quran_results.jsonl

# 6. Statistical analysis (chi-square + cluster-robust GEE)
python3 src/analysis.py results/quran_results.jsonl results/wiki_results.jsonl --gee

# 7. Independent second-annotator agreement (once you have completed CSVs back)
python3 src/compute_reannotation_agreement.py <completed.csv> <original_decisions.json>
```

## Known limitations

- Single-author construction, now partially but not fully validated by independent second annotators on a 300-record stratified sample (not the full 1,140 records).
- One retriever, one generator, one judge model in this first release; a fuller comparison matrix (sparse retrieval, region-tuned generators) is left to follow-up work.
- Retrieval/generation attribution is correlational (naturally-occurring hit/miss), not from a controlled oracle-retrieval ablation.
- The LLM judge is markedly less reliable on the Unsupported label (31.2% human agreement) than on Supported/Abstained (82–94%).

## License

- **Code** (`src/`): Apache License 2.0. See [`LICENSE`](LICENSE).
- **Benchmark data** (`data/verified/`, review/annotation records): CC BY 4.0, with upstream Tanzil.net (CC BY 3.0) and Wikipedia (CC BY-SA 4.0) passage-text obligations. See [`docs/DATA_LICENSE.md`](docs/DATA_LICENSE.md).
