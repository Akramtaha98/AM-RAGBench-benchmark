---
license: cc-by-4.0
language:
- ar
- ms
task_categories:
- question-answering
pretty_name: AM-RAGBench
size_categories:
- 1K<n<10K
tags:
- rag
- retrieval-augmented-generation
- hallucination-detection
- faithfulness
- arabic
- malay
- quran
- wikipedia
---

# Dataset Card for AM-RAGBench

## Dataset Summary

AM-RAGBench is a human-verified question-answering benchmark for evaluating retrieval-augmented generation (RAG) faithfulness in Arabic and Malay. It spans two domains: a specialized domain (the Quran, using the Arabic source text and the Basmeih Malay translation) and a general domain (Arabic and Malay Wikipedia, used as a control set). The benchmark contains 1,140 question-answer pairs, each paired with a gold passage and a human verification decision made during construction.

Almost all existing RAG faithfulness and hallucination benchmarks are English-only. AM-RAGBench targets a bilingual pair chosen for a real, non-arbitrary reason: Arabic and Malay share substantial content overlap through the Quran's translation tradition, while remaining typologically distant (Semitic vs. Austronesian, different scripts), which supports faithfulness research beyond high-resource Western languages.

## Supported Tasks

- **Question Answering / RAG faithfulness evaluation:** given a question and a retrieved passage, generate an answer and evaluate whether it is supported by the passage. This dataset provides the gold questions, answers, and passages needed for that evaluation; it does not itself include model outputs or faithfulness labels.

## Languages

- Arabic (`ar`)
- Malay (`ms`)

## Dataset Structure

### Data Instances

```json
{
  "id": "quran-1:2-ar-q0",
  "language": "ar",
  "domain": "quran",
  "question": "إلى من يُوجه الثناء في هذا النص؟",
  "gold_passage_id": "quran-1:2-ar-q0",
  "gold_passage_text": "الحمد لله رب العالمين",
  "gold_answer": "الله",
  "source_citation": "Quran 1:2 (Arabic, Tanzil.net Uthmani/Simple text, CC BY 3.0)",
  "annotator_id": "A1",
  "verification_pass": "primary"
}
```

### Data Fields

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique record identifier. |
| `language` | string | `ar` (Arabic) or `ms` (Malay). |
| `domain` | string | `quran` (specialized domain) or `wiki` (general domain). |
| `question` | string | The question, in the record's language. |
| `gold_passage_id` | string | Identifier of the source passage the question was drafted from. |
| `gold_passage_text` | string | The gold passage text the answer should be supported by. |
| `gold_answer` | string | The verified gold answer. |
| `source_citation` | string | Exact source and translation/edition for the passage, so claims can be traced to a specific, checkable source. |
| `annotator_id` | string | Anonymized identifier of the human annotator who verified the record. |
| `verification_pass` | string | Verification tier; released records carry `primary`. |

### Data Splits

| Domain | Arabic | Malay | Total |
|---|---|---|---|
| Quran (specialized) | 173 | 196 | 369 |
| Wikipedia (general) | 384 | 387 | 771 |
| **Total** | **557** | **583** | **1,140** |

Provided as two files: `quran_verified.jsonl` (specialized domain) and `wiki_verified.jsonl` (general domain). No train/dev/test split is defined; this is an evaluation-only benchmark.

## Dataset Creation

### Source Data

- **Quran text:** Arabic source text and the Basmeih Malay translation, both from [Tanzil.net](https://tanzil.net), licensed CC BY 3.0.
- **Wikipedia text:** Arabic and Malay Wikipedia, licensed CC BY-SA 4.0.

### Annotations

Candidate questions were drafted independently per language and per passage (not machine-translated from one language into the other, to avoid translation artifacts). Every candidate record was then manually reviewed by a fluent speaker of both Arabic and Malay and either approved as-is, edited, or rejected. Only records that passed this verification (`verification_pass=primary`) are released. A subsequent, independent second-annotator validation (one additional annotator per language) re-reviewed a stratified sample of these decisions to check inter-annotator reliability; see the associated evaluation code repository for that analysis.

## Considerations for Using the Data

- Verification confirmed factual and semantic soundness of each question-answer-passage triple. For the Quran sub-corpus, this is not a claim of interpretive or theological authority, and general Arabic fluency is a different competency from expertise in Quranic exegesis specifically.
- The benchmark is drawn from two domains only (a specialized religious-text domain and general encyclopedic text); results should not be assumed to generalize to arbitrary domains without further validation.
- Construction verification was originally performed by a single primary annotator per the process above, later partially cross-checked by independent second annotators on a sample.

## Licensing Information

The questions, answers, and verification labels in this dataset are released under **CC BY 4.0**. The underlying passage text carries its own upstream license, which still applies to that text specifically:

- Quran passage text: Tanzil.net, **CC BY 3.0** (attribution required; see https://tanzil.net/docs/licensing).
- Wikipedia passage text: **CC BY-SA 4.0** (attribution and share-alike required for redistribution of the `gold_passage_text` field specifically).

## Citation

If you use this dataset, please cite the repository.
