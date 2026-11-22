---
license: other
license_name: mixed-per-subset
license_link: https://huggingface.co/datasets/Akram98/AM-RAGBench#licensing-information
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

- **Quran text:** Arabic source text from [Tanzil.net](https://tanzil.net), distributed under Creative Commons Attribution 3.0 terms (verbatim copies with attribution). The Basmeih Malay translation, also from Tanzil.net, is subject to Tanzil's translation terms: non-commercial use only, and permission from the translator or publisher is required for other uses. We have not obtained such permission, so the Malay Quran passages (and records derived from them) should be used for non-commercial research only.
- **Wikipedia text:** Arabic and Malay Wikipedia, licensed CC BY-SA 4.0.

### Annotations

Candidate questions were drafted independently per language and per passage (not machine-translated from one language into the other, to avoid translation artifacts). Every candidate record was then manually reviewed by a fluent speaker of both Arabic and Malay and either approved as-is, edited, or rejected. Only records that passed this verification (`verification_pass=primary`) are released. A subsequent, independent second-annotator validation (one additional annotator per language) re-reviewed a stratified sample of these decisions to check inter-annotator reliability; see the associated evaluation code repository for that analysis.

## Considerations for Using the Data

- Verification confirmed factual and semantic soundness of each question-answer-passage triple. For the Quran sub-corpus, this is not a claim of interpretive or theological authority, and general Arabic fluency is a different competency from expertise in Quranic exegesis specifically.
- The benchmark is drawn from two domains only (a specialized religious-text domain and general encyclopedic text); results should not be assumed to generalize to arbitrary domains without further validation.
- Construction verification was originally performed by a single primary annotator per the process above, later partially cross-checked by independent second annotators on a sample.

## Licensing Information

Licences differ by subset. One file per licence is provided in `data/verified/licence_split/`; the combined files `quran_verified.jsonl` and `wiki_verified.jsonl` mix licences and are provided for convenience, so use the per-licence files when the licence matters.

| File | Records | Licence |
|---|---|---|
| `quran_ar_CC-BY-4.0.jsonl` | 173 | CC BY 4.0 for the questions, answers and labels. The Arabic passage text is Tanzil.net text under CC BY 3.0 (attribution required, verbatim copies; https://tanzil.net/docs/Text_License). |
| `quran_ms_CC-BY-NC-4.0.jsonl` | 196 | CC BY-NC 4.0 (non-commercial research only). The Malay passage text is the Basmeih translation distributed by Tanzil.net for non-commercial use only; permission from the translator or publisher has not been obtained for other uses (https://tanzil.net/trans/). |
| `wiki_CC-BY-SA-4.0.jsonl` | 771 | CC BY-SA 4.0, including the questions and answers derived from the Wikipedia passages. Records carry the Wikipedia article identifier; revision identifiers were not stored. |

## Citation

If you use this dataset, please cite the repository.

## Answerability flags

`answerability_flags.csv` lists 36 released records (14 Quran, 22 Wikipedia) in which the gold answer shares no word with the gold passage. This is a lexical heuristic, so the list contains candidates for manual answerability review, not confirmed errors, and it can miss unanswerable records whose answers share words with the passage. Example: `quran-2:18-ar-q0` has the passage "صم بكم عمي فهم لا يرجعون" and the answer "المنافقين", which the verse alone does not state. These records have not been corrected or removed pending human review.
