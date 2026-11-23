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
configs:
- config_name: quran_ar
  data_files: quran_ar_CC-BY-4.0.jsonl
- config_name: quran_ms
  data_files: quran_ms_CC-BY-NC-4.0.jsonl
- config_name: wiki
  data_files: wiki_CC-BY-SA-4.0.jsonl
---

# Dataset Card for AM-RAGBench

**This dataset does not have a single licence.** Each of the three data files has its own terms (see [Licensing Information](#licensing-information)). Check the file before reuse, in particular before any commercial use.

## Dataset Summary

AM-RAGBench is a human-reviewed Arabic–Malay question-answering benchmark for evaluating retrieval-augmented generation (RAG) faithfulness. It has two domains: a specialized domain (the Quran, with the Arabic source text and the Basmeih Malay translation) and a general domain (Arabic and Malay Wikipedia, used as a control set). It contains 1,140 question–answer pairs, each with a gold passage and the human decision recorded during construction. Model outputs and faithfulness labels are not part of this dataset; they are in the code repository.

- Paper (title): *AM-RAGBench: An Arabic–Malay Benchmark for Evaluating Retrieval-Augmented Generation Faithfulness* (submitted to *Computation*, MDPI; the citation will be updated when published).
- Code and evaluation: https://github.com/Akramtaha98/AM-RAGBench-benchmark

## Supported Tasks

Question answering and RAG faithfulness evaluation: given a question and a retrieved passage, generate an answer and judge whether it is supported by the passage and whether it is correct against the gold answer. The dataset supplies the gold questions, answers, and passages. It defines no train/dev/test split and is meant for evaluation only.

## Languages

Arabic (`ar`) and Malay (`ms`).

## Dataset Structure

### Files

| File (config) | Records | Domain, language |
|---|---|---|
| `quran_ar_CC-BY-4.0.jsonl` (`quran_ar`) | 173 | Quran, Arabic |
| `quran_ms_CC-BY-NC-4.0.jsonl` (`quran_ms`) | 196 | Quran, Malay |
| `wiki_CC-BY-SA-4.0.jsonl` (`wiki`) | 771 | Wikipedia, 384 Arabic and 387 Malay |
| `unadjudicated_disagreements.csv` | 44 rows | Disagreements between the original decision and the AI-assisted second pass (see below) |
| `llm_suggested_edits.csv` | 44 rows | Candidate corrections proposed by an AI-assisted pass. **Unverified, not human adjudication**; the released records are unchanged |
| `answerability_flags.csv` | 36 records | Records whose gold answer shares no word with the gold passage (candidates for manual review) |

### Data Instance

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
| `language` | string | `ar` or `ms`. |
| `domain` | string | `quran` or `wikipedia_general`. |
| `question` | string | The question, in the record's language. |
| `gold_passage_id` | string | Identifier of the source passage the question was drafted from. |
| `gold_passage_text` | string | The gold passage the answer should be supported by. |
| `gold_answer` | string | The reviewed gold answer. |
| `source_citation` | string | Source and edition of the passage (for Wikipedia, the article identifier). |
| `annotator_id` | string | Anonymized identifier of the reviewing annotator. |
| `verification_pass` | string | Review tier; released records carry `primary`. |

## Dataset Creation

### Source Data

- **Quran, Arabic:** text from [Tanzil.net](https://tanzil.net), distributed under CC BY 3.0 (verbatim copies, attribution required; https://tanzil.net/docs/Text_License).
- **Quran, Malay:** the Basmeih translation as distributed by Tanzil.net. Tanzil states that translations are for non-commercial use and that permission from the translator or publisher is needed for other uses (https://tanzil.net/trans/). We have not obtained such permission, and we have no right to relicense the translation itself.
- **Wikipedia:** Arabic and Malay Wikipedia articles, CC BY-SA 4.0. Records carry the article identifier. **Revision identifiers were not stored**, so exact re-extraction of the passages is not guaranteed.

### Annotations

Candidate questions were drafted with a language model (Qwen3.5, 4B parameters), independently per language and per passage and not translated between languages. A fluent speaker of both languages (the author) reviewed every candidate and approved, edited, or rejected it. Only records that passed (`verification_pass=primary`) are released.

### Validation

- **AI-assisted second pass (not independent human verification).** A second pass over a stratified sample of 150 records per language (300 in total) was produced with the help of an AI model, on sheets that did not show the original decision. No second human reviewed these records. Observed agreement of this pass with the original human decision on the released text is 86.7% (Arabic) and 84.0% (Malay); Cohen's kappa is low (0.36 and 0.20) because almost all records are approved, and Gwet's AC1 is 0.85 and 0.82.
- **Unresolved disagreements.** `unadjudicated_disagreements.csv` lists the 44 sampled records on which the AI-assisted second pass still disagrees with the original decision (20 Arabic, 24 Malay). These have **not been adjudicated by a human**. An AI-assisted pass proposed edits for 41 records and removal of 3 (`llm_suggested_edits.csv`); these proposals are unverified and the released records are unchanged. 39 are in the released data (`in_released_benchmark = yes`); 5 were rejected in primary review and are not released. Users who need the most conservative subset can exclude the 39 released records or re-evaluate them.
- **Answerability flags.** `answerability_flags.csv` lists 36 records whose gold answer shares no word with the gold passage. This lexical screen is a list of candidates for review, not a verified error rate. Excluding these records and the 39 unresolved ones does not change the main results of the paper.

## Considerations for Using the Data

- **Not a religious authority.** Review covered factual and semantic soundness of each question–answer–passage triple. It is not a claim of tafsīr or other theological authority, and the Quran subset has not been reviewed by a specialist in Quranic exegesis. Do not use model outputs evaluated with this benchmark as religious guidance.
- **Unmatched question sets.** Arabic and Malay questions were drafted independently, so the two languages are compared on comparable but not translation-matched questions. Language is confounded with question difficulty and passage choice.
- **Single primary reviewer.** All records were first reviewed by one person; only a 300-record sample received a second pass, and that pass was AI-assisted, not human.
- **Text artifacts.** An automatic screen found empty brackets left from stripped markup in 14 Arabic and 69 Malay Wikipedia records; these were not normalized.
- **Scope.** The benchmark covers two domains and two languages and should not be assumed to generalize.

## Licensing Information

Check the licence of the file you use.

| File | Questions, answers, and other annotations | Passage text embedded in the records |
|---|---|---|
| `quran_ar_CC-BY-4.0.jsonl` | CC BY 4.0 | Tanzil.net Arabic text, CC BY 3.0: attribution required, verbatim copies only. |
| `quran_ms_CC-BY-NC-4.0.jsonl` | CC BY-NC 4.0 (non-commercial research use only) | Basmeih Malay translation, distributed by Tanzil.net for non-commercial use only. **We grant no licence to the translation itself**; its terms are set by the translator, the publisher, and Tanzil. Gold answers in this file quote the translation and are covered by the same restriction. |
| `wiki_CC-BY-SA-4.0.jsonl` | CC BY-SA 4.0, including questions and answers derived from the passages | Wikipedia text, CC BY-SA 4.0. Attribute the Wikipedia articles named in `source_citation`; revision identifiers are not available. |
| `unadjudicated_disagreements.csv`, `llm_suggested_edits.csv`, `answerability_flags.csv` | CC BY 4.0 (annotation metadata only; the quoted passage excerpts and questions keep the licence of the file they come from) | — |

The earlier combined files `quran_verified.jsonl` and `wiki_verified.jsonl` mixed these terms under one licence statement and have been removed. If you downloaded them, use the per-licence files instead.

## Citation

```bibtex
@misc{zeyad2026amragbench,
  title  = {{AM-RAGBench}: An Arabic--Malay Benchmark for Evaluating Retrieval-Augmented Generation Faithfulness},
  author = {Zeyad, Akram Taha},
  year   = {2026},
  note   = {Submitted to Computation (MDPI). Dataset: https://huggingface.co/datasets/Akram98/AM-RAGBench}
}
```

Please also cite Tanzil.net and the Wikipedia articles where your use requires attribution.

## Contact

Akram Taha Zeyad, University of Technology, Baghdad, Iraq. Issues and corrections: https://github.com/Akramtaha98/AM-RAGBench-benchmark/issues
