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
---

# AM-RAGBench

Human-verified Arabic-Malay benchmark for evaluating retrieval-augmented generation (RAG) faithfulness. 1,140 question-answer pairs spanning a specialized domain (Quran, Arabic and Basmeih Malay translation) and a general domain (Arabic and Malay Wikipedia), each with a gold passage, a gold answer, and a verification decision made during construction.

## Files

- `quran_verified.jsonl`: specialized-domain records.
- `wiki_verified.jsonl`: general-domain records.

## Fields

Each record includes: `id`, `domain`, `language`, `question`, `gold_answer`, `gold_passage_text`, `source_citation`, `verification_pass`.

## License

The questions, answers, and verification labels in this dataset are released under CC BY 4.0. The underlying passage text carries its own upstream license: Quran text is from Tanzil.net (CC BY 3.0), and Wikipedia passages are CC BY-SA 4.0. Redistribution of the Wikipedia-derived `gold_passage_text` fields must comply with CC BY-SA 4.0's share-alike requirement in addition to attribution.
