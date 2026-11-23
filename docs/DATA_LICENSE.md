# Data License

Licences differ by subset. The per-licence files are in `data/verified/licence_split/`
and in the Hugging Face dataset (https://huggingface.co/datasets/Akram98/AM-RAGBench).

| File | Licence we grant over our own questions, answers and labels | Upstream passage text (not ours to relicense) |
|---|---|---|
| `quran_ar_CC-BY-4.0.jsonl` | CC BY 4.0 | Tanzil.net Arabic text, CC BY 3.0: attribution required, verbatim copies only (https://tanzil.net/docs/Text_License). |
| `quran_ms_CC-BY-NC-4.0.jsonl` | CC BY-NC 4.0 (non-commercial research use only) | Basmeih Malay translation, distributed by Tanzil.net for non-commercial use only. **We grant no licence to the translation itself.** Permission from the translator or publisher is needed for any other use. |
| `wiki_CC-BY-SA-4.0.jsonl` | CC BY-SA 4.0 | Arabic and Malay Wikipedia text, CC BY-SA 4.0 (and GFDL). Share-alike applies to redistributions that include the passage text. |
| `unadjudicated_disagreements.csv`, `llm_suggested_edits.csv`, `answerability_flags.csv` | CC BY 4.0 for the annotation metadata only | The quoted passage excerpts and questions keep the licence of the file they come from. |

The combined files `data/verified/quran_verified.jsonl` and `wiki_verified.jsonl` mix licences
and are provided for convenience only. Use the per-licence files whenever the licence matters.

Annotation provenance: every record was reviewed by one human annotator (the author). The
300-record second pass and the adjudication proposals (`llm_suggested_edits.csv`) were produced with
AI assistance and are not independent human verification. See the paper (Sections 4.5 and 4.9).
