# Paper Configuration Record

- **Paper type:** Resource paper + empirical evaluation study
- **Discipline:** Computational Linguistics / NLP
- **Target venue:** TACL (full paper, ~20-25 pages)
- **Output format:** LaTeX, ACL template
- **Citation format:** ACL style (author-year)
- **Language:** English
- **Existing materials:** None — literature search and data collection both start from scratch
- **Known constraint:** Native-speaker annotators confirmed for only one of the two languages. Mitigation plan is built into the dataset construction phase below (see Phase A, step 3).

---

## Research Questions

**RQ1 (resource):** Can a human-verified, domain-matched Arabic-Malay parallel QA benchmark be built with sufficient coverage and annotation quality to support retrieval and faithfulness evaluation, despite asymmetric annotator access across the two languages?

**RQ2 (empirical):** How do current retrieval + generation (RAG) pipelines differ in retrieval accuracy and answer faithfulness across Arabic, Malay, and English when evaluated on this benchmark?

**RQ3 (diagnostic):** When RAG systems fail, is the dominant failure mode retrieval miss (relevant passage never surfaced) or generation hallucination (model asserts claims unsupported by a correctly retrieved passage) — and does this failure profile differ by language or by model family (general multilingual vs. region-tuned)?

---

## Why Arabic-Malay (anticipated reviewer objection, addressed up front)

Reviewers will ask why these two languages are paired. The justification to state explicitly in the Introduction:

- Both are large-population, comparatively under-benchmarked languages in RAG/hallucination research relative to English/Chinese.
- They share a natural, high-overlap content domain — the Quran — where Arabic is the source language and Malay has a substantial, well-established translation (the Basmeih translation, hosted openly on Tanzil.net under CC BY 3.0). This gives a real-world, non-arbitrary reason to pair them, rather than pairing two unrelated low-resource languages for convenience.
- **Scope note (decided after a licensing check):** Hadith/fiqh literature was originally considered as a second specialized sub-domain, but no openly-licensed, pre-packaged Hadith source currently offers a Malay translation (checked: fawazahmed0/hadith-api and HadeethEnc.com both cover Indonesian but not Malay). Rather than substitute Indonesian for Malay (which would weaken the paper's stated language pairing) or scrape Malay Hadith translations from a Malaysian religious authority site (reopening the licensing problem this benchmark construction was designed to avoid), the specialized domain is restricted to Quran only. This keeps the Arabic-Malay framing intact and the licensing story clean.
- They are typologically distant (Arabic: Semitic, root-and-pattern morphology, RTL script; Malay: Austronesian, agglutinative, Latin script), which makes the benchmark useful for testing whether RAG faithfulness gaps are morphology/script-driven or more generally "low-resource-driven."

---

## Phase A — Benchmark Construction

1. **Domain and source selection.** Two sub-corpora:
   - *Specialized domain:* Quran only (Arabic source text + the Basmeih Malay translation, both via Tanzil.net, CC BY 3.0). Hadith/fiqh was dropped from scope after the licensing check found no openly-licensed Malay Hadith source (see scope note above).
   - *General domain:* Wikipedia (Arabic + Malay editions) as a control set, so results aren't confounded by domain-specific effects alone.

2. **Question generation.** LLM-drafted candidate questions per source passage, generated independently per language (not machine-translated question pairs — translation artifacts would contaminate the faithfulness signal). Each question anchored to one or more gold passages at generation time.

3. **Verification protocol (handles the single-language-annotator constraint).** This is the methodological detail that needs to be nailed down before data collection starts:
   - Full native-speaker verification (question clarity, answer correctness, gold passage sufficiency) for the language where annotators are already available.
   - For the second language, use one of: (a) paid professional translators/linguists sourced per-task (e.g., ProZ, Upwork) for a genuine second-annotator pass, not just verification of a translation, or (b) a partnership with a university Malay/Arabic linguistics department for annotation-as-collaboration. Machine-translation-plus-spot-check is the fallback but should be flagged as a limitation, not presented as equivalent to (a)/(b) — reviewers will discount data quality claims otherwise.
   - Inter-annotator agreement reported for whichever language has 2+ annotators; single-annotator language reports agreement against a secondary automatic consistency check (e.g., re-answerability check with a strong LLM) as a partial substitute, disclosed explicitly as weaker evidence.

4. **Faithfulness/hallucination label schema.** Adapt an existing taxonomy (e.g., supported / partially supported / unsupported / contradicted, following the FActScore / RAGTruth style of categorization) to both languages, with the schema itself validated by whichever annotator pool is available before scaling.

5. **Benchmark statistics.** Target: report per-language, per-domain counts of questions, average passage length, answer length, and a quality audit sample size and pass rate.

---

## Phase B — Evaluation

- **Retrievers to compare:** BM25 baseline, a general multilingual dense retriever (e.g., BGE-M3 or multilingual E5), and if available, any Arabic-specific and Malay/Indonesian-specific retrieval models, to see whether language-specific retrieval closes gaps that generation cannot.
- **Generators to compare:** a small matrix of general multilingual LLMs vs. region-tuned models (e.g., an Arabic-tuned model such as Jais alongside a general model, and a Southeast-Asian-tuned model such as SEA-LION/Sailor alongside the same general model). This lets the paper make a claim beyond "here are some numbers" — namely whether regional fine-tuning measurably improves faithfulness, which is a more citable finding than a flat leaderboard.
- **Metrics:** retrieval recall@k / nDCG, answer correctness (human-adjudicated on a sample + automatic proxy), faithfulness rate by category, hallucination rate.
- **Oracle-retrieval ablation:** run generation with gold passage forced, to separate retrieval failure from generation failure — this directly answers RQ3.

---

## Phase C — Analysis

- Error taxonomy compared across Arabic vs. Malay vs. English.
- Statistical test for language x model interaction on faithfulness rate (not just descriptive comparison).
- Qualitative error examples per failure category, per language.

---

## Paper Structure (TACL / ACL-style full paper)

1. **Abstract**
2. **Introduction** — resource gap, Arabic-Malay framing (see justification block above), contribution list (dataset + evaluation + failure-mode analysis)
3. **Related Work** — multilingual QA/RAG benchmarks (candidates to verify during literature search: MIRACL, TyDiQA, XOR-QA, MLQA), RAG faithfulness/hallucination evaluation frameworks (candidates: FActScore, RAGTruth), Arabic NLP resources, Malay/Indonesian NLP resources
4. **Benchmark Construction** — Phase A in full, with statistics table and annotation protocol detail (this section carries the resource-paper weight; needs to be the most rigorously documented section)
5. **Evaluation Setup** — Phase B: retrievers, generators, metrics, oracle-retrieval ablation design
6. **Results** — retrieval table, faithfulness-by-language-by-model matrix, statistical tests
7. **Analysis** — error taxonomy, retrieval-vs-generation failure attribution, region-tuned vs. general model comparison
8. **Discussion** — what the language/model interaction implies, whether the gap is script/morphology-driven or resource-driven
9. **Limitations** — explicitly state the single-language-annotator constraint and its mitigation, domain scope (2 domains, not fully general), model coverage
10. **Ethics statement** — handling of religious text, translation sensitivities, annotator compensation
11. **Conclusion**
12. **Data/code availability, CRediT authorship, funding, AI-usage disclosure** — mandatory sections for submission

---

## Immediate Next Steps

1. Literature search pass to verify and cite the candidate related-work papers listed above (none of these are confirmed/verified yet — this outline names them from memory as search targets, not as citations to use as-is).
2. Lock down the second-language annotation plan (professional translator budget vs. university partnership) before committing to a data collection timeline, since this affects both feasibility and how strong a data-quality claim the paper can make.
3. Decide the specific region-tuned model pair (Jais / SEA-LION or equivalents current as of the actual writing date) — verify these are still current, actively maintained models before citing them as baselines.
