# Dataset construction plan

This is the actionable, step-by-step version of Phase A from `paper_outline.md`.
Follow it in order — steps 3 and 4 are the ones that determine whether the
resource-paper claim holds up under review, so don't compress them.

## Step 1: Source selection and licensing (checked)

- **Specialized domain: Quran only.** Arabic source text + the Basmeih Malay
  translation, both hosted on Tanzil.net under CC BY 3.0 — copying/reuse is
  permitted (including in a published benchmark) provided the text is not
  altered and Tanzil.net is credited with a link back. This is settled; no
  further licensing check needed for this source.
  - **Hadith/fiqh was dropped from scope.** Checked two candidate open Hadith
    sources: `fawazahmed0/hadith-api` (Unlicense/public domain, otherwise the
    cleanest option available) and HadeethEnc.com (20+ languages). Neither
    offers a Malay translation — both cover Indonesian instead. Malay Hadith
    translations exist in the real world (e.g. via Malaysian bodies like
    JAKIM) but not as a clean, pre-packaged, openly-licensed dataset, which
    would reopen the scraping/licensing problem this step exists to avoid.
    Decision: keep the Arabic-Malay framing intact, drop Hadith, run the
    specialized domain on Quran alone.
- **General domain (control set):** Arabic Wikipedia + Malay (Bahasa Melayu)
  Wikipedia dumps, both CC BY-SA — straightforward to reuse with attribution.
  Use official Wikimedia dumps, not scraping, to avoid rate-limit/ToS issues.
- **Output of this step:** a source manifest (spreadsheet or JSON) listing every
  document/translation edition, its license, and its access date. For this
  project that manifest is now just two rows (Tanzil Quran + Wikipedia dumps),
  which simplifies Step 2 onward.

## Step 2: Passage extraction

- Chunk source documents into passage-sized units (roughly paragraph-length,
  100-250 words). For the Quran/Hadith corpus, natural units already exist
  (verse/hadith boundaries) — use those instead of arbitrary chunking.
- Deduplicate near-identical passages (common in Hadith collections that
  repeat the same narration across multiple chains).
- **Output:** a passage store per language/domain, each passage with a stable ID.

## Step 3: Question generation

- Generate candidate questions **independently per language** from the
  passage store — do not machine-translate an English question set into
  Arabic/Malay, since translation artifacts (unnatural phrasing, translated
  idioms) would leak into faithfulness scoring and make it unclear whether a
  faithfulness gap is due to the model or due to the question itself.
- Use an LLM to draft 2-3 candidate questions per passage, then filter down
  in Step 4. Keep every draft (even the ones you filter out) — reviewers may
  ask about the filtering rate as an implicit data-quality signal.

## Step 4: Verification protocol — the annotator-asymmetry problem

You've confirmed native-speaker annotators for only one of the two languages.
This is the single biggest risk to the paper's credibility if handled loosely,
so pick one of these paths deliberately and disclose it in the paper's
Limitations section regardless of which you choose:

**Option A — paid professional annotators for the missing language.**
Recruit via a platform like ProZ (for translators/linguists) or Upwork,
screening for native fluency + subject familiarity (Islamic studies
background matters for the specialized domain). Budget for at minimum a
second-pass verification, not just a translation pass — a translator
answering "is this a fair translation" is not the same task as a native
speaker answering "is this question well-formed and is this the correct
answer." This is the strongest option for the paper's credibility.

**Option B — university partnership.** Reach out to a Malay/Indonesian or
Arabic linguistics department for a collaboration (possibly co-authorship in
exchange for annotation time, which is a well-worn and legitimate arrangement
in resource papers). Slower to set up, cheaper, and gives you a
built-in mention of institutional review if human-subjects considerations
come up (annotator compensation, workload).

**Fallback — MT + spot-check.** Machine-translate from the language with
native coverage, then have someone with partial fluency (or the single
annotator pool you have, working with translation aids) spot-check a sample.
Use this only if A and B are truly not feasible on your timeline, and report
the spot-check sample size and disagreement rate explicitly. Don't present
MT-plus-spot-check data with the same confidence as natively-verified data —
tag every record's `verification_pass` field (see `schema.py`) accordingly, so
you can report per-language data quality honestly and, if a reviewer pushes
back, restrict claims to the natively-verified subset if needed.

**Whichever path you pick:** report inter-annotator agreement (Cohen's kappa
or similar) wherever you have 2+ independent judgments, even if that's only
for one language initially.

## Step 5: Gold passage / answer finalization + schema validation

- For every surviving question, confirm exactly which passage(s) support the
  answer, and write the answer in the annotator's own words (not copy-pasted
  from the passage) where possible, to avoid trivial lexical-overlap shortcuts
  during evaluation.
- Run every record through `code/schema.py`'s `validate()` before it's
  admitted to the benchmark. Track the fraction of drafted questions that
  survive to this point — that rejection rate is itself worth reporting.

## Step 6: Faithfulness/hallucination label schema pilot

- Before scaling annotation, pilot the SUPPORTED / PARTIALLY_SUPPORTED /
  UNSUPPORTED / CONTRADICTED / ABSTAINED schema (see `code/faithfulness.py`)
  on ~30-50 example model outputs with whichever annotator pool is available,
  and check that labels are being applied consistently. Adjust label
  definitions if annotators disagree systematically on a category, before
  spending annotation budget on the full set.

## Step 7: Benchmark statistics table

Once built, report (per language, per domain): question count, average
passage length, average answer length, the Step 4 verification-pass
breakdown, and the Step 5 survival/rejection rate. This table is what a
resource-paper reviewer checks first.

## Rough timeline (order-of-magnitude, adjust to your actual annotator/budget situation)

1. Source selection + licensing check: 1-2 weeks
2. Passage extraction + question generation: 1-2 weeks (mostly automated, some manual review)
3. Annotator recruitment (Option A/B) in parallel with the above: 2-4 weeks lead time
4. Verification pass across both languages: 3-6 weeks, depending on annotator availability
5. Schema validation + statistics: a few days once verification is done

Total: roughly 2-3 months to a usable v1 benchmark, dominated by annotator
recruitment and verification, not by engineering.
