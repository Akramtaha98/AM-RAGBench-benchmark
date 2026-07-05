# Data License

The original contributions in this repository's `data/verified/` benchmark
(questions, answers, verification/annotation metadata, and the reannotation
validation records under `review/`) are released under **CC BY 4.0**
(https://creativecommons.org/licenses/by/4.0/). You may use, share, and adapt
this material for any purpose, including commercially, provided you give
appropriate credit to this repository.

This benchmark is built on top of two upstream text sources, each of which
carries its own license that still applies to the passage text itself even
though the questions, answers, and verification labels are original work:

- **Quran text** (Arabic Uthmani/Simple text and the Basmeih Malay
  translation): sourced from Tanzil.net under **CC BY 3.0**. Any
  redistribution of the underlying verse text must retain Tanzil's
  attribution terms; see https://tanzil.net/docs/licensing.
- **Wikipedia passages** (Arabic and Malay Wikipedia): Wikipedia article text
  is licensed **CC BY-SA 4.0** (and dual-licensed under the GFDL). The
  share-alike clause is a stronger copyleft than this repository's own
  CC BY 4.0 choice, so redistributions that include the Wikipedia-derived
  `gold_passage_text` fields specifically must comply with CC BY-SA 4.0's
  share-alike requirement, not just attribution.

In short: cite this benchmark for the questions/answers/labels, and separately
attribute Tanzil.net / Wikipedia for the underlying passage text they
contributed, per their respective licenses.
