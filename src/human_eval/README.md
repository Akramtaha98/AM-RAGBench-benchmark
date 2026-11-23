# Human calibration of the 72B judge

1. Annotation sheets are already generated: `results/human_eval/annotation_ar.xlsx` and `annotation_ms.xlsx`
   (160 items each, 320 total, blind to the judge label and to the generator). To regenerate: `python3 src/human_eval/make_sheets.py`.
2. Annotator 1 fills the green columns (dropdowns): Label 1 faithfulness, Label 2 correctness, NOTES. Read the GUIDELINES sheet first.
   The gold answer is in a hidden column (I): label faithfulness first, then expand it for correctness.
3. For inter-annotator agreement, a second person labels the 60 rows per language with OVERLAP=yes. Save their copy as
   `annotation_ar_2.xlsx` / `annotation_ms_2.xlsx` (copy the original sheet, fill only those rows).
4. Keep `key_private.csv` away from annotators (it contains the judge's labels).
5. Run `python3 src/human_eval/calibrate_judge.py`. It prints the 5x5 confusion matrix, per-label precision/recall/F1,
   population-weighted agreement and kappa with stratified-bootstrap CIs, three-way and correctness tables, human versus judge rates, and
   inter-annotator kappa. Send me `results/human_eval/calibration.json` and I will put the results into the paper.

Time: about 20-25 items per 30 minutes, so roughly 6-8 hours per language sheet. The Arabic and Malay sheets can be split across sittings.
Sampling: strata are domain x language x judge label (SUPPORTED 30, PARTIALLY 20, UNSUPPORTED 15, ABSTAINED 15 per cell; CONTRADICTED
does not occur in the baseline). To enlarge the sample, change `--per-cell`.
