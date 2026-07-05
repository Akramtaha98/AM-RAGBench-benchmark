"""
Statistical analysis of results.jsonl: language x model interaction on
faithfulness rate. Chi-square test of independence as the baseline test;
this now also includes a cluster-robust GEE logistic regression (passage-level
clustering) as a more conservative supplement, since multiple questions in
this benchmark share the same source passage and are therefore not fully
independent observations in the way plain chi-square assumes.

Usage:
    python analysis.py results.jsonl
    python analysis.py quran_results.jsonl wiki_results.jsonl --gee   (clustered regression, both files pooled with a domain column)
"""

import sys
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

try:
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.cov_struct import Exchangeable
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


def load_results(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def faithful_rate_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_faithful"] = df["faithfulness_label"].isin(["SUPPORTED", "PARTIALLY_SUPPORTED"])
    table = df.groupby(["language", "model_name"])["is_faithful"].mean().unstack()
    return table


def chi_square_language_effect(df: pd.DataFrame):
    """
    Tests whether faithfulness label distribution is independent of language.
    H0: faithfulness label is independent of language.
    A significant result (p < .05) means the language x faithfulness
    association is unlikely to be chance -- report chi2, dof, p, and the
    contingency table itself (not just the p-value).
    """
    contingency = pd.crosstab(df["language"], df["faithfulness_label"])
    chi2, p, dof, expected = chi2_contingency(contingency)
    return {
        "contingency_table": contingency,
        "chi2": chi2,
        "dof": dof,
        "p_value": p,
    }


def retrieval_vs_generation_failure_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-tab of retrieval_hit x faithfulness_label. This is the RQ3 diagnostic:
    if UNSUPPORTED/CONTRADICTED cluster where retrieval_hit is False, the
    dominant failure mode is retrieval. If they cluster where retrieval_hit is
    True, the dominant failure mode is generation (the model hallucinated
    despite having the right passage).
    """
    return pd.crosstab(df["retrieval_hit"], df["faithfulness_label"])


def passage_id_from_record_id(record_id: str) -> str:
    """
    Every record id in this benchmark is '<passage-identifying-prefix>-q<N>'
    (e.g. 'quran-2:20-ar-q1' -> passage 'quran-2:20-ar', or
    'ar-wiki-1181-0-q1' -> passage 'ar-wiki-1181-0'). Multiple questions per
    passage means records are clustered, not fully independent -- this
    extracts the cluster key for that.
    """
    return record_id.rsplit("-q", 1)[0]


def cluster_robust_gee(df: pd.DataFrame, outcome_label: str) -> dict:
    """
    Cluster-robust (GEE, exchangeable working correlation, cluster = source
    passage) logistic regression of a binary faithfulness outcome
    (e.g. outcome_label='SUPPORTED' or 'UNSUPPORTED') on retrieval hit and
    language. This is the more conservative supplement to the plain
    chi-square test above: it accounts for multiple questions sharing a
    passage rather than treating every row as an independent draw.

    Requires statsmodels (pip install statsmodels). Excludes
    JUDGE_PARSE_FAILED rows (no real label to model).

    Returns a dict of {"hit": {...}, "is_ms": {...}} each with coef, odds
    ratio, cluster-robust SE, and p-value.
    """
    if not _HAS_STATSMODELS:
        raise ImportError("cluster_robust_gee requires statsmodels: pip install statsmodels")

    sub = df[df["faithfulness_label"] != "JUDGE_PARSE_FAILED"].copy()
    sub["passage_id"] = sub["id"].apply(passage_id_from_record_id)
    sub["hit"] = sub["retrieval_hit"].astype(int)
    sub["is_ms"] = (sub["language"] == "ms").astype(int)
    sub["outcome"] = (sub["faithfulness_label"] == outcome_label).astype(int)

    model = GEE.from_formula("outcome ~ hit + is_ms", groups="passage_id", data=sub,
                              family=Binomial(), cov_struct=Exchangeable())
    res = model.fit()

    out = {}
    for param in ["hit", "is_ms"]:
        out[param] = {
            "coef": res.params[param],
            "odds_ratio": float(np.exp(res.params[param])),
            "robust_se": res.bse[param],
            "p_value": res.pvalues[param],
        }
    out["n_records"] = len(sub)
    out["n_passages"] = sub["passage_id"].nunique()
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_gee = "--gee" in sys.argv
    if len(args) < 1:
        sys.exit("usage: python analysis.py results.jsonl [more_results.jsonl ...] [--gee]")

    domain_names = ["Quran", "Wiki", "Domain3", "Domain4"]  # extend if you have more domain files
    frames = []
    for i, path in enumerate(args):
        d = load_results(path)
        d["domain"] = domain_names[i] if i < len(domain_names) else path
        frames.append(d)
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if "domain" not in df.columns:
        df["domain"] = domain_names[0]

    print("=== Faithful rate by language x model ===")
    print(faithful_rate_table(df))

    print("\n=== Chi-square: language vs faithfulness label (pooled, treats every record as independent) ===")
    result = chi_square_language_effect(df)
    print(result["contingency_table"])
    print(f"chi2={result['chi2']:.3f}, dof={result['dof']}, p={result['p_value']:.4f}")
    if df.shape[0] < 30:
        print("NOTE: sample size is tiny -- this p-value is not meaningful, "
              "it's here to show the test runs correctly.")

    print("\n=== Retrieval hit vs faithfulness label (RQ3 diagnostic) ===")
    print(retrieval_vs_generation_failure_breakdown(df))

    if do_gee:
        if not _HAS_STATSMODELS:
            print("\n--gee requested but statsmodels is not installed: pip install statsmodels")
            return
        print("\n=== Cluster-robust GEE (passage-level clustering), by domain ===")
        for domain in df["domain"].unique():
            sub = df[df["domain"] == domain]
            print(f"\n--- {domain} ---")
            for outcome in ["SUPPORTED", "UNSUPPORTED"]:
                res = cluster_robust_gee(sub, outcome)
                print(f"  outcome={outcome}  (n={res['n_records']}, passages={res['n_passages']})")
                for param in ["hit", "is_ms"]:
                    p = res[param]
                    print(f"    {param:8s} OR={p['odds_ratio']:6.2f}  "
                          f"robust SE={p['robust_se']:.3f}  p={p['p_value']:.4g}")


if __name__ == "__main__":
    main()
