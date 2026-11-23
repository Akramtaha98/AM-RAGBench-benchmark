"""Additional analyses for the second review round. Output: results/review2_analysis.json
(a) Gwet AC1 and PABAK for the second-annotator comparison; (b) faithful x correct 2x2 and hit x language interaction
(H200 matrix, gold-only MPNet baseline); (c) sensitivity to excluding unresolved/flagged records;
(d) calibration sample: raw agreement and strata. Usage: python3 -I src/review2_analysis.py [uploads_dir]"""
import csv, json, os, re, sys, collections, itertools
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
UP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "review", "reannotation")
rng = np.random.default_rng(20261125)
out = {}
def cl(i): return re.sub(r"-q\d+$", "", i)
# ---------- (a) AC1 / PABAK
def ac1_pabak(pairs):
    n = len(pairs); labs = sorted(set(a for a, b in pairs) | set(b for a, b in pairs)); K = len(labs)
    po = sum(a == b for a, b in pairs)/n
    pi = {l: (sum(a == l for a, b in pairs)+sum(b == l for a, b in pairs))/(2*n) for l in labs}
    pe = sum(p*(1-p) for p in pi.values())/(K-1) if K > 1 else 1.0
    return (po-pe)/(1-pe) if pe < 1 else float("nan"), (K*po-1)/(K-1) if K > 1 else float("nan")
def boot(pairs, B=2000):
    v = []
    for _ in range(B):
        s = [pairs[i] for i in rng.integers(0, len(pairs), len(pairs))]; v.append(ac1_pabak(s)[0])
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
M = {"valid": "approve", "edit": "edit", "invalid": "reject"}
Lc = lambda p: list(csv.DictReader(open(p, encoding="utf-8-sig")))
out["agreement"] = {}
for lang in ("arabic", "malay"):
    orig = json.load(open(os.path.join(ROOT, "review", "reannotation", f"{lang}_original_decisions.json")))
    v1 = {r["id"]: r for r in Lc(os.path.join(UP, f"amragbench_{lang}_reannotation_v1_completed.csv"))}
    fx = {r["id"]: r for r in Lc(os.path.join(UP, f"amragbench_{lang}_reannotation_fixup_v1_completed.csv"))}
    fin = {k: fx.get(k, v) for k, v in v1.items()}; P = {"orig3": [], "A_edit_as_approve": [], "B_binary_validity": []}
    for k, r in fin.items():
        o = orig[k]["decision"]; nw = M[r["new_annotator_decision"]]
        P["orig3"].append((o, nw)); P["A_edit_as_approve"].append(("approve" if o == "edit" else o, nw))
        P["B_binary_validity"].append(("ok" if o in ("approve", "edit") else "no", "ok" if nw == "approve" else "no"))
    for nme, pr in P.items():
        a, pb = ac1_pabak(pr); out["agreement"][f"{lang}|{nme}"] = {"n": len(pr), "AC1": a, "AC1_ci": boot(pr), "PABAK": pb}
# ---------- (b) H200
R = os.path.join(ROOT, "results", "h200", "judged")
MODELS = ["Qwen2.5-7B-Instruct", "Qwen2.5-32B-Instruct", "Llama-3.1-8B-Instruct", "gemma-2-9b-it", "Sailor2-8B-Chat"]
load = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
flag = {r["id"] for r in csv.DictReader(open(os.path.join(ROOT, "Computation work", "answerability_flags.csv"), encoding="utf-8-sig"))}
dis = {r["id"] for r in csv.DictReader(open(os.path.join(ROOT, "data", "verified", "unadjudicated_disagreements.csv"), encoding="utf-8")) if r["in_released_benchmark"] == "yes"}
out["n_flag"], out["n_dis_released"], out["n_overlap"] = len(flag), len(dis), len(flag & dis)
def boot_mean(v, c, B=2000):
    v = np.asarray(v, float); ks = np.unique(c); idx = [np.where(c == k)[0] for k in ks]; e = v.mean(); o = []
    for _ in range(B):
        s = rng.integers(0, len(ks), len(ks)); o.append(v[np.concatenate([idx[i] for i in s])].mean())
    return [100*e, 100*float(np.percentile(o, 2.5)), 100*float(np.percentile(o, 97.5))]
out["twobytwo"] = {}; out["sensitivity"] = {}; rows_all = []
for dom in ("quran", "wiki"):
    for cond in ("goldmpnet", "oracle"):
        cnt = collections.Counter()
        for m in MODELS:
            for x in load(f"{R}/{dom}_{m}_{cond}_full.jsonl"):
                f = x["faithfulness_label"] == "SUPPORTED"; c = x["correctness_label"] == "CORRECT"
                cnt[(("faithful" if f else "not_faithful"), ("correct" if c else "incorrect"))] += 1
                if cond == "goldmpnet": rows_all.append((dom, m, x))
        tot = sum(cnt.values()); out["twobytwo"][f"{dom}|{cond}"] = {"n": tot, **{f"{a}|{b}": [v, 100*v/tot] for (a, b), v in cnt.items()}}
    # sensitivity: oracle effect on misses with all records vs excluding flagged / unresolved
    for excl_name, excl in (("all", set()), ("excl_flags", flag), ("excl_disagreements", dis), ("excl_both", flag | dis)):
        effs = []
        for m in MODELS:
            b = load(f"{R}/{dom}_{m}_goldmpnet_full.jsonl"); o = load(f"{R}/{dom}_{m}_oracle_full.jsonl")
            keep = [i for i, x in enumerate(b) if x["id"] not in excl and not x["top1_hit"]]
            effs.append(100*(np.mean([o[i]["faithfulness_label"] == "SUPPORTED" for i in keep]) - np.mean([b[i]["faithfulness_label"] == "SUPPORTED" for i in keep])))
        out["sensitivity"][f"{dom}|{excl_name}"] = {"n_miss_records_per_model": len(keep), "oracle_minus_base_SUP_miss_by_model": [float(e) for e in effs], "min": float(min(effs)), "max": float(max(effs))}
# interaction: logit SUPPORTED ~ hit*language, clustered by passage (pooled over generators, generator fixed effects)
import statsmodels.formula.api as smf, pandas as pd
out["interaction"] = {}
for dom in ("quran", "wiki"):
    d = pd.DataFrame([{"sup": int(x["faithfulness_label"] == "SUPPORTED"), "hit": int(x["top1_hit"]), "ar": int(x["language"] == "ar"), "gen": m, "cl": cl(x["id"])} for (dm, m, x) in rows_all if dm == dom])
    fit = smf.logit("sup ~ hit*ar + C(gen)", data=d).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d.cl)[0]})
    ci = fit.conf_int()
    out["interaction"][dom] = {t: {"coef": float(fit.params[t]), "se": float(fit.bse[t]), "p": float(fit.pvalues[t]), "OR": float(np.exp(fit.params[t])), "OR_ci": [float(np.exp(ci.loc[t, 0])), float(np.exp(ci.loc[t, 1]))]} for t in ("hit", "ar", "hit:ar")}
# ---------- (d) calibration strata
key = list(csv.DictReader(open(os.path.join(ROOT, "results", "human_eval", "key_private.csv"), encoding="utf-8")))
st = collections.OrderedDict()
for r in key: st.setdefault((r["domain"], r["language"], r["judge72_faith"]), [int(r["stratum_N"]), 0])[1] += 1
out["strata"] = [{"domain": a, "language": b, "judge_label": c, "N": v[0], "n": v[1]} for (a, b, c), v in sorted(st.items())]
cal = json.load(open(os.path.join(ROOT, "results", "human_eval", "calibration.json")))
U = np.array(cal["faith5_confusion_counts"]); out["raw_agreement"] = [float(np.trace(U)), float(U.sum()), float(np.trace(U)/U.sum())]
out["raw_unsupported_human"] = {"human_unsup_n": float(U[2].sum()), "judge_unsup_n": float(U[:, 2].sum()), "both": float(U[2, 2])}
json.dump(out, open(os.path.join(ROOT, "results", "review2_analysis.json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ("agreement", "twobytwo", "sensitivity", "interaction", "raw_agreement", "raw_unsupported_human", "n_flag", "n_dis_released", "n_overlap")}, indent=1))
