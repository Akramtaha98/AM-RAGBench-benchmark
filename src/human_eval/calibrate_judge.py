"""Calibrate the 72B LLM judge against human labels (after annotation).

Inputs: results/human_eval/annotation_ar.xlsx and annotation_ms.xlsx (filled by annotator 1), optionally
annotation_ar_2.xlsx / annotation_ms_2.xlsx (second annotator, only 'OVERLAP=yes' rows need labels), and key_private.csv.
Because the sample is stratified by the judge's own label, plain agreement is biased. All headline numbers are
re-weighted to the population (weight of an item = stratum_N / stratum_n) and CIs come from a stratified bootstrap.

Outputs results/human_eval/calibration.json and prints: 5x5 confusion matrix, per-label precision/recall/F1
(judge as predictor, human as reference), weighted agreement and Cohen's kappa, three-way and correctness tables,
population-weighted human vs judge label rates by domain x language, and inter-annotator kappa on the overlap.
Usage: python3 src/human_eval/calibrate_judge.py
"""
import csv, json, os
import numpy as np
from collections import defaultdict
from openpyxl import load_workbook

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "human_eval")
rng = np.random.default_rng(11)
F5 = ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "ABSTAINED"]
COR = ["CORRECT", "INCORRECT", "ABSTAINED"]
T3 = ["SUPPORTED", "NOT_SUPPORTED", "ABSTAINED"]
to3 = lambda l: l if l in ("SUPPORTED", "ABSTAINED") else "NOT_SUPPORTED"

def read_sheet(path):
    ws = load_workbook(path, data_only=True)["ANNOTATE"]; out = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[0]: out[r[0]] = {"faith": (r[4] or "").strip().upper() or None, "corr": (r[5] or "").strip().upper() or None, "notes": r[6]}
    return out

def kappa(a, b, labels, w=None):
    w = np.ones(len(a)) if w is None else np.asarray(w, float)
    idx = {l: i for i, l in enumerate(labels)}; M = np.zeros((len(labels), len(labels)))
    for x, y, wi in zip(a, b, w): M[idx[x], idx[y]] += wi
    n = M.sum(); po = np.trace(M)/n; pe = (M.sum(1) @ M.sum(0))/n**2
    return float((po-pe)/(1-pe)) if pe < 1 else float("nan")

def weighted_table(h, j, labels, w):
    idx = {l: i for i, l in enumerate(labels)}; M = np.zeros((len(labels),)*2)
    for x, y, wi in zip(h, j, w): M[idx[x], idx[y]] += wi
    return M

def metrics(h, j, labels, w):
    M = weighted_table(h, j, labels, w); tot = M.sum()
    res = {"agreement": float(np.trace(M)/tot), "kappa": kappa(h, j, labels, w), "per_label": {}}
    for i, l in enumerate(labels):
        tp = M[i, i]; pj = M[:, i].sum(); ph = M[i, :].sum()
        p = tp/pj if pj else float("nan"); r = tp/ph if ph else float("nan")
        res["per_label"][l] = {"precision": float(p), "recall": float(r), "f1": float(2*p*r/(p+r)) if p == p and r == r and p+r else float("nan"),
                               "human_rate": float(ph/tot), "judge_rate": float(pj/tot)}
    return res, M

def strat_boot(items, stat, B=2000):
    strata = defaultdict(list)
    for it in items: strata[it["stratum"]].append(it)
    vals = []
    for _ in range(B):
        samp = []
        for s, v in strata.items(): samp += [v[i] for i in rng.integers(0, len(v), len(v))]
        vals.append(stat(samp))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

def main():
    key = {r["item_id"]: r for r in csv.DictReader(open(os.path.join(D, "key_private.csv"), encoding="utf-8"))}
    items = []
    for lang in ("ar", "ms"):
        p = os.path.join(D, f"annotation_{lang}.xlsx")
        if not os.path.exists(p): continue
        for iid, v in read_sheet(p).items():
            if v["faith"] and v["corr"]:
                k = key[iid]; items.append({"id": iid, "lang": lang, "dom": k["domain"], "hf": v["faith"], "hc": v["corr"], "jf": k["judge72_faith"],
                    "j3": k["judge72_threeway"], "jc": k["judge72_correct"], "w": float(k["stratum_N"])/float(k["stratum_n"]),
                    "stratum": (k["domain"], lang, k["judge72_faith"])})
    print("labelled items:", len(items))
    assert items, "no labelled items found"
    for it in items: assert it["hf"] in F5 and it["hc"] in COR, f"bad label in {it['id']}"
    w = [it["w"] for it in items]; out = {"n": len(items)}
    # 5-way faithfulness
    m5, M5 = metrics([i["hf"] for i in items], [i["jf"] for i in items], F5, w)
    m5["agreement_ci"] = strat_boot(items, lambda s: float(np.average([x["hf"] == x["jf"] for x in s], weights=[x["w"] for x in s])))
    m5["kappa_ci"] = strat_boot(items, lambda s: kappa([x["hf"] for x in s], [x["jf"] for x in s], F5, [x["w"] for x in s]), B=800)
    out["faith5"] = m5; out["faith5_confusion_weighted_rows_human_cols_judge"] = M5.tolist()
    # unweighted confusion (counts)
    U = weighted_table([i["hf"] for i in items], [i["jf"] for i in items], F5, [1]*len(items)); out["faith5_confusion_counts"] = U.tolist()
    # 3-way
    h3 = [to3(i["hf"]) for i in items]; m3, _ = metrics(h3, [i["j3"] for i in items], T3, w)
    m3["agreement_ci"] = strat_boot(items, lambda s: float(np.average([to3(x["hf"]) == x["j3"] for x in s], weights=[x["w"] for x in s])))
    out["threeway"] = m3
    # correctness
    mc, _ = metrics([i["hc"] for i in items], [i["jc"] for i in items], COR, w)
    mc["agreement_ci"] = strat_boot(items, lambda s: float(np.average([x["hc"] == x["jc"] for x in s], weights=[x["w"] for x in s])))
    out["correctness"] = mc
    # population-weighted rates by domain x language: human vs judge
    rates = {}
    for dom in ("quran", "wiki"):
        for lang in ("ar", "ms"):
            sub = [i for i in items if i["dom"] == dom and i["lang"] == lang]
            if not sub: continue
            ww = np.array([i["w"] for i in sub]); tot = ww.sum()
            f = lambda lab, key_: float((ww*np.array([i[key_] == lab for i in sub])).sum()/tot)
            rates[f"{dom}|{lang}"] = {"SUPPORTED": {"human": f("SUPPORTED", "hf"), "judge": f("SUPPORTED", "jf")},
                "UNSUPPORTED": {"human": f("UNSUPPORTED", "hf"), "judge": f("UNSUPPORTED", "jf")},
                "ABSTAINED": {"human": f("ABSTAINED", "hf"), "judge": f("ABSTAINED", "jf")},
                "CORRECT": {"human": f("CORRECT", "hc"), "judge": f("CORRECT", "jc")}}
    out["rates_population_weighted"] = rates
    # inter-annotator agreement on overlap
    ia = {}
    for lang in ("ar", "ms"):
        p2 = os.path.join(D, f"annotation_{lang}_2.xlsx")
        if not os.path.exists(p2): continue
        s2 = read_sheet(p2); a = [(i, s2[i["id"]]) for i in items if i["lang"] == lang and key[i["id"]]["overlap"] == "yes" and i["id"] in s2 and s2[i["id"]]["faith"]]
        if a:
            ia[lang] = {"n": len(a), "faith_agreement": float(np.mean([x[0]["hf"] == x[1]["faith"] for x in a])),
                        "faith_kappa": kappa([x[0]["hf"] for x in a], [x[1]["faith"] for x in a], F5),
                        "corr_kappa": kappa([x[0]["hc"] for x in a], [(x[1]["corr"] or x[0]["hc"]) for x in a], COR)}
    out["inter_annotator"] = ia
    json.dump(out, open(os.path.join(D, "calibration.json"), "w"), indent=1)
    print(f"5-way weighted agreement {m5['agreement']*100:.1f}% CI {[round(100*x,1) for x in m5['agreement_ci']]}  kappa {m5['kappa']:.2f} CI {[round(x,2) for x in m5['kappa_ci']]}")
    print("per-label (judge as predictor):")
    for l, v in m5["per_label"].items(): print(f"  {l:20s} P {v['precision']:.2f} R {v['recall']:.2f} F1 {v['f1']:.2f}  human rate {100*v['human_rate']:.1f}%  judge rate {100*v['judge_rate']:.1f}%")
    print(f"3-way agreement {100*m3['agreement']:.1f}%  kappa {m3['kappa']:.2f}; correctness agreement {100*mc['agreement']:.1f}%  kappa {mc['kappa']:.2f}")
    print("population-weighted rates (human vs judge):"); print(json.dumps(rates, indent=1))
    print("inter-annotator:", ia)

if __name__ == "__main__":
    main()
