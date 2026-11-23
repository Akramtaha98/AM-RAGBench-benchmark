"""Analysis of the H200 matrix (results/h200). Passage-clustered bootstrap CIs (clusters = gold passage from the
record id). Judge for all labels: Qwen2.5-72B-Instruct (faithfulness 5-way, three-way, gold-answer correctness).
Usage: python3 src/h200_analysis.py -> results/h200_analysis.json"""
import json, re, os, glob, itertools
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "h200")
rng = np.random.default_rng(20261124)
def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
def cl(i): return re.sub(r"-q\d+$", "", i)
MODELS = ["Qwen2.5-7B-Instruct","Qwen2.5-32B-Instruct","Llama-3.1-8B-Instruct","gemma-2-9b-it","Sailor2-8B-Chat"]
B = 2000
def boot(v, c):
    v = np.asarray(v, float); ks = np.unique(c); idx = [np.where(c == k)[0] for k in ks]
    est = v.mean(); out = []
    for _ in range(B):
        s = rng.integers(0, len(ks), len(ks)); out.append(v[np.concatenate([idx[i] for i in s])].mean())
    return [100*est, 100*float(np.percentile(out, 2.5)), 100*float(np.percentile(out, 97.5))]
out = {"retrieval": {}, "gen": {}, "seeds": {}, "pooled": {}}
for f in sorted(glob.glob(os.path.join(R, "retr", "*.jsonl"))):
    n = os.path.basename(f)[:-6]; r = load(f); c = np.array([cl(x["id"]) for x in r])
    out["retrieval"][n] = {"n": len(r), "R1": boot([x["top1_hit"] for x in r], c), "R3": boot([x["retrieval_hit"] for x in r], c),
        "index": {l: next(x["index_size"] for x in r if x["language"] == l) for l in ("ar", "ms")}}
def stats(rows):
    c = np.array([cl(x["id"]) for x in rows]); F = [x["faithfulness_label"] for x in rows]
    sup = np.array([l == "SUPPORTED" for l in F], float); ps = np.array([l == "PARTIALLY_SUPPORTED" for l in F], float)
    un = np.array([l == "UNSUPPORTED" for l in F], float); ab = np.array([l == "ABSTAINED" for l in F], float)
    cor = np.array([x["correctness_label"] == "CORRECT" for x in rows], float)
    t3s = np.array([x["threeway_label"] == "SUPPORTED" for x in rows], float)
    hit = np.array([x["top1_hit"] for x in rows])
    d = {"n": len(rows), "SUP": boot(sup, c), "PSUP": boot(ps, c), "UNS": boot(un, c), "ABS": boot(ab, c), "CORRECT": boot(cor, c),
         "SUP3": boot(t3s, c), "top1": boot(hit.astype(float), c)}
    if hit.any():
        d["SUP_hit"] = boot(sup[hit], c[hit]); d["CORRECT_hit"] = boot(cor[hit], c[hit])
    if (~hit).any():
        d["SUP_miss"] = boot(sup[~hit], c[~hit]); d["CORRECT_miss"] = boot(cor[~hit], c[~hit])
        sm = sup[~hit].sum()
        d["SUP_wrong_share_miss"] = float(100*(sup[~hit]*(1-cor[~hit])).sum()/sm) if sm else None
        d["ABS_miss"] = boot(ab[~hit], c[~hit])
    d["SUP_wrong_share_all"] = float(100*(sup*(1-cor)).sum()/sup.sum())
    for lg in ("ar", "ms"):
        m = np.array([x["language"] == lg for x in rows])
        d[f"SUP_{lg}"] = boot(sup[m], c[m]); d[f"CORRECT_{lg}"] = boot(cor[m], c[m]); d[f"top1_{lg}"] = boot(hit[m].astype(float), c[m])
    return d, sup, cor, hit, c
for dom in ("quran", "wiki"):
    for m in MODELS:
        G = {cond: load(os.path.join(R, "judged", f"{dom}_{m}_{cond}_full.jsonl")) for cond in ("goldmpnet", "fulle5", "oracle")}
        base, sb, cb, hb, c = stats(G["goldmpnet"]); orc, so, co, ho, _ = stats(G["oracle"]); full = stats(G["fulle5"])[0]
        assert [x["id"] for x in G["goldmpnet"]] == [x["id"] for x in G["oracle"]]
        diff = {"SUP_miss": boot((so-sb)[~hb], c[~hb]), "SUP_hit": boot((so-sb)[hb], c[hb]),
                "CORRECT_miss": boot((co-cb)[~hb], c[~hb]), "CORRECT_all": boot(co-cb, c)}
        out["gen"][f"{dom}|{m}"] = {"goldmpnet": base, "oracle": orc, "fulle5": full, "oracle_minus_base": diff}
    runs = [load(os.path.join(R, "judged", f"{dom}_Qwen2.5-7B-Instruct_goldmpnet_t07_s{s}_full.jsonl")) for s in (1, 2, 3)]
    t0 = load(os.path.join(R, "judged", f"{dom}_Qwen2.5-7B-Instruct_goldmpnet_full.jsonl"))
    same = [float(np.mean([a["faithfulness_label"] == b["faithfulness_label"] for a, b in zip(x, y)])*100) for x, y in itertools.combinations(runs, 2)]
    out["seeds"][dom] = {"label_identical_pairs": same,
        "SUP_by_seed": [float(np.mean([x["faithfulness_label"] == "SUPPORTED" for x in r])*100) for r in runs],
        "CORRECT_by_seed": [float(np.mean([x["correctness_label"] == "CORRECT" for x in r])*100) for r in runs],
        "SUP_t0": float(np.mean([x["faithfulness_label"] == "SUPPORTED" for x in t0])*100),
        "CORRECT_t0": float(np.mean([x["correctness_label"] == "CORRECT" for x in t0])*100)}
    keys = ["SUP", "CORRECT", "SUP_miss", "SUP_hit", "CORRECT_miss", "CORRECT_hit", "ABS", "UNS", "top1"]
    for cond in ("goldmpnet", "oracle", "fulle5"):
        out["pooled"][f"{dom}|{cond}"] = {k: float(np.mean([out["gen"][f"{dom}|{m}"][cond][k][0] for m in MODELS if k in out["gen"][f"{dom}|{m}"][cond]])) for k in keys}
json.dump(out, open(os.path.join(R, "..", "h200_analysis.json"), "w"), indent=1)
print("ok")
