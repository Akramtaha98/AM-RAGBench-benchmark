"""Passage-clustered bootstrap 95% CIs for the pilot-run tables (Tables 5, 7, 9) and second-judge agreement.
Cluster = record id without the trailing -q<k> (all questions on one passage form one cluster). 4,000 resamples, seed 7.
Output: results/review3_cis.json.  Usage: python3 src/review3_cis.py"""
import json, re, numpy as np, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
rng = np.random.default_rng(7); B = 4000
def load(p): return [json.loads(l) for l in open(os.path.join(ROOT, p), encoding="utf-8")]
cl = lambda i: re.sub(r"-q\d+$", "", i)
def boot(rows, stat):
    ids = sorted({cl(r["id"]) for r in rows}); by = {i: [] for i in ids}
    for r in rows: by[cl(r["id"])].append(r)
    est = stat(rows); vals = []
    for _ in range(B):
        samp = [r for i in rng.choice(ids, len(ids)) for r in by[i]]; vals.append(stat(samp))
    v = np.array(vals, dtype=float); return est, float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))
def rate(lab): return lambda rs: 100 * np.mean([r["faithfulness_label"] == lab for r in rs]) if rs else float("nan")
LAB = {"Sup": "SUPPORTED", "Uns": "UNSUPPORTED", "Abs": "ABSTAINED", "PSup": "PARTIALLY_SUPPORTED", "Con": "CONTRADICTED"}
out = {"table5": {}, "table7_9": {}, "judge2": {}}
base = {"quran": load("results/quran_results.jsonl"), "wiki": load("results/wiki_results.jsonl")}
orc = {"quran": load("results/new/quran_oracle.jsonl"), "wiki": load("results/new/wiki_oracle.jsonl")}
for d in base:
    for lang in ("ar", "ms"):
        rs = [r for r in base[d] if r["language"] == lang]
        out["table5"][f"{d}|{lang}"] = {k: boot(rs, rate(v)) for k, v in LAB.items()}
        out["table5"][f"{d}|{lang}"]["n"] = len(rs)
gold = {}
for f in ("quran_verified.jsonl", "wiki_verified.jsonl"):
    for r in load("data/verified/" + f): gold[r["id"]] = r["gold_passage_text"]
for d in base:
    hit = {r["id"]: r["passage"] == gold[r["id"]] for r in base[d]}  # text-match hit, as defined in Section 3.7
    for cond, rows in (("baseline", base[d]), ("oracle", orc[d])):
        for sub in ("all", "miss", "hit"):
            rs = [r for r in rows if sub == "all" or (hit[r["id"]] == (sub == "hit"))]
            out["table7_9"][f"{d}|{cond}|{sub}"] = {k: boot(rs, rate(LAB[k])) for k in ("Sup", "Uns", "Abs")} | {"n": len(rs)}
def kappa(rs):
    a = [r["faithfulness_label"] for r in rs]; b = [r["judge2_label"] for r in rs]; labs = sorted(set(a) | set(b)); po = np.mean([x == y for x, y in zip(a, b)])
    pe = sum(np.mean([x == l for x in a]) * np.mean([y == l for y in b]) for l in labs); return (po - pe) / (1 - pe) if pe < 1 else float("nan")
def agree(rs): return 100 * np.mean([r["faithfulness_label"] == r["judge2_label"] for r in rs])
for d in ("quran", "wiki"):
    rs = [r for r in load(f"results/new/{d}_rejudged.jsonl") if r.get("judge2_label")]  # all records with a second-judge label; first-judge parse failures count as disagreements
    out["judge2"][d] = {"n": len(rs), "agreement": boot(rs, agree), "kappa": boot(rs, kappa)}
json.dump(out, open(os.path.join(ROOT, "results", "review3_cis.json"), "w"), indent=1)
for k, v in out["table5"].items(): print(k, v["n"], {m: tuple(round(x, 1) for x in v[m]) for m in ("Sup", "Uns")})
for k, v in out["table7_9"].items(): print(k, v["n"], {m: tuple(round(x, 1) for x in v[m]) for m in ("Sup", "Uns", "Abs")})
print(out["judge2"])
