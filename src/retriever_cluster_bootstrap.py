"""Passage-clustered comparison of retrievers (paired rank-1 hit differences).

Usage (needs the same packages as retriever_mcnemar.py):
    python3 src/retriever_cluster_bootstrap.py
Resamples gold passages (clusters) with replacement and reports, for each domain and retriever
pair, the mean paired difference in rank-1 hit rate with a 95% percentile CI.
"""
import sys, collections, itertools, json
sys.path.insert(0, "src")
import numpy as np
import review_response_analysis as R
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

ver, res = R.load()
models = {"MPNet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "E5": "intfloat/multilingual-e5-base"}
hits = collections.defaultdict(dict)
groups = collections.defaultdict(list)
for r in res: groups[(r["domain"], r["language"])].append(r)
def run(name, scorer):
    for key, rs in groups.items():
        pids, texts, seen = [], [], set()
        for r in rs:
            if r["gold_pid"] not in seen: seen.add(r["gold_pid"]); pids.append(r["gold_pid"]); texts.append(r["gold_text"])
        S = scorer(texts, [r["question"] for r in rs])
        for i, r in enumerate(rs): hits[name][r["id"]] = int(pids[int(np.argmax(S[i]))] == r["gold_pid"])
def bm25(texts, qs):
    b = BM25Okapi([R.tok(t) for t in texts]); return np.array([b.get_scores(R.tok(q)) for q in qs])
run("BM25", bm25)
for nm, m in models.items():
    st = SentenceTransformer(m); pq, pp = ("query: ", "passage: ") if "e5" in m else ("", "")
    def dense(texts, qs, st=st, pq=pq, pp=pp):
        P = st.encode([pp + t for t in texts], normalize_embeddings=True, show_progress_bar=False, batch_size=16)
        Q = st.encode([pq + q for q in qs], normalize_embeddings=True, show_progress_bar=False, batch_size=16)
        return Q @ P.T
    run(nm, dense); del st
dom = {r["id"]: r["domain"] for r in res}
cluster = {r["id"]: (r["domain"], r["language"], r["gold_pid"]) for r in res}
rng = np.random.default_rng(0)
out = {}
for d in ("quran", "wiki"):
    ids = [i for i in hits["BM25"] if dom[i] == d]
    cl = collections.defaultdict(list)
    for i in ids: cl[cluster[i]].append(i)
    keys = list(cl)
    for a, b in itertools.combinations(["BM25", "MPNet", "E5"], 2):
        per = {k: (sum(hits[a][i] - hits[b][i] for i in cl[k]), len(cl[k])) for k in keys}
        diffs = []
        for _ in range(5000):
            pick = rng.integers(0, len(keys), len(keys))
            num = sum(per[keys[j]][0] for j in pick); den = sum(per[keys[j]][1] for j in pick)
            diffs.append(num / den)
        est = sum(v[0] for v in per.values()) / sum(v[1] for v in per.values())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out[f"{d}:{a}-{b}"] = dict(diff=est, ci=[lo, hi], clusters=len(keys))
        print(d, a, "-", b, "diff %.3f CI [%.3f, %.3f]" % (est, lo, hi), "clusters", len(keys))
json.dump(out, open("results/retriever_cluster_bootstrap.json", "w"), indent=1)
