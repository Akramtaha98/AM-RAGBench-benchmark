"""Paired exact McNemar tests on rank-1 hits across BM25, MPNet and E5 (per domain, languages pooled).
Usage: PYTHONPATH=/tmp/pylib python3 src/retriever_mcnemar.py --csv-dir <dir>"""
import sys, collections, itertools, json
sys.path.insert(0, "src")
import numpy as np
from scipy.stats import binomtest
import review_response_analysis as R

ver, res = R.load()
models = {"MPNet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "E5": "intfloat/multilingual-e5-base"}
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
hits = collections.defaultdict(dict)  # name -> id -> hit1
groups = collections.defaultdict(list)
for r in res: groups[(r["domain"], r["language"])].append(r)
def run(name, scorer):
    for key, rs in groups.items():
        pids, texts, seen = [], [], set()
        for r in rs:
            if r["gold_pid"] not in seen: seen.add(r["gold_pid"]); pids.append(r["gold_pid"]); texts.append(r["gold_text"])
        S = scorer(texts, [r["question"] for r in rs])
        for i, r in enumerate(rs): hits[name][(r["id"])] = int(pids[int(np.argmax(S[i]))] == r["gold_pid"])
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
out = {}
for d in ("quran", "wiki"):
    ids = [i for i in hits["BM25"] if dom[i] == d]
    for a, b in itertools.combinations(["BM25", "MPNet", "E5"], 2):
        x = sum(1 for i in ids if hits[a][i] and not hits[b][i]); y = sum(1 for i in ids if hits[b][i] and not hits[a][i])
        p = binomtest(x, x + y, 0.5).pvalue if x + y else 1.0
        out[f"{d}:{a}>{b}"] = dict(a_only=x, b_only=y, p=p, n=len(ids))
        print(d, a, "vs", b, "only-first", x, "only-second", y, "p=%.2e" % p, "(Bonferroni x6: %.2e)" % min(1, 6 * p))
json.dump(out, open("results/retriever_mcnemar.json", "w"), indent=1)
