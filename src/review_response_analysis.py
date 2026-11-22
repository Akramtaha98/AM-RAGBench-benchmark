"""
Re-analyses prepared in response to peer review (top-1 retrieval conditioning,
second-annotator validity reanalysis, judge per-label confidence intervals,
BM25 and alternative dense retrievers).

Everything here is computed from files released with the benchmark
(results/*.jsonl, data/verified/*.jsonl, review/reannotation/*.json) plus the
completed review CSVs. Nothing is simulated.

Usage (from repo root):
    python src/review_response_analysis.py --csv-dir <folder with completed CSVs> [--dense]
"""
import hashlib, argparse, csv, json, math, re, sys, collections
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = {}


def L(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    A = collections.Counter(a for a, _ in pairs)
    B = collections.Counter(b for _, b in pairs)
    pe = sum(A[l] / n * B[l] / n for l in set(A) | set(B))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def boot_kappa(pairs, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(pairs)
    ks = []
    for _ in range(B):
        s = [pairs[i] for i in rng.integers(0, n, n)]
        ks.append(kappa(s))
    return tuple(np.percentile(ks, [2.5, 97.5]))


def tok(t):
    return re.findall(r"\w+", t.lower())


def load():
    ver = {}
    for r in L(ROOT / "data/verified/quran_verified.jsonl") + L(ROOT / "data/verified/wiki_verified.jsonl"):
        ver[r["id"]] = r
    res = L(ROOT / "results/quran_results.jsonl") + L(ROOT / "results/wiki_results.jsonl")
    for r in res:
        g = ver[r["id"]]
        r["domain"] = "quran" if g["domain"].startswith("quran") else "wiki"
        r["gold_pid"] = "t" + hashlib.md5(g["gold_passage_text"].strip().encode()).hexdigest()[:12]  # passage identity = text (gold_passage_id is per-question)
        r["gold_text"] = g["gold_passage_text"]
        r["top1_hit"] = r["passage"].strip() == g["gold_passage_text"].strip()
    return ver, res


def part_top1(res):
    out = {}
    print("\n== A. Recall@1 vs recall@3 (generator sees only top-1) ==")
    for dom in ("quran", "wiki"):
        for lang in ("ar", "ms", None):
            rs = [r for r in res if r["domain"] == dom and (lang is None or r["language"] == lang)]
            n = len(rs)
            r1 = sum(r["top1_hit"] for r in rs)
            r3 = sum(r["retrieval_hit"] for r in rs)
            print(f"{dom:6s} {lang or 'all':3s} n={n:4d} R@1={100*r1/n:5.1f}% ({r1}) R@3={100*r3/n:5.1f}% ({r3})")
            out[f"{dom}_{lang or 'all'}"] = dict(n=n, r1=r1, r3=r3)
    # conditioned faithfulness on top-1 hit
    labs = ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "ABSTAINED"]
    print("\n== A2. Faithfulness conditioned on top-1 hit ==")
    for dom in ("quran", "wiki"):
        rs = [r for r in res if r["domain"] == dom and r["faithfulness_label"] in labs]
        for hit in (False, True):
            sub = [r for r in rs if r["top1_hit"] == hit]
            c = collections.Counter(r["faithfulness_label"] for r in sub)
            n = len(sub)
            print(f"{dom:6s} top1_{'hit ' if hit else 'miss'} n={n:3d} " +
                  " ".join(f"{l[:3]}={100*c[l]/n:5.1f}" for l in labs))
        tab = np.array([[sum(1 for r in rs if r["top1_hit"] == h and r["faithfulness_label"] == l) for l in labs] for h in (False, True)])
        keep = tab.sum(0) > 0
        chi, p, dfree, _ = stats.chi2_contingency(tab[:, keep])
        V = math.sqrt(chi / (tab.sum() * (min(tab[:, keep].shape) - 1)))
        print(f"   chi2={chi:.1f} df={dfree} p={p:.2g} V={V:.2f}")
        out[f"chi_{dom}"] = dict(chi2=chi, df=int(dfree), p=p, V=V)
        for lang in ("ar", "ms"):
            sub = [r for r in rs if r["language"] == lang]
            t = np.array([[sum(1 for r in sub if r["top1_hit"] == h and r["faithfulness_label"] == l) for l in labs] for h in (False, True)])
            k = t.sum(0) > 0
            chi, p, dfree, _ = stats.chi2_contingency(t[:, k])
            V = math.sqrt(chi / (t.sum() * (min(t[:, k].shape) - 1)))
            print(f"   within {lang}: chi2={chi:.1f} df={dfree} p={p:.2g} V={V:.2f}")
            out[f"chi_{dom}_{lang}"] = dict(chi2=chi, df=int(dfree), p=p, V=V)
    # GEE
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    print("\n== A3. GEE (cluster = gold passage) with top-1 hit ==")
    for dom in ("quran", "wiki"):
        rs = [r for r in res if r["domain"] == dom and r["faithfulness_label"] in labs]
        import pandas as pd
        df = pd.DataFrame([{"sup": int(r["faithfulness_label"] == "SUPPORTED"),
                            "uns": int(r["faithfulness_label"] == "UNSUPPORTED"),
                            "hit": int(r["top1_hit"]), "ms": int(r["language"] == "ms"),
                            "pid": r["gold_pid"]} for r in rs])
        for y in ("sup", "uns"):
            m = smf.gee(f"{y} ~ hit + ms", "pid", df, family=sm.families.Binomial(),
                        cov_struct=sm.cov_struct.Exchangeable()).fit()
            for v in ("hit", "ms"):
                lo, hi = np.exp(m.conf_int().loc[v])
                print(f"   {dom} {y}: {v} OR={math.exp(m.params[v]):.2f} [{lo:.2f},{hi:.2f}] p={m.pvalues[v]:.2g}")
                out[f"gee_{dom}_{y}_{v}"] = dict(OR=math.exp(m.params[v]), lo=lo, hi=hi, p=m.pvalues[v])
    return out


def part_judge(csvdir):
    print("\n== D. Judge validation per-label CIs (Wilson) ==")
    rows = list(csv.DictReader(open(csvdir / "agreement_review_completed-2.csv", encoding="utf-8-sig")))
    rows = [r for r in rows if r["faithfulness_label"] != "JUDGE_PARSE_FAILED"]
    n = len(rows)
    k = sum(r["human_decision"] == "agree" for r in rows)
    print(f"overall {k}/{n} = {100*k/n:.1f}% CI {[round(100*x,1) for x in wilson(k,n)]}")
    out = {"overall": (k, n)}
    for lab in ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "ABSTAINED"]:
        s = [r for r in rows if r["faithfulness_label"] == lab]
        kk = sum(r["human_decision"] == "agree" for r in s)
        lo, hi = wilson(kk, len(s))
        print(f"  {lab:20s} {kk}/{len(s)} = {100*kk/len(s):.1f}% CI [{100*lo:.1f},{100*hi:.1f}]")
        out[lab] = (kk, len(s), lo, hi)
    # partial confusion
    conf = collections.Counter((r["faithfulness_label"], r["human_corrected_label"]) for r in rows
                               if r["human_decision"] == "disagree" and r["human_corrected_label"].strip())
    print("  partial confusion (judge -> human, disagreements with a stated correction):", dict(conf))
    unk = sum(1 for r in rows if r["human_decision"] == "disagree" and not r["human_corrected_label"].strip())
    print(f"  disagreements without corrected label: {unk}")
    return out


def part_reannot(csvdir):
    print("\n== C. Second-annotator reanalysis ==")
    M = {"valid": "approve", "edit": "edit", "invalid": "reject"}

    def Lc(p):
        return list(csv.DictReader(open(p, encoding="utf-8-sig")))

    out = {}
    disagreements = []
    allp = {"orig3": [], "A": [], "B": []}
    for lang in ("arabic", "malay"):
        orig = json.load(open(ROOT / f"review/reannotation/{lang}_original_decisions.json"))
        v1 = {r["id"]: r for r in Lc(csvdir / f"amragbench_{lang}_reannotation_v1_completed.csv")}
        fx = {r["id"]: r for r in Lc(csvdir / f"amragbench_{lang}_reannotation_fixup_v1_completed.csv")}
        fin = {k: fx.get(k, v) for k, v in v1.items()}
        pairs3, pairsA, pairsB = [], [], []
        for k, r in fin.items():
            o = orig[k]["decision"]
            nw = M[r["new_annotator_decision"]]
            pairs3.append((o, nw))
            # A: original edit means "released after correction" -> compare as approve (final text valid)
            pairsA.append(("approve" if o == "edit" else o, nw))
            # B: binary final-record validity: original (approve|edit)=acceptable, reject=not;
            #    second annotator valid = acceptable, edit/invalid = not fully valid
            pairsB.append(("ok" if o in ("approve", "edit") else "no", "ok" if nw == "approve" else "no"))
            if pairsA[-1][0] != nw:
                disagreements.append(dict(language=lang, id=k, domain=orig[k]["domain"], original=o, second=r["new_annotator_decision"],
                                          question=r.get("new_corrected_question") or r["question"], gold_answer=r["gold_answer"],
                                          passage=r["gold_passage_text"], second_notes=r.get("new_annotator_notes", ""), adjudicated_decision=""))
        for name, pr in (("orig3", pairs3), ("A_edit_as_approve", pairsA), ("B_binary_validity", pairsB)):
            n = len(pr)
            ag = sum(a == b for a, b in pr)
            lo, hi = wilson(ag, n)
            kp = kappa(pr)
            klo, khi = boot_kappa(pr)
            print(f"{lang:7s} {name:20s} agree {ag}/{n}={100*ag/n:.1f}% CI[{100*lo:.1f},{100*hi:.1f}] kappa={kp:.3f} CI[{klo:.2f},{khi:.2f}]")
            out[f"{lang}_{name}"] = dict(n=n, agree=ag, lo=lo, hi=hi, kappa=kp, klo=float(klo), khi=float(khi))
        # by domain for A
        for dom in ("quran", "wiki"):
            sub = [p for (k, _), p in zip(fin.items(), pairsA) if orig[k]["domain"] == dom]
            if sub:
                print(f"   {dom:6s} A: n={len(sub)} agree={100*sum(a==b for a,b in sub)/len(sub):.1f}% kappa={kappa(sub):.3f}")
                out[f"{lang}_{dom}_A"] = dict(n=len(sub), agree=sum(a == b for a, b in sub), kappa=kappa(sub))
    p = ROOT / "review/reannotation/disagreements_for_adjudication.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(disagreements[0].keys()))
        w.writeheader()
        w.writerows(disagreements)
    print(f"wrote {len(disagreements)} disagreements to {p}")
    return out


def part_retrievers(res, models):
    print("\n== B. Retriever comparison (per domain x language index of unique gold passages) ==")
    from rank_bm25 import BM25Okapi
    groups = collections.defaultdict(list)
    for r in res:
        groups[(r["domain"], r["language"])].append(r)
    out = {}

    def rank_eval(name, scorer):
        for key, rs in sorted(groups.items()):
            pids, texts = [], []
            seen = set()
            for r in rs:
                if r["gold_pid"] not in seen:
                    seen.add(r["gold_pid"]); pids.append(r["gold_pid"]); texts.append(r["gold_text"])
            S = scorer(texts, [r["question"] for r in rs])
            h1 = h3 = 0
            for i, r in enumerate(rs):
                order = np.argsort(-S[i])
                top = [pids[j] for j in order[:3]]
                h1 += top[0] == r["gold_pid"]
                h3 += r["gold_pid"] in top
            n = len(rs)
            out[(name, key)] = (h1, h3, n, len(pids))
        for dom in ("quran", "wiki"):
            tot = [0, 0, 0]
            for lang in ("ar", "ms"):
                h1, h3, n, m = out[(name, (dom, lang))]
                tot[0] += h1; tot[1] += h3; tot[2] += n
                print(f"{name:28s} {dom:6s} {lang} n={n:3d} idx={m:3d} R@1={100*h1/n:5.1f} R@3={100*h3/n:5.1f}")
            print(f"{name:28s} {dom:6s} all R@1={100*tot[0]/tot[2]:5.1f} R@3={100*tot[1]/tot[2]:5.1f}")

    def bm25(texts, qs):
        b = BM25Okapi([tok(t) for t in texts])
        return np.array([b.get_scores(tok(q)) for q in qs])

    rank_eval("BM25", bm25)
    for m in models:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(m)
        pre_q, pre_p = ("query: ", "passage: ") if "e5" in m else ("", "")

        def dense(texts, qs, st=st, pre_q=pre_q, pre_p=pre_p):
            P = st.encode([pre_p + t for t in texts], normalize_embeddings=True, show_progress_bar=False, batch_size=16)
            Q = st.encode([pre_q + q for q in qs], normalize_embeddings=True, show_progress_bar=False, batch_size=16)
            return Q @ P.T

        rank_eval(m.split("/")[-1], dense)
        del st
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--dense", nargs="*", default=[], help="sentence-transformers model names")
    a = ap.parse_args()
    ver, res = load()
    o = {}
    o["top1"] = part_top1(res)
    o["judge"] = part_judge(Path(a.csv_dir))
    o["reannot"] = part_reannot(Path(a.csv_dir))
    ret = part_retrievers(res, a.dense)
    json.dump(o, open(ROOT / "results/review_response_stats.json", "w"), default=str, indent=1)
