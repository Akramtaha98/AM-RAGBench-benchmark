"""Analysis of the v2 runs (repeated baseline/oracle, BM25/E5 generation, Llama generator).
Passage-clustered bootstrap (clusters = gold passage, taken from the record id), token-F1 answer
correctness against gold answers, run-to-run variation, Monte-Carlo exact-style chi-square tests.
Usage: python3 src/v2_analysis.py  -> results/v2_analysis.json"""
import json, re, glob, os, unicodedata, itertools
import numpy as np
from scipy.stats import chi2_contingency
R = os.path.join(os.path.dirname(__file__), "..", "results")
rng = np.random.default_rng(20261123)
LABELS = ["SUPPORTED","PARTIALLY_SUPPORTED","UNSUPPORTED","CONTRADICTED","ABSTAINED"]
def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
def cl(rid): return re.sub(r"-q\d+$","",rid)
def norm(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[ـۖ-ۭ]","",s).replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ى","ي").replace("ة","ه")
    return re.sub(r"[^\w\s]"," ",s.lower()).split()
def f1(a,g):
    a,g=norm(a),norm(g)
    if not a or not g: return 0.0
    c=sum(min(a.count(w),g.count(w)) for w in set(a))
    if c==0: return 0.0
    p,r=c/len(a),c/len(g); return 2*p*r/(p+r)
def correct(row):
    a,g=row["generated_answer"],row["gold_answer"]
    if re.search(r"don.t know|tidak tahu|لا أعرف|لا اعرف",a or "",re.I): return False
    return f1(a,g)>=0.5 or (len(norm(g))>0 and " ".join(norm(g)) in " ".join(norm(a)))
def boot(vals, clusters, B=4000):
    """cluster bootstrap CI of a mean; vals array, clusters array of keys"""
    keys=np.unique(clusters); idx={k:np.where(clusters==k)[0] for k in keys}
    est=vals.mean(); out=[]
    for _ in range(B):
        ks=rng.choice(keys,len(keys)); ii=np.concatenate([idx[k] for k in ks]); out.append(vals[ii].mean())
    return float(est), float(np.percentile(out,2.5)), float(np.percentile(out,97.5))
def mc_chi2(x,y,n=20000):
    """Monte-Carlo permutation p-value for chi-square independence"""
    xs,ys=np.unique(x,return_inverse=True)[1],np.unique(y,return_inverse=True)[1]
    def stat(a,b):
        t=np.zeros((a.max()+1,b.max()+1)); np.add.at(t,(a,b),1)
        t=t[:,t.sum(0)>0]; e=np.outer(t.sum(1),t.sum(0))/t.sum(); return ((t-e)**2/e).sum()
    s0=stat(xs,ys); cnt=sum(stat(xs,rng.permutation(ys))>=s0 for _ in range(n)); return float(s0),(cnt+1)/(n+1)
out={}
doms={"quran":"quran","wiki":"wiki"}
data={}
for f in sorted(glob.glob(os.path.join(R,"v2","*.jsonl"))):
    data[os.path.basename(f)[:-6]]=load(f)
def summ(rows):
    n=len(rows); c={l:0 for l in LABELS+["JUDGE_PARSE_FAILED"]}
    for r in rows: c[r["faithfulness_label"]]=c.get(r["faithfulness_label"],0)+1
    return {"n":n,"counts":c}
for k,rows in data.items(): out.setdefault("conditions",{})[k]=summ(rows)
# per-domain analyses
for d in ["quran","wiki"]:
    D={}
    base=[data[f"{d}_mpnet_run{i}"] for i in (1,2,3)]; orc=[data[f"{d}_oracle_run{i}"] for i in (1,2,3)]
    ids=[r["id"] for r in base[0]]; assert all([r["id"] for r in b]==ids for b in base+orc)
    clu=np.array([cl(i) for i in ids]); top1=np.array([r["top1_hit"] for r in base[0]])
    assert all([r["top1_hit"] for r in b]==list(top1) for b in base)
    sup=lambda rows: np.array([r["faithfulness_label"]=="SUPPORTED" for r in rows],float)
    abst=lambda rows: np.array([r["faithfulness_label"]=="ABSTAINED" for r in rows],float)
    uns=lambda rows: np.array([r["faithfulness_label"]=="UNSUPPORTED" for r in rows],float)
    # run-to-run
    D["run_to_run"]={ "baseline_SUP_by_run":[float(sup(b).mean()) for b in base],
        "oracle_SUP_by_run":[float(sup(o).mean()) for o in orc],
        "baseline_SUP_hit_by_run":[float(sup(b)[top1].mean()) for b in base],
        "oracle_SUP_hit_by_run":[float(sup(o)[top1].mean()) for o in orc],
        "baseline_SUP_miss_by_run":[float(sup(b)[~top1].mean()) for b in base],
        "oracle_SUP_miss_by_run":[float(sup(o)[~top1].mean()) for o in orc],
        "baseline_ABS_miss_by_run":[float(abst(b)[~top1].mean()) for b in base],
        "oracle_ABS_miss_by_run":[float(abst(o)[~top1].mean()) for o in orc],
        "baseline_UNS_by_run":[float(uns(b).mean()) for b in base],
        "oracle_UNS_by_run":[float(uns(o).mean()) for o in orc]}
    # record-level label agreement between runs (identical label)
    ag=[]
    for a,b in itertools.combinations(base,2): ag.append(float(np.mean([x["faithfulness_label"]==y["faithfulness_label"] for x,y in zip(a,b)])))
    D["baseline_label_identical_between_runs"]=ag
    ag=[]
    for a,b in itertools.combinations(orc,2): ag.append(float(np.mean([x["faithfulness_label"]==y["faithfulness_label"] for x,y in zip(a,b)])))
    D["oracle_label_identical_between_runs"]=ag
    # mean over runs per record; cluster bootstrap
    bm=np.mean([sup(b) for b in base],0); om=np.mean([sup(o) for o in orc],0)
    D["SUP_mean3_baseline_all"]=boot(bm,clu); D["SUP_mean3_oracle_all"]=boot(om,clu)
    D["SUP_mean3_baseline_miss"]=boot(bm[~top1],clu[~top1]); D["SUP_mean3_oracle_miss"]=boot(om[~top1],clu[~top1])
    D["SUP_mean3_baseline_hit"]=boot(bm[top1],clu[top1]); D["SUP_mean3_oracle_hit"]=boot(om[top1],clu[top1])
    D["oracle_minus_baseline_SUP_miss"]=boot((om-bm)[~top1],clu[~top1])
    D["oracle_minus_baseline_SUP_hit"]=boot((om-bm)[top1],clu[top1])
    ab=np.mean([abst(b) for b in base],0); ao=np.mean([abst(o) for o in orc],0)
    D["ABS_mean3_baseline_miss"]=boot(ab[~top1],clu[~top1]); D["ABS_mean3_oracle_miss"]=boot(ao[~top1],clu[~top1])
    ub=np.mean([uns(b) for b in base],0); uo=np.mean([uns(o) for o in orc],0)
    D["UNS_mean3_baseline_all"]=boot(ub,clu); D["UNS_mean3_oracle_all"]=boot(uo,clu)
    # hit effect with 3-run mean, by language
    lang=np.array([r["language"] for r in base[0]])
    D["hit_rate_mpnet"]=boot(top1.astype(float),clu)
    # MC chi-square tests on run1
    lab=np.array([r["faithfulness_label"] for r in base[0]]); ok=lab!="JUDGE_PARSE_FAILED"
    D["mc_chi2_hit_x_label_run1"]=mc_chi2(top1[ok].astype(int),lab[ok]); D["mc_chi2_lang_x_label_run1"]=mc_chi2(lang[ok],lab[ok])
    # other retrievers / generator
    for name in ["bm25","e5","mpnet_llama31"]:
        key=f"{d}_{name}"
        if key not in data: continue
        rows=data[key]; assert [r["id"] for r in rows]==ids
        t1=np.array([r["top1_hit"] for r in rows]); s=sup(rows)
        D[name]={"top1_hit":boot(t1.astype(float),clu),"SUP":boot(s,clu),"ABS":boot(abst(rows),clu),"UNS":boot(uns(rows),clu),
                 "SUP_hit":boot(s[t1],clu[t1]) if t1.any() else None,"SUP_miss":boot(s[~t1],clu[~t1]),
                 "parse_failed":int(sum(r["faithfulness_label"]=="JUDGE_PARSE_FAILED" for r in rows))}
        if name in("bm25","e5"):
            D[name]["hit_minus_mpnet"]=boot(t1.astype(float)-top1.astype(float),clu)
            D[name]["SUP_minus_mpnet_mean3"]=boot(s-bm,clu)
        if name=="mpnet_llama31":
            D[name]["SUP_minus_qwen_mean3"]=boot(s-bm,clu)
            D[name]["ABS_minus_qwen_mean3"]=boot(abst(rows)-ab,clu)
            D[name]["UNS_minus_qwen_mean3"]=boot(uns(rows)-ub,clu)
    # answer correctness vs gold (baseline run1, oracle run1, plus run-mean)
    for nm,rows in [("baseline_run1",base[0]),("oracle_run1",orc[0])]:
        c=np.array([correct(r) for r in rows],float); s=sup(rows)
        D[f"correct_{nm}"]={"correct_all":boot(c,clu),"correct_miss":boot(c[~top1],clu[~top1]),"correct_hit":boot(c[top1],clu[top1]),
            "SUP_and_correct":float((s*c).mean()),"SUP_not_correct":float((s*(1-c)).mean()),
            "SUP_not_correct_miss":float((s*(1-c))[~top1].mean()),"SUP_correct_miss":float((s*c)[~top1].mean()),
            "SUP_not_correct_share_of_SUP_miss":float((s*(1-c))[~top1].sum()/max(s[~top1].sum(),1))}
    out[d]=D
json.dump(out,open(os.path.join(R,"v2_analysis.json"),"w"),indent=1,ensure_ascii=False)
print("ok")

# ---- supplementary: 3-run means for correctness and baseline MPNet rows ----
sup2={}
for d in ["quran","wiki"]:
    base=[data[f"{d}_mpnet_run{i}"] for i in (1,2,3)]; orc=[data[f"{d}_oracle_run{i}"] for i in (1,2,3)]
    ids=[r["id"] for r in base[0]]; clu=np.array([cl(i) for i in ids]); top1=np.array([r["top1_hit"] for r in base[0]])
    S={}
    for nm,runs in (("baseline",base),("oracle",orc)):
        C=np.mean([[correct(r) for r in rr] for rr in runs],0)
        Sp=np.mean([[r["faithfulness_label"]=="SUPPORTED" for r in rr] for rr in runs],0)
        Ab=np.mean([[r["faithfulness_label"]=="ABSTAINED" for r in rr] for rr in runs],0)
        Un=np.mean([[r["faithfulness_label"]=="UNSUPPORTED" for r in rr] for rr in runs],0)
        S[nm]={"correct_all":boot(C,clu),"correct_miss":boot(C[~top1],clu[~top1]),"correct_hit":boot(C[top1],clu[top1]),
               "SUP_and_wrong_share_of_SUP_all":float((Sp*(1-C)).sum()/Sp.sum()),
               "SUP_and_wrong_share_of_SUP_miss":float((Sp*(1-C))[~top1].sum()/Sp[~top1].sum()),
               "ABS_all":boot(Ab,clu),"UNS_all":boot(Un,clu)}
    sup2[d]=S
out["supp"]=sup2
json.dump(out,open(os.path.join(R,"v2_analysis.json"),"w"),indent=1,ensure_ascii=False)
