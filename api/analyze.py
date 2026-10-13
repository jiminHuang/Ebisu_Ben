"""Recompute every Ebisu metric from per-sample model outputs and write LaTeX tables.

Inputs: API sample dirs (run_api.py: <dir>/<model>/{icr,te,krc}_samples.jsonl) and lm_eval sample dirs
(samples_ebisu_{jficr,jfte,jfkrc}_*.jsonl). All outputs are re-parsed here with one parser per task so that
API and vLLM models are scored identically.

Usage: python analyze.py --api DIR --lmeval DIR --data DIR --out DIR [--names names.json]
"""
import argparse, collections, glob, json, os, random, re, sys, unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "ebisu_oct26"))
import jp_utils  # noqa: E402

ICR_LABELS = ["-2", "-1", "0", "+1", "+2"]
KRC_LABELS = ["mitsubishi", "mitsui", "sumitomo", "fuyo", "sanwa", "dkb", "other"]
ICR_RE = re.compile(r"([+\-][012]|0)")
KRC_RE = re.compile(r"(mitsubishi|mitsui|sumitomo|fuyo|sanwa|dkb|other)", re.I)
THINK = re.compile(r"<think>.*?</think>", re.S)
FILES = {"icr": "jp-icr/JF-ICR_test_hf_ready_303.jsonl", "te": "jp-te/jf_te.jsonl",
         "krc": "JF-KRC-3/jf_krc_merged_release_compact_v2b.jsonl"}


# ---------------- loading ----------------
def load_api(root):
    models = {}
    for d in sorted(Path(root).glob("*")):
        if not d.is_dir():
            continue
        rec = {}
        for t in ("icr", "te", "krc"):
            f = d / f"{t}_samples.jsonl"
            if f.exists():
                rec[t] = {str(r["id"]): r["output"] for r in map(json.loads, open(f))}
        if rec:
            models[d.name] = rec
    return models


def load_lmeval(root):
    models = {}
    tmap = {"jficr": "icr", "jfte": "te", "jfkrc": "krc"}
    for f in sorted(glob.glob(f"{root}/**/samples_ebisu_*.jsonl", recursive=True)):
        m = re.search(r"samples_ebisu_(jficr|jfte|jfkrc)_", f)
        name = Path(f).parent.parent.name  # results/<model>[_tag]/<sanitized path>/samples_*.jsonl
        rec = models.setdefault(name, {})
        rec[tmap[m.group(1)]] = {str(r["doc"]["id"]): r["resps"][0][0] for r in map(json.loads, open(f))}  # latest file wins
    return models


# ---------------- parsing ----------------
def clean(o):
    if str(o).startswith("__ERROR__"):
        return ""
    o = THINK.sub("", o or "")
    try:  # schema-constrained outputs: {"terms": [...]} or {"label": "..."}
        d = json.loads(o)
        if isinstance(d, dict):
            return json.dumps(d["terms"], ensure_ascii=False) if "terms" in d else str(d.get("label", ""))
    except Exception:
        pass
    return o


def first(rx, o):
    m = rx.search(clean(o))
    return m.group(1).lower() if m else None


def norm_term(t):
    t = unicodedata.normalize("NFKC", str(t))
    return re.sub(r"[\s・･,，、.。()（）「」『』\"'“”]", "", t)


# ---------------- classification metrics ----------------
def cls_metrics(gold, pred, labels):
    n = len(gold)
    acc = sum(g == p for g, p in zip(gold, pred)) / n
    f1s, per = [], {}
    for l in labels:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per[l] = {"p": pr, "r": rc, "f1": f, "support": tp + fn}
        if tp + fn:  # macro over labels present in gold
            f1s.append(f)
    return {"acc": acc, "macro_f1": sum(f1s) / len(f1s), "invalid": sum(p is None for p in pred) / n, "per_label": per}


def icr_scores(gold, pred):
    m = cls_metrics(gold, pred, ICR_LABELS)
    # MAE on the ordinal scale; an unparseable output is scored as the farthest label from gold
    err = [abs(int(g) - int(p)) if p is not None else max(abs(int(g) + 2), abs(int(g) - 2)) for g, p in zip(gold, pred)]
    m["mae"] = sum(err) / len(err)
    m["confusion"] = {g: collections.Counter(p or "invalid" for gg, p in zip(gold, pred) if gg == g) for g in ICR_LABELS}
    return m


# ---------------- JF-TE metrics ----------------
def te_scores(gold_raw, outs):
    pairs = list(zip(gold_raw, outs))
    golds = [jp_utils._parse_term_structure(g) for g in gold_raw]
    preds = [jp_utils._parse_term_structure(clean(o)) for o in outs]
    invalid = sum(1 for o, p in zip(outs, preds) if not p and clean(o).strip() not in ("", "[]")) + \
              sum(1 for o in outs if not clean(o).strip())
    hr = jp_utils.term_hr_agg([(g, clean(o)) for g, o in pairs])

    def maxset(structs, gold, fn):
        s = set()
        for st in structs:
            for grp in st:
                if grp:
                    c = grp[0] if gold else max(grp, key=lambda x: len(jp_utils._normalize_term_text(x)))
                    c = fn(c)
                    if c:
                        s.add(c)
        return s

    def prf(G, P):
        tp = len(G & P)
        p = tp / len(P) if P else 0.0
        r = tp / len(G) if G else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    # maximal-term exact (corpus-level set, as in jp_utils) and normalized match
    ex = prf(maxset(golds, True, jp_utils._normalize_term_text), maxset(preds, False, jp_utils._normalize_term_text))
    nm = prf(maxset(golds, True, norm_term), maxset(preds, False, norm_term))

    # per-instance diagnostics: boundary vs missing errors, partial-overlap F1, all-term (nested) P/R/F1
    tp_o = n_g = n_p = 0
    nest_tp = nest_g = nest_p = 0
    err = collections.Counter()
    for g, p in zip(golds, preds):
        G = {jp_utils._normalize_term_text(grp[0]) for grp in g if grp}
        P = {jp_utils._normalize_term_text(max(grp, key=len)) for grp in p if grp}
        n_g += len(G); n_p += len(P)
        for x in G:
            if x in P:
                err["exact"] += 1; tp_o += 1
            elif any(x in y or y in x for y in P):
                err["boundary"] += 1; tp_o += 1
            else:
                err["missed"] += 1
        err["spurious"] += sum(1 for y in P if not any(x in y or y in x for x in G))
        GA = {jp_utils._normalize_term_text(t) for grp in g for t in grp}
        PA = {jp_utils._normalize_term_text(t) for grp in p for t in grp}
        nest_tp += len(GA & PA); nest_g += len(GA); nest_p += len(PA)
    po_p = tp_o / n_p if n_p else 0.0
    po_r = tp_o / n_g if n_g else 0.0
    np_, nr_ = (nest_tp / nest_p if nest_p else 0.0), (nest_tp / nest_g if nest_g else 0.0)
    return {"p": ex[0], "r": ex[1], "f1": ex[2], "norm_f1": nm[2],
            "overlap_f1": 2 * po_p * po_r / (po_p + po_r) if po_p + po_r else 0.0,
            "all_p": np_, "all_r": nr_, "all_f1": 2 * np_ * nr_ / (np_ + nr_) if np_ + nr_ else 0.0,
            "r1": hr["hr_at_1"], "r5": hr["hr_at_5"], "r10": hr["hr_at_10"],
            "invalid": invalid / len(outs), "errors": dict(err)}


# ---------------- baselines ----------------
def baselines(rows):
    out = {}
    g = [str(r["answer"]) for r in rows["icr"]]
    maj = collections.Counter(g).most_common(1)[0][0]
    rnd = random.Random(0)
    sims = [icr_scores(g, [rnd.choice(ICR_LABELS) for _ in g]) for _ in range(200)]
    out["icr"] = {"majority": icr_scores(g, [maj] * len(g)),
                  "random": {k: sum(s[k] for s in sims) / len(sims) for k in ("acc", "macro_f1", "mae")},
                  "dist": {l: g.count(l) for l in ICR_LABELS}}
    g = [str(r["answer"]).lower() for r in rows["krc"]]
    maj = collections.Counter(g).most_common(1)[0][0]
    sims = [cls_metrics(g, [rnd.choice(KRC_LABELS) for _ in g], KRC_LABELS) for _ in range(200)]
    out["krc"] = {"majority": cls_metrics(g, [maj] * len(g), KRC_LABELS),
                  "random": {k: sum(s[k] for s in sims) / len(sims) for k in ("acc", "macro_f1")},
                  "dist": {l: g.count(l) for l in KRC_LABELS}}
    # JF-TE composition
    n_max = n_nested = n_with_nest = 0
    terms = set()
    for r in rows["te"]:
        for grp in r["answer"]:
            n_max += 1; n_nested += len(grp) - 1; n_with_nest += len(grp) > 1; terms.update(grp)
    n = len(rows["te"])
    out["te"] = {"instances": n, "maximal": n_max, "nested": n_nested, "mentions": n_max + n_nested,
                 "unique_terms": len(terms), "max_per_inst": n_max / n, "nested_per_inst": n_nested / n,
                 "share_max_with_nested": n_with_nest / n_max}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True); ap.add_argument("--lmeval", required=True)
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = {t: [json.loads(l) for l in open(Path(a.data) / f)] for t, f in FILES.items()}
    models = {**load_lmeval(a.lmeval), **load_api(a.api)}
    res = {"baselines": baselines(rows), "models": {}}
    for name, rec in models.items():
        r = {}
        if "icr" in rec and len(rec["icr"]) == len(rows["icr"]):
            r["icr"] = icr_scores([str(x["answer"]) for x in rows["icr"]], [first(ICR_RE, rec["icr"][str(x["id"])]) for x in rows["icr"]])
        if "krc" in rec and len(rec["krc"]) == len(rows["krc"]):
            r["krc"] = cls_metrics([str(x["answer"]).lower() for x in rows["krc"]],
                                   [first(KRC_RE, rec["krc"][str(x["id"])]) for x in rows["krc"]], KRC_LABELS)
        if "te" in rec and len(rec["te"]) == len(rows["te"]):
            r["te"] = te_scores([x["answer"] for x in rows["te"]], [rec["te"][str(x["id"])] for x in rows["te"]])
        res["models"][name] = r
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(res, open(Path(a.out) / "metrics.json", "w"), indent=1, default=dict)
    for n, r in res["models"].items():
        print(f"{n:45s}", *(f"{t}:{k}={r[t][k]:.4f}" for t, k in [("icr", "acc"), ("icr", "macro_f1"), ("icr", "mae"), ("icr", "invalid"),
                                                              ("te", "f1"), ("te", "norm_f1"), ("te", "r10"), ("te", "invalid"),
                                                              ("krc", "acc"), ("krc", "macro_f1"), ("krc", "invalid")] if t in r))
    print(json.dumps(res["baselines"], indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
