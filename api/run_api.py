"""Concurrent API evaluation of Ebisu (jp-icr 303, jp-te 254, JF-KRC-3 346) through an OpenAI-compatible gateway.

Mirrors the lm_eval setup in tasks/ebisu_oct26: the query is sent as a single user message, greedy decoding,
max tokens 512 (ICR) / 1024 (TE, KRC), first-match regex parsing for ICR/KRC and jp_utils metrics for TE.
Usage: python run_api.py --model gpt-4o --data DIR --out DIR [--tasks icr,te,krc] [--conc 32]
Key: GATEWAY_KEY env var; base URL: GATEWAY_BASE_URL (default https://endpoint.greatrouter.com/v1).
"""
import argparse, asyncio, json, os, re, sys, time
from pathlib import Path
import openai

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "ebisu_oct26"))
import jp_utils  # noqa: E402

FILES = {"icr": "jp-icr/JF-ICR_test_hf_ready_303.jsonl", "te": "jp-te/jf_te.jsonl",
         "krc": "JF-KRC-3/jf_krc_merged_release_compact_v2b.jsonl"}
MAXTOK = {"icr": 512, "te": 1024, "krc": 1024}
ICR_RE = re.compile(r"([+\-][012]|0)")
KRC_RE = re.compile(r"(mitsubishi|mitsui|sumitomo|fuyo|sanwa|dkb|other)", re.I)
REASONING = re.compile(r"^(gpt-5|o\d)")


async def call(client, model, prompt, max_tok, sem):
    kw = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if REASONING.match(model):
        kw["max_completion_tokens"] = max_tok
    else:
        kw.update(temperature=0.0, max_tokens=max_tok)
    async with sem:
        for attempt in range(6):
            try:
                r = await client.chat.completions.create(**kw)
                return r.choices[0].message.content or "", r.usage.model_dump() if r.usage else {}
            except Exception as e:  # rate limits / transient gateway errors
                if attempt == 5:
                    return f"__ERROR__ {type(e).__name__}: {e}", {}
                await asyncio.sleep(2 ** attempt)


def first(rx, text):
    m = rx.search(text or "")
    return m.group(1).lower() if m else ""


def score(task, rows, outs):
    if task == "te":
        pairs = [(r["answer"], o) for r, o in zip(rows, outs)]
        hr = jp_utils.term_hr_agg(pairs)
        return {"f1": round(jp_utils.term_max_f1_agg(pairs), 4), **hr}
    rx = ICR_RE if task == "icr" else KRC_RE
    preds = [first(rx, o) for o in outs]
    acc = sum(p == str(r["answer"]).lower() for p, r in zip(preds, rows)) / len(rows)
    return {"acc": round(acc, 4), "unparsed": sum(p == "" for p in preds)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="icr,te,krc"); ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=0, help="override all per-task budgets (reasoning models)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    client = openai.AsyncOpenAI(base_url=os.environ.get("GATEWAY_BASE_URL", "https://endpoint.greatrouter.com/v1"),
                                api_key=os.environ["GATEWAY_KEY"], timeout=600, max_retries=0)
    sem = asyncio.Semaphore(a.conc)
    out = Path(a.out) / (a.model + (f"_{a.tag}" if a.tag else ""));
    if a.max_tokens: MAXTOK.update({k: a.max_tokens for k in MAXTOK})
    out.mkdir(parents=True, exist_ok=True)
    tasks = a.tasks.split(",")
    rows = {t: [json.loads(l) for l in open(Path(a.data) / FILES[t])] for t in tasks}
    t0 = time.time()
    res = await asyncio.gather(*[asyncio.gather(*[call(client, a.model, r["query"], MAXTOK[t], sem) for r in rows[t]]) for t in tasks])
    summary = {"model": a.model, "seconds": round(time.time() - t0)}
    for t, rr in zip(tasks, res):
        outs = [o for o, _ in rr]
        with open(out / f"{t}_samples.jsonl", "w") as f:
            for r, (o, u) in zip(rows[t], rr):
                f.write(json.dumps({"id": r["id"], "answer": r["answer"], "output": o, "usage": u}, ensure_ascii=False) + "\n")
        summary[t] = {**score(t, rows[t], outs), "errors": sum(o.startswith("__ERROR__") for o in outs)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))

asyncio.run(main())
