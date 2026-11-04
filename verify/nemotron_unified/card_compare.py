"""Tables for the model card check: per language (explicit prompt) and `auto`.

usage: python card_compare.py RUN_DIR [--first N]   (RUN_DIR holds clips/manifest.jsonl and *.jsonl)

Per language and latency: clips whose tokens equal NeMo's (offline: text, for the Swift
driver), and WER against FLEURS for NeMo and each port. For `auto`: parity with NeMo's
`auto`, WER with `auto` vs the explicit prompt, any `<..>` tag in the text, and how often
the first N words differ from the explicit prompt's first N words (printed for reading).
"""

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

from compare import edits, norm_words

LATS = ["160", "320", "1120", "offline"]


def load(root):
    manifest = {r["id"]: r for r in map(json.loads, open(root / "clips/manifest.jsonl", encoding="utf-8"))}
    recs = defaultdict(dict)  # (id, latency) -> impl -> rec
    for f in glob.glob(str(root / "**/*.jsonl"), recursive=True):
        if f.endswith("manifest.jsonl"):
            continue
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            recs[(r["id"], str(r["latency"]))][r["impl"]] = r
    return manifest, recs


def same(a, n):
    if a.get("error") or n.get("error"):
        return False
    if "tokens" in a and "tokens" in n:
        return a["tokens"] == n["tokens"]
    return (a.get("text") or "").strip() == (n.get("text") or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--first", type=int, default=3)
    args = ap.parse_args()
    manifest, recs = load(Path(args.root))
    impls = sorted({i for v in recs.values() for i in v if i != "nemo"})
    langs = sorted({r.get("true_lang", r["lang"]) for r in manifest.values()})
    print(f"{len(manifest)} rows, {len(recs)} (row, latency) pairs, impls nemo + {impls}")
    errs = [(k, i, v[i]["error"][:100]) for k, v in recs.items() for i in v if v[i].get("error")]
    print(f"errors: {len(errs)} {errs[:10]}")
    missing = [(cid, lat) for cid, r in manifest.items()
               for lat in [str(x) for x in r["latencies"]] + (["offline"] if r.get("offline") else []) + list(r.get("contexts", {}))
               if "nemo" not in recs.get((cid, lat), {})]
    print(f"rows without a NeMo result: {len(missing)} {missing[:8]}")

    def table(kind, title):
        print(f"\n## {title}: tokens identical to NeMo per impl (n/N over {LATS}), WER NeMo | impl...")
        tot = defaultdict(lambda: [0, 0])
        wer = defaultdict(lambda: [0, 0])
        for lang in langs:
            cells = []
            for lat in LATS:
                ident = defaultdict(lambda: [0, 0])
                for cid, r in manifest.items():
                    if r["kind"] != kind or r.get("true_lang", r["lang"]) != lang:
                        continue
                    v = recs.get((cid, lat), {})
                    if "nemo" not in v:
                        continue
                    ref = norm_words(r["text"])
                    for impl in ["nemo"] + impls:
                        if impl in v:
                            e = edits(norm_words(v[impl].get("text")), ref)
                            for key in ((lang, lat, impl), ("all", lat, impl)):
                                wer[key][0] += e
                                wer[key][1] += len(ref)
                        if impl != "nemo" and impl in v:
                            s = same(v[impl], v["nemo"])
                            ident[impl][0] += s
                            ident[impl][1] += 1
                            tot[(lat, impl)][0] += s
                            tot[(lat, impl)][1] += 1
                w = " | ".join(f"{100 * wer[(lang, lat, i)][0] / max(wer[(lang, lat, i)][1], 1):.1f}"
                               for i in ["nemo"] + impls if wer[(lang, lat, i)][1])
                cells.append(f"{lat}: " + " ".join(f"{ident[i][0]}/{ident[i][1]}" for i in impls if ident[i][1]) + f" WER {w}")
            print(f"  {lang}: " + "; ".join(cells))
        for lat in LATS:
            w = " | ".join(f"{i} {100 * wer[('all', lat, i)][0] / max(wer[('all', lat, i)][1], 1):.2f}"
                           for i in ["nemo"] + impls if wer[("all", lat, i)][1])
            print(f"  all {lat}: identical " + ", ".join(f"{i} {tot[(lat, i)][0]}/{tot[(lat, i)][1]}" for i in impls) + f"; WER {w}")

    table("fleurs", "Explicit prompt (bare code)")
    table("auto", "auto prompt")
    table("locale", "Locale prompt (es-ES, pt-BR)")

    print("\n## auto: leading tags, first-word divergence from the explicit prompt (NeMo, and each impl vs its own explicit)")
    for impl in ["nemo"] + impls:
        tags = [(cid, lat) for (cid, lat), v in recs.items() if impl in v and "<" in (v[impl].get("text") or "")]
        print(f"  {impl}: texts containing '<': {len(tags)} {tags[:5]}")
    n = args.first
    for impl in ["nemo"] + impls:
        for lat in LATS:
            diff, total, by_lang, worse = 0, 0, defaultdict(int), 0
            shown = []
            for cid, r in manifest.items():
                if r["kind"] != "auto":
                    continue
                base = cid[len("auto-"):]
                a = recs.get((cid, lat), {}).get(impl)
                x = recs.get((base, lat), {}).get(impl)
                if not a or not x:
                    continue
                fa, fx, fr = norm_words(a.get("text"))[:n], norm_words(x.get("text"))[:n], norm_words(r["text"])[:n]
                total += 1
                if fa != fx:
                    diff += 1
                    by_lang[r["true_lang"]] += 1
                    worse += edits(fa, fr) > edits(fx, fr)
                    shown.append(f"      {r['true_lang']} {base}: auto {' '.join(fa)!r} | explicit {' '.join(fx)!r} | ref {' '.join(fr)!r}")
            if total:
                print(f"  {impl} {lat}: first {n} words differ {diff}/{total} (auto further from the reference in {worse}); by language {dict(sorted(by_lang.items()))}")
                if impl == "nemo" and lat in ("160", "320"):
                    print("\n".join(shown))

    print("\n## odd latencies and left contexts (explicit prompt): tokens identical to NeMo")
    for impl in impls:
        cnt = defaultdict(lambda: [0, 0])
        for (cid, lat), v in recs.items():
            if lat in LATS or impl not in v or "nemo" not in v:
                continue
            cnt[lat][0] += same(v[impl], v["nemo"])
            cnt[lat][1] += 1
        print(f"  {impl}: " + ", ".join(f"{lat} {a}/{b}" for lat, (a, b) in sorted(cnt.items())))
    wer = defaultdict(lambda: [0, 0])
    for (cid, lat), v in recs.items():
        if "nemo" in v and manifest[cid]["kind"] == "fleurs" and manifest[cid].get("contexts"):
            ref = norm_words(manifest[cid]["text"])
            wer[lat][0] += edits(norm_words(v["nemo"]["text"]), ref)
            wer[lat][1] += len(ref)
    print("  NeMo WER on the first clip of each language: " + ", ".join(
        f"{lat} {100 * a / max(b, 1):.2f}" for lat, (a, b) in sorted(wer.items())))

    print("\n## non-token-identical pairs (impl, row, latency, NeMo | impl)")
    for (cid, lat), v in sorted(recs.items()):
        for impl in impls:
            if impl in v and "nemo" in v and not same(v[impl], v["nemo"]):
                print(f"  {impl} {cid} {lat}\n     nemo: {(v['nemo'].get('text') or '')[:120]}\n     port: {(v[impl].get('text') or v[impl].get('error') or '')[:120]}")


if __name__ == "__main__":
    main()
