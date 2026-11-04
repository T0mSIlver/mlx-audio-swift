"""Compare NeMo, Python-port and Swift-port outputs from the verification run.

usage: python compare.py RUN_DIR   (RUN_DIR holds clips/manifest.jsonl and the *.jsonl artifacts)

Prints, per implementation and quantization: token-exact agreement with NeMo by latency
and by clip kind, token edit rate, WER against FLEURS references (NeMo's and the port's),
session/delta text consistency, errors; then the worst disagreements.
"""

import glob
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path


def edits(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def norm_words(text):
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return text.split()


def main():
    root = Path(sys.argv[1])
    manifest = {r["id"]: r for r in map(json.loads, open(root / "clips/manifest.jsonl", encoding="utf-8"))}
    recs = defaultdict(dict)  # (id, latency) -> impl -> rec
    for f in glob.glob(str(root / "**/*.jsonl"), recursive=True):
        if f.endswith("manifest.jsonl"):
            continue
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            recs[(r["id"], str(r["latency"]))][r["impl"]] = r
    impls = sorted({i for v in recs.values() for i in v if i != "nemo"})
    print(f"{len(recs)} (clip, latency) pairs; implementations: nemo + {impls}")
    nemo_pairs = sum(1 for v in recs.values() if "nemo" in v)
    print(f"NeMo reference present for {nemo_pairs} pairs; NeMo errors: "
          f"{[(k, v['nemo']['error']) for k, v in recs.items() if 'nemo' in v and v['nemo'].get('error')][:5]}")

    worst = []
    for impl in impls:
        by_lat = defaultdict(lambda: [0, 0, 0, 0])  # exact, total, token edits, nemo tokens
        by_kind = defaultdict(lambda: [0, 0])
        text_only = defaultdict(lambda: [0, 0])
        errs, incons = [], []
        wer_impl = defaultdict(lambda: [0, 0])
        wer_nemo = defaultdict(lambda: [0, 0])
        for (cid, lat), v in recs.items():
            if impl not in v:
                continue
            r = v[impl]
            row = manifest.get(cid, {})
            if r.get("error"):
                errs.append((cid, lat, r["error"][:80]))
                continue
            for key in ("session_text", "delta_text"):
                if key in r and r[key].strip() != (r.get("text") or "").strip():
                    incons.append((cid, lat, key))
            if row.get("text") and row["kind"] in ("fleurs", "long", "clipped", "padded", "music") and lat != "offline":
                ref = norm_words(row["text"])
                wer_impl[lat][0] += edits(norm_words(r.get("text")), ref)
                wer_impl[lat][1] += len(ref)
                if "nemo" in v:
                    wer_nemo[lat][0] += edits(norm_words(v["nemo"]["text"]), ref)
                    wer_nemo[lat][1] += len(ref)
            if "nemo" not in v or v["nemo"].get("error"):
                continue
            n = v["nemo"]
            if "tokens" in r:
                same = r["tokens"] == n["tokens"]
                e = edits(r["tokens"], n["tokens"])
                b = by_lat[lat]
                b[0] += same
                b[1] += 1
                b[2] += e
                b[3] += max(len(n["tokens"]), 1)
                k = by_kind[row.get("kind", "?")]
                k[0] += same
                k[1] += 1
                if not same:
                    worst.append((e / max(len(n["tokens"]), 1), impl, cid, lat, n["text"][:90], r.get("text", "")[:90]))
            else:
                t = text_only[lat]
                t[0] += (r.get("text") or "").strip() == (n.get("text") or "").strip()
                t[1] += 1
        print(f"\n== {impl}")
        for lat in sorted(by_lat, key=lambda x: (x == "offline", int(x) if x.isdigit() else 0)):
            b = by_lat[lat]
            print(f"  latency {lat:>7}: tokens identical to NeMo {b[0]}/{b[1]}, token edit rate {b[2] / b[3]:.4f}")
        for lat, t in sorted(text_only.items()):
            print(f"  latency {lat:>7}: text identical to NeMo {t[0]}/{t[1]}")
        print("  by kind: " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in sorted(by_kind.items())))
        if wer_impl:
            print("  WER vs FLEURS (port | NeMo): " + ", ".join(
                f"{lat} {wer_impl[lat][0] / max(wer_impl[lat][1], 1):.3f} | {wer_nemo[lat][0] / max(wer_nemo[lat][1], 1):.3f}"
                for lat in sorted(wer_impl, key=lambda x: int(x) if x.isdigit() else 0)))
        print(f"  errors: {len(errs)} {errs[:6]}")
        print(f"  session/delta text != final text: {len(incons)} {incons[:6]}")
    print("\n== worst token disagreements (edit rate, impl, clip, latency, NeMo | port)")
    for w in sorted(worst, reverse=True)[:25]:
        print(f"  {w[0]:.3f} {w[1]} {w[2]} {w[3]}\n     nemo: {w[4]}\n     port: {w[5]}")


if __name__ == "__main__":
    main()
