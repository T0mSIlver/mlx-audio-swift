"""Score the LFM3 LiveASR spike: WER of LFM3 live, LFM3 offline and NeMo Unified 320 ms
on the same English clips, plus LiveASR's chunk timing.

usage: python score.py ROOT   (ROOT holds clips/manifest.jsonl and the *.jsonl results)
"""

import glob
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nemotron_unified"))
from compare import edits, norm_words  # noqa: E402


def main():
    root = Path(sys.argv[1])
    manifest = {r["id"]: r for r in map(json.loads, open(root / "clips/manifest.jsonl", encoding="utf-8"))}
    hyps = {}  # arm -> id -> text
    live = {}
    for f in glob.glob(str(root / "**/*.jsonl"), recursive=True):
        if f.endswith("manifest.jsonl"):
            continue
        for rec in map(json.loads, open(f, encoding="utf-8")):
            if rec.get("impl") == "lfm3-live":
                hyps.setdefault("LFM3 live", {})[rec["id"]] = rec["text"]
                if rec.get("offline_text") is not None:
                    hyps.setdefault("LFM3 offline", {})[rec["id"]] = rec["offline_text"]
                live[rec["id"]] = rec
            elif rec.get("impl") == "nemo" and rec.get("latency") == 320:
                hyps.setdefault("Unified 320 ms (NeMo)", {})[rec["id"]] = rec["text"]

    print("| Arm | FLEURS en clips | WER | 2-min monologue WER |")
    print("|---|---|---|---|")
    for arm, by_id in sorted(hyps.items()):
        e = w = 0
        n = 0
        for cid, text in by_id.items():
            m = manifest[cid]
            if m["kind"] != "fleurs":
                continue
            ref = norm_words(m["text"])
            e += edits(norm_words(text), ref)
            w += len(ref)
            n += 1
        longs = [(cid, t) for cid, t in by_id.items() if manifest[cid]["kind"] == "long"]
        lw = "-"
        if longs:
            ref = norm_words(manifest[longs[0][0]]["text"])
            lw = f"{100 * edits(norm_words(longs[0][1]), ref) / len(ref):.2f} %"
        print(f"| {arm} | {n} | {100 * e / max(w, 1):.2f} % | {lw} |")

    if live:
        compute = [d["compute_s"] for r in live.values() for d in r["deltas"]]
        spans = []
        first = []
        empty = 0
        for r in live.values():
            prev = 0.0
            for d in r["deltas"]:
                spans.append(d["audio_s"] - prev)
                prev = d["audio_s"]
            nonempty = [d for d in r["deltas"] if d["text"].strip()]
            if nonempty:
                first.append(nonempty[0]["audio_s"] + nonempty[0]["compute_s"])
            empty += sum(1 for d in r["deltas"] if not d["text"].strip())
        glued = sum(
            1 for r in live.values() for a, b in zip(r["deltas"], r["deltas"][1:])
            if a["text"] and b["text"] and not a["text"][-1].isspace() and not b["text"][0].isspace()
            and a["text"][-1].isalnum() and b["text"][0].isalnum()
        )
        print()
        print(f"LiveASR: {len(compute)} chunk calls over {len(live)} clips; audio between calls "
              f"median {statistics.median(spans):.2f} s; first text at audio+compute median "
              f"{statistics.median(first):.2f} s (CPU runner); empty chunks {empty}; "
              f"chunk pairs joined mid-word with no space {glued}; CPU compute per chunk median "
              f"{statistics.median(compute):.2f} s (not a Mac number); "
              f"text-context restarts {sum(r['restarts'] for r in live.values())}.")


if __name__ == "__main__":
    main()
