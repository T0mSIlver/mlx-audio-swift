"""English clip set for the LFM3-Audio LiveASR spike (LVX-13): public FLEURS en_us test audio.

N clips of 4-20 s, plus a 2-minute monologue (FLEURS sentences after the first N, 0.3 s gaps)
that crosses LiveASR's 30 s text-context restart four times. Manifest rows match
../nemotron_unified so nemo_ref.py reads them (kind fleurs/long, latencies [320]).

usage: python clips_en.py OUT [--n 60]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nemotron_unified"))
from make_clips import SR, fleurs, write  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cid, audio, text in fleurs("en_us", args.n, 4.0, 20.0):
        write(out, rows, f"en-{cid}", audio, "en", text, "fleurs", [320], offline=True)
    parts, texts, total = [], [], 0
    gap = np.zeros(int(0.3 * SR), dtype=np.float32)
    for _, audio, text in fleurs("en_us", 0, 4.0, 20.0, skip=args.n):
        parts += [audio.astype(np.float32), gap]
        texts.append(text)
        total += len(audio) + len(gap)
        if total >= 120 * SR:
            break
    write(out, rows, "long-en-2min", np.concatenate(parts), "en", " ".join(texts), "long", [320], offline=False)
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "clips", file=sys.stderr)


if __name__ == "__main__":
    main()
