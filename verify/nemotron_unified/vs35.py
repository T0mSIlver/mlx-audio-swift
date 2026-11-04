"""Nemotron Unified against Nemotron 3.5 in the Python port: same runner, clips and harness.

Per model: FLEURS WER offline and through the live session (80 ms pushes) at each latency
both models support; then one 2-minute English monologue through the live session at
each latency, timing every step and reading MLX's peak memory.

usage: python vs35.py MANIFEST OUT.json LABEL=MODEL[@JOBS] ...

JOBS is a comma list of latencies in ms and/or "offline" (default: offline,160,560,1120).
A cache-aware (3.5) checkpoint gets its latency by setting its default context to
[left, lookahead] directly: its session ignores transcription_delay_ms unless the config
says model_type nemotron_asr_streaming, which the mlx-community 3.5 conversions do not.
"""

import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_audio.stt.utils import load, load_audio

LATENCIES = [160, 560, 1120]
LOOKAHEAD_35 = {80: 0, 160: 1, 320: 3, 560: 6, 1120: 13}  # cache-aware: latency = (lookahead + 1) * 80 ms
PUSH = 1280  # 80 ms at 16 kHz


def words(text):
    text = unicodedata.normalize("NFKC", text or "").lower()
    return re.sub(r"[^\w\s']", " ", text).split()


def edits(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(per_clip, ids):
    ok = [per_clip[i] for i in ids if isinstance(per_clip.get(i), list)]
    return round(100 * sum(e for e, _ in ok) / max(1, sum(n for _, n in ok)), 2)


def live(model, samples, lang, latency):
    """Feed 80 ms pushes as a server would; return text, step times, first-text audio time."""
    if not model.buffered:
        model.default_att_context_size = [model.default_att_context_size[0], LOOKAHEAD_35[latency]]
    s = model.create_streaming_session(language=lang, transcription_delay_ms=latency if model.buffered else None)
    if not model.buffered:
        assert s.att_context_size[1] == LOOKAHEAD_35[latency], s.att_context_size
    pieces, steps, first = [], [], None
    for i in range(0, len(samples), PUSH):
        s.feed(samples[i : i + PUSH])
        t = time.perf_counter()
        out = s.step(max_decode_tokens=64)
        steps.append(time.perf_counter() - t)
        if out and first is None and "".join(out).strip():
            first = (i + PUSH) / 16000 + steps[-1]
        pieces += out
    s.close()
    while not s.done:
        pieces += s.step(max_decode_tokens=64)
    return "".join(pieces), steps, first


def main():
    manifest, out_path, specs = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    rows = [json.loads(l) for l in open(manifest, encoding="utf-8")]
    fleurs = [r for r in rows if r["kind"] == "fleurs"]
    mono = next(r for r in rows if r["id"] == "long-en-2min")
    result = {}
    for spec in specs:
        label, path = spec.split("=", 1)
        path, _, jobs = path.partition("@")
        jobs = [j if j == "offline" else int(j) for j in (jobs or "offline,160,560,1120").split(",")]
        lats = [j for j in jobs if j != "offline"]
        model = load(path)
        res = {"model": path, "wer": {}, "speed": {}}
        for lat in jobs:
            per_clip = {}
            for r in fleurs:
                audio = np.array(load_audio(str(manifest.parent / r["file"]), sr=16000), dtype=np.float32)
                try:
                    if lat == "offline":
                        text = model.generate(mx.array(audio), language=r["lang"]).text
                    else:
                        text = live(model, audio, r["lang"], lat)[0]
                except Exception as e:  # e.g. a language the checkpoint lacks
                    per_clip[r["id"]] = f"{type(e).__name__}: {e}"[:200]
                    continue
                hyp, gold = words(text), words(r["text"])
                per_clip[r["id"]] = [edits(hyp, gold), len(gold)]
                mx.clear_cache()
            res["wer"][str(lat)] = per_clip
            print(label, lat, wer(per_clip, per_clip), file=sys.stderr, flush=True)
        audio = np.array(load_audio(str(manifest.parent / mono["file"]), sr=16000), dtype=np.float32)
        live(model, audio[: 3 * 16000], "en", lats[0])  # warm-up
        for lat in lats:
            mx.clear_cache()
            mx.reset_peak_memory()
            _, steps, first = live(model, audio, "en", lat)
            ms = np.array(steps[38:]) * 1000  # drop the first 3 s
            res["speed"][str(lat)] = {
                "mean_step_ms": round(float(ms.mean()), 1),
                "p95_step_ms": round(float(np.percentile(ms, 95)), 1),
                "gpu_share_of_realtime": round(float(ms.mean()) / 80, 3),
                "first_text_s": round(first, 2) if first else None,
                "peak_memory_mb": round(mx.get_peak_memory() / 2**20),
            }
            print(label, lat, res["speed"][str(lat)], file=sys.stderr, flush=True)
        result[label] = res
        del model
        mx.clear_cache()
    # WER over the clips every model transcribed, so the rows compare like for like.
    for lat in sorted({l for r in result.values() for l in r["wer"]}):
        have = [r for r in result.values() if lat in r["wer"]]
        common = set.intersection(*({i for i, v in r["wer"][lat].items() if isinstance(v, list)} for r in have))
        for r in have:
            r.setdefault("wer_common", {})[lat] = {"wer": wer(r["wer"][lat], common), "clips": len(common)}
    json.dump(result, open(out_path, "w"), indent=1)
    print(json.dumps({k: {"wer_common": v["wer_common"], "speed": v["speed"]} for k, v in result.items()}, indent=1))


if __name__ == "__main__":
    main()
