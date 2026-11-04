"""Run the Python mlx-audio Nemotron Unified port over the verification manifest.

For each clip: offline `generate`, and at each latency both `stream_generate` (token ids)
and the live `create_streaming_session` path fed in 80 ms pushes (text, as a server sees
it). Records any exception instead of stopping.

usage: python mlx_run.py MODEL_DIR LABEL MANIFEST OUT.jsonl [--latencies 320,...] [--kinds ...]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_audio.stt.models.nemotron_asr.buffered import DELAY_CONTEXTS
from mlx_audio.stt.utils import load, load_audio


def result_tokens(result):
    return [t.id for s in result.sentences for t in s.tokens]


def session_text(model, samples, lang, latency, push=1280):
    s = model.create_streaming_session(language=lang, transcription_delay_ms=latency)
    pieces = []
    for i in range(0, len(samples), push):
        s.feed(samples[i : i + push])
        pieces += s.step(max_decode_tokens=64)
    s.close()
    while not s.done:
        pieces += s.step(max_decode_tokens=64)
    return "".join(pieces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("label")
    ap.add_argument("manifest")
    ap.add_argument("out")
    ap.add_argument("--latencies", default="")
    ap.add_argument("--kinds", default="")
    ap.add_argument("--no-session", action="store_true")
    args = ap.parse_args()
    root = Path(args.manifest).parent
    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    if args.kinds:
        rows = [r for r in rows if r["kind"] in args.kinds.split(",")]
    only = [int(v) for v in args.latencies.split(",") if v]
    t0 = time.time()
    model = load(args.model)
    print(f"load {time.time() - t0:.1f}s", file=sys.stderr)
    with open(args.out, "a", encoding="utf-8") as out:
        for r in rows:
            audio = load_audio(str(root / r["file"]), sr=16000)
            samples = np.array(audio, dtype=np.float32)
            jobs = (["offline"] if r.get("offline") else []) + [
                lat for lat in r["latencies"] if not only or lat in only
            ]
            for lat in jobs:
                rec = {"id": r["id"], "latency": lat, "impl": f"py-{args.label}", "error": None}
                t1 = time.time()
                try:
                    if lat == "offline":
                        res = model.generate(audio, language=r["lang"])
                        rec["tokens"], rec["text"] = result_tokens(res), res.text
                    else:
                        c, rc = DELAY_CONTEXTS[lat]
                        res = None
                        for res in model.stream_generate(audio, language=r["lang"], att_context_size=[70, c, rc]):
                            pass
                        rec["tokens"] = result_tokens(res) if res else []
                        rec["text"] = res.text if res else ""
                        rec["stream_seconds"] = round(time.time() - t1, 2)
                        if not args.no_session:
                            rec["session_text"] = session_text(model, samples, r["lang"], lat).strip()
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {e}"[:300]
                rec["seconds"] = round(time.time() - t1, 2)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                mx.clear_cache()
                print(r["id"], lat, rec["seconds"], rec["error"] or "", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
