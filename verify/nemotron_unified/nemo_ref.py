"""NeMo reference for Nemotron Unified: offline and buffered streaming token ids.

The streaming loop is NeMo's examples/asr/asr_chunked_inference/rnnt/
speech_to_text_streaming_infer_rnnt.py at batch 1 (same StreamingBatchedAudioBuffer,
same model forward, same slicing); the chunk frames are then decoded greedily in one
pass, which equals the script's stateful per-chunk greedy decoding.

usage: python nemo_ref.py NEMO MANIFEST OUT.jsonl --shard i/n [--ids a,b] [--latencies 320,...]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import librosa
import torch

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.utils.streaming_utils import ContextSize, StreamingBatchedAudioBuffer

# latency ms -> (chunk, right) encoder frames, NVIDIA's table; left is 5.6 s (70 frames).
CONTEXTS = {160: (1, 1), 240: (1, 2), 320: (1, 3), 560: (2, 5), 1120: (7, 7), 2080: (13, 13)}
LEFT = 70


def context_for(latency):
    """(chunk, right) for any latency: the table row at or below it gives the chunk,
    the rest is right context (the ports' rule; NeMo itself takes both as inputs)."""
    if latency in CONTEXTS:
        return CONTEXTS[latency]
    frames = min(max(int(latency / 80 + 0.5), 2), 26)
    chunk = CONTEXTS[max(ms for ms in CONTEXTS if ms <= frames * 80)][0]
    return chunk, frames - chunk


def tokens_of(hyp):
    seq = hyp.y_sequence
    return seq.tolist() if hasattr(seq, "tolist") else list(seq)


def decode(m, frames):
    if frames.shape[1] == 0:
        return []
    x = frames.transpose(1, 2)
    hyp = m.decoding.rnnt_decoder_predictions_tensor(
        encoder_output=x, encoded_lengths=torch.tensor([frames.shape[1]])
    )[0]
    return tokens_of(hyp)


def offline(m, audio, prompt):
    m.encoder.set_default_att_context_size([-1, -1, -1])
    enc, enc_len = m(
        input_signal=audio, input_signal_length=torch.tensor([audio.shape[1]]), language_prompt_indices=prompt
    )
    return decode(m, enc[:, :, : int(enc_len[0])].transpose(1, 2))


def streaming(m, audio, prompt, chunk, right, left=LEFT):
    sub = m.encoder.subsampling_factor
    f2s = 160 * sub
    ctx = ContextSize(left=left * f2s, chunk=chunk * f2s, right=right * f2s)
    m.encoder.set_default_att_context_size([left, chunk, right])
    n = audio.shape[1]
    buf = StreamingBatchedAudioBuffer(batch_size=1, context_samples=ctx, dtype=audio.dtype, device="cpu")
    rest = torch.tensor([n])
    left_sample, right_sample = 0, min(ctx.chunk + ctx.right, n)
    chunks, steps = [], 0
    while left_sample < n:
        chunk_len = min(right_sample, n) - left_sample
        is_last_b = torch.tensor([chunk_len]) >= rest
        is_last = right_sample >= n
        lens = torch.where(is_last_b, rest, torch.full_like(rest, chunk_len))
        buf.add_audio_batch_(
            audio[:, left_sample:right_sample], audio_lengths=lens, is_last_chunk=is_last, is_last_chunk_batch=is_last_b
        )
        enc, enc_len = m(
            input_signal=buf.samples,
            input_signal_length=buf.context_size_batch.total(),
            language_prompt_indices=prompt,
        )
        enc = enc.transpose(1, 2)
        ec = buf.context_size.subsample(factor=f2s)
        ecb = buf.context_size_batch.subsample(factor=f2s)
        k = int((enc_len - ecb.left)[0]) if bool(is_last_b[0]) else int(ecb.chunk[0])
        chunks.append(enc[:, ec.left : ec.left + k])
        steps += 1
        rest -= lens
        left_sample = right_sample
        right_sample = min(right_sample + ctx.chunk, n)
    frames = torch.cat(chunks, dim=1) if chunks else torch.zeros(1, 0, enc.shape[-1])
    return decode(m, frames), steps, frames.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nemo")
    ap.add_argument("manifest")
    ap.add_argument("out")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--kinds", default="")
    ap.add_argument("--ids", default="")
    ap.add_argument("--latencies", default="")
    ap.add_argument("--contexts", default="", help="NAME:L,C,R;... run these contexts instead of the manifest latencies")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(os.cpu_count() or 4)
    root = Path(args.manifest).parent
    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    if args.kinds:
        rows = [r for r in rows if r["kind"] in args.kinds.split(",")]
    if args.ids:
        rows = [r for r in rows if r["id"] in args.ids.split(",")]
    i, n = (int(v) for v in args.shard.split("/"))
    # Interleave by cost so shards finish together.
    rows = sorted(rows, key=lambda r: -r["seconds"])[i::n]
    only = [int(v) for v in args.latencies.split(",") if v]
    alt = {}
    for spec in [x for x in args.contexts.split(";") if x]:
        name, ctx = spec.split(":")
        alt[name] = tuple(int(v) for v in ctx.split(","))

    m = ASRModel.restore_from(args.nemo, map_location="cpu")
    m.eval()
    m.preprocessor.featurizer.dither = 0.0
    m.preprocessor.featurizer.pad_to = 0
    vocab = m.tokenizer

    with open(args.out, "a", encoding="utf-8") as out:
        for r in rows:
            audio, _ = librosa.load(str(root / r["file"]), sr=16000)  # NeMo's own loader: mono, soxr
            audio = torch.tensor(audio, dtype=torch.float32)[None]
            prompt = torch.tensor([m.resolve_prompt_language(r["lang"])], dtype=torch.long)
            if alt:
                jobs = list(alt.items())
            else:
                jobs = [("offline", None)] if r.get("offline") else []
                jobs += [(lat, (LEFT,) + context_for(lat)) for lat in r["latencies"] if not only or lat in only]
                jobs += [(name, tuple(ctx)) for name, ctx in r.get("contexts", {}).items()]
            for lat, cr in jobs:
                t0 = time.time()
                steps = frames = None
                err = None
                try:
                    if lat == "offline":
                        toks = offline(m, audio, prompt)
                    else:
                        left, chunk, right = cr
                        toks, steps, frames = streaming(m, audio, prompt, chunk, right, left=left)
                except Exception as e:  # the reference itself may reject degenerate input
                    toks, err = [], f"{type(e).__name__}: {e}"[:300]
                rec = {
                    "id": r["id"],
                    "latency": lat,
                    "impl": "nemo",
                    "tokens": toks,
                    "text": vocab.ids_to_text(toks) if toks else "",
                    "steps": steps,
                    "frames": frames,
                    "seconds": round(time.time() - t0, 2),
                    "error": err,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                print(r["id"], lat, rec["seconds"], "s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
