"""Profile one steady-state Nemotron Unified streaming step, phase by phase.

MLX is lazy: every phase's output is evaluated at its own boundary, so each timer
charges only its own work. Steady state = a full window (left + chunk + right).
Each arm prints ms per step and ms of GPU per second of audio (steps/s x step).

usage: python profile_step.py MODEL_DIR [MODEL_DIR ...]
"""

import json
import sys
import time

import mlx.core as mx
import numpy as np

from mlx_audio.stt.models.nemotron_asr.audio import log_mel_spectrogram
from mlx_audio.stt.models.nemotron_asr.conformer import NEG_INF, create_chunked_rc_visibility
from mlx_audio.stt.utils import load

FS = 1280
REPS = 20


def timed(fn, reps=REPS, warm=4):
    for _ in range(warm):
        mx.eval(fn())
    t = time.perf_counter()
    for _ in range(reps):
        mx.eval(fn())
    return (time.perf_counter() - t) / reps * 1000


def encoder_phases(m, mel, ctx, dtype):
    enc = m.encoder
    lengths = mx.array([mel.shape[1] - 1] * mel.shape[0])
    x0 = mel.astype(dtype)
    out = {}
    out["subsampling"] = timed(lambda: enc.pre_encode(x0, lengths)[0])
    x, ol = enc.pre_encode(x0, lengths)
    x, pos = enc.pos_enc(x)
    mx.eval(x, pos)
    seq = x.shape[1]
    valid = mx.arange(seq)[None, :] < ol[:, None]
    vis = valid[:, None, None, :] & create_chunked_rc_visibility(seq, *ctx)[None, None]
    mask = mx.where(vis, 0.0, NEG_INF).astype(x.dtype)
    mx.eval(mask)
    layer = enc.layers[0]
    out["one_layer"] = timed(lambda: layer(x, pos_emb=pos, mask=mask, valid=valid))
    y = layer.norm_feed_forward1(x)
    out["  ff1"] = timed(lambda: layer.feed_forward1(layer.norm_feed_forward1(x)))
    out["  attn"] = timed(lambda: layer.self_attn(layer.norm_self_att(x), pos_emb=pos, mask=mask))
    out["  conv"] = timed(lambda: layer.conv(layer.norm_conv(x), valid=valid))
    out["  ff2"] = timed(lambda: layer.feed_forward2(layer.norm_feed_forward2(x)))

    def all_layers():
        h = x
        for l in enc.layers:
            h = l(h, pos_emb=pos, mask=mask, valid=valid)
        return h

    out["24_layers"] = timed(all_layers)
    return out, x


def step_fn(m, samples, ctx, dtype, lang="en", compiled=None):
    left = ctx[0]

    def run():
        mel = log_mel_spectrogram(mx.array(samples), m.preprocessor_config).astype(dtype)
        n = samples.shape[-1] // 160
        if compiled is not None:
            enc = compiled(mel)
        else:
            enc, _ = m.encoder(mel, lengths=mx.array([n]), att_context_size=list(ctx))
        return m.apply_prompt(enc[:, left : left + ctx[1]], lang)

    return run


def main():
    rng = np.random.default_rng(0)
    results = []
    for model_dir in sys.argv[1:]:
        t = time.perf_counter()
        m = load(model_dir)
        mx.eval(m.parameters())
        load_s = time.perf_counter() - t
        label = model_dir.rstrip("/").split("/")[-1]
        for dname, dtype in (("f32-act", mx.float32), ("bf16-act", mx.bfloat16)):
            arms = [
                ("320ms L70 C1 R3", (70, 1, 3), 1),
                ("320ms L70 C2 R2", (70, 2, 2), 2),
                ("320ms L40 C1 R3", (40, 1, 3), 1),
                ("320ms L25 C1 R3", (25, 1, 3), 1),
                ("160ms L70 C1 R1", (70, 1, 1), 1),
                ("560ms L70 C2 R5", (70, 2, 5), 2),
                ("1120ms L70 C7 R7", (70, 7, 7), 7),
            ]
            for name, ctx, chunk_frames in arms:
                n = sum(ctx) * FS
                samples = rng.normal(0, 0.05, n).astype(np.float32)
                ms = timed(step_fn(m, samples, ctx, dtype))
                row = {
                    "model": label,
                    "act": dname,
                    "arm": name,
                    "step_ms": round(ms, 2),
                    "gpu_ms_per_audio_s": round(ms * 12.5 / chunk_frames, 1),
                    "load_s": round(load_s, 1),
                }
                if name == "320ms L70 C1 R3":
                    mel = log_mel_spectrogram(mx.array(samples), m.preprocessor_config)
                    row["mel_ms"] = round(timed(lambda: log_mel_spectrogram(mx.array(samples), m.preprocessor_config)), 2)
                    phases, x = encoder_phases(m, mel, ctx, dtype)
                    row.update({k: round(v, 2) for k, v in phases.items()})
                    enc, _ = m.encoder(mel.astype(dtype), lengths=mx.array([n // 160]), att_context_size=list(ctx))
                    chunk = m.apply_prompt(enc[:, 70:71], "en")
                    mx.eval(chunk)
                    row["prompt_ms"] = round(timed(lambda: m.apply_prompt(enc[:, 70:71], "en")), 2)
                    # one greedy RNN-T frame with 2 symbols: decoder + joint, synced per token
                    def decode_frame():
                        out, hidden = m.decoder(None, None)
                        for tok in range(2):
                            logits = m.joint(chunk[:, :1], out.astype(chunk.dtype))
                            k = int(mx.argmax(logits))
                            out, hidden = m.decoder(mx.array([[k % 4000]], dtype=mx.int32), hidden)
                        return out
                    row["decode_2_tokens_ms"] = round(timed(decode_frame), 2)
                    # two windows batched vs one at a time
                    two = np.stack([samples, samples])
                    def batched():
                        mels = mx.concatenate([log_mel_spectrogram(mx.array(s), m.preprocessor_config) for s in two]).astype(dtype)
                        return m.encoder(mels, lengths=mx.array([n // 160] * 2), att_context_size=list(ctx))[0]
                    row["batch2_ms"] = round(timed(batched), 2)
                    # mx.compile on the fixed steady-state shape
                    lens = mx.array([n // 160])
                    comp = mx.compile(lambda mel: m.encoder(mel, lengths=lens, att_context_size=list(ctx))[0])
                    try:
                        row["compiled_step_ms"] = round(timed(step_fn(m, samples, ctx, dtype, compiled=comp)), 2)
                    except Exception as e:
                        row["compiled_step_ms"] = f"{type(e).__name__}: {e}"[:120]
                print(json.dumps(row), flush=True)
                results.append(row)
        del m
        mx.clear_cache()
    print("PEAK_MB", round(mx.get_peak_memory() / 2**20))


if __name__ == "__main__":
    main()
