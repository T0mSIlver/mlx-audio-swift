"""Check the Python port's buffered windows against NeMo's own buffer classes.

For every latency context, many audio lengths and several live feed patterns, the
port's BufferedEncoderState must encode exactly the windows NeMo's streaming script
encodes: same buffer start/end, same left-context frames dropped, same number of
chunk frames kept, same last-step flag. NeMo side: StreamingBatchedAudioBuffer and the
script's loop, run on lengths only (no model).

usage: python verify_windows.py   (needs nemo and the port on sys.path)
"""

import random
import sys

import numpy as np
import torch

from mlx_audio.stt.models.nemotron_asr.buffered import BufferedContext, BufferedEncoderState
from nemo.collections.asr.parts.utils.streaming_utils import ContextSize, StreamingBatchedAudioBuffer

HOP, SUB = 160, 8
FS = HOP * SUB


def enc_len(samples: int) -> int:
    # NeMo: mel frames = samples // hop; three stride-2 convs, kernel 3, pad 1.
    m = samples // HOP
    for _ in range(3):
        m = (m + 2 - 3) // 2 + 1 if m > 0 else 0
    return m


def nemo_windows(n, left, chunk, right):
    ctx = ContextSize(left=left * FS, chunk=chunk * FS, right=right * FS)
    audio = torch.zeros(1, n)
    buf = StreamingBatchedAudioBuffer(batch_size=1, context_samples=ctx, dtype=audio.dtype, device="cpu")
    rest = torch.tensor([n])
    ls, rs = 0, min(ctx.chunk + ctx.right, n)
    out, end = [], 0
    while ls < n:
        cl = min(rs, n) - ls
        last_b = torch.tensor([cl]) >= rest
        last = rs >= n
        lens = torch.where(last_b, rest, torch.full_like(rest, cl))
        buf.add_audio_batch_(audio[:, ls:rs], audio_lengths=lens, is_last_chunk=last, is_last_chunk_batch=last_b)
        end += rs - ls
        total = int(buf.context_size_batch.total()[0])
        lf = buf.context_size.left // FS
        lfb = int(buf.context_size_batch.left[0]) // FS
        keep = enc_len(total) - lfb if bool(last_b[0]) else int(buf.context_size_batch.chunk[0]) // FS
        assert buf.samples.shape[1] == total, (buf.samples.shape, total)
        out.append((end - total, end, lf, keep, bool(last_b[0])))
        rest -= lens
        ls = rs
        rs = min(rs + ctx.chunk, n)
    return out


class FakeModel:
    class P:
        hop_length = HOP

    class E:
        subsampling_factor = SUB

    preprocessor_config = P()
    encoder_config = E()


def port_windows(n, left, chunk, right, feeds):
    state = BufferedEncoderState(FakeModel(), BufferedContext(left, chunk, right), "en")
    seen = []

    def record(starts, length, lft, *, last):
        keep = enc_len(length) - lft if last else chunk
        seen.extend((s, s + length, lft, keep, last) for s in starts)
        return []

    state._encode = record
    pos = 0
    for f in feeds:
        if pos >= n:
            break
        state.push(np.zeros(min(f, n - pos), np.float32))
        pos += f
    state.flush()
    return seen


def main():
    rng = random.Random(0)
    contexts = [(70, c, r) for c, r in [(1, 1), (1, 2), (1, 3), (2, 5), (7, 7), (13, 13)]]
    contexts += [(25, 13, 13), (35, 2, 5), (5, 1, 3), (0, 1, 1), (125, 125, 62)]
    lengths = [1, 80, 159, 160, 161, 1279, 1280, 1281, 2559, 2560, 2561, 4000]
    lengths += [k * FS + e for k in (3, 4, 13, 14, 26, 27, 70, 71, 72, 83, 84, 96, 97, 200) for e in (0, 1, 159, 160, 700)]
    lengths += [rng.randrange(1, 40 * 16000) for _ in range(60)]
    bad = checked = 0
    for left, chunk, right in contexts:
        for n in lengths:
            ref = nemo_windows(n, left, chunk, right)
            feeds_list = [[n], [1280] * (n // 1280 + 1), [97] * (n // 97 + 1)]
            feeds_list.append([rng.randrange(1, 9000) for _ in range(n // 100 + 2)])
            for feeds in feeds_list:
                got = port_windows(n, left, chunk, right, feeds)
                checked += 1
                if got != ref:
                    bad += 1
                    if bad <= 10:
                        diff = next(i for i, (a, b) in enumerate(zip(got + [None] * 99, ref)) if a != b)
                        print(f"MISMATCH ctx={left, chunk, right} n={n} feed0={feeds[0]} step {diff}: port {got[diff:diff+2]} nemo {ref[diff:diff+2]}")
    zero = [(c, n) for c in contexts for n in lengths if any(w[3] <= 0 for w in nemo_windows(n, *c))]
    print(f"checked {checked} (context, length, feed) cases, {bad} mismatches")
    print(f"cases where NeMo's last step keeps <= 0 frames: {len(zero)} e.g. {zero[:6]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
