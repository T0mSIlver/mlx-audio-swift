"""Build the Nemotron Unified verification clip set from public FLEURS audio.

Writes WAVs and manifest.jsonl into OUT. Each manifest row:
  {"id", "file", "lang", "text" (reference or null), "kind", "latencies", "offline"}

usage: python make_clips.py OUT [--per-lang N] [--long-minutes 2,10]
"""

import argparse
import csv
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly

SR = 16000
ALL_LATENCIES = [160, 240, 320, 560, 1120, 2080]
# FLEURS config -> prompt language passed to every implementation.
LANGS = {
    "en_us": "en",
    "fr_fr": "fr",
    "de_de": "de",
    "es_419": "es",
    "it_it": "it",
    "pt_br": "pt",
    "pl_pl": "pl",
    "ru_ru": "ru",
    "el_gr": "el",
    "fi_fi": "fi",
}


def fleurs(config: str, wanted: int, min_s: float, max_s: float, skip: int = 0):
    """Return [(id, audio, transcription)] for `wanted` test clips in [min_s, max_s]."""
    tsv = hf_hub_download("google/fleurs", f"data/{config}/test.tsv", repo_type="dataset")
    rows = []
    with open(tsv, newline="", encoding="utf-8") as f:
        for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            # id, file_name, raw_transcription, transcription, phonemes, num_samples, gender
            rows.append((r[1], r[2], int(r[5])))
    rows = sorted({r[0]: r for r in rows}.values())
    picked = [r for r in rows if min_s * SR <= r[2] <= max_s * SR][skip:]
    picked = picked[: wanted if wanted > 0 else len(picked)]
    names = {r[0]: r for r in picked}
    tar_path = hf_hub_download("google/fleurs", f"data/{config}/audio/test.tar.gz", repo_type="dataset")
    out = {}
    with tarfile.open(tar_path) as tar:
        for member in tar:
            base = Path(member.name).name
            if base in names:
                data = tar.extractfile(member).read()
                audio, sr = sf.read(io.BytesIO(data), dtype="float32")
                assert sr == SR, (base, sr)
                out[base] = (Path(base).stem, audio, names[base][1])
                if len(out) == len(names):
                    break
    return [out[n] for n in sorted(out)]


def write(out: Path, rows: list, cid: str, audio, lang, text, kind, latencies, sr=SR, offline=True):
    path = out / "wav" / f"{cid}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, subtype="PCM_16")
    rows.append(
        {
            "id": cid,
            "file": f"wav/{cid}.wav",
            "lang": lang,
            "text": text,
            "kind": kind,
            "latencies": latencies,
            "offline": offline,
            "seconds": round(len(audio) / sr, 3),
            "sr": sr,
        }
    )


def music(seconds: float, seed: int) -> np.ndarray:
    """Chords with a beat: harmonic tones, no speech."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    y = np.zeros_like(t)
    for i, start in enumerate(np.arange(0, seconds, 0.5)):
        root = 110 * 2 ** (rng.integers(0, 12) / 12)
        seg = (t >= start) & (t < start + 0.5)
        env = np.exp(-3 * (t[seg] - start))
        for ratio in (1, 1.25, 1.5, 2):
            for h in range(1, 5):
                y[seg] += env * np.sin(2 * np.pi * root * ratio * h * t[seg]) / (h * 4)
        # a kick on every beat
        kick = (t >= start) & (t < start + 0.05)
        y[kick] += 0.6 * np.sin(2 * np.pi * 60 * (t[kick] - start)) * np.exp(-40 * (t[kick] - start))
    return (0.3 * y / np.abs(y).max()).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--per-lang", type=int, default=5)
    ap.add_argument("--long-minutes", default="2,10")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    per_lang = {}
    for config, lang in LANGS.items():
        clips = fleurs(config, args.per_lang, 4.0, 16.0)
        per_lang[lang] = clips
        for cid, audio, text in clips:
            write(out, rows, f"{lang}-{cid}", audio, lang, text, "fleurs", ALL_LATENCIES)

    en = per_lang["en"][0][1]
    fr = per_lang["fr"][0][1]
    rng = np.random.default_rng(7)
    # Non-speech and degenerate input, at every latency.
    write(out, rows, "edge-silence-5s", np.zeros(5 * SR, np.float32), "en", "", "silence", ALL_LATENCIES)
    noise = (10 ** (-30 / 20) * rng.standard_normal(5 * SR)).astype(np.float32)
    write(out, rows, "edge-noise-5s", noise, "en", "", "noise", ALL_LATENCIES)
    write(out, rows, "edge-music-8s", music(8, 1), "en", "", "music", ALL_LATENCIES)
    mixed = en + 0.3 * np.resize(music(len(en) / SR + 1, 2), en.shape)
    write(out, rows, "edge-speech-over-music", mixed.astype(np.float32), "en", per_lang["en"][0][2], "music", ALL_LATENCIES)
    write(out, rows, "edge-clipped", np.clip(en * 8, -1, 1).astype(np.float32), "en", per_lang["en"][0][2], "clipped", ALL_LATENCIES)
    pad = np.zeros(3 * SR, np.float32)
    write(out, rows, "edge-padded-silence", np.concatenate([pad, en, pad]), "en", per_lang["en"][0][2], "padded", ALL_LATENCIES)
    # Tiny inputs: under one STFT hop, under one encoder frame, about one chunk.
    for n in (80, 159, 160, 1279, 1280, 1281, 2560, 4000, 8000):
        write(out, rows, f"edge-tiny-{n}", en[:n].copy(), "en", None, "tiny", ALL_LATENCIES)
    # Short commands: the first 1.5 s of a few clips in three languages.
    for lang in ("en", "fr", "de"):
        for cid, audio, _ in per_lang[lang][1:3]:
            write(out, rows, f"cmd-{lang}-{cid}", audio[: int(1.5 * SR)].copy(), lang, None, "command", ALL_LATENCIES)
    # Sample rates and channels: each implementation's own loader resamples or downmixes.
    for name, audio, lang in (("en", en, "en"), ("fr", fr, "fr")):
        for sr in (8000, 44100, 48000):
            g = np.gcd(sr, SR)
            write(out, rows, f"sr-{name}-{sr}", resample_poly(audio, sr // g, SR // g).astype(np.float32),
                  lang, None, "samplerate", [320], sr=sr)
        stereo = np.stack([audio, 0.5 * audio], axis=1)
        write(out, rows, f"sr-{name}-16000-stereo", stereo, lang, None, "stereo", [320])
        g = np.gcd(48000, SR)
        st48 = resample_poly(stereo, 48000 // g, SR // g, axis=0).astype(np.float32)
        write(out, rows, f"sr-{name}-48000-stereo", st48, lang, None, "stereo", [320], sr=48000)
        write(out, rows, f"sr-{name}-16000-mono", audio, lang, None, "samplerate", [320])

    # Long monologues: FLEURS sentences of one language joined with 0.3 s gaps.
    gap = np.zeros(int(0.3 * SR), np.float32)
    for minutes in [float(m) for m in args.long_minutes.split(",") if m]:
        for config, lang in (("en_us", "en"), ("fr_fr", "fr")):
            if minutes > 2 and lang != "en":
                continue
            target = int(minutes * 60 * SR)
            parts, texts, total = [], [], 0
            for cid, audio, text in fleurs(config, 0, 3.0, 30.0, skip=args.per_lang):
                parts += [audio, gap]
                texts.append(text)
                total += len(audio) + len(gap)
                if total >= target:
                    break
            long = np.concatenate(parts)[:target]
            lat = [320, 1120] if minutes <= 2 else [320, 2080]
            write(out, rows, f"long-{lang}-{int(minutes)}min", long, lang, " ".join(texts), "long", lat, offline=minutes <= 1)

    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} clips, {sum(r['seconds'] for r in rows) / 60:.1f} min")


if __name__ == "__main__":
    main()
