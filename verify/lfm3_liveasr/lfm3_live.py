"""LFM3-Audio-2B reference LiveASR (Transformers, CPU fp32), the card's loop verbatim,
plus offline ASR on the same clip for comparison.

Per clip writes {"id", "impl": "lfm3-live", "text", "deltas": [{"audio_s", "compute_s", "text"}],
"restarts", "offline_text", "seconds"}. audio_s = audio fed when the chunk was decoded,
so a delta's text cannot appear on screen before that point plus its compute time.
The text context restarts every RESTART_S of audio, as the card says for streams over 30 s.

usage: python lfm3_live.py MANIFEST OUT.jsonl --shard i/n [--rev SHA]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import librosa
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

MODEL = "LiquidAI/LFM3-Audio-2B"
RESTART_S = 30.0
MIC = 1_600  # 100 ms at 16 kHz, as in the card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("out")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--rev", default=None)
    args = ap.parse_args()
    i, n = map(int, args.shard.split("/"))
    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")][i::n]
    root = Path(args.manifest).parent

    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL, revision=args.rev, dtype=torch.float32, device_map="cpu", trust_remote_code=True
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL, revision=args.rev, trust_remote_code=True)
    tok = processor.tokenizer
    chunk_frames = 8 * model.config.interleaved_n_audio
    prompt = tok(
        "<|startoftext|><|im_start|>system\nPerform live ASR.<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids
    suppress = tok.convert_tokens_to_ids(["<|mixed_end|>", "<|endoftext|>"])
    stop = tok.convert_tokens_to_ids(["<|endoftext|>", "<|im_start|>"])

    with open(args.out, "a", encoding="utf-8") as out:
        for r in rows:
            t0 = time.time()
            wav, _ = librosa.load(str(root / r["file"]), sr=16_000)
            state, cache, deltas, restarts = None, None, [], 0
            mel = torch.empty((1, processor.feature_extractor.feature_size, 0))
            fed = 0
            context_start = 0  # samples fed when the current text context began

            def step(frames, final=False):
                nonlocal state, restarts, context_start
                restart = state is not None and fed - context_start >= RESTART_S * 16_000
                if restart:
                    restarts += 1
                    context_start = fed
                c0 = time.time()
                with torch.no_grad():
                    o = model.transcribe_live(
                        frames,
                        state,
                        input_ids=prompt if (state is None or restart) else None,
                        is_final=final,
                        max_new_tokens=96 if final else 48,
                        suppress_token_ids=suppress,
                        stop_token_ids=stop,
                    )
                state = o.state
                deltas.append({
                    "audio_s": round(fed / 16_000, 3),
                    "compute_s": round(time.time() - c0, 3),
                    "text": tok.decode(o.sequences[0], skip_special_tokens=True),
                    "restart": restart,
                })

            for start in range(0, len(wav), MIC):
                piece = wav[start:start + MIC]
                fed += len(piece)
                feats, cache = processor.feature_extractor.stream(piece, cache)
                mel = torch.cat((mel, feats.input_features), dim=-1)
                while mel.shape[-1] >= chunk_frames:
                    step(mel[..., :chunk_frames])
                    mel = mel[..., chunk_frames:]
            feats, cache = processor.feature_extractor.stream([], cache, is_final=True)
            mel = torch.cat((mel, feats.input_features), dim=-1)
            while mel.shape[-1] > chunk_frames:
                step(mel[..., :chunk_frames])
                mel = mel[..., chunk_frames:]
            step(mel, final=True)
            live = "".join(d["text"] for d in deltas)

            offline = None
            if r.get("offline"):
                conv = [
                    {"role": "system", "content": "Perform ASR in English."},
                    {"role": "user", "content": [{"type": "audio", "audio": str(root / r["file"])}]},
                ]
                inputs = processor.apply_chat_template(
                    conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
                )
                with torch.no_grad():
                    g = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                                       suppress_mixed=True, suppress_audio_out=True)
                seq = g.sequences if hasattr(g, "sequences") else g
                offline = tok.decode(seq[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

            rec = {"id": r["id"], "impl": "lfm3-live", "latency": "live", "text": live, "deltas": deltas,
                   "restarts": restarts, "offline_text": offline, "chunk_frames": chunk_frames,
                   "seconds": round(time.time() - t0, 2)}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(r["id"], rec["seconds"], "s", len(deltas), "chunks", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
