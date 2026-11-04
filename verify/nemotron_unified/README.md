# Nemotron Unified: checks against NeMo

These scripts compare the MLX ports of [nvidia/nemotron-labs-unified-asr-0.6b](https://huggingface.co/nvidia/nemotron-labs-unified-asr-0.6b) (mlx-audio and mlx-audio-swift) with NeMo, token by token, on public audio. Everything runs on GitHub's runners. NeMo runs on Linux CPU in fp32, and the ports run on the macOS runners.

| Script | What it does |
|---|---|
| `make_clips.py` | Builds 86 clips (25.6 min): 5 FLEURS test sentences in each of 10 languages, silence, noise, music, clipped and padded speech, inputs of 80 to 8,000 samples, 8/44.1/48 kHz and stereo copies, and 2- and 10-minute monologues. |
| `nemo_ref.py` | Runs NeMo's own `speech_to_text_streaming_infer_rnnt.py` loop at batch 1, with 5.6 s of left context and NVIDIA's chunk table at 160 to 2080 ms, and offline. It writes the token ids for each clip and latency. |
| `mlx_run.py`, `../../Tests/NemotronUnifiedVerifyTests.swift` | Run the Python and Swift ports on the same clips at fp32, bf16, 8-bit and 4-bit. |
| `compare.py` | Reports, per port and precision, the share of clips token-identical to NeMo, the token edit rate, WER on the FLEURS clips, and errors. |
| `verify_windows.py` | Checks that the Python port cuts a stream into exactly NeMo's buffer windows, using NeMo's own classes, over 6,248 lengths, layouts and feed sizes. |
| `profile_step.py` | Times each part of one streaming step. |
| `vs35.py` | Runs Unified against Nemotron 3.5 in the Python port, both 8-bit, on the same runner: FLEURS WER, time per step and peak memory. |

The workflows that run them:
- `.github/workflows/verify-nemotron-unified.yml`: NeMo in 16 shards, then each port and precision.
- `verify-nemotron-unified-card.yml`: the model card's 25 languages and the `auto` prompt.
- `verify-nemotron-unified-vs35.yml`: Unified against 3.5.

Start any of them from the Actions tab (`workflow_dispatch`). The pinned revisions are at the top of each file: NeMo `NVIDIA-NeMo/Speech@0c7af33` and checkpoint `023ce907`.

Results so far:
- At fp32 both ports give NeMo's tokens on every streaming clip-latency pair: Swift 527/527 ([run](https://github.com/T0mSIlver/mlx-audio-swift/actions/runs/41763554258)), Python 442/442.
- Offline, Python matches NeMo's text on 82/83 clips. The one miss is an 8 kHz input, where the two resamplers feed the model different samples.
- On all 25 of the model card's languages, with the language given and with `auto`, both ports at fp32 give NeMo's tokens on all 1,190 clip-latency pairs ([run](https://github.com/T0mSIlver/mlx-audio-swift/actions/runs/41811398966)).
- In bf16, 89-96 % of clips stay token-identical at each latency, and FLEURS WER stays within 0.3 points of NeMo's.
