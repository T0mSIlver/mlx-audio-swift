"""Clip set for the model card's claims: 25 languages, explicit and `auto` prompts.

Per language, the same 5 FLEURS test clips as make_clips.py (4-16 s):
  - kind "fleurs": the bare code as prompt (the card's `target_lang="de"`), at
    160, 320, 1120 ms and offline. The first clip of each language also runs the
    odd latencies 480, 800, 1600 ms and three left contexts (0.8, 2.0, 2.8 s).
  - kind "auto": the same audio with `auto` (prompt 127), same latencies.
  - kind "locale" (es, pt only): the locale whose prompt differs from the bare
    code (FLEURS es_419 with es-ES, pt_br with pt-BR).

usage: python card_clips.py OUT [--per-lang N]
"""

import argparse
import json
from pathlib import Path

from make_clips import fleurs, write

CARD_LATENCIES = [160, 320, 1120]
ODD_LATENCIES = [480, 800, 1600]
# name -> [left, chunk, right] encoder frames (80 ms each), at 320 ms latency.
LEFT_CONTEXTS = {"320l10": [10, 1, 3], "320l25": [25, 1, 3], "320l35": [35, 1, 3]}
LANGS = {
    "bg_bg": "bg", "hr_hr": "hr", "cs_cz": "cs", "da_dk": "da", "nl_nl": "nl",
    "en_us": "en", "et_ee": "et", "fi_fi": "fi", "fr_fr": "fr", "de_de": "de",
    "el_gr": "el", "hu_hu": "hu", "it_it": "it", "lv_lv": "lv", "lt_lt": "lt",
    "mt_mt": "mt", "pl_pl": "pl", "pt_br": "pt", "ro_ro": "ro", "sk_sk": "sk",
    "sl_si": "sl", "es_419": "es", "sv_se": "sv", "ru_ru": "ru", "uk_ua": "uk",
}
LOCALES = {"es": "es-ES", "pt": "pt-BR"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--per-lang", type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for config, lang in LANGS.items():
        for i, (cid, audio, text) in enumerate(fleurs(config, args.per_lang, 4.0, 16.0)):
            base = f"{lang}-{cid}"
            write(out, rows, base, audio, lang, text, "fleurs", CARD_LATENCIES + (ODD_LATENCIES if i == 0 else []))
            if i == 0:
                rows[-1]["contexts"] = LEFT_CONTEXTS
            for kind, prompt in (("auto", "auto"), ("locale", LOCALES.get(lang))):
                if prompt:
                    write(out, rows, f"{kind}-{base}", audio, prompt, text, kind, CARD_LATENCIES)
                    rows[-1].update(file=f"wav/{base}.wav", true_lang=lang)
                    (out / "wav" / f"{kind}-{base}.wav").unlink()
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows, {sum(r['seconds'] for r in rows if r['kind'] == 'fleurs') / 60:.1f} min of audio")


if __name__ == "__main__":
    main()
