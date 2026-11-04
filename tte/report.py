"""Render the README results tables from results/<config>/*.json.

The tables sit between <!-- results:start --> and <!-- results:end --> in README.md. `--check` fails
when the README does not match what the saved results say, so no number in the table can drift from
its file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tte.config import RunConfig
from tte.data import ROOT

START, END = "<!-- results:start -->", "<!-- results:end -->"


def _load(d: Path, name: str) -> dict[str, Any]:
    out: dict[str, Any] = json.loads((d / name).read_text(encoding="utf-8"))
    return out


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render(cfg: RunConfig) -> str:
    d = cfg.results_dir
    rel = d.relative_to(ROOT)
    tr = _load(d, "train_log.json")
    ev = {k: _load(d, f"eval_{k}.json") for k in ("bigram", "base", "sft", "dpo")}
    q = _load(d, "quant.json")
    dl = _load(d, "dpo_log.json")
    lines: list[str] = []
    add = lines.append

    add(f"Model: {tr['params']:,} parameters, trained for {tr['config']['train']['steps']:,} steps "
        f"({tr['tokens_seen']:,} characters seen) in {tr['train_seconds']:.0f} s on `{tr['device']}`, torch {tr['torch']}. "
        f"Source: `{rel}/`.")
    add("")
    add("**1. Pretraining from scratch vs the bigram baseline** (held-out test split, "
        f"{ev['base']['heldout']['chars_scored']:,} characters; `eval_bigram.json`, `eval_base.json`)")
    add("")
    add("| Metric | Bigram | Transformer |")
    add("|---|---|---|")

    def ci(e: dict[str, Any]) -> str:
        h = e["heldout"]
        lo, hi = h["bits_per_char_ci95"]
        return f"{h['bits_per_char']:.3f} ({lo:.3f} to {hi:.3f})"

    add(f"| Test perplexity per character | {ev['bigram']['heldout']['perplexity']:.2f} | {ev['base']['heldout']['perplexity']:.2f} |")
    add(f"| Test bits per character (95% CI) | {ci(ev['bigram'])} | {ci(ev['base'])} |")
    add(f"| Word completion, exact match (n={ev['base']['word_completion']['n']}) | "
        f"{pct(ev['bigram']['word_completion']['exact_match'])} | {pct(ev['base']['word_completion']['exact_match'])} |")
    for kind, label in (("word_swap", "Minimal pairs, word order"), ("char_swap", "Minimal pairs, spelling")):
        add(f"| {label} (n={ev['base']['minimal_pairs'][kind]['n']}, chance 50%) | "
            f"{pct(ev['bigram']['minimal_pairs'][kind]['accuracy'])} | {pct(ev['base']['minimal_pairs'][kind]['accuracy'])} |")
    add("")
    add("**2. Behaviour shaping: before and after, same harness** (`eval_base.json`, `eval_sft.json`, `eval_dpo.json`)")
    add("")
    add("| Metric | Base | After SFT | After SFT + DPO |")
    add("|---|---|---|---|")
    stages = ("base", "sft", "dpo")
    n_id, n_long = ev["base"]["reversal"]["in_dist"]["n"], ev["base"]["reversal"]["long"]["n"]
    add(f"| Reversal, 3 to 8 letters, exact match (n={n_id}) | " + " | ".join(pct(ev[s]["reversal"]["in_dist"]["exact_match"]) for s in stages) + " |")
    add(f"| Reversal, 9 to 12 letters (unseen lengths), exact match (n={n_long}) | " + " | ".join(pct(ev[s]["reversal"]["long"]["exact_match"]) for s in stages) + " |")
    add("| Reversal, 9 to 12 letters, per-character accuracy | " + " | ".join(pct(ev[s]["reversal"]["long"]["char_accuracy"]) for s in stages) + " |")
    add("| Test perplexity per character | " + " | ".join(f"{ev[s]['heldout']['perplexity']:.3f}" for s in stages) + " |")
    add("| Word completion, exact match | " + " | ".join(pct(ev[s]["word_completion"]["exact_match"]) for s in stages) + " |")
    add("| Minimal pairs, word order | " + " | ".join(pct(ev[s]["minimal_pairs"]["word_swap"]["accuracy"]) for s in stages) + " |")
    add("")
    add(f"DPO used {dl['pairs']} preference pairs: the SFT model was wrong on {dl['pairs']} of the "
        f"{dl['words_tried_to_find_pairs']} unseen training words it was asked to reverse "
        f"({pct(dl['sft_error_rate_on_dpo_words'])}). Final training-batch reward accuracy "
        f"{pct(dl['log'][-1]['reward_accuracy'])} (`dpo_log.json`, pairs in `dpo_pairs.jsonl`).")
    add("")
    pilot = _load(d, "pilot_shaping.json")
    add(f"**2b. How the DPO settings were chosen: pilot on validation words only** (`pilot_shaping.json`, "
        f"{pilot['sft']['in_dist']['n']} + {pilot['sft']['long']['n']} words from the validation split; "
        f"rule: {pilot['rule']})")
    add("")
    add("| Model | 3 to 8 letters, exact match | 9 to 12 letters, exact match | Mean length error, 3 to 8 letters |")
    add("|---|---|---|---|")
    ps = pilot["sft"]
    add(f"| SFT only | {pct(ps['in_dist']['exact_match'])} | {pct(ps['long']['exact_match'])} | {ps['in_dist']['mean_length_error']:+.2f} |")
    for c in pilot["dpo"]:
        kind = "DPO + NLL on chosen" if c["nll_weight"] > 0 else "Plain DPO"
        mark = " (chosen)" if c["lr"] == pilot["chosen"]["lr"] and c["nll_weight"] == pilot["chosen"]["nll_weight"] else ""
        add(f"| {kind}, lr {c['lr']:g}{mark} | {pct(c['in_dist']['exact_match'])} | {pct(c['long']['exact_match'])} | "
            f"{c['in_dist']['mean_length_error']:+.2f} |")
    add("")
    add("**3. Robustness: bits per character, lower is better** (`eval_base.json`, `eval_bigram.json`)")
    add("")
    add("| Input | Bigram | Transformer | Transformer change vs clean | Unknown chars |")
    add("|---|---|---|---|---|")
    rb, rt = ev["bigram"]["robustness"], ev["base"]["robustness"]
    names = {
        "clean": "Clean test text",
        "typos_5pct_context": "5% of letters in the context replaced",
        "typos_15pct_context": "15% of letters in the context replaced",
        "lowercase": "All lowercase",
        "uppercase": "All uppercase",
        "ood_danish_other_author": "Other Danish author (Jacobsen, Fru Marie Grubbe)",
        "ood_english": "English (Andersen in translation)",
    }
    for k, label in names.items():
        delta = "" if k == "clean" else f"+{rt[k]['delta_vs_clean']:.3f}"
        unk = "" if k == "clean" else pct(rt[k]["unknown_char_rate"])
        add(f"| {label} | {rb[k]['bits_per_char']:.3f} | {rt[k]['bits_per_char']:.3f} | {delta} | {unk} |")
    add("")
    add(f"**4. Inference: int8 dynamic quantization of the final model** (`quant.json`; CPU, {q['threads']} threads, "
        f"engine {q['engine']}, {q['runs']} timed runs each, prompt {q['prompt_len']} characters, {q['new_tokens']} new tokens)")
    add("")
    add("| Metric | fp32 | int8 dynamic |")
    add("|---|---|---|")
    f, i = q["fp32"], q["int8_dynamic"]
    add(f"| Weights on disk | {f['state_dict_bytes'] / 1e6:.2f} MB | {i['state_dict_bytes'] / 1e6:.2f} MB |")
    add(f"| Test perplexity per character | {f['heldout']['perplexity']:.3f} | {i['heldout']['perplexity']:.3f} |")
    add(f"| Reversal, 3 to 8 letters, exact match | {pct(f['reversal']['in_dist'])} | {pct(i['reversal']['in_dist'])} |")
    add(f"| Word completion, exact match | {pct(f['word_completion_exact_match'])} | {pct(i['word_completion_exact_match'])} |")
    for key, label in (("prefill", "Prefill (one forward pass)"), ("generate_kv_cache", "Generate with KV cache"),
                       ("generate_no_cache", "Generate without KV cache")):
        a, b = f["latency"][key], i["latency"][key]
        add(f"| {label}, p50 / p95 / p99 ms | {a['p50_ms']:.2f} / {a['p95_ms']:.2f} / {a['p99_ms']:.2f} | "
            f"{b['p50_ms']:.2f} / {b['p95_ms']:.2f} / {b['p99_ms']:.2f} |")
    return "\n".join(lines)


def update_readme(cfg: RunConfig, check: bool) -> int:
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        print("README.md has no results markers")
        return 1
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    new = f"{head}{START}\n{render(cfg)}\n{END}{tail}"
    if check:
        if new != text:
            print("README.md results tables differ from results/; run: uv run python -m tte report")
            return 1
        print("README.md results tables match results/")
        return 0
    path.write_text(new, encoding="utf-8")
    return 0
