"""Command line: python -m tte <command> --config configs/full.json

  build-evals   rebuild evals/*.jsonl from the test split (deterministic)
  train         pretrain from scratch and fit the bigram baseline
  eval          run every eval on one checkpoint (or the bigram) and write JSON
  sft           supervised fine-tuning on the reversal task
  dpo           DPO on on-policy preference pairs
  quant         int8 dynamic quantization: accuracy and latency
  seeds         SFT and DPO again with the extra seeds in the config; reversal results and paired tests
  report        render the README results tables from results/ (--check fails if README differs)
  all           train, eval base and bigram, sft, eval, dpo, eval, quant, seeds, report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from tte import checkpoint
from tte.config import RunConfig, load_config, seed_everything
from tte.data import ROOT, CharTokenizer, load_splits
from tte.evals import BigramScorer, TransformerScorer, pop_predictions, run_all


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def predictions_path(eval_json: Path) -> Path:
    return eval_json.with_name(eval_json.stem.replace("eval_", "predictions_", 1) + ".jsonl")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def cmd_eval(cfg: RunConfig, ckpt: str, out: Path) -> dict[str, Any]:
    seed_everything(cfg.seed)
    if ckpt == "bigram":
        from tte.train import load_bigram

        tok = CharTokenizer.from_text(load_splits().train)
        scorer: Any = BigramScorer(load_bigram(cfg), int(cfg.model["max_len"]))
        label = "bigram"
    else:
        model, tok, meta = checkpoint.load(Path(ckpt))
        scorer = TransformerScorer(model, model.cfg.max_len)
        label = str(meta.get("stage", Path(ckpt).stem))
    res = run_all(scorer, tok, cfg.evals.max_items, cfg.evals.max_chars, label)
    write_jsonl(predictions_path(out), pop_predictions(res))
    write_json(out, res)
    h = res["heldout"]
    print(f"[{label}] test ppl {h['perplexity']}  bpc {h['bits_per_char']}  "
          f"reversal {res['reversal']['in_dist']['exact_match']}  -> {out.relative_to(ROOT)}")
    return res


def cmd_quant(cfg: RunConfig) -> dict[str, Any]:
    from tte.evals import eval_heldout, eval_reversal, eval_word_completion
    from tte.quant import latency, quantize_int8, state_bytes

    model, tok, _ = checkpoint.load(cfg.runs_dir / "dpo.pt")
    q = quantize_int8(checkpoint.load(cfg.runs_dir / "dpo.pt")[0])
    test = load_splits().test
    text = test[: cfg.evals.max_chars] if cfg.evals.max_chars else test
    prompt = tok.encode(text[: cfg.quant.prompt_len])
    res: dict[str, Any] = {"model": "dpo", "threads": cfg.threads, "engine": torch.backends.quantized.engine,
                           "prompt_len": cfg.quant.prompt_len, "new_tokens": cfg.quant.new_tokens, "runs": cfg.quant.runs,
                           "timing": "runs interleaved across fp32 and int8, order flipped every run"}
    models = {"fp32": model, "int8_dynamic": q}
    for name, m in models.items():
        sc = TransformerScorer(m, m.cfg.max_len)
        res[name] = {
            "state_dict_bytes": state_bytes(m),
            "heldout": eval_heldout(sc, tok, text),
            "word_completion_exact_match": eval_word_completion(sc, tok, cfg.evals.max_items)["exact_match"],
            "reversal": {k: v["exact_match"] for k, v in eval_reversal(sc, tok, cfg.evals.max_items).items()},
        }
    for name, lat in latency(models, prompt, cfg.quant.new_tokens, cfg.quant.runs, cfg.quant.warmup).items():
        res[name]["latency"] = lat
    write_json(cfg.results_dir / "quant.json", res)
    for name in ("fp32", "int8_dynamic"):
        r = res[name]
        print(f"[{name}] {r['state_dict_bytes'] / 1e6:.2f} MB  ppl {r['heldout']['perplexity']}  "
              f"gen p50 {r['latency']['generate_kv_cache']['p50_ms']} ms")
    return res


def cmd_seeds(cfg: RunConfig) -> dict[str, Any]:
    """Rerun SFT and DPO from the same base model with each extra seed and score reversal.

    The seed changes which words SFT and DPO train on, the batches and dropout; pretraining is not
    repeated. The main run (cfg.seed) is read back from its prediction files.
    """
    import shutil
    from dataclasses import replace

    from tte.evals import eval_reversal
    from tte.shaping import dpo, sft
    from tte.stats import mcnemar_exact

    def outcomes(preds: list[dict[str, Any]]) -> dict[str, list[bool]]:
        return {k: [bool(p["correct"]) for p in preds if p["set"] == f"reversal_{k}"] for k in ("in_dist", "long")}

    def row(seed: int, before: dict[str, list[bool]], after: dict[str, list[bool]], pairs: int) -> dict[str, Any]:
        return {
            "seed": seed,
            "dpo_pairs": pairs,
            **{stage: {k: round(sum(v) / len(v), 4) for k, v in o.items()} for stage, o in (("sft", before), ("dpo", after))},
            "paired_sft_to_dpo": {k: mcnemar_exact(before[k], after[k]) for k in before},
        }

    r = cfg.results_dir

    def read_preds(stage: str) -> list[dict[str, Any]]:
        lines = predictions_path(r / f"eval_{stage}.json").read_text(encoding="utf-8").splitlines()
        return [json.loads(x) for x in lines if x]

    main_pairs = json.loads((r / "dpo_log.json").read_text())["pairs"]
    rows = [row(cfg.seed, outcomes(read_preds("sft")), outcomes(read_preds("dpo")), main_pairs)]
    for seed in cfg.shaping_seeds:
        c = replace(cfg, seed=seed, name=f"{cfg.name}/seed_{seed}")
        c.runs_dir.mkdir(parents=True, exist_ok=True)
        c.results_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(cfg.runs_dir / "base.pt", c.runs_dir / "base.pt")
        sft(c)
        log = dpo(c)
        got: dict[str, dict[str, list[bool]]] = {}
        for stage in ("sft", "dpo"):
            model, tok, _ = checkpoint.load(c.runs_dir / f"{stage}.pt")
            rev = eval_reversal(TransformerScorer(model, model.cfg.max_len), tok, cfg.evals.max_items)
            got[stage] = {k: [bool(p["correct"]) for p in rev[k]["predictions"]] for k in ("in_dist", "long")}
        rows.append(row(seed, got["sft"], got["dpo"], log["pairs"]))
        print(f"[seed {seed}] {rows[-1]['sft']} -> {rows[-1]['dpo']}", flush=True)
    res = {
        "note": "Same pretrained base model for every seed. The seed changes the SFT and DPO word split, batches "
        "and dropout. The first row is the main run.",
        "runs": rows,
    }
    write_json(r / "seeds.json", res)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tte", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build-evals", "train", "eval", "sft", "dpo", "quant", "seeds", "report", "all"])
    ap.add_argument("--config", default="configs/full.json")
    ap.add_argument("--checkpoint", help="eval: path to a .pt file, or 'bigram'")
    ap.add_argument("--out", help="eval: output JSON path")
    ap.add_argument("--check", action="store_true", help="report: fail if README.md is not up to date")
    args = ap.parse_args(argv)
    torch.use_deterministic_algorithms(True)
    # Denormal floats appear as weights shrink and make CPU matmuls several times slower.
    torch.set_flush_denormal(True)

    if args.command == "build-evals":
        from tte.tasks import build_all, write_all

        write_all(build_all(load_splits()))
        return 0

    cfg = load_config(ROOT / args.config if not Path(args.config).is_absolute() else args.config)
    torch.set_num_threads(cfg.threads)
    r = cfg.results_dir

    if args.command == "train":
        from tte.train import pretrain

        pretrain(cfg)
    elif args.command == "eval":
        if not args.checkpoint:
            ap.error("eval needs --checkpoint")
        out = Path(args.out) if args.out else r / f"eval_{Path(args.checkpoint).stem}.json"
        cmd_eval(cfg, args.checkpoint, out)
    elif args.command == "sft":
        from tte.shaping import sft

        sft(cfg)
    elif args.command == "dpo":
        from tte.shaping import dpo

        dpo(cfg)
    elif args.command == "quant":
        cmd_quant(cfg)
    elif args.command == "seeds":
        cmd_seeds(cfg)
    elif args.command == "report":
        from tte.report import update_readme

        return update_readme(cfg, check=args.check)
    elif args.command == "all":
        from tte.report import update_readme
        from tte.shaping import dpo, sft
        from tte.train import pretrain

        pretrain(cfg)
        cmd_eval(cfg, "bigram", r / "eval_bigram.json")
        cmd_eval(cfg, str(cfg.runs_dir / "base.pt"), r / "eval_base.json")
        sft(cfg)
        cmd_eval(cfg, str(cfg.runs_dir / "sft.pt"), r / "eval_sft.json")
        dpo(cfg)
        cmd_eval(cfg, str(cfg.runs_dir / "dpo.pt"), r / "eval_dpo.json")
        cmd_quant(cfg)
        cmd_seeds(cfg)
        if cfg.name == "full":
            update_readme(cfg, check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
