#!/usr/bin/env python3
"""Compare a short CPU run with an interrupted/resumed run on real caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from latent_diffusion.data import load_cache, sha256_file  # noqa: E402
from latent_diffusion.training import DenoiserTrainer, load_frozen_model  # noqa: E402


def equal(a: object, b: object) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, type(a)) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def require_split(path: Path, expected: str) -> None:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("split") != expected:
        raise ValueError(f"Expected {expected} cache; no test cache opened")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--split-after", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.split_after < args.steps <= 16:
        raise ValueError("Require 1 <= split-after < steps <= 16")
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory for this deterministic check")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(int(config["training"].get("intraop_threads", 2)))
    require_split(args.train_cache, "train")
    require_split(args.val_cache, "val")
    train, val = load_cache(args.train_cache), load_cache(args.val_cache)
    args.output_dir.mkdir(parents=True)
    uninterrupted_path = args.output_dir / "uninterrupted.pt"
    resumed_path = args.output_dir / "resumed.pt"

    uninterrupted = DenoiserTrainer(train, val, config, device="cpu")
    uninterrupted.run(args.steps, uninterrupted_path)
    interrupted = DenoiserTrainer(train, val, config, device="cpu")
    interrupted.run(args.split_after, resumed_path)
    continued = DenoiserTrainer(train, val, config, device="cpu")
    continued.resume(resumed_path)
    continued.run(args.steps, resumed_path)

    first, second = uninterrupted.checkpoint_payload(), continued.checkpoint_payload()
    checks = {
        "model_state_exact": equal(first["model_state"], second["model_state"]),
        "optimizer_state_exact": equal(first["optimizer_state"], second["optimizer_state"]),
        "diffusion_schedule_exact": equal(first["diffusion_state"], second["diffusion_state"]),
        "training_history_exact": equal(first["history"], second["history"]),
        "rng_streams_exact": equal(first["rng"]["streams"], second["rng"]["streams"]),
        "same_step": first["step"] == second["step"] == args.steps,
    }
    loaded, schedule, scalers, metadata = load_frozen_model(resumed_path, "cpu")
    with torch.no_grad():
        u = continued.train_u[:2]
        c = continued.train_c[:2]
        t = torch.tensor([0, continued.diffusion.timesteps - 1])
        checks["inference_reload_exact"] = torch.equal(continued.model(u, t, c), loaded(u, t, c))
        checks["schedule_reload_exact"] = torch.equal(continued.diffusion.alpha_bars, schedule.alpha_bars)
    checks["scaler_hash_exact"] = scalers["hash"] == continued.scalers["hash"]
    checks["inference_step_exact"] = metadata["step"] == args.steps
    result = {
        "status": "pass" if all(checks.values()) else "fail",
        "device": "cpu", "steps": args.steps, "split_after": args.split_after,
        "checks": checks,
        "train_cache_fingerprint": train["manifest"]["fingerprint"],
        "val_cache_fingerprint": val["manifest"]["fingerprint"],
        "uninterrupted_checkpoint_sha256": sha256_file(uninterrupted_path),
        "resumed_checkpoint_sha256": sha256_file(resumed_path),
        "note": "Extra fixed-bank validation at the interruption boundary changes validation_history only; separate RNG streams leave updates identical on this CPU.",
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "pass":
        raise AssertionError("Interrupted and uninterrupted CPU updates diverged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
