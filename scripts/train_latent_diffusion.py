"""Train the new denoiser from explicit, previously encoded train/val caches.

CPU example (from the repository root):
  .venv/Scripts/python.exe scripts/train_latent_diffusion.py \
    --config configs/diffusion/cpu_smoke.json \
    --train-cache data/cache/diffusion_cpu/train \
    --val-cache data/cache/diffusion_cpu/val \
    --output-dir artifacts/diffusion/cpu_smoke_seed0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from latent_diffusion.data import load_cache  # noqa: E402
from latent_diffusion.training import DenoiserTrainer  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="Explicit device, default cpu; no accelerator auto-selection")
    parser.add_argument("--mode", choices=("smoke", "overfit"), default="smoke")
    parser.add_argument("--max-steps", type=int, help="Actual stopping step; cannot exceed the config cap")
    parser.add_argument("--seed", type=int, help="Override the config seed")
    parser.add_argument("--batch-size", type=int, help="Override the config batch size")
    parser.add_argument("--validation-frequency", type=int, help="Override the config validation interval")
    parser.add_argument("--resume", type=Path, help="Existing checkpoint for strict same-configuration continuation")
    return parser.parse_args()


def _require_split(cache_dir: Path, expected: str) -> None:
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("split") != expected:
        raise ValueError(f"Expected {expected} cache, found {manifest.get('split')!r}; no test data opened")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> int:
    args = arguments()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for argument, key in (("seed", "seed"), ("batch_size", "batch_size"),
                          ("validation_frequency", "validation_frequency")):
        value = getattr(args, argument)
        if value is not None:
            config["training"][key] = value
    threads = int(config["training"].get("intraop_threads", 2))
    if threads < 1:
        raise ValueError("intraop_threads must be positive")
    torch.set_num_threads(threads)
    cap = int(config["training"]["max_steps"])
    if args.mode == "overfit":
        cap = min(cap, 200)
    target = args.max_steps if args.max_steps is not None else cap
    if not 1 <= target <= cap:
        raise ValueError(f"Requested steps must be in [1, {cap}]")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    _require_split(args.train_cache, "train")
    _require_split(args.val_cache, "val")
    train_cache = load_cache(args.train_cache)
    val_cache = load_cache(args.val_cache)
    checkpoint = args.output_dir / "checkpoint.pt"
    metrics_path = args.output_dir / "metrics.json"
    if args.resume is None and (checkpoint.exists() or metrics_path.exists()):
        raise FileExistsError("Output directory already contains a run; use --resume or a new directory")
    trainer = DenoiserTrainer(train_cache, val_cache, config, mode=args.mode, device=args.device)
    if args.resume is not None:
        trainer.resume(args.resume)
    try:
        metrics = trainer.run(target, checkpoint)
    except KeyboardInterrupt:
        trainer.save(checkpoint)
        print(f"Interrupted after {trainer.step_number} completed updates; partial checkpoint: {checkpoint}", file=sys.stderr)
        raise
    metrics.update({
        "device": args.device,
        "torch_version": torch.__version__,
        "intraop_threads": torch.get_num_threads(),
        "num_workers": config["training"]["num_workers"],
        "config": str(args.config),
        "train_cache": str(args.train_cache),
        "val_cache": str(args.val_cache),
        "checkpoint": str(checkpoint),
        "resumed_from": str(args.resume) if args.resume else None,
        "original_splits_recovered": False,
        "pretrained_training_overlap": "unknown",
        "checkpoint_cmr_label_compatibility": "unresolved",
    })
    _atomic_json(metrics_path, metrics)
    print(json.dumps({key: metrics[key] for key in (
        "mode", "completed_steps", "elapsed_seconds", "updates_per_second",
        "fixed_corruption_mse_before", "fixed_corruption_mse_after", "checkpoint",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
