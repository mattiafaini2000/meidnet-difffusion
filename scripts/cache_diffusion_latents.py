#!/usr/bin/env python3
"""Create or resume a deterministic train/validation MEIDNet joint-latent cache.

Example CPU engineering sample (run from the repository root)::

    .venv/Scripts/python.exe scripts/cache_diffusion_latents.py --split train \
      --csv data/processed/cmr_reconstructed/train.csv \
      --cif-root data/processed/cmr_reconstructed/cifs/train \
      --checkpoint MEIDNet-main/checkpoints/dual_autoencoder_clip_earlyfusion_propertyaware_2k.pth \
      --output-dir data/cache/diffusion_cpu/train --limit 1024 --device cpu

Use ``--limit 0`` for a later full-split cache, not for this CPU smoke task.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "MEIDNet-main") not in sys.path:
    sys.path.insert(0, str(ROOT / "MEIDNet-main"))

from latent_diffusion.data import (  # noqa: E402
    cache_batches, ids_hash, reject_test_path, select_rows, sha256_file,
)
from latent_diffusion.integration import (  # noqa: E402
    DEFAULT_CHECKPOINT, DEFAULT_TEMPLATE_MANIFEST, DEFAULT_TRAINING_SUPPORT,
    FrozenMEIDNet,
)
from meidnet.model import parse_cif_to_dense  # noqa: E402


FAMILY_BY_ANION = {
    "O3": "oxide", "N3": "nitride", "O2N": "oxynitride", "ON2": "oxynitride",
    "O2F": "oxyhalide", "O2S": "oxychalcogenide", "OFN": "other_mixed",
}
ANION_ORDER = ("O", "F", "N", "S", "Cl", "Br", "I", "Se", "Te")


def exact_anion_key(symbols: list[str]) -> str:
    counts = Counter(symbols)
    ordered = [symbol for symbol in ANION_ORDER if symbol in counts]
    ordered += sorted(set(counts) - set(ordered))
    return "".join(symbol + (str(counts[symbol]) if counts[symbol] > 1 else "")
                   for symbol in ordered)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--cif-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--template-manifest", type=Path, default=DEFAULT_TEMPLATE_MANIFEST)
    parser.add_argument("--training-support", type=Path, default=DEFAULT_TRAINING_SUPPORT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0,
                        help="maximum source-only selected rows; zero means full split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--allow-test", action="store_true",
                        help="requires separate authorization; NEVER use in this milestone")
    return parser.parse_args()


def read_source_rows(csv_path: Path) -> list[dict]:
    """Read source strings exactly, preserving zero labels and hex material IDs."""
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        needed = {"material_id", "heat_all", "dir_gap", "cif", "cif_filename",
                  "composition_group", "source_site_symbols", "meidnet_site_symbols",
                  "site_permutation", "source_anion"}
        if not needed <= set(reader.fieldnames or []):
            raise ValueError(f"CSV missing cache fields: {sorted(needed - set(reader.fieldnames or []))}")
        for raw in reader:
            material_id = raw["material_id"]
            symbols = json.loads(raw["source_site_symbols"])
            parsed = json.loads(raw["meidnet_site_symbols"])
            permutation = [int(index) for index in json.loads(raw["site_permutation"])]
            if (len(symbols) != 5 or len(parsed) != 5 or sorted(permutation) != list(range(5))
                    or any(symbols[i] != parsed[permutation[i]] for i in range(5))):
                raise ValueError(f"Invalid preserved role/parser metadata: {material_id}")
            if raw["cif_filename"] != f"{material_id}.cif":
                raise ValueError(f"CIF filename and material ID disagree: {material_id}")
            if exact_anion_key(symbols[2:]) != raw["source_anion"]:
                raise ValueError(f"Source anion key and exact source-site multiset disagree: {material_id}")
            row = {
                "material_id": material_id,
                "heat_all": float(raw["heat_all"]),
                "dir_gap": float(raw["dir_gap"]),
                "composition_group": raw["composition_group"],
                "source_anion_key": raw["source_anion"],
                "source_family": FAMILY_BY_ANION.get(raw["source_anion"], "other_unresolved"),
                "source_roles": symbols,
                "exact_anion_multiset": sorted(symbols[2:]),
                "parser_roles": parsed,
                "site_permutation": permutation,
                "cif_filename": raw["cif_filename"],
                "source_id": raw.get("source_id"),
                "source_unique_id": raw.get("source_unique_id"),
                "embedded_cif": raw["cif"],
            }
            rows.append(row)
    if not rows or len({row["material_id"] for row in rows}) != len(rows):
        raise ValueError("Empty source CSV or duplicate material IDs")
    return rows


def cif_digest_and_rows(rows: list[dict], cif_root: Path) -> tuple[str, list[dict]]:
    digest = hashlib.sha256()
    clean = []
    for row in rows:
        path = cif_root / row["cif_filename"]
        raw = path.read_bytes()
        if raw.decode("utf-8") != row["embedded_cif"]:
            raise ValueError(f"CSV/CIF text mismatch: {row['material_id']}")
        digest.update(row["material_id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(raw)
        clean.append({key: value for key, value in row.items() if key != "embedded_cif"})
    return digest.hexdigest(), clean


def main() -> int:
    args = arguments()
    reject_test_path(args.split, args.csv, args.cif_root, args.allow_test)
    if args.threads <= 0 or args.batch_size <= 0:
        raise ValueError("threads and batch size must be positive")
    torch.set_num_threads(args.threads)
    csv_path = args.csv.resolve(strict=True)
    cif_root = args.cif_root.resolve(strict=True)
    source_rows = read_source_rows(csv_path)
    source_ids = sorted(row["material_id"] for row in source_rows)
    selected, selection = select_rows(source_rows, limit=args.limit, seed=args.seed)
    cif_sha, selected = cif_digest_and_rows(selected, cif_root)
    adapter = FrozenMEIDNet(args.checkpoint, args.template_manifest,
                            args.training_support, device=args.device)

    def encode(batch: list[dict]) -> torch.Tensor:
        vectors = []
        for row in batch:
            path = cif_root / row["cif_filename"]
            try:
                vectors.append(torch.from_numpy(parse_cif_to_dense(str(path))).float())
            except Exception as error:
                raise ValueError(
                    f"PARSE_FAILURE:{row['material_id']}:{type(error).__name__}:{error}"
                ) from error
        crystal = torch.stack(vectors).to(adapter.device)
        conditions = torch.tensor([[row["heat_all"], row["dir_gap"]] for row in batch],
                                  dtype=torch.float32, device=adapter.device)
        return adapter.encode_joint(crystal, conditions)

    base = {
        "split": args.split,
        "source_csv_sha256": sha256_file(csv_path),
        "source_ids_sha256": ids_hash(source_ids),
        "selected_cif_bytes_sha256": cif_sha,
        "selection": selection,
        "checkpoint_sha256": adapter.checkpoint_sha256,
        "model_state_sha256": adapter.model_state_sha256,
        "model_source_sha256": sha256_file(ROOT / "MEIDNet-main" / "meidnet" / "model.py"),
        "parser": "meidnet.model.parse_cif_to_dense:unmodified",
        "template_manifest_sha256": adapter.template_sha256,
        "template_input_coordinates_sha256": adapter.template_coordinates_sha256,
        "output_template_sha256": adapter.output_template_sha256,
        "policy_source_sha256": adapter.policy_sha256,
        "training_support_sha256": adapter.training_support_sha256,
        "vocab": {role: list(adapter.vocab[role]) for role in ("A", "B", "X")},
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "original_splits_recovered": False,
        "pretrained_training_overlap": "unknown",
        "checkpoint_cmr_heat_label_compatibility": "unresolved",
    }
    cache = cache_batches(args.output_dir, selected, encode, base, batch_size=args.batch_size)
    adapter.assert_frozen()
    print(json.dumps({
        "status": cache["manifest"]["status"], "split": args.split,
        "selected": cache["manifest"]["selected_rows"],
        "processed": cache["manifest"]["processed_rows"],
        "latent_dim": cache["manifest"]["latent_dim"],
        "fingerprint": cache["manifest"]["fingerprint"],
        "output_dir": str(Path(args.output_dir).resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
