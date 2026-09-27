"""Small, asset-independent helpers for joint-latent caches and scalers.

Only the CLI integration layer parses CIFs.  This module deliberately needs no
materials packages, which makes cache integrity and scaling easy to test.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch


CACHE_SCHEMA = "joint_latent_cache_v1"
SCALER_SCHEMA = "train_featurewise_scalers_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def ids_hash(ids: list[str]) -> str:
    return json_hash(ids)


def reject_test_path(split: str, csv_path: Path, cif_root: Path, allow_test: bool) -> None:
    """Block accidental final-test encoding before either path is opened."""
    if not allow_test and (split == "test" or csv_path.stem.lower() == "test"
                           or cif_root.name.lower() == "test"):
        raise ValueError("Final-test cache access is disabled by default")


def select_rows(rows: list[dict], limit: int = 0, seed: int = 42) -> tuple[list[dict], dict]:
    """Select without model outcomes, round-robin over exact-anion/gap strata.

    The source split is unchanged.  Selection order is sorted by material ID so a
    resumed cache is a stable prefix of the same source-only engineering sample.
    ``limit=0`` selects the full supplied split (for the later VM).
    """
    if limit < 0:
        raise ValueError("limit must be nonnegative (zero means all rows)")
    if len({str(row["material_id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate material IDs in source rows")
    ordered = sorted(rows, key=lambda row: str(row["material_id"]))
    if not ordered:
        raise ValueError("No rows to select")
    if limit == 0 or limit >= len(ordered):
        chosen = ordered
    else:
        strata: dict[str, list[dict]] = defaultdict(list)
        for row in ordered:
            gap = float(row["dir_gap"])
            if not math.isfinite(gap) or gap < 0:
                raise ValueError(f"Invalid gap for {row['material_id']}: {gap}")
            key = f"{row['source_anion_key']}|{'zero' if gap == 0 else 'positive'}"
            strata[key].append(row)
        for key, members in strata.items():
            members.sort(key=lambda row: (
                hashlib.sha256(f"{seed}:{key}:{row['material_id']}".encode()).hexdigest(),
                row["material_id"],
            ))
        keys = sorted(strata, key=lambda key: (len(strata[key]), key))
        chosen = []
        while len(chosen) < limit:
            progressed = False
            for key in keys:
                if strata[key]:
                    chosen.append(strata[key].pop(0))
                    progressed = True
                    if len(chosen) == limit:
                        break
            if not progressed:
                break
        chosen.sort(key=lambda row: row["material_id"])
    counts: dict[str, int] = defaultdict(int)
    for row in chosen:
        key = f"{row['source_anion_key']}|{'zero' if float(row['dir_gap']) == 0 else 'positive'}"
        counts[key] += 1
    manifest = {
        "algorithm": "source_only_exact_anion_x_zero_positive_round_robin_sha256_v1",
        "seed": int(seed),
        "requested_limit": int(limit),
        "source_rows": len(ordered),
        "selected_rows": len(chosen),
        "selected_ids_sha256": ids_hash([str(row["material_id"]) for row in chosen]),
        "stratum_counts": dict(sorted(counts.items())),
        "purpose": "engineering_subset_not_population_benchmark",
    }
    return chosen, manifest


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                     dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                                     delete=False) as handle:
        temp = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp, path)


def _atomic_torch(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp",
                                     delete=False) as handle:
        temp = Path(handle.name)
        torch.save(value, handle)
    os.replace(temp, path)


def _check_arrays(latents: torch.Tensor, conditions: torch.Tensor, rows: list[dict]) -> None:
    if latents.ndim != 2 or conditions.shape != (latents.shape[0], 2):
        raise ValueError("Cache tensors require [N,D] latents and [N,2] conditions")
    if latents.dtype != torch.float32 or conditions.dtype != torch.float32:
        raise ValueError("Cache tensors must be float32")
    if latents.device.type != "cpu" or conditions.device.type != "cpu":
        raise ValueError("Cache tensors must be on CPU")
    if len(rows) != len(latents):
        raise ValueError("Cache metadata row count disagrees with tensors")
    ids = [row["material_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate material IDs in cache")
    if not bool(torch.isfinite(latents).all() and torch.isfinite(conditions).all()):
        raise ValueError("Nonfinite cache tensors")
    for index, row in enumerate(rows):
        expected = torch.tensor([float(row["heat_all"]), float(row["dir_gap"])], dtype=torch.float32)
        if not torch.equal(conditions[index], expected):
            raise ValueError(f"Condition/metadata mismatch for {row['material_id']}")


def load_cache(cache_dir: Path, expected_fingerprint: str | None = None) -> dict:
    """Load a verified snapshot with ordered IDs, float32 latents and raw properties.

    The manifest fingerprint covers source and preprocessing metadata; the
    snapshot digest covers the stored tensors and per-row metadata.
    """
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != CACHE_SCHEMA:
        raise ValueError("Unknown cache schema")
    snapshot_fields = {"fingerprint", "status", "processed_rows", "processed_ids_sha256",
                       "latent_dim", "dtype", "device", "latent_definition", "condition_order",
                       "snapshot_file", "snapshot_sha256"}
    if json_hash({key: value for key, value in manifest.items() if key not in snapshot_fields}) != manifest["fingerprint"]:
        raise ValueError("Cache source/preprocessing fingerprint mismatch")
    if expected_fingerprint is not None and manifest["fingerprint"] != expected_fingerprint:
        raise ValueError("Stale or incompatible cache fingerprint")
    snapshot = cache_dir / manifest["snapshot_file"]
    if sha256_file(snapshot) != manifest["snapshot_sha256"]:
        raise ValueError("Cache snapshot hash mismatch")
    payload = torch.load(snapshot, map_location="cpu", weights_only=True)
    if payload["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Cache manifest/snapshot fingerprint mismatch")
    latents, conditions, rows = payload["latents"], payload["conditions"], payload["rows"]
    _check_arrays(latents, conditions, rows)
    if manifest["processed_rows"] != len(rows) or manifest["latent_dim"] != latents.shape[1]:
        raise ValueError("Cache manifest count or dimension mismatch")
    if ids_hash([row["material_id"] for row in rows]) != manifest["processed_ids_sha256"]:
        raise ValueError("Cache ordered-ID hash mismatch")
    if manifest["status"] not in {"partial", "complete"}:
        raise ValueError("Invalid cache status")
    if manifest["status"] == "complete" and len(rows) != manifest["selected_rows"]:
        raise ValueError("Complete cache does not include every selected row")
    if (manifest["status"] == "complete"
            and ids_hash([row["material_id"] for row in rows]) != manifest["selected_ids_sha256"]):
        raise ValueError("Complete cache selected-ID hash mismatch")
    return {"latents": latents, "conditions": conditions, "rows": rows,
            "ids": [row["material_id"] for row in rows], "manifest": manifest}


def cache_batches(cache_dir: Path, selected_rows: list[dict], encode_batch: Callable[[list[dict]], torch.Tensor],
                  base_manifest: dict, batch_size: int = 32) -> dict:
    """Encode a stable selected-ID suffix and commit an atomic snapshot per batch.

    ``encode_batch`` receives ordered source rows and returns [batch, latent_dim]
    latents. Conditions retain the original [heat_all, dir_gap] values.
    """
    if batch_size <= 0 or not selected_rows:
        raise ValueError("Positive batch size and nonempty selection required")
    ids = [row["material_id"] for row in selected_rows]
    if len(set(ids)) != len(ids) or ids != sorted(ids):
        raise ValueError("Selected IDs must be unique and sorted")
    expected = dict(base_manifest)
    expected.update({"schema": CACHE_SCHEMA, "selected_rows": len(ids),
                     "selected_ids_sha256": ids_hash(ids)})
    fingerprint = json_hash(expected)
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        prior = load_cache(cache_dir, expected_fingerprint=fingerprint)
        if prior["ids"] != ids[:len(prior["ids"])]:
            raise ValueError("Partial cache ID ordering is incompatible with selection")
        if prior["manifest"]["status"] == "complete":
            return prior
        latents, conditions, cached_rows = prior["latents"], prior["conditions"], prior["rows"]
    else:
        latents = torch.empty((0, 0), dtype=torch.float32)
        conditions = torch.empty((0, 2), dtype=torch.float32)
        cached_rows = []
    for start in range(len(cached_rows), len(selected_rows), batch_size):
        batch = selected_rows[start:start + batch_size]
        try:
            encoded = encode_batch(batch)
            encoded = encoded.detach().to(device="cpu", dtype=torch.float32)
            batch_conditions = torch.tensor(
                [[float(row["heat_all"]), float(row["dir_gap"])] for row in batch],
                dtype=torch.float32,
            )
            if encoded.ndim != 2 or encoded.shape[0] != len(batch):
                raise ValueError("Encoder returned a tensor with wrong batch shape")
            if len(cached_rows) and encoded.shape[1] != latents.shape[1]:
                raise ValueError("Encoder changed latent dimension during resume")
            latents = encoded if not len(cached_rows) else torch.cat([latents, encoded], dim=0)
            conditions = torch.cat([conditions, batch_conditions], dim=0)
            cached_rows.extend(batch)
            _check_arrays(latents, conditions, cached_rows)
        except Exception as error:
            _atomic_json(cache_dir / "failure.json", {
                "failed_batch_ids": [row["material_id"] for row in batch],
                "processed_rows_before_failed_batch": start,
                "error_type": type(error).__name__, "error": str(error),
            })
            raise
        count = len(cached_rows)
        snapshot_file = f"cache_{count:06d}.pt"
        _atomic_torch(cache_dir / snapshot_file, {
            "fingerprint": fingerprint, "latents": latents,
            "conditions": conditions, "rows": cached_rows,
        })
        manifest = expected | {
            "fingerprint": fingerprint,
            "status": "complete" if count == len(selected_rows) else "partial",
            "processed_rows": count,
            "processed_ids_sha256": ids_hash(ids[:count]),
            "latent_dim": int(latents.shape[1]),
            "dtype": "float32", "device": "cpu",
            "latent_definition": "z_joint=(normalize(proj_crystal(normalize(z_crystal_raw)))+normalize(proj_prop(normalize(z_prop_raw))))/2; no final unit normalization",
            "condition_order": ["heat_all", "dir_gap"],
            "snapshot_file": snapshot_file,
            "snapshot_sha256": sha256_file(cache_dir / snapshot_file),
        }
        _atomic_json(manifest_path, manifest)
    return load_cache(cache_dir, expected_fingerprint=fingerprint)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.dtype != torch.float32 or tensor.device.type != "cpu":
        raise ValueError("Scaler statistics must be CPU float32")
    return tensor.contiguous().numpy().astype("<f4", copy=False).tobytes(order="C")


def scaler_hash(scalers: dict) -> str:
    digest = hashlib.sha256()
    header = {key: scalers[key] for key in ("schema", "train_cache_fingerprint", "std_floor")}
    for name in ("latent", "condition"):
        header[name] = {"floored_dimensions": scalers[name]["floored_dimensions"],
                        "dimensions": int(scalers[name]["mean"].numel())}
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    for name in ("latent", "condition"):
        for statistic in ("mean", "std"):
            digest.update(_tensor_bytes(scalers[name][statistic]))
    return digest.hexdigest()


def validate_scalers(scalers: dict) -> None:
    if scalers.get("schema") != SCALER_SCHEMA:
        raise ValueError("Unknown scaler schema")
    if not math.isfinite(float(scalers["std_floor"])) or float(scalers["std_floor"]) <= 0:
        raise ValueError("Scaler floor must be finite and positive")
    for name in ("latent", "condition"):
        mean, std = scalers[name]["mean"], scalers[name]["std"]
        if mean.ndim != 1 or mean.shape != std.shape:
            raise ValueError(f"Invalid {name} scaler shape")
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
            raise ValueError(f"Invalid {name} scaler values")
    if scalers.get("hash") != scaler_hash(scalers):
        raise ValueError("Scaler hash mismatch")


def fit_scalers(train_cache: dict, std_floor: float = 1e-6) -> dict:
    """Fit TRAIN-only featurewise population statistics for latent and property axes.

    Values below ``std_floor`` are floored, with affected dimensions recorded
    in the scaler dictionary and covered by its hash.
    """
    if train_cache["manifest"].get("split") != "train":
        raise ValueError("Only a training cache may fit diffusion scalers")
    if not math.isfinite(std_floor) or std_floor <= 0:
        raise ValueError("std_floor must be finite and positive")
    scalers = {"schema": SCALER_SCHEMA, "train_cache_fingerprint": train_cache["manifest"]["fingerprint"],
               "std_floor": float(std_floor), "training_rows": len(train_cache["ids"]),
               "training_cache_status": train_cache["manifest"]["status"]}
    for name, values in (("latent", train_cache["latents"]),
                         ("condition", train_cache["conditions"])):
        if values.ndim != 2 or len(values) == 0 or not bool(torch.isfinite(values).all()):
            raise ValueError(f"Invalid training values for {name} scaler")
        mean = values.mean(dim=0)
        observed_std = values.std(dim=0, unbiased=False)
        floored = torch.nonzero(observed_std < std_floor).flatten().tolist()
        scalers[name] = {"mean": mean.cpu().float(),
                         "std": observed_std.clamp_min(std_floor).cpu().float(),
                         "floored_dimensions": [int(index) for index in floored]}
    scalers["hash"] = scaler_hash(scalers)
    validate_scalers(scalers)
    return scalers


def apply_scalers(latents: torch.Tensor, conditions: torch.Tensor,
                  scalers: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return standardized latents and [heat_all, dir_gap] properties."""
    validate_scalers(scalers)
    latent_values, property_values = latents.float(), conditions.float()
    if latent_values.shape[-1] != scalers["latent"]["mean"].numel() or property_values.shape[-1] != 2:
        raise ValueError("Data/scaler dimensions do not match")
    return ((latent_values - scalers["latent"]["mean"].to(latent_values.device))
            / scalers["latent"]["std"].to(latent_values.device),
            (property_values - scalers["condition"]["mean"].to(property_values.device))
            / scalers["condition"]["std"].to(property_values.device))


def invert_latents(standardized: torch.Tensor, scalers: dict) -> torch.Tensor:
    """Invert TRAIN-fitted latent scaling before frozen MEIDNet decoding."""
    validate_scalers(scalers)
    if standardized.shape[-1] != scalers["latent"]["mean"].numel():
        raise ValueError("Latent/scaler dimensions do not match")
    return standardized * scalers["latent"]["std"].to(standardized.device) + scalers["latent"]["mean"].to(standardized.device)
