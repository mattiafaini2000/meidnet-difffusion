#!/usr/bin/env python3
"""Validate the prepared MEIDNet audit data and the documented checkpoint.

This is deliberately a setup smoke test, not a decoder-quality audit.  It
checks that the real repository loader can read every audit example and that
the checkpoint is structurally compatible with the repository model.  It does
not run an encode/decode pass while the CMR heat-label reference compatibility
with the pretrained checkpoint remains unresolved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEIDNET_ROOT = PROJECT_ROOT / "MEIDNet-main"
DEFAULT_AUDIT_CSV = PROJECT_ROOT / "data" / "processed" / "cmr_reconstructed" / "audit.csv"
DEFAULT_VALIDATION_CIFS = (
    PROJECT_ROOT / "data" / "processed" / "cmr_reconstructed" / "cifs" / "val"
)
DEFAULT_CHECKPOINT = (
    MEIDNET_ROOT
    / "checkpoints"
    / "dual_autoencoder_clip_earlyfusion_propertyaware_2k.pth"
)
DEFAULT_REPORT = PROJECT_ROOT / "setup_smoke_report.json"
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--cif-root", type=Path, default=DEFAULT_VALIDATION_CIFS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Small non-shuffled loader batch used for the smoke test (default: 4).",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def require_directory(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def read_audit_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"material_id", "heat_all", "dir_gap", "cif_filename"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Audit CSV is missing fields: {sorted(missing)}")

        rows: list[dict[str, Any]] = []
        for line_number, raw in enumerate(reader, start=2):
            material_id = (raw.get("material_id") or "").strip()
            if not material_id:
                raise ValueError(f"Blank material_id at CSV line {line_number}")
            heat_all = float(raw["heat_all"])
            dir_gap = float(raw["dir_gap"])
            if not math.isfinite(heat_all) or not math.isfinite(dir_gap):
                raise ValueError(f"Non-finite label for {material_id} at CSV line {line_number}")
            rows.append(
                {
                    "material_id": material_id,
                    "heat_all": heat_all,
                    "dir_gap": dir_gap,
                    "cif_filename": (raw.get("cif_filename") or "").strip(),
                }
            )

    if not rows:
        raise ValueError(f"Audit CSV has no data rows: {path}")
    counts = Counter(row["material_id"] for row in rows)
    duplicates = sorted(material_id for material_id, count in counts.items() if count != 1)
    if duplicates:
        raise ValueError(f"Duplicate audit material_id values: {duplicates}")
    return rows


def checkpoint_summary(path: Path) -> tuple[Any, dict[str, Any]]:
    import torch

    with path.open("rb") as handle:
        prefix = handle.read(len(LFS_POINTER_PREFIX))
    if prefix == LFS_POINTER_PREFIX:
        raise ValueError(f"Checkpoint is a Git LFS pointer, not model weights: {path}")

    file_size = path.stat().st_size
    if file_size <= len(LFS_POINTER_PREFIX):
        raise ValueError(f"Checkpoint is unexpectedly small ({file_size} bytes): {path}")

    # weights_only prevents arbitrary global imports during deserialization.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint root must be a dict, got {type(checkpoint).__name__}")

    required_metadata = {
        "latent_dim_struct",
        "latent_dim_prop",
        "latent_dim_common",
        "max_sites",
        "num_species",
        "species_embedding_dim",
        "model_state_dict",
    }
    missing = required_metadata.difference(checkpoint)
    if missing:
        raise KeyError(f"Checkpoint is missing fields: {sorted(missing)}")

    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, dict) or not state_dict:
        raise TypeError("model_state_dict must be a non-empty dict")
    bad_entries = [key for key, value in state_dict.items() if not isinstance(key, str) or not torch.is_tensor(value)]
    if bad_entries:
        raise TypeError(f"Non-tensor or non-string state_dict entries: {bad_entries[:5]}")

    tensor_inventory = [
        {
            "key": key,
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
        }
        for key, value in sorted(state_dict.items())
    ]
    metadata = {
        key: int(checkpoint[key])
        for key in sorted(required_metadata - {"model_state_dict"})
    }
    summary = {
        "path": str(path),
        "size_bytes": file_size,
        "sha256": sha256_file(path),
        "git_lfs_pointer": False,
        "safe_weights_only_load": True,
        "metadata": metadata,
        "state_dict_tensor_count": len(state_dict),
        "state_dict_value_count": sum(value.numel() for value in state_dict.values()),
        "state_dict_tensors": tensor_inventory,
    }
    return checkpoint, summary


def instantiate_model(checkpoint: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    import torch

    source_root = str(MEIDNET_ROOT.resolve())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

    from meidnet.model import (
        MAX_SITES,
        NUM_SPECIES,
        DualAutoencoderModel,
        PropertyDecoder,
        PropertyEncoder,
        SE3Decoder,
        SE3Encoder,
    )

    latent_struct = int(checkpoint["latent_dim_struct"])
    latent_prop = int(checkpoint["latent_dim_prop"])
    latent_common = int(checkpoint["latent_dim_common"])
    max_sites = int(checkpoint["max_sites"])
    num_species = int(checkpoint["num_species"])
    species_embedding_dim = int(checkpoint["species_embedding_dim"])

    if max_sites != MAX_SITES or num_species != NUM_SPECIES:
        raise ValueError(
            "Checkpoint dimensions disagree with repository constants: "
            f"checkpoint max_sites={max_sites}, num_species={num_species}; "
            f"source max_sites={MAX_SITES}, num_species={NUM_SPECIES}"
        )

    crystal_encoder = SE3Encoder(
        max_sites,
        num_species,
        latent_struct,
        node_hidden_dim=128,
        species_embedding_dim=species_embedding_dim,
    )
    crystal_decoder = SE3Decoder(
        max_sites,
        num_species,
        latent_common,
        node_hidden_dim=128,
        species_embedding_dim=species_embedding_dim,
    )
    property_encoder = PropertyEncoder(hidden_dim=128, latent_dim=latent_prop)
    property_decoder = PropertyDecoder(latent_dim_common=latent_common)
    model = DualAutoencoderModel(
        crystal_encoder,
        crystal_decoder,
        property_encoder,
        property_decoder,
        latent_dim_common=latent_common,
        max_sites=max_sites,
        num_species=num_species,
    )

    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict state_dict load returned incompatible keys: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("Model was not fully frozen")

    return model, {
        "constructor_compatible": True,
        "strict_state_dict_load": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_mode": model.training,
        "all_parameters_frozen": True,
        "device": str(next(model.parameters()).device),
        "torch_version": torch.__version__,
    }


def validate_dataset(audit_csv: Path, cif_root: Path, rows: list[dict[str, Any]], batch_size: int) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    source_root = str(MEIDNET_ROOT.resolve())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from meidnet.model import MAX_SITES, NUM_SPECIES, TripleModalityDataset

    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    expected = {row["material_id"]: row for row in rows}
    missing_cifs = []
    mismatched_filenames = []
    for row in rows:
        expected_name = f"{row['material_id']}.cif"
        if row["cif_filename"] != expected_name:
            mismatched_filenames.append(
                {"material_id": row["material_id"], "csv": row["cif_filename"], "expected": expected_name}
            )
        if not (cif_root / expected_name).is_file():
            missing_cifs.append(expected_name)
    if mismatched_filenames:
        raise ValueError(f"Audit CIF filename mismatches: {mismatched_filenames[:5]}")
    if missing_cifs:
        raise FileNotFoundError(f"Audit CIF files missing from validation directory: {missing_cifs[:5]}")

    dataset = TripleModalityDataset(str(cif_root), str(audit_csv))
    loaded_ids = [material_id for _, material_id in dataset.samples]
    loaded_counts = Counter(loaded_ids)
    duplicate_loaded = sorted(material_id for material_id, count in loaded_counts.items() if count != 1)
    missing_ids = sorted(set(expected).difference(loaded_counts))
    unexpected_ids = sorted(set(loaded_counts).difference(expected))
    if duplicate_loaded or missing_ids or unexpected_ids or len(dataset) != len(rows):
        raise AssertionError(
            "The real loader did not preserve the exact audit membership: "
            f"expected={len(rows)}, loaded={len(dataset)}, duplicate={duplicate_loaded}, "
            f"missing={missing_ids}, unexpected={unexpected_ids}"
        )

    expected_vector_length = 6 + MAX_SITES * MAX_SITES + MAX_SITES * NUM_SPECIES + MAX_SITES * 3
    for index in range(len(dataset)):
        sample = dataset[index]
        material_id = sample["material_id"]
        if sample["crystal_vec"].shape != (expected_vector_length,):
            raise AssertionError(
                f"Unexpected crystal vector shape for {material_id}: {tuple(sample['crystal_vec'].shape)}"
            )
        if not torch.isfinite(sample["crystal_vec"]).all():
            raise AssertionError(f"Non-finite crystal vector for {material_id}")
        if not torch.isfinite(sample["heat_all"]) or not torch.isfinite(sample["dir_gap"]):
            raise AssertionError(f"Non-finite loader label for {material_id}")

        expected_heat = torch.tensor(expected[material_id]["heat_all"], dtype=torch.float32)
        expected_gap = torch.tensor(expected[material_id]["dir_gap"], dtype=torch.float32)
        if not torch.equal(sample["heat_all"], expected_heat):
            raise AssertionError(f"heat_all changed for {material_id}")
        if not torch.equal(sample["dir_gap"], expected_gap):
            raise AssertionError(f"dir_gap changed for {material_id}")

    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)
    batch = next(iter(loader))
    batch_ids = list(batch["material_id"])
    expected_batch_size = min(batch_size, len(dataset))
    if batch["crystal_vec"].shape != (expected_batch_size, expected_vector_length):
        raise AssertionError(f"Unexpected batch crystal shape: {tuple(batch['crystal_vec'].shape)}")
    if batch["heat_all"].shape != (expected_batch_size,) or batch["dir_gap"].shape != (expected_batch_size,):
        raise AssertionError("Unexpected batch label shapes")
    if not all(torch.isfinite(batch[key]).all() for key in ("crystal_vec", "heat_all", "dir_gap")):
        raise AssertionError("Non-finite values in smoke-test batch")
    for position, material_id in enumerate(batch_ids):
        expected_heat = torch.tensor(expected[material_id]["heat_all"], dtype=torch.float32)
        expected_gap = torch.tensor(expected[material_id]["dir_gap"], dtype=torch.float32)
        if not torch.equal(batch["heat_all"][position], expected_heat):
            raise AssertionError(f"Batched heat_all changed for {material_id}")
        if not torch.equal(batch["dir_gap"][position], expected_gap):
            raise AssertionError(f"Batched dir_gap changed for {material_id}")

    return {
        "audit_row_count": len(rows),
        "loader_sample_count": len(dataset),
        "all_intended_ids_loaded_exactly_once": True,
        "labels_match_csv_after_expected_float32_conversion": True,
        "all_crystal_vectors_finite": True,
        "crystal_vector_length": expected_vector_length,
        "batch_size": expected_batch_size,
        "batch_shapes": {
            "crystal_vec": list(batch["crystal_vec"].shape),
            "heat_all": list(batch["heat_all"].shape),
            "dir_gap": list(batch["dir_gap"].shape),
            "material_id_count": len(batch_ids),
        },
        "batch_finite": True,
        "audit_material_ids": sorted(expected),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    args = parse_args()
    report_path = args.report.expanduser().resolve()
    report: dict[str, Any] = {
        "status": "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpreter": sys.executable,
        "project_root": str(PROJECT_ROOT),
        "meidnet_source_root": str(MEIDNET_ROOT),
        "model_forward": {
            "status": "blocked_not_run",
            "reason": (
                "The CMR heat_of_formation_all reference convention is not yet established as "
                "compatible with the pretrained checkpoint's heat_all training labels. A "
                "property-conditioned encode/decode result would therefore be scientifically "
                "ambiguous. No encode/decode forward pass was executed."
            ),
        },
    }

    try:
        audit_csv = require_file(args.audit_csv, "audit CSV")
        cif_root = require_directory(args.cif_root, "validation CIF directory")
        checkpoint_path = require_file(args.checkpoint, "checkpoint")
        rows = read_audit_rows(audit_csv)

        checkpoint, checkpoint_report = checkpoint_summary(checkpoint_path)
        model, model_report = instantiate_model(checkpoint)
        # Retain the model until all structural checks finish; deliberately do not call it.
        del model
        dataset_report = validate_dataset(audit_csv, cif_root, rows, args.batch_size)

        report.update(
            {
                "status": "passed",
                "checkpoint": checkpoint_report,
                "model_compatibility": model_report,
                "dataset": {
                    "audit_csv": str(audit_csv),
                    "cif_root": str(cif_root),
                    **dataset_report,
                },
            }
        )
        write_json_atomic(report_path, report)
        print(f"Setup smoke validation passed: {report_path}")
        print(
            f"  checkpoint: {checkpoint_report['sha256']} "
            f"({checkpoint_report['state_dict_tensor_count']} tensors, strict load)"
        )
        print(
            f"  dataset: {dataset_report['loader_sample_count']}/"
            f"{dataset_report['audit_row_count']} audit IDs loaded; one "
            f"batch of {dataset_report['batch_size']} is finite"
        )
        print("  encode/decode forward: BLOCKED/NOT RUN (property-label semantics unresolved)")
        return 0
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        write_json_atomic(report_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
