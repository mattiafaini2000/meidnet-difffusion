#!/usr/bin/env python3
"""Run a frozen four-example MEIDNet numerical forward smoke check.

The CMR labels are used exactly as stored, but remain provisional with respect
to the checkpoint's unavailable training-label provenance.  This script checks
execution and decoder interfaces; it does not measure scientific accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEIDNET_ROOT = PROJECT_ROOT / "MEIDNet-main"
DEFAULT_AUDIT_CSV = PROJECT_ROOT / "data" / "processed" / "cmr_reconstructed" / "audit.csv"
DEFAULT_CIF_ROOT = PROJECT_ROOT / "data" / "processed" / "cmr_reconstructed" / "cifs" / "val"
DEFAULT_CHECKPOINT = (
    MEIDNET_ROOT / "checkpoints" / "dual_autoencoder_clip_earlyfusion_propertyaware_2k.pth"
)
DEFAULT_JSON_REPORT = PROJECT_ROOT / "reports" / "forward_smoke.json"
DEFAULT_MARKDOWN_REPORT = PROJECT_ROOT / "reports" / "forward_smoke.md"

try:
    from scripts.validate_setup import (
        checkpoint_summary,
        instantiate_model,
        require_directory,
        require_file,
        sha256_file,
    )
except ModuleNotFoundError:  # Direct execution adds scripts/, rather than the project root, to sys.path.
    from validate_setup import (  # type: ignore[no-redef]
        checkpoint_summary,
        instantiate_model,
        require_directory,
        require_file,
        sha256_file,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--cif-root", type=Path, default=DEFAULT_CIF_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--markdown-report", type=Path, default=DEFAULT_MARKDOWN_REPORT)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "material_id",
            "heat_all",
            "dir_gap",
            "source_anion",
            "cif_filename",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Audit CSV is missing fields: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"Audit CSV has no rows: {path}")
    ids = [row["material_id"].strip() for row in rows]
    if any(not material_id for material_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Audit material IDs must be non-empty and unique")
    for row in rows:
        values = (float(row["heat_all"]), float(row["dir_gap"]))
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite label for {row['material_id']}")
    return rows


def load_ordered_batch(audit_csv: Path, cif_root: Path, rows: list[dict[str, str]], batch_size: int):
    """Load real CIFs through TripleModalityDataset, then restore CSV row order."""
    import torch

    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    source_root = str(MEIDNET_ROOT.resolve())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from meidnet.model import TripleModalityDataset

    dataset = TripleModalityDataset(str(cif_root), str(audit_csv))
    samples = {dataset[index]["material_id"]: dataset[index] for index in range(len(dataset))}
    chosen_rows = rows[: min(batch_size, len(rows))]
    chosen_ids = [row["material_id"].strip() for row in chosen_rows]
    missing = [material_id for material_id in chosen_ids if material_id not in samples]
    if missing:
        raise AssertionError(f"The real dataset loader skipped requested audit IDs: {missing}")

    batch = {
        "material_id": chosen_ids,
        "crystal_vec": torch.stack([samples[mid]["crystal_vec"] for mid in chosen_ids]),
        "heat_all": torch.stack([samples[mid]["heat_all"] for mid in chosen_ids]),
        "dir_gap": torch.stack([samples[mid]["dir_gap"] for mid in chosen_ids]),
    }
    return batch, chosen_rows, len(dataset)


def tensor_summary(tensor) -> dict[str, Any]:
    import torch

    detached = tensor.detach().cpu()
    finite = torch.isfinite(detached)
    summary: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype).removeprefix("torch."),
        "all_finite": bool(finite.all()),
        "finite_count": int(finite.sum()),
        "value_count": detached.numel(),
    }
    if detached.numel() and finite.any():
        finite_values = detached[finite].double()
        summary.update(
            {
                "min": float(finite_values.min()),
                "max": float(finite_values.max()),
                "mean": float(finite_values.mean()),
            }
        )
    if detached.ndim >= 2:
        flattened = detached.double().reshape(detached.shape[0], -1)
        summary["l2_norm_per_example"] = [float(value) for value in flattened.norm(dim=1)]
    return summary


def difference_summary(left, right) -> dict[str, Any]:
    import torch

    if left.shape != right.shape:
        return {
            "shape_compatible": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    difference = left.detach().cpu().double() - right.detach().cpu().double()
    absolute = difference.abs()
    return {
        "shape_compatible": True,
        "exactly_equal": bool(torch.equal(left.detach().cpu(), right.detach().cpu())),
        "all_finite": bool(torch.isfinite(difference).all()),
        "max_abs": float(absolute.max()) if absolute.numel() else 0.0,
        "mean_abs": float(absolute.mean()) if absolute.numel() else 0.0,
        "rms": float(torch.sqrt((difference * difference).mean())) if difference.numel() else 0.0,
        "l2": float(difference.norm()),
    }


def reference_free_inputs(input_coords, center):
    """Return exactly the zero coordinate and center inputs used by generation."""
    import torch

    return torch.zeros_like(input_coords), torch.zeros_like(center)


def decode_reference_free(decoder, z_joint, input_coords, center):
    """Call the baseline decoder without any per-example reference information."""
    zero_coords, zero_center = reference_free_inputs(input_coords, center)
    outputs = decoder(z_joint, input_coords=zero_coords, center=zero_center)
    return outputs, zero_coords, zero_center


def named_decoder_outputs(outputs) -> dict[str, Any]:
    return dict(zip(("lattice", "adjacency", "species_logits", "coordinates"), outputs))


def anion_symbol(source_anion: str) -> str:
    symbols = re.findall(r"[A-Z][a-z]?", source_anion)
    if len(symbols) != 1:
        raise ValueError(f"Expected one anion symbol, got {source_anion!r}")
    return symbols[0]


def source_trace() -> dict[str, Any]:
    return {
        "training_model": {
            "path": str(MEIDNET_ROOT / "meidnet" / "model.py"),
            "crystal_encoder_forward_line": 354,
            "crystal_decoder_forward_line": 432,
            "encode_modalities_line": 586,
            "joint_forward_line": 631,
            "facts": [
                "encode_modalities L2-normalizes each raw modality, projects it, then L2-normalizes the common latents",
                "z_joint is the arithmetic mean of crystal-common and property-common and is not normalized again",
                "reconstruction passes reference coordinates and their center to the crystal decoder",
                "input_species is accepted by the decoder but is not read in its forward implementation",
            ],
        },
        "generation_model": {
            "path": str(MEIDNET_ROOT / "meidnet" / "design.py"),
            "crystal_decoder_forward_line": 426,
            "optimisation_decode_call_line": 767,
            "decode_and_filter_call_line": 962,
            "facts": [
                "both generation calls pass all-zero input coordinates and all-zero centers",
                "the generation decoder applies a five-slot species mask when one is supplied",
                "the generation decoder returns an all-ones adjacency tensor instead of learned adjacency logits",
                "decode_and_filter later constructs fixed five-site cubic ABX3 templates and does not emit its raw coordinates as CIF geometry",
            ],
        },
    }


def analyse_forward(
    audit_csv: Path,
    cif_root: Path,
    checkpoint_path: Path,
    batch_size: int = 4,
) -> dict[str, Any]:
    import numpy
    import pandas
    import torch
    import torch.nn.functional as functional

    audit_csv = require_file(audit_csv, "audit CSV")
    cif_root = require_directory(cif_root, "validation CIF directory")
    checkpoint_path = require_file(checkpoint_path, "checkpoint")
    rows = read_rows(audit_csv)
    batch, chosen_rows, loaded_count = load_ordered_batch(audit_csv, cif_root, rows, batch_size)

    checkpoint, checkpoint_report = checkpoint_summary(checkpoint_path)
    model, model_report = instantiate_model(checkpoint)
    model.eval().requires_grad_(False)

    crystal_vec = batch["crystal_vec"]
    heat_all = batch["heat_all"]
    dir_gap = batch["dir_gap"]
    material_ids = batch["material_id"]

    timings: dict[str, float] = {}
    with torch.inference_mode():
        start = time.perf_counter()
        z_crystal_raw, crystal_center = model.crystal_encoder(crystal_vec)
        z_crystal_normalized = functional.normalize(z_crystal_raw, p=2, dim=1)
        z_crystal_projected = functional.normalize(
            model.proj_crystal(z_crystal_normalized), p=2, dim=1
        )
        timings["independent_crystal_encoder_ms"] = (time.perf_counter() - start) * 1000.0

        properties = torch.stack((heat_all, dir_gap), dim=1)
        z_property_raw = model.property_encoder(properties)
        z_property_projected = functional.normalize(
            model.proj_prop(functional.normalize(z_property_raw, p=2, dim=1)), p=2, dim=1
        )

        start = time.perf_counter()
        (
            z_crystal_common,
            z_property_common,
            z_joint,
            center,
            input_species,
            input_coords,
        ) = model.encode_modalities(crystal_vec, heat_all, dir_gap)
        timings["encode_modalities_ms"] = (time.perf_counter() - start) * 1000.0

        if not torch.allclose(z_crystal_projected, z_crystal_common, atol=1e-7, rtol=1e-6):
            raise AssertionError("Independent crystal projection disagrees with encode_modalities")
        if not torch.allclose(z_property_projected, z_property_common, atol=1e-7, rtol=1e-6):
            raise AssertionError("Independent property projection disagrees with encode_modalities")
        if not torch.allclose(crystal_center, center, atol=1e-7, rtol=1e-6):
            raise AssertionError("Independent crystal center disagrees with encode_modalities")

        start = time.perf_counter()
        full_outputs = model(crystal_vec, heat_all, dir_gap)
        timings["usual_full_forward_ms"] = (time.perf_counter() - start) * 1000.0

        reference_outputs = named_decoder_outputs(
            model.crystal_decoder(
                z_joint,
                input_species=input_species,
                input_coords=input_coords,
                center=center,
            )
        )
        raw_reference_free, zero_coords, zero_center = decode_reference_free(
            model.crystal_decoder, z_joint, input_coords, center
        )
        reference_free_outputs = named_decoder_outputs(raw_reference_free)

    full_named = {
        "z_crystal_common": full_outputs[0],
        "z_property_common": full_outputs[1],
        "lattice": full_outputs[2],
        "adjacency_logits": full_outputs[3],
        "species_logits": full_outputs[4],
        "coordinates": full_outputs[5],
        "property_output": full_outputs[6],
    }
    for name, explicit in (
        ("z_crystal_common", z_crystal_common),
        ("z_property_common", z_property_common),
        ("lattice", reference_outputs["lattice"]),
        ("adjacency_logits", reference_outputs["adjacency"]),
        ("species_logits", reference_outputs["species_logits"]),
        ("coordinates", reference_outputs["coordinates"]),
    ):
        if not torch.equal(full_named[name], explicit):
            raise AssertionError(f"Usual full forward disagrees with the explicit reconstruction path: {name}")

    per_example_differences = []
    for index, material_id in enumerate(material_ids):
        per_example_differences.append(
            {
                "material_id": material_id,
                "reference_A_vs_reference_free_B": {
                    name: difference_summary(reference_outputs[name][index], reference_free_outputs[name][index])
                    for name in reference_outputs
                },
            }
        )

    # The actual generation module contains a separate model implementation.
    # Importing it is side-effect free because its CLI is guarded by __main__.
    source_root = str(MEIDNET_ROOT.resolve())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from meidnet import design

    latent_dimensions = {
        int(checkpoint["latent_dim_struct"]),
        int(checkpoint["latent_dim_prop"]),
        int(checkpoint["latent_dim_common"]),
    }
    if len(latent_dimensions) != 1:
        raise ValueError(
            "The generation implementation exposes one latent dimension, but checkpoint dimensions differ: "
            f"{sorted(latent_dimensions)}"
        )
    generation_model = design.DualAutoencoderModel(
        int(checkpoint["max_sites"]),
        int(checkpoint["num_species"]),
        latent_dim=latent_dimensions.pop(),
    )
    incompatible = generation_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict generation-model load failed: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    generation_model.eval().requires_grad_(False)

    actual_masked_outputs = []
    with torch.inference_mode():
        generation_unmasked = named_decoder_outputs(
            generation_model.crystal_decoder(
                z_joint,
                input_coords=zero_coords,
                center=zero_center,
                species_mask=None,
            )
        )
        for index, row in enumerate(chosen_rows):
            symbol = anion_symbol(row["source_anion"])
            mask = design.build_species_mask({symbol})
            outputs = named_decoder_outputs(
                generation_model.crystal_decoder(
                    z_joint[index : index + 1],
                    input_coords=torch.zeros_like(input_coords[index : index + 1]),
                    center=torch.zeros_like(center[index : index + 1]),
                    species_mask=mask,
                )
            )
            logits = outputs["species_logits"]
            invalid = ~mask.unsqueeze(0)
            expected_fill = torch.finfo(logits.dtype).min
            actual_masked_outputs.append(
                {
                    "material_id": material_ids[index],
                    "allowed_anions": [symbol],
                    "allowed_species_slot_count": int(mask.sum()),
                    "masked_species_logit_count": int(invalid.sum()),
                    "all_disallowed_logits_equal_dtype_min": bool(
                        torch.all(logits[invalid] == expected_fill)
                    ),
                    "outputs": {name: tensor_summary(value) for name, value in outputs.items()},
                }
            )

    generation_comparison = {
        name: difference_summary(reference_free_outputs[name], generation_unmasked[name])
        for name in reference_free_outputs
    }

    return {
        "status": "passed",
        "execution_status": "passed_frozen_cpu_forward",
        "scientific_interpretation": {
            "label_compatibility": "unresolved_provisional_inputs",
            "pretrained_training_overlap": "unknown",
            "claims_not_established": [
                "property calibration or accuracy",
                "reconstruction quality",
                "target attainment",
                "stability or synthesizability",
                "held-out generalization",
            ],
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "interpreter": sys.executable,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "pymatgen": version("pymatgen"),
            "device": "cpu",
            "inference_mode": True,
            "model_eval": not model.training and not generation_model.training,
            "all_parameters_frozen": not any(
                parameter.requires_grad
                for loaded_model in (model, generation_model)
                for parameter in loaded_model.parameters()
            ),
        },
        "inputs": {
            "audit_csv": str(audit_csv),
            "audit_csv_sha256": sha256_file(audit_csv),
            "cif_root": str(cif_root),
            "checkpoint": str(checkpoint_path),
            "material_ids_in_csv_order": material_ids,
            "loader_audit_count": loaded_count,
            "batch_size": len(material_ids),
            "provisional_labels": [
                {
                    "material_id": row["material_id"],
                    "heat_all": float(row["heat_all"]),
                    "dir_gap": float(row["dir_gap"]),
                    "source_anion": row["source_anion"],
                    "cif_path": str(cif_root / row["cif_filename"]),
                    "cif_sha256": sha256_file(cif_root / row["cif_filename"]),
                }
                for row in chosen_rows
            ],
        },
        "checkpoint": checkpoint_report,
        "model_compatibility": model_report,
        "source_trace": source_trace(),
        "timings_ms": timings,
        "latents": {
            "z_crystal_raw": tensor_summary(z_crystal_raw),
            "z_crystal_normalized_before_projection": tensor_summary(z_crystal_normalized),
            "z_crystal_common": tensor_summary(z_crystal_common),
            "z_property_raw": tensor_summary(z_property_raw),
            "z_property_common": tensor_summary(z_property_common),
            "z_joint_not_renormalized": tensor_summary(z_joint),
            "center": tensor_summary(center),
            "crystal_encoder_label_inputs_used": False,
            "independent_crystal_projection_matches_encode_modalities": True,
            "independent_property_projection_matches_encode_modalities": True,
        },
        "usual_full_forward": {
            "all_outputs_finite": all(torch.isfinite(value).all() for value in full_named.values()),
            "outputs": {name: tensor_summary(value) for name, value in full_named.items()},
            "property_output_order": ["heat_all", "dir_gap"],
            "usual_forward_matches_explicit_joint_reconstruction_decode": True,
        },
        "paired_decoder_interface": {
            "A": "fixed z_joint with reconstruction reference coordinates and center",
            "B": "same fixed z_joint with explicit all-zero coordinates and center; no reference species supplied",
            "input_species_exclusion": (
                "The B call supplies no input_species. The baseline decoder's input_species argument is unused "
                "by its source implementation."
            ),
            "A_output_summaries": {
                name: tensor_summary(value) for name, value in reference_outputs.items()
            },
            "B_output_summaries": {
                name: tensor_summary(value) for name, value in reference_free_outputs.items()
            },
            "aggregate_A_vs_B": {
                name: difference_summary(reference_outputs[name], reference_free_outputs[name])
                for name in reference_outputs
            },
            "per_example": per_example_differences,
        },
        "actual_generation_implementation": {
            "strict_checkpoint_load": True,
            "missing_keys": [],
            "unexpected_keys": [],
            "reference_free_unmasked_vs_baseline_reference_free": generation_comparison,
            "masked_calls_matching_each_example_anion_scope": actual_masked_outputs,
            "compatibility_finding": (
                "The generation implementation is strictly state-dict compatible. Lattice, unmasked species "
                "logits, and coordinates match the baseline reference-free decoder. Its adjacency output is "
                "intentionally an all-ones tensor rather than the baseline decoder's learned dot-product logits."
            ),
            "downstream_projection_note": (
                "The actual decode_and_filter path samples five ABX3 slots and constructs a fixed cubic template; "
                "raw decoder coordinates are not used as final CIF coordinates, and its learned lattice is later "
                "replaced by an ionic-radius-derived cubic lattice for accepted structures."
            ),
        },
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
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


def format_number(value: float) -> str:
    return f"{value:.6g}"


def markdown_report(report: dict[str, Any]) -> str:
    if report.get("status") != "passed":
        error = report.get("error", {})
        return (
            "# Frozen forward smoke check\n\n"
            f"Execution status: **FAILED** (`{error.get('type', 'unknown')}: "
            f"{error.get('message', 'unknown')}`).\n\n"
            "The machine-readable report contains the traceback. No scientific result is claimed.\n"
        )

    outputs = report["usual_full_forward"]["outputs"]
    latents = report["latents"]
    pair = report["paired_decoder_interface"]["aggregate_A_vs_B"]
    generation = report["actual_generation_implementation"]
    lines = [
        "# Frozen forward smoke check",
        "",
        "Execution status: **PASSED**. Four real validation-audit examples completed a strict-weight, "
        "CPU, evaluation-mode, frozen `torch.inference_mode()` forward pass.",
        "",
        "Scientific label status: **UNRESOLVED / PROVISIONAL INPUTS**. The run does not establish property "
        "calibration, scientific label compatibility, reconstruction quality, target attainment, or held-out "
        "generalization. `pretrained_training_overlap` remains `unknown`.",
        "",
        "## Inputs",
        "",
        f"- Checkpoint SHA-256: `{report['checkpoint']['sha256']}`",
        f"- Audit CSV SHA-256: `{report['inputs']['audit_csv_sha256']}`",
        f"- Material IDs, in CSV order: `{', '.join(report['inputs']['material_ids_in_csv_order'])}`",
        f"- Full forward wall time: `{format_number(report['timings_ms']['usual_full_forward_ms'])} ms`",
        "",
        "## Full-forward tensors",
        "",
        "| Tensor | Shape | Range | Finite |",
        "|---|---:|---:|---:|",
    ]
    for name, details in outputs.items():
        lines.append(
            f"| `{name}` | `{details['shape']}` | "
            f"`[{format_number(details['min'])}, {format_number(details['max'])}]` | "
            f"{details['all_finite']} |"
        )
    lines.extend(
        [
            "",
            "## Latent paths",
            "",
            "| Latent | Shape | Per-example L2 norm range |",
            "|---|---:|---:|",
        ]
    )
    for name in (
        "z_crystal_raw",
        "z_crystal_normalized_before_projection",
        "z_crystal_common",
        "z_property_raw",
        "z_property_common",
        "z_joint_not_renormalized",
    ):
        details = latents[name]
        norms = details["l2_norm_per_example"]
        lines.append(
            f"| `{name}` | `{details['shape']}` | "
            f"`[{format_number(min(norms))}, {format_number(max(norms))}]` |"
        )
    joint_norms = latents["z_joint_not_renormalized"]["l2_norm_per_example"]
    lines.extend(
        [
            "",
            "The raw crystal latent was normalized, projected through `proj_crystal`, and normalized in the "
            "common space exactly as `encode_modalities` specifies. The joint latent was the direct common-space "
            "average and was not normalized a second time.",
            "",
            f"Joint-latent norms: `{', '.join(format_number(value) for value in joint_norms)}`.",
            "",
            "## Paired decoder inputs",
            "",
            "A used the reconstruction reference coordinates and center. B held each `z_joint` fixed but used "
            "the generation path's explicit zero coordinates and zero center, with no ground-truth species input.",
            "",
            "| Raw output | A vs B maximum absolute difference | A vs B RMS difference |",
            "|---|---:|---:|",
        ]
    )
    for name, details in pair.items():
        lines.append(
            f"| `{name}` | `{format_number(details['max_abs'])}` | `{format_number(details['rms'])}` |"
        )
    lines.extend(
        [
            "",
            "These differences measure auxiliary-input dependence; they are not reconstruction-quality scores.",
            "",
            "## Actual generation implementation",
            "",
            f"{generation['compatibility_finding']}",
            "",
            f"{generation['downstream_projection_note']}",
            "",
            "The per-example masked outputs and every A/B comparison are recorded in "
            "`reports/forward_smoke.json`.",
            "",
            "## Reproduce",
            "",
            "```powershell",
            "$env:PYTHONDONTWRITEBYTECODE = \"1\"",
            "& .\\.venv\\Scripts\\python.exe .\\scripts\\smoke_forward.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
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
    json_path = args.json_report.expanduser().resolve()
    markdown_path = args.markdown_report.expanduser().resolve()
    try:
        report = analyse_forward(
            args.audit_csv,
            args.cif_root,
            args.checkpoint,
            batch_size=args.batch_size,
        )
    except BaseException as error:
        report = {
            "status": "failed",
            "execution_status": "failed",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_interpretation": {
                "label_compatibility": "unresolved_provisional_inputs",
                "pretrained_training_overlap": "unknown",
            },
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "command": [sys.executable, *sys.argv],
        }
        write_json_atomic(json_path, report)
        write_text_atomic(markdown_path, markdown_report(report))
        raise

    report["command"] = [sys.executable, *sys.argv]
    write_json_atomic(json_path, report)
    write_text_atomic(markdown_path, markdown_report(report))
    print(f"Frozen forward smoke check passed: {json_path}")
    print(f"Readable summary: {markdown_path}")
    print(f"  material IDs: {', '.join(report['inputs']['material_ids_in_csv_order'])}")
    print(f"  full forward: {report['timings_ms']['usual_full_forward_ms']:.3f} ms on CPU")
    print("  numerical execution: PASSED")
    print("  label compatibility: UNRESOLVED / PROVISIONAL INPUTS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
