#!/usr/bin/env python3
"""Bounded, fresh-process CPU generation comparison through unchanged MF_P4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latent_diffusion.data import invert_latents, load_cache, sha256_file  # noqa: E402
from latent_diffusion.evaluation import (  # noqa: E402
    choose_training_targets, full_composition_key,
    nearest_condition_indices, stable_seed, summarize_attempts,
)
from latent_diffusion.integration import FrozenMEIDNet  # noqa: E402
from latent_diffusion.training import load_frozen_model  # noqa: E402
from scripts.multifamily_census import anion_key  # noqa: E402


METHODS = (
    "conditional_diffusion", "property_only_meidnet",
    "conditional_resample", "unconditional_diffusion",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="new denoiser checkpoint")
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--meidnet-checkpoint", type=Path, required=True)
    parser.add_argument("--template-manifest", type=Path, required=True)
    parser.add_argument("--training-support", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


def check_compatibility(metadata: dict, train_cache: dict, val_cache: dict,
                        adapter: FrozenMEIDNet) -> None:
    compatible = metadata["compatibility"]
    expected = {
        "train_cache_fingerprint": train_cache["manifest"]["fingerprint"],
        "val_cache_fingerprint": val_cache["manifest"]["fingerprint"],
    }
    for key, value in expected.items():
        if compatible[key] != value:
            raise ValueError(f"Denoiser checkpoint/cache mismatch: {key}")
    if compatible["mode"] != "smoke":
        raise ValueError("Generation requires the fresh smoke checkpoint, not overfit weights")
    source = compatible["train_cache_manifest"]
    for key, actual in (
        ("checkpoint_sha256", adapter.checkpoint_sha256),
        ("model_state_sha256", adapter.model_state_sha256),
        ("template_manifest_sha256", adapter.template_sha256),
        ("policy_source_sha256", adapter.policy_sha256),
        ("training_support_sha256", adapter.training_support_sha256),
        ("output_template_sha256", adapter.output_template_sha256),
    ):
        if source[key] != actual:
            raise ValueError(f"Frozen MEIDNet/MF_P4 input changed: {key}")
    for key in (
        "checkpoint_sha256", "model_source_sha256", "model_state_sha256",
        "template_manifest_sha256", "policy_source_sha256",
        "training_support_sha256", "output_template_sha256",
        "template_input_coordinates_sha256", "latent_definition", "latent_dim",
        "condition_order", "vocab",
    ):
        if val_cache["manifest"][key] != source[key]:
            raise ValueError(f"Validation cache disagrees with training cache: {key}")
    adapter.assert_frozen()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict, tuple))
                             else value for key, value in row.items()})


def require_cache_split(cache_dir: Path, expected: str) -> None:
    """Reject a final-test cache before opening its tensor snapshot."""
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("split") != expected:
        raise ValueError(f"Expected {expected} cache, found {manifest.get('split')!r}; no test data opened")


def run() -> int:
    args = arguments()
    if args.threads < 1:
        raise ValueError("Thread count must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable; no fallback")
    torch.set_num_threads(args.threads)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    sample_config = config["sampling"]
    if not (1 <= sample_config["max_targets"] <= 8
            and sample_config["proposals_per_target"] == 8
            and sample_config["ddim_steps"] in (20, 50, 100)
            and sample_config["guidance"] == 1.0):
        raise ValueError("This engineering panel fixes <=8 targets, eight draws, 20/50/100 DDIM steps, guidance 1")
    for cache_dir, split in ((args.train_cache, "train"), (args.val_cache, "val")):
        require_cache_split(cache_dir, split)
    train = load_cache(args.train_cache)
    val = load_cache(args.val_cache)
    if train["manifest"]["split"] != "train" or val["manifest"]["split"] != "val":
        raise ValueError("Only train/validation cache inputs are allowed; no test cache opened")
    if train["manifest"]["status"] != "complete" or val["manifest"]["status"] != "complete":
        raise ValueError("Cache must be complete before generation")
    model, diffusion, scalers, metadata = load_frozen_model(args.checkpoint, args.device)
    adapter = FrozenMEIDNet(args.meidnet_checkpoint, args.template_manifest,
                            args.training_support, device=args.device)
    check_compatibility(metadata, train, val, adapter)
    if model.latent_dim != train["latents"].shape[1] or model.latent_dim != adapter.latent_dim:
        raise ValueError("Denoiser/cache/MEIDNet latent dimensions differ")
    if config["model"] != {key: value for key, value in metadata["compatibility"]["model"].items()
                           if key != "latent_dim"} or config["diffusion"] != metadata["compatibility"]["diffusion"]:
        raise ValueError("Evaluation architecture or noise schedule differs from checkpoint")
    targets = choose_training_targets(train["rows"], sample_config["max_targets"])
    if len(targets) == 0:
        raise ValueError("No training-supported requested property pairs")
    target_ids = [row["material_id"] for row in targets]
    if len(set(target_ids)) != len(target_ids):
        raise AssertionError("Target IDs are not unique")
    # Selection is complete before model outputs are inspected.
    output = args.output_dir
    if output.exists():
        raise FileExistsError(f"Use a new run directory; refusing to overwrite {output}")
    output.mkdir(parents=True)
    (output / "examples").mkdir()
    train_ids = train["ids"]
    subset_compositions = Counter(row["composition_group"] for row in train["rows"])
    subset_anions = Counter(row["source_anion_key"] for row in train["rows"])
    full_compositions = adapter.training_support["full_composition_counts"]
    full_anions = adapter.training_support["exact_anion_counts"]
    raw_latents = train["latents"].to(args.device)
    property_mean = scalers["condition"]["mean"].to(args.device)
    property_std = scalers["condition"]["std"].to(args.device)
    latent_mean = scalers["latent"]["mean"].to(args.device)
    latent_std = scalers["latent"]["std"].to(args.device)
    target_records: list[dict] = []
    attempts: list[dict] = []
    saved_examples = Counter()
    with torch.no_grad():
        for target in targets:
            target_id = target["material_id"]
            raw_properties = torch.tensor([float(target["heat_all"]), float(target["dir_gap"])],
                                          dtype=torch.float32, device=args.device)
            standardized_properties = (raw_properties - property_mean) / property_std
            neighbors = nearest_condition_indices(train["conditions"], raw_properties.cpu(),
                                                  property_std.cpu(), train_ids, k=16)
            support = [(index, distance) for index, distance in neighbors if distance <= 0.5]
            target_records.append({
                "target_id": target_id,
                "heat_all": float(raw_properties[0]), "dir_gap": float(raw_properties[1]),
                "source_anion_stratum_selection_only": target["source_anion_key"],
                "source_family_selection_only": target["source_family"],
                "neighbor_radius_standardized": 0.5,
                "neighbors_within_radius_among_top16": len(support),
                "distinct_groups_within_radius": len({train["rows"][index]["composition_group"]
                                                       for index, _ in support}),
                "nearest16": [{"material_id": train_ids[index], "distance": distance}
                              for index, distance in neighbors],
            })
            property_latent = adapter.property_only_latent(raw_properties)
            for proposal_index in range(sample_config["proposals_per_target"]):
                noise_seed = stable_seed("ddim-noise", target_id, proposal_index) % (2**63 - 1)
                categorical_seed = stable_seed("mf-p4-draw", target_id, proposal_index)
                noise_rng = torch.Generator(device=args.device).manual_seed(noise_seed)
                initial_noise = torch.randn((1, model.latent_dim), device=args.device,
                                            generator=noise_rng)
                batched_properties = standardized_properties[None, :]
                conditional_u = diffusion.ddim_sample(model, batched_properties, initial_noise,
                                                       steps=sample_config["ddim_steps"], guidance=1.0)
                unconditional_u = diffusion.ddim_sample(model, batched_properties, initial_noise,
                                                         steps=sample_config["ddim_steps"], guidance=0.0)
                neighbor_rng = random.Random(stable_seed("empirical-neighbor", target_id, proposal_index))
                sampled_index = neighbors[neighbor_rng.randrange(len(neighbors))][0]
                generated = {
                    "conditional_diffusion": invert_latents(conditional_u, scalers),
                    "property_only_meidnet": property_latent,
                    "conditional_resample": raw_latents[sampled_index:sampled_index + 1],
                    "unconditional_diffusion": invert_latents(unconditional_u, scalers),
                }
                paired_diffusion_l2 = float(torch.linalg.vector_norm(
                    generated["conditional_diffusion"] - generated["unconditional_diffusion"]))
                paired_standardized_l2 = float(torch.linalg.vector_norm(conditional_u - unconditional_u))
                for method in METHODS:
                    latent = generated[method]
                    if not bool(torch.isfinite(latent).all()):
                        raise FloatingPointError(f"Nonfinite generated latent: {method}, {target_id}")
                    decoded = adapter.decode_mf_p4(latent, categorical_seed)
                    roles = decoded["sampled_roles"]
                    composition = full_composition_key(roles) if roles else None
                    anion = anion_key(tuple(roles[2:])) if roles else None
                    standardized_latent = (latent - latent_mean) / latent_std
                    latent_distances = torch.linalg.vector_norm(raw_latents - latent, dim=1)
                    record = {
                        "target_id": target_id,
                        "target_heat_all": float(raw_properties[0]),
                        "target_dir_gap": float(raw_properties[1]),
                        "proposal_index": proposal_index,
                        "method": method,
                        "guidance": (1.0 if method == "conditional_diffusion" else
                                     0.0 if method == "unconditional_diffusion" else None),
                        "initial_noise_seed": noise_seed if method.endswith("diffusion") else None,
                        "paired_condition_unconditioned_latent_l2": (
                            paired_diffusion_l2 if method.endswith("diffusion") else None),
                        "paired_condition_unconditioned_standardized_l2": (
                            paired_standardized_l2 if method.endswith("diffusion") else None),
                        "categorical_seed": categorical_seed,
                        "resampled_training_id": train_ids[sampled_index] if method == "conditional_resample" else None,
                        "latent_norm": float(latent.norm()),
                        "latent_min": float(latent.min()), "latent_max": float(latent.max()),
                        "standardized_latent_norm": float(standardized_latent.norm()),
                        "nearest_cached_latent_distance": float(latent_distances.min()),
                        "sampled": decoded["sampled"],
                        "sampled_roles": roles,
                        "sampled_anion_key": anion,
                        "full_composition": composition,
                        "anion_seen_in_cpu_train_subset": anion in subset_anions if anion else None,
                        "composition_seen_in_cpu_train_subset": composition in subset_compositions if composition else None,
                        "anion_seen_in_full_train_metadata": anion in full_anions if anion else None,
                        "composition_seen_in_full_train_metadata": composition in full_compositions if composition else None,
                        "raw_lengths_angstrom": decoded["raw_lengths_angstrom"],
                        "raw_angles_degrees": decoded["raw_angles_degrees"],
                        "projected_cubic_length_angstrom": decoded.get("projected_cubic_length_angstrom"),
                        "constructed": decoded["constructed"],
                        "minimal_valid": decoded["minimal_valid"],
                        "minimum_periodic_distance_angstrom": decoded.get("minimum_periodic_distance_angstrom"),
                        "accepted": decoded["accepted"],
                        "rejection_reason": decoded["rejection_reason"],
                        "cif_write_error": decoded.get("cif_write_error"),
                    }
                    attempts.append(record)
                    if decoded["accepted"] and decoded["cif"] is not None and saved_examples[method] < 2:
                        name = f"{method}_{target_id}_{proposal_index}.cif"
                        (output / "examples" / name).write_text(decoded["cif"], encoding="utf-8")
                        saved_examples[method] += 1
    adapter.assert_frozen()
    expected_attempts = len(targets) * sample_config["proposals_per_target"] * len(METHODS)
    if len(attempts) != expected_attempts:
        raise AssertionError(f"Incomplete proposal ledger: {len(attempts)}/{expected_attempts}")
    write_csv(output / "attempts.csv", attempts)
    summaries = summarize_attempts(attempts)
    write_csv(output / "by_target_method.csv", summaries)
    (output / "targets.json").write_text(json.dumps(target_records, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8")
    methods = {}
    training_median_latent_norm = float(torch.linalg.vector_norm(raw_latents, dim=1).median())
    for method in METHODS:
        rows = [row for row in attempts if row["method"] == method]
        accepted = [row for row in rows if row["accepted"]]
        methods[method] = {
            "attempts": len(rows), "sampled": sum(row["sampled"] for row in rows),
            "constructed": sum(row["constructed"] for row in rows),
            "minimal_valid": sum(row["minimal_valid"] for row in rows),
            "accepted": len(accepted),
            "cif_serialization_failures": sum(bool(row["cif_write_error"]) for row in rows),
            "unique_accepted_compositions": len({row["full_composition"] for row in accepted}),
            "median_latent_norm": float(torch.tensor([row["latent_norm"] for row in rows]).median()),
            "median_paired_condition_unconditioned_latent_l2": (
                float(torch.tensor([row["paired_condition_unconditioned_latent_l2"]
                                    for row in rows]).median()) if method.endswith("diffusion") else None),
            "rejection_reasons": dict(Counter(row["rejection_reason"] or "NONE"
                                              for row in rows if not row["accepted"])),
        }
    summary = {
        "status": "passed_bounded_software_smoke",
        "device": args.device, "torch": torch.__version__, "platform": platform.platform(),
        "denoiser_steps": metadata["step"], "ddim_steps": sample_config["ddim_steps"],
        "cpu_train_cache_median_latent_norm": training_median_latent_norm,
        "target_count": len(targets), "attempts_per_target_method": sample_config["proposals_per_target"],
        "total_attempts": len(attempts), "methods": methods,
        "original_splits_recovered": False,
        "pretrained_training_overlap": "unknown",
        "checkpoint_cmr_heat_label_compatibility": "unresolved",
        "independent_property_or_stability_validation": "not_performed",
        "gpu_execution": "not_run_not_validated",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8")
    report_lines = [
        "# Bounded CPU latent-to-MF_P4 generation smoke", "",
        f"Device: `{args.device}`; denoiser updates: `{metadata['step']}`; {sample_config['ddim_steps']} DDIM steps; "
        f"{len(targets)} training-supported property requests; eight proposals per method/request.",
        "This is an engineering subset, not a held-out inverse-design benchmark.", "",
        "| Method | Attempts | Constructed | Minimal accepted | Unique accepted compositions | Median latent norm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = methods[method]
        report_lines.append(f"| {method} | {row['attempts']} | {row['constructed']} | "
                            f"{row['accepted']} | {row['unique_accepted_compositions']} | "
                            f"{row['median_latent_norm']:.3f} |")
    report_lines += [
        "", f"CPU training-cache median latent norm: `{training_median_latent_norm:.3f}`.",
        "Minimal acceptance means only the unchanged MF_P4 geometry/finite checks passed;",
        "it is not evidence that a generated latent lies on the learned manifold or meets a property target.",
        "", "Every rejected attempt and raw lattice prediction is retained in `attempts.csv`.",
        "Paired conditional/unconditional latent L2 distances are recorded in the attempt ledger;",
        "this diagnostic was added after inspecting the initial output panel, without changing targets or weights.",
        "Per-request diversity and rejection reasons are in `by_target_method.csv`;",
        "selected requests and their 16-nearest cached training neighbors are in `targets.json`.",
        "CIFs under `examples/` are minimal-rule outputs only, not physically validated materials.",
        "", "The property-only path uses raw `[heat_all, dir_gap]` values in frozen MEIDNet.",
        "Diffusion alone uses train-only standardized conditions and inverted latents.",
        "All four methods share the unchanged MF_P4 global role vocabulary, independent",
        "X1/X2/X3 draws, 1.25 temperature, fixed decoder input and output templates,",
        "and one-attempt budget. Target source-family information was used only to select",
        "requests; it was not passed to any sampler. Conditional and unconditional DDIM",
        "use paired initial noise, with separate categorical draw seeds.",
        "", "No source CIF or reference latent was provided to a generation method.",
        "No independent property, stability, novelty, or DFT evaluation was performed.",
        "CMR heat-label/checkpoint compatibility remains unresolved; checkpoint",
        "training overlap is unknown. The original MEIDNet optimization baseline",
        "was not run, so no superiority comparison is possible. GPU execution is untested.",
    ]
    (output / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    output_hashes = {str(path.relative_to(output)): sha256_file(path)
                     for path in sorted(output.rglob("*")) if path.is_file()}
    manifest = {
        "command": [sys.executable, *sys.argv],
        "config_sha256": sha256_file(args.config),
        "denoiser_checkpoint_sha256": sha256_file(args.checkpoint),
        "train_cache_fingerprint": train["manifest"]["fingerprint"],
        "val_cache_fingerprint": val["manifest"]["fingerprint"],
        "train_cache_manifest_sha256": sha256_file(args.train_cache / "manifest.json"),
        "val_cache_manifest_sha256": sha256_file(args.val_cache / "manifest.json"),
        "meidnet_checkpoint_sha256": adapter.checkpoint_sha256,
        "meidnet_state_sha256_before_after_equal": True,
        "template_manifest_sha256": adapter.template_sha256,
        "policy_sha256": adapter.policy_sha256,
        "training_support_sha256": adapter.training_support_sha256,
        "scaler_sha256": scalers["hash"],
        "output_hashes": output_hashes,
        "test_split_opened": False,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                              encoding="utf-8")
    print(json.dumps({"status": summary["status"], "total_attempts": len(attempts),
                      "report": str(output / "report.md")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
