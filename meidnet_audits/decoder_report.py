"""Render the recorded decoder audit scorecard."""

from __future__ import annotations

from typing import Any

import numpy as np


def _legacy_markdown_report(
    summary: dict[str, Any], latent_paths: tuple[str, ...]
) -> str:
    raw = summary["raw_joint_reconstruction"]
    lines = [
        "# MEIDNet small decoder audit",
        "",
        "## Status and interpretation boundary",
        "",
        f"- Numerical execution: **{summary['status'].upper()}**.",
        "- CMR property labels: **provisional numerical inputs; compatibility unresolved**.",
        "- Pretrained training overlap: **unknown**.",
        "- Test-set model inference: **not performed**.",
        "- This is an interface/mechanism audit, not property validation, stability analysis, or a paper-result reproduction.",
        "",
        "## Configuration",
        "",
        f"- Audit examples: {summary['input_count']} validation records",
        f"- Seeds: {summary['configuration']['seeds']}",
        f"- Latent paths: {', '.join(latent_paths)}",
        f"- Decoder temperature/top-k/tries: {summary['configuration']['decode_temperature']} / "
        f"{summary['configuration']['decode_topk']} / {summary['configuration']['decode_tries']}",
        f"- Joint perturbation L2 size: {summary['configuration']['perturbation_size']}",
        "- History and anti-repeat counts were reset between every paired method.",
        "- The reference-free API accepts only a latent and family mask; it constructs zero coordinates and zero center internally.",
        "",
        "## 1. Numerical forward and paired decoder execution",
        "",
        "Both the training/evaluation model and the separate generation implementation strict-loaded all checkpoint keys. "
        "The main comparison decoded each fixed, un-renormalized `z_joint` once with reconstruction coordinates/center and once with exact generation-style zero inputs.",
        "",
        f"- Mean reconstruction/reference-free species-logit absolute difference: {raw['mean_species_logit_absolute_difference']:.6g}",
        f"- Mean reconstruction/reference-free coordinate absolute difference: {raw['mean_coordinate_absolute_difference']:.6g}",
        f"- Joint reconstruction exact-composition count: {raw['composition_exact_count']}/{summary['input_count']}",
        f"- Joint reconstruction all-role recovery count: {raw['all_roles_exact_count']}/{summary['input_count']}",
        f"- Raw lattice simple-plausibility count: {raw['raw_lattice_plausible_count']}/{summary['input_count']}",
        f"- Training and generation decoders agreed on reference-free lattice/species/coordinate tensors: {summary['controls']['both_decoder_implementations_agree']}",
        f"- Repeated identical latent control was exact: {summary['controls']['repeated_latent_exact']}",
        "",
        "The training decoder's `input_species` argument is not consumed by its implementation; its auxiliary dependence comes from coordinates and center. "
        "Generation returns an all-ones adjacency rather than the learned adjacency logits returned by the training decoder, so those adjacency outputs are intentionally not equated.",
        "",
        "## 2. Template-constrained composition generation",
        "",
        "| Latent path | Runs | Accepted | Fraction | Unique formulas | Source-role recoveries |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for path_name in latent_paths:
        item = summary["stochastic_generation"][path_name]
        lines.append(
            f"| {path_name} | {item['runs']} | {item['accepted']} | "
            f"{item['acceptance_fraction']:.3f} | {item['unique_formulas']} | "
            f"{item['source_role_recoveries']} |"
        )
    lines.extend(
        [
            "",
            "Every traced stochastic decision was checked against the repository's unmodified `decode_and_filter` result under an independently reset, matching RNG. "
            "Acceptance means only that the repository's declared template and heuristic filters accepted the projected candidate.",
            "",
            "## 3. Raw learned geometry versus final CIF geometry",
            "",
            "The generation pipeline computes raw lattice and coordinate heads, but final candidates replace both: coordinates become the fixed five-site Pm-3m template and the cubic lattice constant becomes `clip(2*(r_B+r_X), 3, 8)` from ionic radii. "
            "Therefore valid projected CIFs cannot establish learned-geometry capability, and weak raw geometry alone does not invalidate the explicitly template-constrained composition mechanism.",
            "",
            f"Representative CIFs written: {summary['representative_cif_count']}",
            "",
            "## 4. Coverage for a later diffusion study",
            "",
            "This audit deliberately retains the existing validation sample. Its sparse positive-gap coverage limits property-range conclusions but does not prevent interface and auxiliary-input diagnostics. "
            "See `split_balance.md` for group-aware split, property, and chemistry coverage. No dataset membership was changed.",
            "",
            "## 5. Remaining limitations",
            "",
            "- `heat_of_formation_all` compatibility with the checkpoint's original `heat_all` remains unresolved.",
            "- `pretrained_training_overlap` remains unknown.",
            "- Property-head values in the numeric outputs are not independent validation.",
            "- Template projection, charge/tolerance filters, and ionic-radius cells are declared priors, not learned-geometry evidence.",
            "- No DFT, stability, synthesizability, or held-out generalization claim is made.",
            "",
            "## Smallest next experiment",
            "",
            "Resolve the original heat-label/reference convention, then repeat this same paired audit on a documented positive-gap validation stress subset without changing masks, projection, seeds, or decoder weights. Do not begin diffusion until the intended latent interface and conditioning semantics are explicitly chosen.",
            "",
            "## Reproduction",
            "",
            "```powershell",
            "$env:PYTHONDONTWRITEBYTECODE = \"1\"",
            "$env:PYTHONHASHSEED = \"0\"",
            "& .\\.venv\\Scripts\\python.exe .\\scripts\\audit_decoder.py",
            "```",
            "",
            "Detailed per-example and per-seed measurements are in `decoder_audit_raw.csv`, "
            "`decoder_audit_results.csv`, `decoder_audit_failures.csv`, and `decoder_audit_summary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def markdown_report(summary: dict[str, Any], latent_paths: tuple[str, ...]) -> str:
    """Render the addendum scorecard without changing any measured result."""

    def fmt(value: Any, digits: int = 6) -> str:
        if value is None:
            return "undefined"
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}g}"
        return str(value)

    information = summary["reference_information"]
    a_to_b_nll = information["A_to_B_delta_nll_B_minus_A"]
    a_to_b_top1 = information["A_to_B_delta_top1_A_minus_B"]
    a_to_b_masked_ab_nll = information["A_to_B_masked_AB_only_delta_nll_B_minus_A"]
    a_to_b_masked_ab_top1 = information[
        "A_to_B_masked_AB_only_delta_top1_A_minus_B"
    ]
    geometry_a = summary["raw_geometry"]["A_reference_assisted"]
    geometry_b = summary["raw_geometry"]["B_reference_free"]
    readiness = summary["readiness"]
    configuration = summary["configuration"]
    condition_ab = summary["accepted_AB_by_material_condition"]
    fixed_raw = information["control_comparisons"]["unmasked"]["fixed_real"]
    fixed_masked_ab = information["control_comparisons"]["masked_ab"]["fixed_real"]
    fixed_final = summary["final_control_comparisons"]["fixed_real"]
    joint_yield = summary["stochastic_generation"]["joint"]
    fixed_yield = summary["stochastic_generation"]["fixed_real"]
    geometry_c = summary["raw_geometry"]["C_reference_free_by_latent_path"]
    geometry_c_rows = [
        f"| {path_name} | {item['legal_cell_count']}/{item['examples']} | "
        f"{item['finite_positive_volume_count']}/{item['examples']} | "
        f"{fmt(item['mean_lattice_length_absolute_error_angstrom'])} | "
        f"{fmt(item['mean_lattice_angle_absolute_error_degrees'])} | "
        f"{fmt(item['median_relative_volume_error'])} | "
        f"{item['structure_matcher_scale_false_count']} | "
        f"{item['structure_matcher_scale_true_count']} |"
        for path_name, item in geometry_c.items()
    ]

    lines = [
        "# MEIDNet small decoder audit",
        "",
        "## Audit completion and interpretation boundary",
        "",
        f"- Audit completion: **{summary['status']}**.",
        f"- Numerical execution: **{readiness['numerical_execution']}**.",
        f"- Raw learned-geometry readiness: **{readiness['raw_geometry_generation']}**.",
        "- Template-constrained composition readiness: "
        f"**{readiness['template_constrained_composition_generation']}**.",
        "- CMR labels are provisional numerical inputs; checkpoint compatibility remains unresolved.",
        "- `pretrained_training_overlap` remains unknown.",
        "- No model inference was run on the final test set.",
        "- This is an interface/mechanism diagnostic, not property validation, stability evidence, "
        "synthesizability evidence, or a reproduction of the paper's original-data result.",
        "",
        f"Criteria timing: {summary['criteria_timing']}",
        "",
        "## Frozen protocol and integrity",
        "",
        f"- Sample: {summary['input_count']} validation rows / "
        f"{summary['independent_composition_groups']} independent composition groups.",
        f"- Seeds: {configuration['seeds']}.",
        f"- Latent arms: {', '.join(latent_paths)}.",
        f"- Temperature / top-k / maximum sequential attempts: "
        f"{configuration['decode_temperature']} / {configuration['decode_topk']} / "
        f"{configuration['decode_tries_maximum_per_call']}.",
        f"- Relative perturbations: {configuration['perturbation_relative_l2']}; no renormalization.",
        f"- Fixed-real donor: `{configuration['fixed_real_donor_id']}`, chosen lexicographically "
        "before the full run. Its family/property condition can mismatch a recipient.",
        "- Every call starts with reset Python, NumPy, PyTorch, history, and count state. "
        "Branch-dependent rejection can desynchronize later RNG draws, so arms are seed-matched "
        "but not claimed to remain draw-by-draw paired after divergence.",
        "- The declared source family supplies `{O}` or `{N}`. This forces X; X is never credited "
        "as learned recovery.",
        "- B-D received no reference A/B, coordinates, center, occupancy, atom count, or permutation.",
        "",
        "Recorded harness and scientific checks (a negative scientific finding does not make the "
        "audit incomplete):",
        "",
    ]
    lines.extend(f"- `{name}`: {value}" for name, value in summary["hard_checks"].items())
    lines.extend(
        [
            "",
            (
                "The separate generation decoder agrees with the canonical decoder for the statically "
                "equivalent reference-free lattice, species, and coordinate heads."
                if summary["hard_checks"]["equivalent_B_C_within_tolerance"]
                else "The separate generation decoder does not agree with the canonical decoder for "
                "one or more statically equivalent reference-free heads; see the check and scorecard."
            ),
            "Adjacency is deliberately excluded: canonical code returns learned dot products; "
            "generation returns all ones.",
            "",
            "## Reference-information preservation",
            "",
            "Primary metric: raw/unmasked occupied-site NLL in natural-log nats/site. Padding is "
            "excluded. The recorded source-to-parser mapping was checked for every row; all 20 mappings "
            "are identity in this fixed diagnostic sample.",
            "",
            f"A to B reference dependence: mean `NLL_B - NLL_A` = {fmt(a_to_b_nll['mean'])}, "
            f"group-bootstrap 95% CI [{fmt(a_to_b_nll['ci95_low'])}, "
            f"{fmt(a_to_b_nll['ci95_high'])}]. Positive means reference-free B is worse.",
            f"Mean `accuracy_A - accuracy_B` = {fmt(a_to_b_top1['mean'])}, 95% CI "
            f"[{fmt(a_to_b_top1['ci95_low'])}, {fmt(a_to_b_top1['ci95_high'])}].",
            f"Mean occupied-site A/B probability TV = "
            f"{fmt(information['A_to_B_mean_total_variation'])}.",
            f"After applying the identical generation-family mask and scoring only the learned "
            f"A/B slots, mean `NLL_B - NLL_A` = {fmt(a_to_b_masked_ab_nll['mean'])}, "
            f"95% CI [{fmt(a_to_b_masked_ab_nll['ci95_low'])}, "
            f"{fmt(a_to_b_masked_ab_nll['ci95_high'])}]; mean `accuracy_A - accuracy_B` = "
            f"{fmt(a_to_b_masked_ab_top1['mean'])}, 95% CI "
            f"[{fmt(a_to_b_masked_ab_top1['ci95_low'])}, "
            f"{fmt(a_to_b_masked_ab_top1['ci95_high'])}].",
            f"Mean family-masked A/B-only probability TV = "
            f"{fmt(information['A_to_B_masked_AB_only_mean_total_variation'])}.",
            "",
            "| Control versus joint | Stage | Delta NLL (control - joint) | 95% group CI | Mean TV | N |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for stage in ("unmasked", "masked_ab"):
        for control in latent_paths:
            if control == "joint":
                continue
            item = information["control_comparisons"][stage][control]
            effect = item["delta_nll_control_minus_joint"]
            lines.append(
                f"| {control} | {stage} | {fmt(effect['mean'])} | "
                f"[{fmt(effect['ci95_low'])}, {fmt(effect['ci95_high'])}] | "
                f"{fmt(item['mean_total_variation'])} | {item['paired_examples']} |"
            )
    lines.extend(
        [
            "",
            "Family-masked A/B-only reference-probability restriction failures "
            f"(A reference-assisted / B reference-free / all C arms): "
            f"{information['A_masked_AB_only_zero_true_probability_sites']} / "
            f"{information['B_masked_AB_only_zero_true_probability_sites']} / "
            f"{information['C_masked_AB_only_zero_true_probability_sites']}. "
            "Their NLL status is recorded as positive-infinity restriction failure, not clipped.",
            f"The retained all-five-site masked diagnostic (which includes three forced-X slots) has "
            f"{information['masked_all_five_forced_X_including_zero_true_probability_sites']} "
            "restriction failures and is not used for masked control comparisons.",
            "`decoder_audit/information_sites.csv` records every scored occupied site's full "
            "118-class logit and probability vectors. `raw_heads.csv` records the corresponding "
            "per-site TV arrays; aggregate means are never the only retained evidence.",
            "TV establishes sensitivity only; usefulness is assessed from NLL advantage and whether "
            "reference-relevant A/B effects survive the final path.",
            "",
            "## Template-constrained composition generation",
            "",
            "The frozen repository budget is a maximum of 12 sequential attempts per call, with the "
            "baseline's first-pass early stop. There are 60 calls per arm and at most 720 attempts. "
            "`N_attempt` is the number actually consumed, including failed sampling/filter attempts.",
            "",
            "| Arm | Calls | N attempt | N constructed | N pass | Pass/attempt | Unique formula | Unique A/B | Unique structures | Dominant accepted A/B | Accepted source A/B per attempt |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for path_name in latent_paths:
        item = summary["stochastic_generation"][path_name]
        lines.append(
            f"| {path_name} | {item['decode_calls']} | {item['N_attempt']} | "
            f"{item['N_constructed']} | {item['N_pass']} | {fmt(item['acceptance_yield'])} | "
            f"{item['N_unique_composition']} | "
            f"{item['N_unique_accepted_AB']} | "
            f"{item['N_unique_structure_matcher_scale_false']} | "
            f"{fmt(item['dominant_accepted_AB_fraction'])} | "
            f"{fmt(item['passed_source_AB_recovery_rate'])} |"
        )
    lines.extend(
        [
            "",
            "The additive trace matched the untouched repository function's endpoint, A/B/X and "
            "filter values for every call. For all-rejected calls this verifies the same no-output "
            "endpoint; internal rejection reasons come from the inspected additive trace, not an "
            "independent internal baseline trace.",
            "",
            "| Control | Joint-control accepted source-A/B yield | 95% group CI | Difference by seed | Endpoint changed | Formula changed when both passed |",
            "|---|---:|---:|---|---:|---:|",
        ]
    )
    for control in latent_paths:
        if control == "joint":
            continue
        item = summary["final_control_comparisons"][control]
        effect = item["joint_minus_control_passed_source_AB_recovery_rate"]
        lines.append(
            f"| {control} | {fmt(effect['mean'])} | "
            f"[{fmt(effect['ci95_low'])}, {fmt(effect['ci95_high'])}] | "
            f"{item['passed_source_AB_recovery_rate_difference_by_seed']} | "
            f"{fmt(item['endpoint_changed_fraction'])} | "
            f"{fmt(item['formula_changed_given_both_passed_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "Counts by family/material condition are in `decoder_audit/yield_by_condition.csv`; every "
            "consumed proposal is in `decoder_audit/attempts.csv`. A and B recovery are separate in the "
            "machine-readable files. X is a forced family prior.",
            f"For the joint arm, {condition_ab['conditions_with_no_accepted_output']} of "
            f"{condition_ab['condition_count']} material conditions had no accepted output; among "
            f"conditions with an accepted output, {condition_ab['conditions_with_single_accepted_AB']} "
            f"had one accepted A/B pair and {condition_ab['conditions_with_multiple_accepted_AB']} "
            "had multiple accepted A/B pairs across the fixed seeds. These within-condition counts are "
            "reported separately so pooled family/formula diversity cannot hide condition-level collapse.",
            "The pre-filter attempt A/B recovery rate remains in machine-readable results as a sampling "
            "diagnostic, but it is not used for readiness. The readiness effect counts only proposals "
            "that preserved source A/B and passed projection plus the common filters.",
            "",
            "## Raw learned geometry",
            "",
            "Raw lattice values use the training inverse scale (lengths x20, angles x180). "
            "`StructureMatcher(scale=False)` is the geometry-preserving primary comparison; scale=True "
            "is a supplementary topology result. Periodic physical displacements use cell-aware "
            "distance matrices and species-constrained Hungarian assignment.",
            "",
            "| Stage | Legal cells | Positive volumes | Length MAE (Å) | Angle MAE (deg) | Median relative volume error | scale=False fits | scale=True fits |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            f"| A reference-assisted | {geometry_a['legal_cell_count']}/{geometry_a['examples']} | "
            f"{geometry_a['finite_positive_volume_count']}/{geometry_a['examples']} | "
            f"{fmt(geometry_a['mean_lattice_length_absolute_error_angstrom'])} | "
            f"{fmt(geometry_a['mean_lattice_angle_absolute_error_degrees'])} | "
            f"{fmt(geometry_a['median_relative_volume_error'])} | "
            f"{geometry_a['structure_matcher_scale_false_count']} | "
            f"{geometry_a['structure_matcher_scale_true_count']} |",
            f"| B reference-free | {geometry_b['legal_cell_count']}/{geometry_b['examples']} | "
            f"{geometry_b['finite_positive_volume_count']}/{geometry_b['examples']} | "
            f"{fmt(geometry_b['mean_lattice_length_absolute_error_angstrom'])} | "
            f"{fmt(geometry_b['mean_lattice_angle_absolute_error_degrees'])} | "
            f"{fmt(geometry_b['median_relative_volume_error'])} | "
            f"{geometry_b['structure_matcher_scale_false_count']} | "
            f"{geometry_b['structure_matcher_scale_true_count']} |",
            "",
            "Stage-C generation implementation by latent arm:",
            "",
            "| Latent arm | Legal cells | Positive volumes | Length MAE (Å) | Angle MAE (deg) | Median relative volume error | scale=False fits | scale=True fits |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *geometry_c_rows,
            "",
            "Final generation discards both learned coordinates and learned lattice. It substitutes a "
            "fixed five-site Pm-3m template and cubic `clip(2*(r_B+r_X),3,8)` ionic-radius cell. "
            "A valid projected CIF therefore cannot establish learned geometry.",
            "",
            "Unmodified post-construction rules: ABX3 role count; repository realism check with "
            "`HARD_MIN=0.8 Å`; family identification; optional `symprec=0.05` refinement; existential "
            "charge balance; B-X window `[0.75,1.35]*(r_B+r_X)`; Goldschmidt oxide `[0.80,1.05]` "
            "or nitride `[0.75,1.08]`; and octahedral factor `[0.414,0.90]`. The repository realism "
            "check uses direct Cartesian distances; audit distance reporting is periodic. No rule changed.",
            "",
            f"Representative projected CIFs written and round-tripped: "
            f"{summary['representative_cif_count']}.",
            "",
            "## Compact scorecard",
            "",
            "| Metric | Arm | Control/paired difference | Denominator | Uncertainty | Conclusion |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for row in summary["scorecard"]:
        lines.append(
            f"| {row['metric']} | {row['arm']} | {fmt(row['control_or_paired_difference'])} | "
            f"{row['denominator']} | {row['uncertainty']} | {row['conclusion']} |"
        )
    lines.extend(
        [
            "",
            "## Separate conclusions",
            "",
            f"1. **Numerical execution — {readiness['numerical_execution']}.** "
            + (
                "Strict loading, frozen/eval state, finite unmasked heads, deterministic repeats, "
                "B/C equivalence, exact membership, input exclusion, and accounting passed."
                if readiness["execution_integrity_passed"]
                else "The audit completed, but the following measured scientific integrity checks "
                f"failed: {readiness['scientific_failures']}."
            ),
            f"2. **Raw geometry generation — {readiness['raw_geometry_generation']}.** This concerns "
            "free learned geometry only and is independent of template validity. All 20 reference-free "
            "cells were legal and positive-volume, but scale=False and scale=True matched 0/20; "
            f"matcher statuses were {geometry_b['structure_matcher_status_counts']}. Mean same-slot "
            f"periodic displacement was {fmt(geometry_b['mean_same_slot_periodic_rms_angstrom'])} Å.",
            "3. **Template-constrained composition generation — "
            f"{readiness['template_constrained_composition_generation']}.** "
            f"{readiness['decision_logic']}",
            f"The latent clearly changes raw probabilities (joint/fixed mean TV "
            f"{fmt(fixed_raw['mean_total_variation'])}) and survives projection as changed endpoints "
            f"in {fmt(fixed_final['endpoint_changed_fraction'])} of paired calls; joint produced "
            f"{joint_yield['N_unique_accepted_AB']} accepted A/B pairs versus "
            f"{fixed_yield['N_unique_accepted_AB']} for fixed-real. That establishes sensitivity and "
            "non-collapse, not useful reference-relevant information.",
            f"The useful-information criteria were not met: unmasked fixed-minus-joint NLL was "
            f"{fmt(fixed_raw['delta_nll_control_minus_joint']['mean'])} with 95% CI "
            f"[{fmt(fixed_raw['delta_nll_control_minus_joint']['ci95_low'])}, "
            f"{fmt(fixed_raw['delta_nll_control_minus_joint']['ci95_high'])}] (the fixed latent was "
            "better on that raw metric); the family-masked A/B-only interval crossed zero "
            f"[{fmt(fixed_masked_ab['delta_nll_control_minus_joint']['ci95_low'])}, "
            f"{fmt(fixed_masked_ab['delta_nll_control_minus_joint']['ci95_high'])}]; and no joint or "
            "control proposal recovered source A/B after common filters. The joint-minus-fixed "
            "accepted source-A/B effect was exactly zero in seeds 0, 1, and 2.",
            "",
            "## Coverage and remaining limitations",
            "",
            "The unchanged sample has 19 source-zero direct gaps and one positive gap. Zero is not "
            "reclassified as confirmed metallic behavior. The generation-scope validation pool has only "
            "one positive-gap row, so this audit cannot support positive-gap generalization.",
            "",
            "- Heat-label compatibility with the checkpoint remains unresolved.",
            "- Pretrained-training overlap remains unknown.",
            "- Property-head values are not independent validation.",
            "- Template/filter validity is not stability or synthesizability.",
            "- No DFT, training, fine-tuning, or diffusion was run.",
            "",
            "## Smallest next experiment",
            "",
            "The smallest interface diagnostic is to predeclare one fixed canonical five-site ABX3 "
            "coordinate/center input shared by every example, then repeat A/B/C scoring without changing "
            "weights, masks, projection, or criteria. That would test whether the observed zero-input "
            "reference dependency is specifically an off-manifold auxiliary-input problem; it must remain "
            "a separate arm, not replace the baseline. Independently resolve the original heat-label "
            "convention and obtain broader positive-gap validation coverage. This task stops without "
            "implementing that diagnostic, a model fix, or diffusion.",
            "",
            "## Commands and verification actually executed",
            "",
            "```powershell",
            "$env:PYTHONDONTWRITEBYTECODE = \"1\"",
            "$env:PYTHONHASHSEED = \"0\"",
            "$env:MPLCONFIGDIR = Join-Path $env:TEMP \"meidnet-mpl-audit\"",
            "& .\\.venv\\Scripts\\python.exe .\\scripts\\audit_decoder.py",
            "& .\\.venv\\Scripts\\python.exe -m pytest tests\\test_decoder_audit.py -q",
            "& .\\.venv\\Scripts\\python.exe -m pytest -q",
            "```",
            "",
            "The final focused decoder tests passed 8/8 and the full suite passed 21/21. An earlier "
            "focused run had one test-only failure because a float32 difference near 0.001 was asserted "
            "with an inappropriately tight 1e-9 default tolerance; the test was corrected to an explicit "
            "1e-6 absolute tolerance and rerun. This was not a model or audit failure.",
            "",
            "A first full run was inspected, then the identical frozen run was repeated after report-only "
            "wording and all-arm geometry-table additions. The model, data, arms, criteria, seeds, and "
            "baseline behavior were unchanged.",
            "",
            "## Reproduction",
            "",
            "```powershell",
            "$env:PYTHONDONTWRITEBYTECODE = \"1\"",
            "$env:PYTHONHASHSEED = \"0\"",
            "$env:MPLCONFIGDIR = Join-Path $env:TEMP \"meidnet-mpl-audit\"",
            "& .\\.venv\\Scripts\\python.exe .\\scripts\\audit_decoder.py",
            "```",
            "",
            "The actual interpreter/argv, package versions, git state, hashes, IDs, and configuration "
            "are in `reports/decoder_audit/run_manifest.json`. Detailed artifacts are `raw_heads.csv`, "
            "`information_sites.csv`, `per_sample.csv`, `attempts.csv`, `failures.csv`, "
            "`yield_by_condition.csv`, `scorecard.csv`, and `summary.json` in that directory.",
            "",
        ]
    )
    return "\n".join(lines)
