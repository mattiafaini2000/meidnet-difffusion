"""Render the fixed-template reachability report."""

from __future__ import annotations

from typing import Any

import numpy as np


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}g}"
    return str(value)


def markdown_report(summary: dict[str, Any], latent_arms: tuple[str, ...]) -> str:
    reachability = summary["reference_reachability"]
    comparisons = summary["raw_comparisons"]
    generation_rows = summary["template_generation"]["yield_by_family"]
    charge_space = summary["enumerated_constraint_space"]
    joint_family = {
        row["family"]: row
        for row in generation_rows
        if row["latent_arm"] == "joint" and row["family"] in {"oxide", "nitride"}
    }
    old_family = summary["previous_zero_baseline_by_family"]
    fixed_effect = comparisons["template_fixed_minus_joint_masked_ab_nll"]
    zero_effect = comparisons["zero_minus_template_joint_masked_ab_nll"]
    a_gap = comparisons["template_minus_A_joint_masked_ab_nll"]
    raw_joint_rows = [
        row for row in summary["raw_by_family"] if row["latent_arm"] == "joint"
    ]
    sampling_template_rows = [
        row
        for row in summary["sampling_by_family"]
        if row["geometry_mode"] == "template"
    ]
    previous_geometry = summary["previous_reference_geometry_controls"]
    final_fixed_effect = summary["template_generation"]["paired_control_effects"][
        "fixed_real"
    ]

    lines = [
        "# Canonical-template reachability follow-up",
        "",
        "## Outcome",
        "",
        f"- Execution integrity: **{summary['execution_integrity']}**.",
        f"- Fixed geometry restores reference-relevant composition information: "
        f"**{summary['decisions']['fixed_geometry_restoration']}**.",
        f"- Recommended next scope: **{summary['decisions']['next_experiment']}**.",
        "- CMR property labels remain provisional inputs; checkpoint label compatibility is unresolved.",
        "- `pretrained_training_overlap` remains unknown. No final-test inference was run.",
        "- This follow-up is an interface and constraint diagnostic, not diffusion, target-property validation, DFT evidence, or unrestricted geometry validation.",
        "",
        "The protocol was recorded before this follow-up inference, after the completed decoder audit had already been inspected. It is informed follow-up planning, not retroactive preregistration.",
        "",
        "## Reference-composition admissibility under current rules",
        "",
        f"All {reachability['all_count']} audit rows satisfy the dataset's declared species-list scope, but only "
        f"{reachability['charge_neutral_count']} satisfy the repository's existential formal-charge rule and only "
        f"{reachability['deterministically_exportable_count']} survive the complete fixed-template/refinement/filter path.",
        "",
        "| Family | References | Charge-neutral | Deterministically exportable |",
        "|---|---:|---:|---:|",
    ]
    for family in ("oxide", "nitride"):
        item = reachability["by_family"][family]
        lines.append(
            f"| {family} | {item['count']} | {item['charge_neutral']} | "
            f"{item['deterministically_exportable']} |"
        )
    lines.extend(
        [
            "",
            "Charge-neutral source formulas: "
            + ", ".join(f"`{value}`" for value in reachability["charge_neutral_roles"])
            + ". Full-pipeline exportable source formulas: "
            + ", ".join(
                f"`{value}`" for value in reachability["deterministically_exportable_roles"]
            )
            + ".",
            "",
            "`CaPbO3` is charge-neutral under the declared tables but fails the unchanged oxide Goldschmidt window; its measured oracle values and every per-material reason are in `reference_reachability.csv`. Formal-valence rejection is a repository-rule result, not proof of physical impossibility.",
            "",
            "The finite table enumeration shows that nitride failure is structurally imposed by the current rules:",
            "",
            "| Family | B choices | B with compatible A | Charge-compatible A/B pairs | B with any full-filter survivor | Full-filter A/B pairs |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for family in ("oxide", "nitride"):
        item = charge_space[family]
        lines.append(
            f"| {family} | {item['B_choices']} | {item['B_with_compatible_A']} | "
            f"{item['charge_compatible_AB_pairs']} | {item['B_with_exportable_A']} | "
            f"{item['exportable_AB_pairs']} |"
        )
    lines.extend(
        [
            "",
            "No enumerated nitride A/B/N combination survives the frozen tolerance/μ rules. Therefore the prior and current zero nitride accepted yield cannot be attributed to the decoder alone; under these exact rules the accepted nitride output space is empty.",
            "",
            "## Universal fixed input T",
            "",
            "The training parser scan selected one global slot convention before inference. Across "
            f"{summary['template_input']['training_convention']['eligible_training_rows']} compatible training rows, the chosen mapping from `generation.TEMPLATE` into decoder slots is "
            f"`{summary['template_input']['training_convention']['selected_slot_from_generation_template']}`. "
            "Slots 5–19 remain zero. The center is the mean of the five occupied slots, "
            f"`{summary['template_input']['center_from_first_five']}`, not the 20-slot mean "
            f"`{summary['template_input']['all_twenty_mean_not_used']}`.",
            "",
            f"Coordinate tensor SHA-256: `{summary['template_input']['coordinates_sha256']}`. "
            f"Center SHA-256: `{summary['template_input']['center_sha256']}`. Exact tensors, relative coordinates, pairwise distances, padding, and hash method are serialized in `input_manifest.json`.",
            "",
            "The coordinates are fractional, no species or occupancy tensor is supplied, and the decoder retains all 20 nodes and its all-ones adjacency. Periodic parser wrapping creates more than one ordinary-Euclidean origin branch in the current training export; the fixed analytic x=0 branch is the training-majority/median convention, not a claim about the unavailable historical checkpoint data.",
            "",
            "A center translation alone cancels from pairwise differences. The template changes the relative-coordinate/distance tensor, while the final coordinate head separately adds the center. The learned lattice head is latent-only and, as expected, is unchanged between zero and T for identical latents.",
            "",
            "## Raw decoder comparisons",
            "",
            "Signs were frozen in `protocol.md`: positive `NLL_zero - NLL_T` favors T; positive `NLL_fixed - NLL_joint` favors the per-example joint latent; positive `NLL_T - NLL_A` means T remains worse than reference-assisted A.",
            "",
            "| Paired metric | Mean | 95% whole-group bootstrap interval | Groups |",
            "|---|---:|---:|---:|",
            f"| Masked A/B-only NLL: zero - T, joint | {fmt(zero_effect['mean'])} | [{fmt(zero_effect['ci95_low'])}, {fmt(zero_effect['ci95_high'])}] | {zero_effect['independent_groups']} |",
            f"| Masked A/B-only NLL: T - A, joint | {fmt(a_gap['mean'])} | [{fmt(a_gap['ci95_low'])}, {fmt(a_gap['ci95_high'])}] | {a_gap['independent_groups']} |",
            f"| Masked A/B-only NLL: fixed-real - joint within T | {fmt(fixed_effect['mean'])} | [{fmt(fixed_effect['ci95_low'])}, {fmt(fixed_effect['ci95_high'])}] | {fixed_effect['independent_groups']} |",
            f"| Masked A/B-only NLL: property - joint within T | {fmt(comparisons['template_property_minus_joint_masked_ab_nll']['mean'])} | [{fmt(comparisons['template_property_minus_joint_masked_ab_nll']['ci95_low'])}, {fmt(comparisons['template_property_minus_joint_masked_ab_nll']['ci95_high'])}] | {comparisons['template_property_minus_joint_masked_ab_nll']['independent_groups']} |",
            "",
            f"For the joint latent, T versus zero raw/masked probability TV means are `{fmt(comparisons['joint_template_vs_zero_unmasked_tv_mean'])}` / `{fmt(comparisons['joint_template_vs_zero_masked_ab_tv_mean'])}`. TV and changed formulas establish sensitivity only. NLL is a likelihood diagnostic, not calibrated uncertainty, and forced O/N slots are excluded from the primary masked A/B comparison.",
            "",
            "The raw lattice head is latent-only, so its error is identical between zero and T. The species/coordinate heads change because T changes the relative-distance tensor:",
            "",
            "| Joint geometry mode | Family | Source A/B top-1 | Exact raw composition | scale=False match | Mean lattice-length MAE (Å) | Mean periodic coordinate RMS (Å) | Mean minimum distance (Å) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in raw_joint_rows:
        lines.append(
            f"| {item['geometry_mode']} | {item['family']} | "
            f"{item['raw_source_AB_top1_count']}/{item['material_count']} | "
            f"{item['raw_exact_composition_count']}/{item['material_count']} | "
            f"{item['scale_false_match_count']}/{item['material_count']} | "
            f"{fmt(item['mean_lattice_length_mae_angstrom'])} | "
            f"{fmt(item['mean_periodic_coordinate_rms_angstrom'])} | "
            f"{fmt(item['mean_minimum_periodic_distance_angstrom'])} |"
        )
    lines.extend(
        [
            "",
            f"For context, the preserved A reference-assisted control matched `StructureMatcher(scale=False)` in {previous_geometry['A_reference_assisted']['structure_matcher_scale_false_count']}/20 cases with mean same-slot periodic RMS `{fmt(previous_geometry['A_reference_assisted']['mean_same_slot_periodic_rms_angstrom'])}` Å. The zero B control matched 0/20 with mean RMS `{fmt(previous_geometry['B_reference_free']['mean_same_slot_periodic_rms_angstrom'])}` Å. T reaches 16/20 raw scale=False matches overall, but this remains a universal template-assisted result rather than unrestricted learned geometry.",
            "",
            "Per-example source probabilities, lattice errors, periodic coordinate errors, minimum distances, and explicit composition-incompatible versus geometrically unmatched statuses are in `per_example_metrics.csv`.",
            "",
            "## Final template-constrained generation",
            "",
            "Each arm used the same 20 examples, three seeds, temperature 1.25, top-k 12, and maximum 12 sequential attempts with the unchanged first-pass early stop. Consumed attempts, successful calls, unique compositions, and source-material coverage remain separate denominators.",
            "",
            "| Arm | Family | Source materials | Attempts consumed | Accepted calls | Materials with output | Unique accepted compositions | Source A/B sampled | Source A/B accepted |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in latent_arms:
        for family in ("oxide", "nitride"):
            item = next(
                row
                for row in generation_rows
                if row["latent_arm"] == arm and row["family"] == family
            )
            lines.append(
                f"| {arm} | {family} | {item['unique_source_materials']} | "
                f"{item['consumed_attempts']} | {item['accepted_calls']} | "
                f"{item['source_materials_with_accepted_output']} | "
                f"{item['unique_accepted_compositions']} | "
                f"{item['attempt_source_AB_recovery_count']} | "
                f"{item['accepted_source_AB_recovery_count']} |"
            )
    lines.extend(
        [
            "",
            "The unchanged earlier joint zero-interface baseline was oxide "
            f"{old_family['oxide']['N_pass']}/{old_family['oxide']['N_attempt']} accepted calls per consumed attempt and nitride "
            f"{old_family['nitride']['N_pass']}/{old_family['nitride']['N_attempt']}. Under T, joint is oxide "
            f"{joint_family['oxide']['accepted_calls']}/{joint_family['oxide']['consumed_attempts']} and nitride "
            f"{joint_family['nitride']['accepted_calls']}/{joint_family['nitride']['consumed_attempts']}.",
            "",
            f"Only {reachability['deterministically_exportable_count']} source materials form the full constraint-admissible denominator, all oxides. Recovery on that subset is reported explicitly in `yield_by_family.csv`; nitride admissible-subset recovery is undefined because its denominator is zero, not a measured zero rate.",
            "",
            f"Joint T recovered the accepted source A/B pair in 3 consumed attempts/calls, all three seeds for `NaVO3`; neither fixed-real nor property did so. The paired joint-minus-fixed endpoint-recovery mean is `{fmt(final_fixed_effect['all_examples']['joint_minus_control_endpoint_source_AB_recovery']['mean'])}` over all 20 groups with interval [{fmt(final_fixed_effect['all_examples']['joint_minus_control_endpoint_source_AB_recovery']['ci95_low'])}, {fmt(final_fixed_effect['all_examples']['joint_minus_control_endpoint_source_AB_recovery']['ci95_high'])}]. On the two exportable references it is `{fmt(final_fixed_effect['oracle_exportable_only']['joint_minus_control_endpoint_source_AB_recovery']['mean'])}` with interval [{fmt(final_fixed_effect['oracle_exportable_only']['joint_minus_control_endpoint_source_AB_recovery']['ci95_low'])}, {fmt(final_fixed_effect['oracle_exportable_only']['joint_minus_control_endpoint_source_AB_recovery']['ci95_high'])}]. Both intervals include zero because recovery occurred for one of only two eligible materials, so final exact-recovery superiority remains small-sample evidence rather than a population-level estimate.",
            "",
            "For the other exportable reference, `LaNiO3`, T moved source Ni from zero-mode B rank 17 (outside top-12) to rank 2 (inside top-12), but its one-attempt source A/B probability was only about 0.0184 and it was not drawn in the bounded run. A correct known latent is not expected to decode only one target-compatible material.",
            "",
            "Branch-dependent RNG divergence is expected: zero and T calls start from the same resolved seed, but changed categorical probabilities and early rejections can map the same random draws to different species and consume different numbers of draws. `per_seed_metrics.csv` records sequence and endpoint divergence; it is not treated as draw-by-draw pairing after a branch diverges.",
            "",
            "## B-first sampling and failure attribution",
            "",
            f"Across joint/T rows, the mean actual post-top-k B probability mass on choices with no charge-compatible A is oxide `{fmt(summary['sampling_summary']['joint_template']['oxide_no_compatible_A_mass'])}` and nitride `{fmt(summary['sampling_summary']['joint_template']['nitride_no_compatible_A_mass'])}`. The corresponding raw 118-class and pre-top-k masses, source-B ranks/top-k inclusion, and conditional source-A ranks are kept separate in `sampling_reachability.csv`.",
            "",
            "| T latent arm | Family | Raw 118-class invalid-B mass | Family-pool pre-temperature mass | Temperature/pre-top-k mass | Actual post-top-k mass | Source B in top-k | Source A charge-compatible | Source A/B top-k reachable |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in sampling_template_rows:
        lines.append(
            f"| {item['latent_arm']} | {item['family']} | "
            f"{fmt(item['mean_B_raw_all_species_mass_without_compatible_A'])} | "
            f"{fmt(item['mean_B_raw_family_pool_mass_without_compatible_A'])} | "
            f"{fmt(item['mean_B_temperature_pre_topk_mass_without_compatible_A'])} | "
            f"{fmt(item['mean_B_actual_post_topk_mass_without_compatible_A'])} | "
            f"{item['source_B_in_topk_count']}/{item['material_count']} | "
            f"{item['source_A_charge_compatible_count']}/{item['material_count']} | "
            f"{item['source_AB_topk_reachable_count']}/{item['material_count']} |"
        )
    lines.extend(
        [
            "",
            "Failures can therefore arise before model preference is tested: the family B table can offer no compatible A, the source A can be absent from the B/X-conditioned pool, source B or A can be top-k truncated, or the deterministic tolerance/μ filter can reject an otherwise charge-valid template. These are declared generator-rule/B-first limitations, not physical impossibility claims.",
            "",
            "## Separate conclusions",
            "",
            "1. **Fixed geometry and reference information.** " + summary["decisions"]["fixed_geometry_explanation"],
            "2. **Declared-rule and B-first failures.** " + summary["decisions"]["constraint_explanation"],
            "3. **Pilot decision.** " + summary["decisions"]["pilot_explanation"],
            "",
            "A fixed-template improvement, if any, is an interface intervention rather than a diffusion result. It does not validate free geometry, positive-gap controllability, property attainment, DFT stability, or checkpoint generalization. Any future latent optimization and diffusion comparison would need the same declared interface, with the untouched zero-input baseline retained separately.",
            "",
            "## Integrity, limitations, and commands actually run",
            "",
        ]
    )
    for name, passed in summary["hard_checks"].items():
        lines.append(f"- `{name}`: {passed}")
    lines.extend(
        [
            "",
            "Unresolved issues: checkpoint heat-label/reference compatibility; checkpoint training membership; the checkpoint's original slot/origin distribution; and independent physical validation. The audit sample still contains only one positive direct-gap record.",
            "",
            "Verification actually observed:",
            "",
            "- The first focused test run reported 5 passed and 1 failed. The new training-convention check had incorrectly read stoichiometric `source_anion` values such as `O3`/`N3` as per-site symbols. It was corrected to use `source_site_symbols`; this was a test/harness bug before the full run, not a model result.",
            "- The corrected focused suite passed 6/6 both before and after the full run.",
            "- The final complete suite passed 27/27. Two spglib deprecation warnings were emitted; no test failed.",
            "- `pip check` reported no broken requirements.",
            "- An initial complete run was inspected, then the identical frozen experiment was repeated after adding only requested oracle/refinement fields and clearer report tables. Inputs, criteria, model calls, seeds, masks, filters, and scientific results were unchanged.",
            "",
            "Commands:",
            "",
            "```powershell",
            '$env:PYTHONDONTWRITEBYTECODE = "1"',
            '$env:PYTHONHASHSEED = "0"',
            '$env:MPLCONFIGDIR = Join-Path $env:TEMP "meidnet-template-reachability"',
            "& .\\.venv\\Scripts\\python.exe .\\scripts\\audit_template_reachability.py",
            "& .\\.venv\\Scripts\\python.exe -m pytest tests\\test_template_reachability.py -q",
            "& .\\.venv\\Scripts\\python.exe -m pytest -q",
            "```",
            "",
            "The exact executed command, environment, current revision/differences, input hashes, template tensors, configuration, and artifact hashes are in `run_manifest.json` and `summary.json`.",
        ]
    )
    return "\n".join(lines) + "\n"
