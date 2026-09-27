from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "multifamily_panel.py"
spec = importlib.util.spec_from_file_location("multifamily_panel", SCRIPT)
assert spec and spec.loader
panel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(panel)


def row(material_id, group, anion, gap, **extra):
    return {
        "material_id": material_id,
        "composition_group": group,
        "anion_key": anion,
        "dir_gap": gap,
        "split": "val",
        **extra,
    }


def test_sparse_strata_are_kept_and_groups_are_indivisible():
    rows = [
        row("o1", "oxide_one", "O3", 0),
        row("o2", "oxide_one", "O3", 0),
        row("o3", "oxide_two", "O3", 0),
        row("o4", "oxide_two", "O3", 0),
        row("n1", "nitride", "N3", 1.2),
        row("m1", "mixed", "O2N", 0),
    ]
    selected, draws, manifest = panel.select_validation_panels(
        rows, seed=42, max_panel=4, max_stochastic=3
    )
    assert {"n1", "m1"}.issubset(selected)
    assert len(selected) <= 4 and len(draws) <= 3
    assert set(draws).issubset(selected)
    assert ("o1" in selected) == ("o2" in selected)
    assert ("o3" in selected) == ("o4" in selected)
    assert manifest["panel"]["sparse_strata_all_included"]
    assert manifest["strata"]["N3|positive"]["panel"] == 1
    assert manifest["strata"]["O2N|zero"]["panel"] == 1


def test_selection_is_order_invariant_and_ignores_decoder_results():
    rows = [
        row(f"id_{i:02d}", f"group_{i // 2}", "O3" if i % 4 < 2 else "O2N", i % 3)
        for i in range(30)
    ]
    first = panel.select_validation_panels(rows, max_panel=16, max_stochastic=8)
    changed = [dict(item, decoder_success=i % 2) for i, item in enumerate(reversed(rows))]
    second = panel.select_validation_panels(changed, max_panel=16, max_stochastic=8)
    assert first == second
    assert first[0] == sorted(first[0])
    assert first[1] == sorted(first[1])
    assert len({row["composition_group"] for row in rows if row["material_id"] in first[0]}) <= 8


def test_zero_gap_is_a_real_stratum_and_source_anion_fallback_is_supported():
    rows = [
        {"material_id": "a", "composition_group": "g1", "source_anion": "OFN", "dir_gap": 0.0},
        {"material_id": "b", "composition_group": "g2", "source_anion": "OFN", "dir_gap": 0.4},
    ]
    selected, draws, manifest = panel.select_validation_panels(rows)
    assert selected == draws == ["a", "b"]
    assert manifest["strata"]["OFN|zero"]["population"] == 1
    assert manifest["strata"]["OFN|positive"]["population"] == 1


def test_nonvalidation_and_missing_labels_are_rejected():
    with pytest.raises(ValueError, match="non-validation"):
        panel.select_validation_panels([row("test", "g", "O3", 0, split="test")])
    with pytest.raises(ValueError, match="Invalid direct-gap"):
        panel.select_validation_panels([row("missing", "g", "O3", float("nan"))])
    with pytest.raises(ValueError, match="repeated"):
        panel.select_validation_panels([row("a", "g", "O3", 0), row("a", "g", "O3", 1)])
