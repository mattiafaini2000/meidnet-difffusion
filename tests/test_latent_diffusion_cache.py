"""Cache/scaler integrity tests; the real-asset check skips explicitly if absent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from latent_diffusion.data import (
    apply_scalers, cache_batches, fit_scalers, invert_latents, load_cache,
    reject_test_path, select_rows, validate_scalers,
)


ROOT = Path(__file__).resolve().parents[1]


def rows(n: int = 8) -> list[dict]:
    anions = ("O3", "N3", "O2F", "O2S")
    return [
        {"material_id": f"cmr_{i:04d}", "composition_group": f"g{i}",
         "source_anion_key": anions[i % len(anions)], "source_family": anions[i % len(anions)],
         "source_roles": ["Ca", "Ti", "O", "O", "O"], "exact_anion_multiset": ["O"] * 3,
         "heat_all": float(i), "dir_gap": float((i // len(anions)) % 2)}
        for i in range(n)
    ]


def fake_encode(batch: list[dict]) -> torch.Tensor:
    return torch.tensor([[float(row["heat_all"]), float(row["dir_gap"]), 3.0]
                         for row in batch], dtype=torch.float32)


def test_selection_is_source_only_stable_and_stratified() -> None:
    source = rows(40)
    first, info = select_rows(source, limit=16, seed=42)
    second, _ = select_rows(list(reversed(source)), limit=16, seed=42)
    assert [row["material_id"] for row in first] == [row["material_id"] for row in second]
    assert len(first) == 16 and len(info["stratum_counts"]) == 8
    assert [row["material_id"] for row in first] == sorted(row["material_id"] for row in first)


def test_partial_resume_preserves_order_without_reencoding(tmp_path: Path) -> None:
    source = rows()
    calls: list[list[str]] = []

    def fail_on_second(batch: list[dict]) -> torch.Tensor:
        calls.append([row["material_id"] for row in batch])
        if len(calls) == 2:
            raise RuntimeError("synthetic parse failure")
        return fake_encode(batch)

    with pytest.raises(RuntimeError, match="synthetic parse failure"):
        cache_batches(tmp_path, source, fail_on_second, {"split": "train", "source": "fixture"}, batch_size=3)
    partial = load_cache(tmp_path)
    assert partial["manifest"]["status"] == "partial"
    assert partial["ids"] == [row["material_id"] for row in source[:3]]
    failure = json.loads((tmp_path / "failure.json").read_text())
    assert failure["failed_batch_ids"] == [row["material_id"] for row in source[3:6]]

    resumed_calls: list[list[str]] = []

    def resume(batch: list[dict]) -> torch.Tensor:
        resumed_calls.append([row["material_id"] for row in batch])
        return fake_encode(batch)

    complete = cache_batches(tmp_path, source, resume, {"split": "train", "source": "fixture"}, batch_size=3)
    assert resumed_calls[0] == [row["material_id"] for row in source[3:6]]
    assert complete["ids"] == [row["material_id"] for row in source]
    assert complete["manifest"]["status"] == "complete"
    assert cache_batches(tmp_path, source, resume, {"split": "train", "source": "fixture"})["ids"] == complete["ids"]
    assert len(resumed_calls) == 2  # completed cache performed no new encoding


def test_stale_cache_and_corrupt_snapshot_are_rejected(tmp_path: Path) -> None:
    source = rows(3)
    cache_batches(tmp_path, source, fake_encode, {"split": "train", "source": "one"})
    with pytest.raises(ValueError, match="fingerprint"):
        cache_batches(tmp_path, source, fake_encode, {"split": "train", "source": "two"})
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    changed = manifest | {"source": "silently_changed"}
    (tmp_path / "manifest.json").write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_cache(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / manifest["snapshot_file"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_cache(tmp_path)


def test_train_only_scalers_floor_inverse_and_hash(tmp_path: Path) -> None:
    source = rows(8)
    train = cache_batches(tmp_path / "train", source, fake_encode, {"split": "train"})
    scalers = fit_scalers(train, std_floor=1e-6)
    assert scalers["latent"]["floored_dimensions"] == [2]
    u, c = apply_scalers(train["latents"], train["conditions"], scalers)
    assert torch.allclose(invert_latents(u, scalers), train["latents"], atol=1e-6)
    assert torch.equal(c[:, 1], (train["conditions"][:, 1] - scalers["condition"]["mean"][1]) /
                       scalers["condition"]["std"][1])
    scalers["latent"]["mean"][0] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_scalers(scalers)
    val = cache_batches(tmp_path / "val", source, fake_encode, {"split": "val"})
    with pytest.raises(ValueError, match="training cache"):
        fit_scalers(val)


def test_no_final_test_path_by_default() -> None:
    with pytest.raises(ValueError, match="Final-test"):
        reject_test_path("test", Path("train.csv"), Path("cifs/train"), False)
    with pytest.raises(ValueError, match="Final-test"):
        reject_test_path("train", Path("test.csv"), Path("cifs/train"), False)
    with pytest.raises(ValueError, match="Final-test"):
        reject_test_path("train", Path("train.csv"), Path("cifs/test"), False)


def test_real_encoder_cache_agreement_and_mf_p4_parity(tmp_path: Path) -> None:
    checkpoint = ROOT / "MEIDNet-main" / "checkpoints" / "dual_autoencoder_clip_earlyfusion_propertyaware_2k.pth"
    csv_path = ROOT / "data" / "processed" / "cmr_reconstructed" / "val.csv"
    if not checkpoint.is_file() or not csv_path.is_file():
        pytest.skip("Real checkpoint or reconstructed validation CSV is absent")

    from scripts.cache_diffusion_latents import read_source_rows
    from latent_diffusion.integration import FrozenMEIDNet
    from meidnet.model import parse_cif_to_dense
    from scripts import multifamily_policies as policies

    torch.set_num_threads(2)
    row = read_source_rows(csv_path)[0]
    cif = csv_path.parent / "cifs" / "val" / row["cif_filename"]
    crystal = torch.from_numpy(parse_cif_to_dense(str(cif))).float().unsqueeze(0)
    conditions = torch.tensor([[row["heat_all"], row["dir_gap"]]], dtype=torch.float32)
    adapter = FrozenMEIDNet(device="cpu")
    assert row["source_anion_key"] in adapter.training_support["exact_anion_counts"]
    joint = adapter.encode_joint(crystal, conditions)
    with torch.no_grad():
        crystal_common, property_common, expected, _, _, _ = adapter.model.encode_modalities(
            crystal, conditions[:, 0], conditions[:, 1]
        )
    assert torch.equal(joint, expected)
    assert torch.equal(joint, (crystal_common + property_common) / 2)
    assert torch.equal(adapter.property_only_latent(conditions), property_common)
    cached = cache_batches(
        tmp_path / "real", [{key: value for key, value in row.items() if key != "embedded_cif"}],
        lambda batch: adapter.encode_joint(crystal, conditions),
        {"split": "val", "checkpoint_sha256": adapter.checkpoint_sha256},
    )
    assert cached["ids"] == [row["material_id"]]
    assert torch.equal(cached["latents"], expected)
    assert torch.equal(cached["conditions"], conditions)
    assert not joint.requires_grad
    assert all(not parameter.requires_grad for parameter in adapter.model.parameters())
    heads = adapter.decode_heads(joint)
    seed = 12345
    direct = policies.sample_roles(heads[2][0], adapter.policy, adapter.vocab, seed)
    actual = adapter.decode_mf_p4(joint, seed)
    assert actual["sampled_roles"] == direct["roles"]
    if direct["sampled"]:
        candidate, cell = policies.construct_candidate(direct["roles"], adapter.policy, heads[0][0])
        minimal = policies.minimal_check(candidate, direct["roles"], adapter.vocab)
        assert actual["accepted"] == bool(candidate is not None and minimal["valid"])
        assert actual["projected_cubic_length_angstrom"] == cell.get("projected_cubic_length_angstrom")
        if candidate is not None:
            assert actual["structure"] is not None
            assert actual["structure"].species == candidate.species
            assert torch.equal(torch.tensor(actual["structure"].frac_coords), torch.tensor(candidate.frac_coords))
            assert torch.equal(torch.tensor(actual["structure"].lattice.matrix), torch.tensor(candidate.lattice.matrix))
        expected_reason = None if minimal["valid"] else (cell["reason"] or ";".join(minimal["failure_reasons"]))
        assert actual["rejection_reason"] == expected_reason
    else:
        assert actual["rejection_reason"] == direct["reason"]
    assert "reference" not in adapter.decode_mf_p4.__code__.co_varnames
    assert adapter.decode_mf_p4(joint, seed)["sampled_roles"] == actual["sampled_roles"]
    adapter.assert_frozen()
