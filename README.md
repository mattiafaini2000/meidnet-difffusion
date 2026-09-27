# MEIDNet Latent Diffusion

This exploratory project asks whether conditional generation in a useful structure–property latent space can produce perovskite candidates, and which decoder interfaces and output constraints limit that experiment. It combines frozen [MEIDNet](https://github.com/ABnano/MEIDNet/tree/ed62f6af466f132217fd0e71871bb765d96b1166) representations with decoder and policy audits, then trains a separate conditional latent denoiser. The MEIDNet architecture and pretrained weights are upstream work; the CMR conversion, audits, output policies, and diffusion pipeline are this repository's research code by Mattia Faini. This is a bounded software and mechanism study, not a validated inverse-design result.

## What is here

The pipeline converts the DTU Computational Materials Repository (CMR) cubic-perovskite records into MEIDNet-format CIF/CSV inputs, examines the frozen decoder and its output rules, caches joint latents, trains an epsilon-prediction residual MLP on those latents, and compares four generation arms through a common MF_P4 decoder. The original MEIDNet dataset and split membership were not recovered. The CMR-derived split and checkpoint-label compatibility are discussed in [Methods](docs/methods.md).

| Path | Role |
|---|---|
| `scripts/prepare_cmr_dataset.py` | CMR conversion and composition-group split |
| `scripts/audit_decoder.py`, `scripts/audit_template_reachability.py`, `scripts/audit_multifamily_decoder.py` | Frozen-decoder and output-policy diagnostics |
| `meidnet_audits/` | Small reusable audit and panel helpers |
| `latent_diffusion/` | Denoiser, Gaussian schedule, cache, checkpoint, and decoding adapter |
| `scripts/cache_diffusion_latents.py`, `scripts/train_latent_diffusion.py`, `scripts/evaluate_latent_diffusion.py` | Explicit diffusion workflow entry points |
| `configs/diffusion/` | Executed CPU smoke settings and a separate, unexecuted GPU configuration |
| `docs/experiments_summary.md` | Findings from the split audit, decoder studies, and CPU diffusion run |
| `reports/` | CSV and JSON results, input manifests, and example CIFs |

The complete upstream `MEIDNet-main/` snapshot is a separate reference input and is not part of the public source tree. Its recorded revision is `ABnano/MEIDNet@ed62f6af466f132217fd0e71871bb765d96b1166`. Cache and checkpoint fingerprints depend on the policy adapter `scripts/multifamily_policies.py` and the supplied template and training-support manifests.

## Recorded CPU observations

The 24 September 2026 run used a stratified engineering cache of 1,024 TRAIN and 128 validation records, a frozen MEIDNet checkpoint, 500 denoiser updates, eight TRAIN-supported property requests, and eight one-attempt proposals per request and method. The [experiments summary](docs/experiments_summary.md#conditional-latent-diffusion) covers this run and the preceding decoder studies. The paired-metric panel replayed the first panel with the same checkpoint, targets, and seeds; it is not an independent replication.

| Method | Attempts | Minimally accepted | Distinct accepted compositions | Median latent norm |
|---|---:|---:|---:|---:|
| Conditional DDIM | 64 | 56 | 49 | 78.985 |
| Frozen MEIDNet property-only latent | 64 | 64 | 63 | 1.000 |
| Nearest-property TRAIN-latent resampling | 64 | 64 | 57 | 0.940 |
| Same denoiser with guidance zero | 64 | 56 | 48 | 79.502 |

The TRAIN-cache joint-latent median norm was 0.943. The much larger DDIM norms are a central negative quality observation. Fixed-bank denoising MSE fell from about 1.11535 to 0.52792, only narrowly below the 0.53919 noisy-latent-as-epsilon comparator. Minimal acceptance checks site, cell, and periodic overlap; it does not establish property attainment, stability, novelty, or synthesizability. No original optimized-latent baseline or full-data/GPU denoiser run is recorded. See [Methods](docs/methods.md) for denominators and earlier decoder findings.

## Inputs and commands

Use a local Python environment with PyTorch and the direct dependencies in `requirements-setup.txt`. `requirements-lock.txt` records a Windows CPU environment, not a tested cross-platform or GPU lockfile. The denoiser core uses PyTorch and NumPy; conversion and structural audits also use ASE, pandas, pymatgen, SciPy, and scikit-learn as applicable. `pytest` is needed only to run the existing tests. The provided scripts do not silently download model weights.

The raw CMR ASE/SQLite database belongs at `data/raw/cmr/cubic_perovskites.db`; conversion can retrieve it if absent, so supply an existing copy to avoid a download. The converted `data/processed/cmr_reconstructed/` CSV/CIF tree, latent caches, denoiser checkpoints, and the upstream MEIDNet weight file are not distributed here. The frozen weight path expected by the current scripts is `MEIDNet-main/checkpoints/dual_autoencoder_clip_earlyfusion_propertyaware_2k.pth`. The required template input manifest and TRAIN-role support manifest are included at `reports/template_reachability/input_manifest.json` and `reports/multifamily_rule_relaxation/training_support.json`. The examples below use these files; `--template-manifest` and `--training-support` accept other locations. CMR database terms do not grant permission to redistribute upstream code or weights.

These are example commands from the repository root for **new local output directories**. The first conversion command can download the CMR database if `--raw-db` is missing. Caching performs MEIDNet inference, training performs denoiser optimization, and evaluation generates and decodes candidates.

```bash
python scripts/prepare_cmr_dataset.py --raw-db data/raw/cmr/cubic_perovskites.db --output-dir data/processed/cmr_reconstructed
python scripts/cache_diffusion_latents.py --split train --csv data/processed/cmr_reconstructed/train.csv --cif-root data/processed/cmr_reconstructed/cifs/train --checkpoint MEIDNet-main/checkpoints/dual_autoencoder_clip_earlyfusion_propertyaware_2k.pth --template-manifest reports/template_reachability/input_manifest.json --training-support reports/multifamily_rule_relaxation/training_support.json --output-dir data/cache/example_train --limit 1024
python scripts/cache_diffusion_latents.py --split val --csv data/processed/cmr_reconstructed/val.csv --cif-root data/processed/cmr_reconstructed/cifs/val --checkpoint MEIDNet-main/checkpoints/dual_autoencoder_clip_earlyfusion_propertyaware_2k.pth --template-manifest reports/template_reachability/input_manifest.json --training-support reports/multifamily_rule_relaxation/training_support.json --output-dir data/cache/example_val --limit 128
python scripts/train_latent_diffusion.py --config configs/diffusion/cpu_smoke.json --train-cache data/cache/example_train --val-cache data/cache/example_val --output-dir artifacts/diffusion/example_seed0
python scripts/evaluate_latent_diffusion.py --config configs/diffusion/cpu_smoke.json --checkpoint artifacts/diffusion/example_seed0/checkpoint.pt --train-cache data/cache/example_train --val-cache data/cache/example_val --meidnet-checkpoint MEIDNet-main/checkpoints/dual_autoencoder_clip_earlyfusion_propertyaware_2k.pth --template-manifest reports/template_reachability/input_manifest.json --training-support reports/multifamily_rule_relaxation/training_support.json --output-dir artifacts/diffusion/example_eval_seed0
```

Fresh training refuses to overwrite an existing run; `--resume` is only for a compatible checkpoint with the same caches, scalers, configuration, and device type. The separate `gpu_train.json` describes a future full-data configuration and has not been executed. Dataset and checkpoint provenance, policy definitions, and evidence paths are in [Methods](docs/methods.md).
