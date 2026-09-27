# Experiments Summary

The experiments examined how much the frozen MEIDNet decoder depends on its coordinate inputs and output rules, then used the resulting interface for a small conditional diffusion run. Fixed input geometry and broader species support improved source reconstruction. The diffusion samples, however, had latent norms far above the training distribution, despite often passing the decoder's minimum geometry checks.

## Dataset and split

The CMR export contains 18,928 five-atom structures in 9,646 reduced-composition groups, after removing 441 reference records. A grouped split with seed 42 assigned 15,143 records to training, 1,893 to validation and 1,892 to test. The split audit found no material-ID or composition-group overlap between these sets. These memberships do not identify the data used to train the pretrained MEIDNet checkpoint.

The dataset includes 5,408 single-anion and 13,520 mixed-anion records. Most direct-gap labels are zero: 18,193 out of 18,928. Only 280 records fit the initial generation vocabulary, and its validation subset contains just one positive-gap example. The initial 20-composition decoder panel was selected from validation for property coverage; it was not a random sample. The `heat_all` input comes from CMR's finite-reference-pool energy in eV/atom. Its correspondence with the checkpoint's training label remains uncertain, and a reported zero gap is not independently verified metallicity.

## Decoder inputs

A frozen CPU forward pass on four validation structures produced finite outputs. Both common-space embeddings have 128 dimensions and unit norm. Their average, used as the joint latent, was not normalized again; its norms ranged from 0.940 to 0.953. Replacing reference coordinates and their center with zeros changed the species and coordinate outputs substantially while leaving the lattice head unchanged. The generation path also replaces learned adjacency with an all-ones matrix and constructs a fixed five-site structure, so successful export alone says little about the raw learned geometry.

The first decoder audit compared six latent choices on 20 validation compositions with three seeds, temperature 1.25, top-k 12 and up to 12 sequential attempts per call. Removing reference geometry increased occupied-site negative log-likelihood by 32.385 nats/site on average, with a 95% composition-group bootstrap interval of 26.921–38.022. Raw structures matched the source in 18/20 reference-assisted cases and 0/20 zero-coordinate cases, using species-aware matching without cell rescaling.

The joint-latent arm returned 33 accepted outputs from 60 calls and 370 consumed attempts, with ten distinct accepted A/B pairs. The fixed-real-latent control also returned 33 accepted outputs, but only one A/B pair. None of the six arms recovered the source A/B pair after the common filters. Changing the latent clearly changed the output, but this did not establish useful source recovery through that interface.

## Fixed-template decoding

The follow-up supplied the same canonical coordinate tensor to every example. With the joint latent, raw source matches rose to 16/20: eight of eleven oxides and eight of nine nitrides. The masked A/B negative log-likelihood improvement over zero-coordinate decoding was 33.020 nats/site, with a 95% interval of 25.508–41.331. This improvement depended on the fixed input geometry.

The remaining output rules were restrictive. Only three of the twenty source compositions satisfied the charge rule, and only NaVO3 and LaNiO3 survived the complete deterministic pipeline. Enumeration found 106 admissible oxide A/B pairs and no admissible nitride pairs under those rules. This was a restriction of the chosen filters, not evidence that nitrides are physically impossible.

The fixed-template joint arm produced 15 accepted outputs from 60 calls and 567 attempts, with six distinct compositions. It recovered NaVO3 once in each seed and did not recover LaNiO3. Thus better raw reconstruction did not produce a higher acceptance rate. The recovery difference from the fixed-real control had a confidence interval that included zero; only two source compositions were eligible to be recovered at all.

## Mixed-anion policies

The next study separated the coverage of an output policy from the decoder's ability to recover a source structure. All 18,928 source structures passed the minimum site, cell and periodic-overlap checks. Their eligibility under the successive policies was:

| Policy | Eligible source records | Main restriction |
|---|---:|---|
| MF_P0 | 40 | Original role pools, projection and chemical vetoes |
| MF_P1/P2 | 280 | Original role pools with the old vetoes removed |
| MF_P3 | 5,408 | Training-derived role pools and one homogeneous X species |
| MF_P4 | 18,928 | Training-derived role pools with independent X1, X2 and X3 |

MF_P1 used the radius-based cell; MF_P2 replayed the same sampled elements with a learned cubic cell. MF_P0 to MF_P1 changed sampling as well as filtering, so that comparison does not isolate a single veto.

These counts describe the source catalogue, not successful neural outputs. The decoder evaluation extended well beyond the initial twenty examples:

| Evaluation | Structures | Role |
|---|---:|---|
| Main deterministic decoder evaluation | 384 | Validation structures spanning 196 reduced-composition groups |
| Fixed-budget sampling comparison | 64 | A subset of the main validation panel |
| Revisit of the earlier examples | 20 | A separate comparison of all five policies on the initial panel |

Rare positive-gap cases were deliberately overrepresented in the main panel. The measurements below describe these selected validation structures rather than population performance across CMR.

### Deterministic evaluation: 384 structures

Each structure was evaluated once with its joint structure–property latent, once with a fixed donor latent, and once with its property-only latent: 1,152 structure/latent-arm evaluations. The donor was a single real encoded material reused across references. The decoder received the same input coordinate template in every case.

| Outcome | Joint latent | Fixed donor | Property-only |
|---|---:|---:|---:|
| Correct raw full composition | 141/384 | 0/384 | 1/384 |
| Raw structure match | 133/384 | 0/384 | 0/384 |
| MF_P4 projected structure match | 133/384 | 0/384 | 1/384 |
| MF_P4 minimum structural checks passed | 384/384 | 384/384 | 384/384 |

The raw metric used the learned atomic coordinates and all six predicted lattice parameters. MF_P4 instead used the selected elements on a fixed five-site template, with a cubic cell whose length was the mean of the three predicted lengths. The raw and projected joint totals both equal 133, but four references had different match outcomes. The [per-reference results](../reports/multifamily_rule_relaxation/deterministic_per_reference.csv) and [family totals](../reports/multifamily_rule_relaxation/deterministic_by_family.csv) keep these metrics separate.

Matching used `scale=False`, fractional length tolerance 0.3, angle tolerance 10°, and site tolerance 0.5 in units of the average free length per atom. These are tolerance-based matches, not exact geometric equality. Joint-latent angles in the larger panel ranged from 89.865° to 90.416°. On cubic references, those near-right-angle outputs lie well inside the angular tolerance and do not demonstrate prediction of meaningful non-cubic distortions.

The joint latent recovered more source information than either control, while all three arms passed the minimum checks. This supports source reconstruction through the audited decoder interface. It does not establish the quality of structures generated from diffusion latents.

### Fixed-budget sampling: 64 structures

The sampling subset contained 64 structures in 32 composition groups. Each applicable policy used twelve one-attempt draws per reference at each of three seeds, with no early stopping. For MF_P4 with learned anions, this gives 36 proposals per reference and 2,304 proposals per latent arm.

| Latent arm | Accepted proposals | Source structures recovered within the budget |
|---|---:|---:|
| Joint | 2,304/2,304 | 31/64 |
| Fixed donor | 2,304/2,304 | 0/64 |
| Property-only | 2,304/2,304 | 2/64 |

A source counted as recovered when at least one accepted MF_P4 output matched it, including species and geometry, with `scale=False`. These were matches of the constructed cubic-template outputs. The 31/64 recovery rate cannot be compared directly with the deterministic 133/384 rate: the subset differs, and each source had 36 chances instead of one. Counts are recorded in the [sampling results](../reports/multifamily_rule_relaxation/stochastic_by_family_policy_arm_seed.csv).

Joint-latent recovery was 4/8 for nitrides, 6/8 for mixed O/F/N references, 5/10 for oxides, 4/10 for oxychalcogenides, 4/10 for oxyhalides and 8/18 for oxynitrides. The [saved bootstrap](../reports/multifamily_rule_relaxation/summary.json) over 32 composition groups gave a joint-minus-donor recovery difference of 48.4 percentage points, with a 95% interval of 39.1–57.8 points. All arms passed the minimum checks despite their different recovery rates. Of the 2,304 joint-latent outputs, 1,323 full compositions and every anion multiset occurred in training; absence from training was a support flag, not established novelty.

### Revisit of the earlier twenty structures

The separate joint-latent comparison on the earlier twenty references recovered 2, 11, 18, 17 and 19 sources under MF_P0 through MF_P4 respectively, using 720 attempts per policy. This comparison was added after finding no shared old-role references in the broader sampling panel, so its results were kept separate. The study supported using MF_P4 for a small latent-generation experiment; it did not establish property control or materials stability.

## Conditional latent diffusion

The CPU experiment cached 1,024 training and 128 validation joint latents, selected across anion and zero/positive-gap strata. Training-only means and standard deviations scaled both latents and the two conditions. A 128-dimensional residual denoiser with width 256 and four blocks learned to predict Gaussian noise over a 1,000-level schedule. The run used 500 updates, batch size 32 and condition dropout 0.10; only the denoiser was trained.

A separate sixteen-example, fixed-noise check reduced MSE from 1.12983 to about 0.00000112 in 200 updates, demonstrating memorization of that small fixture. In the fresh 500-update run, fixed-bank validation MSE fell from 1.11535 to 0.52792, only slightly below the 0.53919 comparator that treats the noisy latent itself as the noise estimate. A recorded CPU continuation check also matched six uninterrupted updates to a two-plus-four resumed run. These checks concern optimization and state handling, not generation quality.

Generation used eight property requests selected from the training cache, eight proposals per request and method, and twenty deterministic DDIM steps. All four methods used the same MF_P4 decoder settings and one-attempt budget. The nearest-property comparator sampled among sixteen cached training neighbors and could select the request's own example.

| Method | Attempts | Minimally accepted | Distinct accepted compositions | Median latent norm |
|---|---:|---:|---:|---:|
| Conditional DDIM | 64 | 56 | 49 | 78.985 |
| Frozen MEIDNet property-only latent | 64 | 64 | 63 | 1.000 |
| Nearest-property training-latent resampling | 64 | 64 | 57 | 0.940 |
| Same denoiser with guidance zero | 64 | 56 | 48 | 79.502 |

The training-cache median norm was 0.943. Eight attempts in each diffusion arm failed because of invalid learned lattice values; there were no CIF serialization failures. The large latent norms were the clearest problem in this run. Passing the minimum decoder checks did not show that a sample resembled the encoded training distribution.

The second saved generation panel replayed the same weights, requests and seeds to add a conditional-versus-guidance-zero distance measurement. It was not an independent experiment. The median paired latent distance was 2.107, small beside norms near 79, and did not establish requested-property control.

The saved CIFs were examples of accepted geometry, not independently validated materials. No final-test model inference, full-data GPU run, original MEIDNet latent-optimization comparison, or independent property and stability evaluation was recorded. The decoder studies identify useful interface and policy changes; the bounded diffusion run leaves the quality of conditional generation unresolved.
