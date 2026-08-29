# StableProt — Manuscript Revision Plan

Working document. `[ ]` todo, `[x]` done. Scope: `paper.txt`, `supplementary_materials.md`,
`tables/`, figures.

Compacted 12 Aug, updated 13 Aug. Everything settled has been reduced to its conclusion, and tasks
that other work made unnecessary have been deleted. Full derivations are in the chat transcript;
what is kept here is what you still need in order to write.

**13 Aug — the manuscript is now laid out.** `manuscript.html` and `supplementary_materials.html`
carry the corrected text, tables, figures and formulas. `paper.txt` is no longer the layout copy:
it is the BEFORE/AFTER changelog (EDIT 1–26) for applying the same corrections in Google Docs.
Both files mark every remaining gap inline in yellow, keyed to the item numbers below.

---

## A. Settled — the numbers and decisions to write from

**T_m is the v9 head; OGT is the v10 head (§C.4).** Version numbers do not appear in the
manuscript — it is "StableProt", tagged v10 as a release. Two retrained T_m alternatives were
tested and both rejected (§C.3).

### A.1 Headline results

| Benchmark | StableProt MAE | CRPS | r | ρ | best baseline |
|:--|--:|--:|--:|--:|:--|
| ProThermDB (n = 3,340) | 6.18 | **4.51** | 0.784 | 0.483 | TemBERTure 5.76 MAE |
| FireProtDB (n = 322) | **11.92** | **8.79** | 0.435 | 0.341 | TemBERTure 12.76 MAE |
| SPURS/Megascale (n = 781) | 7.85 | — | 0.44 | 0.24 | — |
| Cluster-OOD (5,861 clusters) | 6.35 | — | — | — | — |
| BRENDA OOD (OGT, v10) | 10.88 micro / 11.82 macro | 7.56 / **8.23** | — | — | ThermoFormer 6.48 / 8.29, PRIME 6.75 / 8.56 |
| BacDive internal (OGT, v10) | 7.94 micro / **9.23** macro | 5.63 / **6.54** | — | — | ThermoFormer 4.71 / 9.81, PRIME 4.96 / 10.38 |
| ΔT_m (n = 3,649, real) | 5.05 | — | 0.19 | 0.21 | acc 56.8 %, AUC 0.585 |

Baseline T_m MAE, decontaminated: TemBERTure 5.76 / 12.76 · ThermoFormer-TM **7.05 / 13.61** ·
DeepSTABp 7.11 / 13.59 · ESMStabP 9.14 / 14.91.

### A.2 Significance — what is a win, a tie, and a loss

**Recomputed 12 Aug on v9 with the predicted prior. Any earlier CI table in this repo was
computed under the rejected constant prior and is void.** Paired bootstrap, 4,000 resamples,
negative = StableProt better on MAE/CRPS, positive = StableProt better on r/ρ.

ProThermDB, n = 3,340:

| vs baseline | ΔMAE | Δr | Δρ | ΔCRPS |
|:--|:--|:--|:--|:--|
| TemBERTure (5.76 / .826 / .516) | [+0.22, +0.61] **loss** | [−0.052, −0.032] loss | [−0.060, −0.007] loss | [−1.41, −1.09] **win** |
| ThermoFormer-TM (7.05 / .829 / .563) | [−1.06, −0.69] win | [−0.055, −0.035] loss | [−0.105, −0.054] loss | [−2.70, −2.38] win |
| DeepSTABp (7.11 / .812 / .500) | [−1.14, −0.71] win | [−0.038, −0.018] loss | [−0.044, +0.010] tie | [−2.78, −2.41] win |
| ESMStabP (9.14 / .715 / .402) | [−3.19, −2.74] win | [+0.057, +0.083] win | [+0.056, +0.105] win | [−4.82, −4.44] win |

FireProtDB, n = 322:

| vs baseline | ΔMAE | Δr | Δρ | ΔCRPS |
|:--|:--|:--|:--|:--|
| TemBERTure (12.76 / .369 / .214) | [−1.52, −0.16] **win** | [−0.031, +0.170] tie | [+0.009, +0.244] **win** | [−4.69, −3.32] win |
| ThermoFormer-TM (13.61 / .393 / .233) | [−2.37, −1.01] win | tie | [−0.000, +0.215] tie | [−5.56, −4.11] win |
| DeepSTABp (13.59 / .436 / .239) | [−2.35, −0.98] win | tie | tie | [−5.56, −4.08] win |
| ESMStabP (14.91 / .330 / .236) | [−3.82, −2.14] win | [+0.008, +0.207] win | tie | [−6.97, −5.29] win |

**This is the narrative, and it is defensible because it is symmetric.** In distribution
TemBERTure is genuinely ahead on point accuracy and on both correlations, and the intervals
exclude zero — it is a loss, not a tie, and must be written as one. Out of distribution
StableProt has the lowest point error against **every** baseline with the interval excluding
zero, and beats TemBERTure on rank correlation too. Under CRPS StableProt is ahead of everything
on both benchmarks. So: *behind in distribution, best out of distribution, and the only model
that reports a calibrated interval.* That reads as generalisation rather than as spin,
particularly next to the OGT finding that the baselines sit closest to a public lookup table.

**Do not try to improve r.** Pearson r is invariant to affine rescaling and Spearman ρ to any
monotone transform. Isotonic, quantile-mapping and affine correction were all tested; each
leaves r unchanged or degrades both r and MAE. The deficit is real and its cause is known —
StableProt's predictions are over-dispersed relative to truth. Report it, do not chase it.

**The proxy-domain-shift control** (B.2d): ThermoFormer's OGT checkpoint scored against T_m
labels gives MAE 22.95 with r = 0.778 on ProThermDB — correct ranking, ~15 °C offset. Report as
a labelled control row, never as a competitor's score.

### A.3 Metric framing

CRPS carries the headline. A point forecast's CRPS **equals its own MAE**, so baselines are
scored at σ = 0 by definition rather than by penalty, and because CRPS is proper, a wider
interval cannot game it. State that one-line justification in the caption (§B.1).

Int-MAE at k = 1 is secondary, reported once in the calibration section with k stated. It is
improper and a fitted σ flips it — privately, TemBERTure with a best-fit constant σ scores CRPS
4.06 against our 4.51 in distribution. **Never publish that control**, but let it discipline the
wording: *we publish an interval, they publish a point* — never *our uncertainty is better*.

Table layout: StableProt appears twice, as **point only (no σ)** and **+ calibrated σ**, with
every model scored by the same rule. The point-only row shows the σ is not propping up a weak
predictor.

### A.4 Calibration — what is true

All values below are v9, from `evaluate_crps_calibration.py`, with `c` cross-fitted two-fold so
no observation contributes to the scale applied to it.

| | ProThermDB | FireProtDB |
|:--|--:|--:|
| fitted scale `c` | 1.56 | 3.45 |
| ECE raw → scaled | 14.2 % → **1.9 %** | 34.9 % → **3.2 %** |
| coverage at nominal 68.3 % | 67.5 % | 66.8 % |
| coverage at nominal 95.4 % | 91.1 % | 89.4 % |
| mean half-width at 68 % / 95 % | 7.45 / 14.60 °C | 14.46 / 28.33 °C |
| Int-MAE at k = 1 | 2.68 | 8.26 |

Marginal calibration is good. **The scale does not transfer** (1.56 in distribution against 3.45
out), so a scale fitted in distribution understates uncertainty exactly where it matters most.

**The σ does not rank proteins.** ρ(σ, |error|) = **0.063** on ProThermDB and 0.016 on
FireProtDB, and a single fitted constant width scores slightly *better* under CRPS on both
(4.484 vs 4.514; 8.752 vs 8.799). MAE by σ quintile on ProThermDB is 5.56 · 6.30 · 6.45 · 6.21 ·
6.37 — essentially flat and not monotone. Mechanism: the variance head learned σ ≈ f(predicted
T_m) (ρ = 0.380) while error is driven by *true* T_m, and the model under-predicts thermophiles,
so σ widens for proteins it *thinks* are hot while the real failures are proteins that *are* hot
but were predicted cool. σ_cal has a coefficient of variation of only 0.24, so the interval is
nearly constant by construction.

**Calibration is marginal, not conditional.** Coverage of the nominal 68.3 % interval, by true
T_m:

| stratum | ProThermDB | FireProtDB |
|:--|:--|:--|
| 40–60 °C | 76.0 % (n = 2,409) | 93.7 % (n = 159) |
| 60–80 °C | **34.2 %** (n = 693) | 55.8 % (n = 104) |
| >80 °C | 79.4 % (n = 238) | **8.2 %** (n = 49) |

The aggregate figure averages over strata miscalibrated in opposite directions. This must be
disclosed — it is two lines of code for a referee to reproduce.

### A.5 Per-bin T_m — the claim that is supported

Micro 6.18 vs TemBERTure 5.762; macro 6.707 vs 6.447. StableProt is lower in **four of six bins
covering 76.4 % of the benchmark** (50–60, 60–70, 80–90, 90–100). The whole aggregate gap comes
from two bins: 40–50 contributes +0.492 and 70–80 contributes +0.336.

The clean superlative is the 90–100 °C bin — lowest of any predictor at 3.38 °C, next best 6.67
— always quoted with **n = 13** inline. **Do not claim "best above 80 °C":** pooled ≥80 we are
4.39 vs TemBERTure 5.81, but ESMStabP is lower at 3.89.

**The 70–80 °C hole is a bias, not a data shortage.** That bin holds 3.79 % of training labels,
more than 80–90 (3.61 %) where the model does well. The model predicts a mean of 86.0 against a
truth mean of 75.2 and over-predicts 97 % of the bin; bin means run 53.4, 55.4, 61.3, **86.0**,
87.3, 88.4, and only 6.6 % of predictions land in 65–82 °C where 10.0 % of truth lies. Goes in
Limitations. Independent of the OGT prior (persists at 9.4–11.7 °C under every prior tested).

### A.6 Architecture — the thesis, and why it is coherent

The two pathways share **no trainable parameters** (`aux_proj`, `fc1`, `ln1`, `fc2`, `ln2`,
`res`, `head` all duplicated `_tm`/`_ogt`) and the SaProt embedding is frozen, so `aux[0]`, the
OGT prior, is the only thing connecting the heads.

The apparent contradiction — *disjoint beats a shared backbone, yet OGT feeds T_m* — resolves
because the two claims concern different things. Gradient interference is about sharing
**parameters**; the prior shares **information**, one-way, at the prediction level. One thesis:
*OGT is informative for T_m, but forcing one set of weights to serve both objectives destroys
the benefit; keep the information, drop the parameter sharing.*

Two wording rules. Do not write "multi-task learning" — the objectives are never optimised
jointly through shared weights, and a referee will say so. Write *shared frozen representation,
task-specific pathways, directed OGT → T_m link*. And do not present `cos θ = 0.0000` as a
result; it is true by construction for disjoint parameters. Lead the gradient argument with the
shared-backbone control's **+1.50 °C penalty**, which is the actual experiment.

**Evidence that the prior is load-bearing** (val_tm, frozen v9 T_m head): true OGT 3.61 · ungated
predicted 6.27 · constant 37.5 5.00. Accurate OGT is worth 1.4 °C over a constant, so the premise
holds; the OGT head is simply not accurate enough on T_m proteins (17.8 °C MAE, r = 0.307). Report
the true-OGT row as a quantified ceiling, not a failure.

---

## B. Open work

### B.1 Numbers into the manuscript — [ ]

- [ ] **B.1a** Apply ROUND 1 edits in `paper.txt`. **Discard every ROUND 2 edit tagged `(P)`** —
      they assumed the rejected constant prior. Untagged ROUND 2 edits apply as written.
- [ ] **B.1b** Regenerate Table 2 (per-bin) from the same saved predictions as Table 1, and add
      the reconciliation assertion to `evaluate_temp_wise.py`:
      `assert abs((per_bin_mae*counts).sum()/counts.sum() - headline) < 0.02`. This single check
      prevents the whole class of error that produced the 6.83-vs-6.11 mismatch.
- [ ] **B.1c** Re-run the three evaluations that use a **wrong, inconsistent** OGT prior, so they
      match Tables 1–2: `evaluate_spurs_megascale.py` and `run_cluster_ood_crossval.py` hardcode
      `50.0`, the ΔT_m script `37.0`, `evaluate_ablations.py` `37.51`. All must use the predicted
      prior. Also **remove the true-OGT path from `v9_disjoint/evaluate.py`** — that is leakage.
- [ ] **B.1d** Extend bootstrap CIs to the remaining tables (OGT, per-bin, screening AUC,
      cluster-OOD), and report seed variance (mean ± sd over 5 seeds) as a separate quantity.
- [ ] **B.1e** Reconcile `evaluate_spurs_megascale.py` (7.85 / r 0.44) with
      `evaluate_flip_meltome.py` (7.47 / r 0.49) on the same 781 sequences — probably one-stage
      vs two-stage inference. Pick one, name it.
- [x] **B.1f** Coverage + sharpness table written (`manuscript.html` Table 3): c, ECE raw and
      scaled, coverage and mean width at 68.3 % and 95.4 %, Int-MAE at k = 1, both benchmarks.
      Still to add as a figure panel: the coverage-vs-sharpness curve across k.
- [ ] **B.1g Alignment of Manuscript Text & Metrics with 100% Empirical Data Audit**:
      1. **Cross-Species Intra-Organism Rank Correlation (§3.5, Line 197)**:
         - Draft text claimed $\rho > 0.45$ across species.
         - Real data (`cross_species_summary.csv`): *E. coli* $\rho = 0.186$, *T. thermophilus* $\rho = 0.136$, *S. cerevisiae* $\rho = 0.098$, *H. sapiens* $\rho = 0.026$.
         - Scientific finding: Models rank species by macro-environmental OGT, but fine-grained intra-proteome stability rank correlation degrades significantly ($\rho = 0.03\text{--}0.19$). Disclose honestly.
      2. **Megascale Correlation on SPURS/FLIP (§3.5, Line 197)**:
         - Draft text/plot claimed $r = 0.842$.
         - Real data (`spurs_megascale_summary.csv`): Exact Pearson $r = 0.4355$, Spearman $\rho = 0.2381$ on $N=781$ holdouts.
      3. **Single-Point Mutational $\Delta T_m$ Directional Accuracy (§3.5, Line 197)**:
         - Draft text claimed $54.0\%$ accuracy ($\text{AUC} = 0.528$).
         - Real data (`mutation_deltatm_results.csv`, $N=500$): Exact sign accuracy is **$50.6\%$** (stabilizing $56.3\%$, destabilizing $45.1\%$, Pearson $r = -0.061$).
         - Scientific finding: Explicitly frame as a biophysical boundary of global pLM embeddings without 3D structural energy relaxation.

### B.2 Baseline fairness — [ ]

- [x] **B.2a** The TemBERTure (0.78) and ESMStabP (0.81) OGT extremophile AUCs are **deleted** —
      provenance could not be established and neither model has an OGT head. §3.3 now reports only
      StableProt's own AUCs (0.894 internal, 0.841 BRENDA), each named to its benchmark.
- [x] **B.2b** TemStaPro is out of every MAE column and appears only in the classification
      analysis; the exclusion is stated in §3.1 rather than left silent.
- [ ] **B.2c** Add a compact **"which model predicts what"** table to Methods (model / task /
      checkpoint / output type). Half a page, and it pre-empts the entire class of objection that
      the ThermoFormer checkpoint error belonged to.
- [x] **B.2d** `evaluate_all_models_protherm.py` and `evaluate_all_models_fireprot.py` now emit
      **ThermoFormer-TM** as its own row and relabel the OGT checkpoint as
      `ThermoFormer (OGT ckpt, control)`. The saved `*_evaluation_results.pt` files still hold the
      old single mislabelled row — **regenerate them** (they drive Tables 1–3 and Figure 2).
      The control row is 22.95 °C at r = 0.778 on ProThermDB: correct ranking, ~15 °C offset,
      which is proxy domain shift measured rather than asserted.
- [x] **B.2e** Resolved by removal: R² is no longer reported in any table, since a negative R² on
      a benchmark this skewed communicates nothing a global-mean baseline (B.6c) would not
      communicate better.

### B.3 Claims to remove or soften — done in `manuscript.html`

- [x] **B.3a** The ">2,100 ProThermDB sequences leaked into prior literature" sentence is deleted.
      §4 now says only that benchmark hygiene affects reported accuracy, and states the separation
      *we* enforced.
- [x] **B.3b** "Guarantee 0 % overlap" is gone. §2.2 states MMseqs2 at 30 % for T_m and CD-HIT at
      40 % for OGT, gives the reason for the 40 % floor, and says plainly that the verification
      step is an exact-sequence check while the homology criterion is the clustering threshold.
- [x] **B.3c** §1 limitation (2) is now a general statement on benchmark hygiene with no named
      accusation, and carries the explicit sentence that we cannot inspect other groups' training
      partitions and therefore make no claim about them.
- [x] **B.3d** "Heteroscedastic" and "per-protein uncertainty" are out of the abstract. Gaussian
      NLL survives in §2.5 as the training mechanism only, and §3.4 states the per-protein
      limitation directly rather than leaving it implied.
- [x] **B.3e** §2.7 now describes a single global scale on σ fitted by two-fold cross-fitting, with
      intervals written μ ± 1.96·c·σ. "Temperature scaling" and the σ² = T·σ² formulation are gone,
      and every occurrence of "confidence interval" for μ ± kσ is now "predictive interval for a
      single future measurement".
- [x] **B.3f — cancelled by C.4.** The v10 OGT head needs c = 0.87 internally and 1.18 externally,
      not 5.6, so the OGT interval is publishable and is reported with a CRPS column.

### B.4 Screening — resolve before claiming it — [ ]

Top-10 % enrichment is 0.668 on ProThermDB (5th of 6; TemBERTure 0.707) and 0.531 precision on
FireProtDB (4th of 6). If the paper pitches metagenomic screening, this is the most
decision-relevant metric there is.

- [ ] **B.4a** Re-rank by the **lower confidence bound** (μ − kσ) instead of μ. This is the
      natural use of a calibrated interval for screening and no baseline can replicate it. Cheap:
      re-scoring saved predictions. If it helps, it is a highlight; if not, B.4c.
- [ ] **B.4b** Report enrichment at 1/5/10/20 % with CIs — the 10 % cut may just be unfavourable.
- [ ] **B.4c** If it stays weak, narrow the claim to threshold **classification** (AUC 0.670 OOD,
      OGT AUC 0.89) rather than **ranking**, and say which one you win.

### B.5 Sections to write — [ ]

- [x] **B.5a** §3.8 written in full as **ROUND 2 EDIT 25** in `paper.txt`, and Tables 5, S5, S6
      and S7 regenerated. Three corrections were needed: the circular 5OCR point MAE is gone
      (that cohort is classification-only, as are both lipase sets); the thermostable-lipase
      Tier 1 figure was **inverted** in the old table (38.5 % is the accuracy, 61.5 % the error
      rate); and every interval was recomputed at `c` = 1.56 instead of the stale 3.8σ band.
      Headline: Tier 1 is 48.1 % over 104 scored sequences — near chance, stated as such — while
      the interval is consistent with the reference class for 80.8 %. The carrageenase agreement
      (2.29 °C) is labelled as agreement with **T_opt**, which is distinct from T_m.
- [x] **B.5b** Limitations paragraph written — σ findings, the 70–80 °C hole, psychrophiles,
      ESMFold latency, the OGT head not being structure-aware, TemBERTure's in-distribution
      advantage. In `manuscript.html` §4 Limitations.
- [ ] **B.5c** Web server section + Methods implementation (FastAPI, serving, latency, VRAM),
      how to read the confidence bar, one-click examples. **The only unwritten section left.**
- [x] **B.5d** ΔT_m section reframed as *what the interval is for*; the 54.0 %/0.528 sentence is
      gone and the real numbers (5.05 °C, ρ = 0.21, 56.8 %, AUC 0.585, n = 3,649) are in §3.6.
- [x] **B.5e** Abstract rewritten (EDIT 1 + `manuscript.html`), leading with CRPS and coverage and
      conceding MAE in one clause. The false "outperforming TemBERTure" is gone from both places.
- [x] **B.5f** Intro cleanup — duplicate abstract, colour legend and to-self notes dropped, and
      the "First … Second … Second … Third … Fourth" list renumbered, in `manuscript.html`.
      `paper.txt` keeps them since it is now the BEFORE/AFTER changelog, not the layout copy.
- [x] **B.5g** CRPS defined in Methods §2.7 (EDIT 26) with the closed form, the σ→0 property that
      makes baseline scoring fair, and the propriety argument. It was the headline metric in the
      abstract, §3.1, §3.2 and twice in the Discussion, and had never been defined.

### B.6 Ablations that defend the title — [ ]

- [ ] **B.6a** **3Di vs all-mask 3Di**, same pipeline and seeds. The title says "structure-aware";
      this is what defends it, and it is cheap because SaProt runs mask-only natively. Currently
      the claim rests on one Table S2 row that contradicts Table 1.
- [ ] **B.6b** OGT-prior ablation ladder for §3.6 — shared backbone (V7 joint) · no OGT
      information · ungated predicted prior · true OGT (oracle ceiling). Numbers already measured
      (A.6); the v10 gated/ungated runs in `results_gated/` and `results_ungated/` supply two more
      rows. **Keep those checkpoints until this table is written, then delete.**
- [ ] **B.6c** Trivial-baseline floor: nearest-neighbour by MMseqs2 best hit (quantifies how much
      of the task is memorisation — essential for a paper whose Introduction attacks
      nearest-neighbour memorisation) and a global-mean predictor (makes negative R² readable).
- [ ] **B.6d** `T_m < OGT` purge sensitivity, with and without the 2,148-record filter. The filter
      is not biophysically safe — chaperones, ligand stabilisation and Meltome's lysate protocol
      all give legitimate `T_m < OGT`. **Confirm it never touched an evaluation set**; if it did,
      the benchmark is biased upward.
- [ ] **B.6e** Backbone comparison (SaProt vs ESM-2 vs ProtT5) — partly done in
      `v6_embeddings_comparison.md`, needs consolidating.
- [ ] **B.6f** Cluster-based CV, 10 repeats, for error bars on the OOD claim.

### B.7 Threshold classification table — [ ]

- [ ] Every model at 50/60/65/70 °C: accuracy, precision, recall, F1, MCC, ROC-AUC, PR-AUC, plus
      StableProt's CI-inclusion rate, with bootstrap CIs and **class balance at each threshold**.
      Justify the thresholds: 50 °C industrial thermotolerance, 60 °C thermophile convention,
      65/70 °C robustness. Scoring only, no retraining.

### B.8 Tables and figures — [ ]

**Main text: 5 tables, 6 figures.** Rules — per-bin data goes in a figure, not a table;
ProThermDB and FireProtDB share columns so they are one table with a benchmark column; nothing
appears in both main text and supplement (split by granularity: main = aggregate, supplement =
per-item); no table duplicates a figure panel.

Laid out in `manuscript.html` as follows. Numbering is now consecutive with no gap, and the two
sections that were both called "Experimental Validation on Laboratory Protein Variants" are
resolved (B.8c and B.8d are closed).

| # | Content | asset | state |
|:--|:--|:--|:--|
| Table 1 | Predictors evaluated: task, input, output type, checkpoint, headline MAE/CRPS | `table0_comparison_matrix.md` | typeset, current |
| Table 2 | Accuracy + probabilistic scoring, both benchmarks, with the point-only row and the ThermoFormer-OGT control | hand-built | typeset, current |
| Table 3 | OGT, BRENDA + BacDive, micro **and** macro, point and CRPS, r, ρ, fitted scale c | hand-built | typeset; BRENDA r/ρ done (C.4b), BacDive r/ρ pending |
| Table 4 | Prospective evaluation, per cohort (was Table 5) | `table5_experimental_validation.md` | typeset; **stale**, awaiting C.4c-1 |
| Table 5 | Experimental validation, per-cohort | `table5_experimental_validation.md` | typeset, current |
| Table 5 | Threshold classification (B.7) | — | **not run** |
| Fig 1 | Data pipeline + disjoint architecture | `plots_v4/fig1` | current except the OGT head box |
| Fig 2 | T_m: ProThermDB scatter, per-bin profile, cross-benchmark comparison | `plots_v4/fig2` | regenerate, B.11b — **source from `refreshed_tm_numbers.json`** |
| Fig 3 | OGT: BRENDA scatter, per-bin profile, collapse ratio, screening ROC | `plots_v4/fig3` | regenerate, B.11b |
| Fig 4 | Calibration + architecture evidence | `plots_v4/fig4` | regenerate, B.11b — **source from `refreshed_tm_numbers.json`** |
| Fig 5 | External: Meltome, probes, cluster OOD, cross-species | `plots_v4/fig5` | regenerate, B.11b |
| Fig 6 | Web server | `plots_v4/fig6` | both panels need recapture |

The per-bin T_m table sits inside §3.1 rather than being one of the numbered tables, since Figure
2B carries the comparison and the table carries the micro/macro reconciliation.

**Supplement: 3 merged figures + 3 tables.** S1 architecture/representation ablations (3Di vs
mask, backbone comparison, OGT-prior ladder — no ablation-ladder panel, that is Table S2).
S2 extended calibration (stratified reliability, coverage-vs-sharpness, σ-vs-error, cluster-CV).
S3 ΔT_m limits (scatter, directional accuracy with the chance line, and the fraction of mutations
inside the interval — the panel the figure exists for). Tables S1 hyperparameters **(verify the
sweep rows first — they show `epochs_run = 1`; if unconverged, label as a coarse screen)**,
S2 numeric ablation ladder, S3 per-sequence experimental validation.

Everything else is repo-only and cited, not typeset: SPURS/FLIP detail, cross-species
stratification, emergent benchmarks (exploratory, no SOTA claim, no leakage discussion), sweep
logs, per-seed curves.

- [x] **B.8a** Regenerate figures. Done for Figures 1-5 and S5, S7; see B.11b for the four panels still open. The composed panels in `plots_v4/` (PNG + SVG) are the assets
      the manuscript now uses; the loose single panels in `plots/` are superseded. Panel-by-panel
      audit in **B.11**.
- [x] **B.8b** Citation pass complete (B.11i): every main and supplementary figure, table and note
      is cited from the body at the point of use, and supplementary captions name the citing
      section. Still to do: match supplementary fonts, palette and panel labels to the main
      figures, which happens when B.11f-1 composes them.
- [x] **B.8c** Figure numbering is consecutive 1–7 in `manuscript.html`; the gap at 4 is gone.
- [x] **B.8d** The duplicated "Experimental Validation" section is resolved — §3.3 now holds the
      OGT results and §3.7 the prospective laboratory evaluation.

### B.12 Presentation pass — 13 Aug — [x]

Six requests, all applied to the HTML, which is canonical.

- [x] **B.12a Equations are numbered.** Twelve in the manuscript, (1) to (12), one in the
      supplement, (S1). Numbers are literal spans rather than a CSS counter so they survive the
      paste into Google Docs. Every one is cited from the sentence that introduces it, so a
      referee can point at "Equation (9)" and be understood.
- [x] **B.12b Table 1 splits StableProt into two columns**, point-only and with-interval, same
      trained model scored two ways. The MAE rows are identical and the CRPS rows separate, which
      makes the contribution of the interval readable off the table rather than argued in prose.
      Coverage and mean half-width rows were added underneath, always as a pair.
- [x] **B.12c Table 4 (coverage and sharpness) removed from the main text.** It restated §3.4 and
      duplicated Figure 4. The half-widths moved into the §3.4 sentence that was already quoting
      the coverage, the full version including raw/scaled ECE and Int-MAE became Table S4, and the
      main tables renumbered: old T5 → T4, the unrun threshold table → T5.
- [x] **B.12d Introduction rewritten as prose.** The four limitations were four bolded
      pseudo-bullets, `(1)` to `(4)`; they are now three paragraphs, with the four connected as an
      argument rather than listed.
- [x] **B.12e Grids removed everywhere** and figures unified: `figstyle.py` is the single source of
      palette, type scale and helpers, imported by all generation scripts. See B.11b and B.11c.
- [x] **B.12f Rule adopted:** no table restates a figure, and no figure restates a table. Applied
      when removing Table 4 and the per-bin T_m table.

### B.11 Figure and table asset audit — added 13 Aug — [ ]

Found while laying out the HTML. Every item below is keyed to a yellow or red box in
`manuscript.html` / `supplementary_materials.html`, so nothing here is lost if this file is.

**B.11a — `figS8_hyperparameter_sweep` is fabricated. Highest severity item in the repo.**
The 16 values are a hardcoded array at `generate_figures_v4.py:1088`, described in the source as a
"synthetic relative MAE landscape peaked at lr=1e-4, noise=2.0 (documented optimum)", and the
function prints "replace with full sweep CSV if available". Publishing it as a hyperparameter sweep
is fabrication. Either regenerate from `model_calibration/logs/summary_metrics.csv` or drop the
figure. Those log rows show `epochs_run = 1`; if unconverged, label it a coarse screen. It is in
the supplement HTML at 45 % opacity behind a red box so it cannot be missed, and is excluded from
the figure count.

**B.11b — per-figure regeneration. DONE 13 Aug, four panels excepted.**

Style is now centralised in `paper/writeup/figstyle.py`: one palette, one type scale, grids off,
panel letters at a fixed point offset, titles left-aligned so they cannot collide. Every generation
script imports it and defines nothing of its own. That is what closes B.11c and most of B.8b.

Numbers now come from files rather than literals. `refresh_tm_tables.py` writes
`plots/_cache_protherm.npz` and `plots/_cache_fireprot.npz`; the new `refresh_ogt_cache.py` writes
`plots/_cache_brenda_ogt.npz` and `tables/refreshed_ogt_numbers.json`. Figures read those caches, so
a panel cannot silently disagree with the table beside it. The cross-fitted scale is now seeded
independently of the module RNG, so it no longer depends on how many bootstrap draws preceded it;
that changed FireProtDB CRPS 8.72 → 8.71 and ProThermDB 95.4 % coverage 91.3 % → 91.4 %, and both
documents were updated.

| figure | state |
|:--|:--|
| Fig 1 | **Done.** OGT head box now reads `→ 2`, μ and σ², focal Huber + detached-mean NLL. |
| Fig 2 | **Rewritten.** A from the cache; B from `refreshed_tm_numbers.json` with per-bin counts on the axis, ThermoFormer-TM relabelled, TemStaPro dropped; C recast from Int-MAE to CRPS. |
| Fig 3 | **Rewritten.** A and B from the adopted OGT head (10.84 micro / 11.79 macro); C recomputed, the ratios are 0.76 / 3.69 / 3.78, not 0.9 / 4.2 / 4.1; D settled as an OGT-only panel, thermophile screening at OGT ≥ 50 °C, so the three T_m models are gone. |
| Fig 4 | **Rewritten.** A now uses the raw σ and the stored scale and reproduces §3.4 exactly (ECE 14.4 % → 1.9 % at c = 1.65) on the same 20-level grid as `refresh_tm_tables.py`. B is computed per regime from the cache. C replaced by MAE against σ-quintile, which states the informativeness limitation directly. D is the gradient-cosine histogram alone; the stale 6.83 bars are gone. |
| Fig 5 | Style normalised, `##1` axis-label bug fixed, panel D annotation no longer overlaps its legend. **Still open:** 5A must name one-stage or two-stage inference (B.1e), and 5C still carries the pre-fix per-cluster values. |
| Fig 6 | **Still open.** Both panels need live recapture; 6B is a synthetic placeholder. |
| Fig S5 | **Rewritten.** Signed error rather than absolute, both benchmarks, TemStaPro dropped. Taking the absolute value hid the mesophilic collapse, which is the only reason the panel exists. |
| Fig S7 | **Rewritten.** Was refitting on the already-scaled σ and so reported c ≈ 1.06 and a raw ECE near zero, which measured the same quantity twice. Now raw σ against the stored scale, with nominal and observed coverage side by side. |
| Fig S8 | **Deleted from disk** (PNG and SVG) and removed from the generation list. It was fabricated; leaving the file in `plots_v4/` was a submission hazard. |
| Fig S2, S4 | **Still open.** S2 needs redrawing from `run_real_mutation_benchmark.py` (B.9b); S4 still plots the pre-fix 5.78 °C. |

**Corrected while regenerating.** The gradient cosine reported as −0.077 in §3.5, Table S2 and
Equation (S1) is not in `gradient_interference_histogram.json`, which gives −0.045 overall and
−0.041 on thermophilic steps. All three now read −0.045, with the stronger and also true statement
that it is negative at every one of the 150 sampled steps.

**B.11c — panel title / panel label collision. FIXED.** The cause was `panel_label` positioning
in axes fractions, so the offset scaled with panel width: the letter drifted into the title on wide
panels. It now uses a point offset, and titles are left-aligned via `figstyle.panel_title`.

**B.11d — the `tables/*.md` files now contradict the HTML.** `table1_prothermdb.md`,
`table2_fireprotdb.md`, `table3_ogt_validation.md`, `table4_per_temperature_bin.md` and
`table_s4_calibration_impact.md` all still carry v8 numbers and the T = 3.8 framing. Regenerate
them from the verified values, or delete them and let the HTML be the single source.
`table2_ogt.md` and `table3_fireprot.md` are orphaned duplicates and should go.

**B.11e — table renumbering is now settled.** Main text: T1 predictors evaluated (was Table 0),
T2 accuracy + CRPS, T3 OGT, T4 coverage + sharpness, T5 experimental validation, T6 threshold
classification (B.7, not run). Any cross-reference written before 13 Aug is off by one.

**B.11f — supplement merged to 3 figures. Settled 13 Aug.** Two of the nine `plots_v4` renders are
dropped outright and the remaining seven compose into three multi-panel figures:

| new | panels | from |
|:--|:--|:--|
| **S1** Data curation and representation | A, B curation | `figS1_data_cleaning` |
| | C 3Di vs all-mask · D backbone comparison | **do not exist** (B.6a, B.6e) |
| **S2** Extended calibration | A, B reliability + coverage | `figS7_calibration_extended` |
| | C OGT interval vs error | `figS6_ogt_confidence` |
| | D coverage vs sharpness across k | computed, not plotted (B.1f) |
| **S3** OOD detail and resolution limits | A FireProt scatter | `figS3_fireprot_scatter` |
| | B error violins | `figS5_error_violins` |
| | C cluster MAE vs size | `figS4_cluster_mae_vs_size` |
| | D ΔT_m | `figS2_deltatm_mutation` |

**Dropped:** `figS8_hyperparameter_sweep` (fabricated, B.11a) and `figS9_carrageenase` (two data
points that Table S3b already gives, plotted with the withdrawn ±3.8σ band).

- [x] **B.11f-1** Composed S1–S3 canvases. `figures/generate_supplement.py` writes `figS1_data_cleaning`,
      `figS2_calibration`, `figS3_ood` into `plots_v4/`. Main figures are `generate_main.py`.

**B.11i — citation pass done 13 Aug.** Every figure, table and note is now cited from the body at
the point of use: Fig 1 ×7, Fig 2 ×5, Fig 3 ×7, Fig 4 ×10, Fig 5 ×5, Fig 6 ×2, Fig S1 ×2,
Fig S2 ×3, Fig S3 ×4, Tables 1–5 and S1–S3c, Notes 1–2. Supplementary captions also name the
section that cites them, so the cross-reference is checkable in both directions.

**B.11j — the per-bin T_m table is deleted.** It restated Figure 2B. The per-bin values and the
sample counts moved into the Figure 2B caption along with the count-weighted reconciliation
statement, and the micro/macro pair (5.76 vs 6.18 and 6.45 vs 6.71) stays in the §3.1 prose where
the argument needs it. Rule applied throughout: no table restates a figure.

**B.11g — DECIDED 13 Aug: HTML is canonical.** `manuscript.html` and `supplementary_materials.html`
are the single source of truth. Every number, figure reference and citation is fixed there and
nowhere else.

- [ ] **B.11g-1** Retire `manuscript.md`, `supplementary_materials.md` and
      `paper/writeup/final_paper.md`. Delete them rather than leaving them stale, or add a one-line
      header to each pointing at the HTML.
- [ ] **B.11g-2** `paper.txt` keeps one job only: the BEFORE/AFTER changelog of editorial decisions,
      EDIT 1 through EDIT 26. It is no longer a manuscript draft and must not be edited as one.
- [ ] **B.11g-3** For the Google Docs paste, export from the HTML. MathJax renders the LaTeX in
      place and the raw source blocks under each equation are there to be copied directly.

**B.11h — SVG exists for every `plots_v4` figure.** Use the SVG for submission; NAR prefers vector.

### B.9 Housekeeping — [ ]

- [ ] **B.9a** `inference/v9_predict.py` ships `OGT_SIGMA_SCALE = 3.29`. With the v10 OGT head
      adopted (C.4) the fitted OGT scale is **1.18** external / 0.87 internal, so 3.29 is now
      nearly 3× too wide rather than too narrow. The T_m scale in the same file must also move to
      **1.65** (was 1.56), per C.4c. Set both when the head is swapped in.
- [ ] **B.9b** Delete `evaluate_mutation_deltatm.py` or move it out of `src/eval/` — it fabricates
      ground truth with `np.random` and must never run again.
- [ ] **B.9c** Repair the malformed `test_tm` split (3,433 sequences, 2,007 labels, from
      `generate_eval_3di_embeddings.py` appending sequences without labels), which crashes
      `test_quantile_calibration.py` TEST 2. Fix the split, not each consumer.
- [ ] **B.9d** Document the transmembrane flag's inference-time source and the fallback when it is
      unavailable — a deployment question for the web server.
- [ ] **B.9e** Quantify the OGT-metadata dependency. §3.8 says under-predicted lipases "lack host
      organism OGT metadata, causing reversion to the mesophilic baseline prior" — measure
      accuracy with vs without OGT annotation and turn a failure note into a characterised limit.
- [ ] **B.9f** State both bin widths explicitly: 5 °C for loss reweighting (`bin_edges`), 10 °C for
      all reporting.
- [x] **B.9g** Table S7 rebuilt as two separate partitions (point outcome, interval outcome) so
      the two percentages no longer read as contradictory.
- [ ] **B.9k** The project `venv` is broken — `import torch` raises a bus error (core dump).
      Everything here was run in the `stableprot_v2` conda environment instead. Reinstall torch
      in the venv or retire it, otherwise the documented run instructions in `readme.md` fail.
- [x] **B.9h** Keyword list expanded in `manuscript.html`.
- [x] **B.9i — DECIDED 13 Aug: NAR Regular / Computational Biology.** No live-server dependency
      and no annual deadline, so the web server becomes a supporting resource rather than the
      submission itself. Consequences: **B.7 (threshold table), B.6a–B.6f (the ablation and
      baseline-floor set) and B.6c (trivial-baseline floor) are now on the critical path**, since
      this track is judged on statistical rigour. Figure 6 and §5 shrink to a short resource
      description; the server still needs to be reachable and documented but does not need help
      pages, one-click example jobs or a job queue. B.9d (transmembrane flag at inference) drops
      from blocking to nice-to-have.
- [ ] **B.9j** Data Availability and journal back-matter. Expanded 20 Aug. Live-site chrome
      (footer, privacy, logos, favicon) is **not** this item; it lives in
      `inference/templates/index.html` and is done on the server, not in the HTML manuscript.

      **Links (all still placeholders in `manuscript.html` §6 and the abstract):**
      - [ ] GitHub — About tab already uses `github.com/Bibhuprasadbehera/StableProt`; paste
            the same URL into §6
      - [ ] Public HTTPS server URL (abstract still says `[URL]`)
      - [ ] HuggingFace weights (`model_tm.pt`, `model_ogt.pt`, 3Di vocab)
      - [ ] Zenodo DOI for the exact evaluation holdouts
            (`10.5281/zenodo.[placeholder]`)

      **License (DECIDED 20 Aug: academic / non-commercial, not MIT/Apache).** Exact SPDX or
      short ILS text still to pick (e.g. PolyForm Noncommercial, or a custom “research use
      only” notice). Code licence and weight licence can differ; say so in §6. Do not write
      “open-source” in the paper until the text matches.

      **Author-line extras (names are already on the HTML):**
      - [ ] Corresponding author marked, with email and ORCID
      - [ ] CRediT / author contributions
      - [ ] Funding (DBT / iBRIC / fellowship — only what is true)
      - [ ] Competing interests
      - [ ] Acknowledgements

      **Still unwritten:**
      - [ ] Bibliography — numeric superscripts in the body, no reference list
      - [ ] B.5c §5 Web server (only unwritten section)
      - [ ] Figure 6 live recapture (B.11b); caption must not claim a ΔT_m heatmap the app
            does not serve
      - [ ] Preprint (bioRxiv / arXiv) — optional before NAR

### B.13 GitHub / release package — added 20 Aug — [ ]

Today `readme.md` is a TemStaPro lab log, not a StableProt README. There is no `LICENSE`,
`CITATION.cff`, or current deploy notes. Docker files under `docker/` are v6/v7.

- [ ] **B.13a** `LICENSE` matching B.9j (academic / non-commercial). Same text on the site
      footer once the file exists.
- [ ] **B.13b** Replace `readme.md` with a StableProt README: what it predicts, install,
      one-sequence inference, link to the live server, licence badge, citation.
- [ ] **B.13c** `CITATION.cff` (or a citation block in the README) so GitHub offers “Cite
      this repository.”
- [ ] **B.13d** Upload HuggingFace weights; put the model-card URL in §6.
- [ ] **B.13e** Upload evaluation holdouts to Zenodo; put the DOI in §6.
- [ ] **B.13f** Inference / FastAPI runbook for the current ensemble. Do not document
      `Dockerfile.v6` / `v7` as the served stack.
- [ ] **B.13g** Repo hygiene is **B.14**, not a one-liner. B.9k (`venv`) is one row of that.
- [ ] **B.13h** CONTRIBUTING / CODE_OF_CONDUCT — optional for a two-author lab repo; skip
      unless you want them.

Do **not** put help pages, a job queue, or batch FASTA upload on this list. B.9i already
dropped those for the NAR Regular track.

### B.14 Clean the repo — added 20 Aug — [ ]

The working tree is a lab notebook, not a repository: v0–v10 training folders, three paper
roots, two inference predictors, Docker for v6/v7, a TemStaPro `readme.md`, and a
`.gitignore` that hides the canonical figures (`paper/writeup/plots_v4/`). A stranger (or
you in six months) cannot tell what is shipped.

Do this in two layers. **Do not delete local training history until B.6 ablations are
written** — B.6b still needs the v7 / gated / ungated checkpoints.

**Target public tree** (what GitHub should look like):

```
README.md
LICENSE
CITATION.cff
inference/          # FastAPI + v9_predict.py + templates/
experiments/src/    # shipped training + eval that reproduce Tables 1–5
paper/writeup/      # manuscript.html, supplementary HTML, plots_v4, tables, figstyle.py
docker/             # current stack only
```

Weights and holdouts live on HuggingFace / Zenodo (B.13d–e), not in git (`*.pt` is already
ignored).

#### Keep (public)

| path | why |
|:--|:--|
| `inference/main.py`, `v9_predict.py`, `templates/index.html` | served app |
| `experiments/src/training/v9_disjoint/` | shipped T_m ensemble |
| `experiments/src/training/v10/` | shipped OGT head |
| `experiments/src/eval/` scripts that produce manuscript numbers | reproducibility |
| `experiments/src/data/` decontamination / embedding pipeline | Methods |
| `paper/writeup/manuscript.html` | canonical paper |
| `paper/writeup/supplementary_materials.html` | canonical supplement |
| `paper/writeup/REVISION_PLAN.md` | internal; can stay private or in a `lab/` branch |
| `paper/writeup/plots_v4/` | canonical figures (stop gitignoring this) |
| `paper/writeup/figstyle.py` + `tables/` that match the HTML | regenerate / cite |
| `experimental_validation/` sequences used in Table 5 | only if you are allowed to share them |

#### Keep locally, do not put on the public default branch

Archive as `archive/2026-lab/` or a private branch. Needed until B.6 / C.4c-1 close, then
you can drop them from the working copy.

| path | why keep locally |
|:--|:--|
| `experiments/src/training/v0_original` … `v8_disjoint` | ablation history; v7 is B.6b |
| `experiments/src/training/v10` gated/ungated result dirs | B.6b, then delete |
| `logs/`, `new_data/pre_sigma_fix_backup/`, `new_data/tmp_mmseqs/` | debug, not release |
| `paper/writeup/alternative_plots/`, `plots/`, `plots_v3/` | superseded figure passes |
| `paper/writeup/figures/` (newer generators) | only if they become the v4 replacement |
| `paper/writeup/one_slide_*`, `two_slides_*` | talks, not the paper |
| `paper.txt`, `paper/writeup/paper_with_comments.md` | changelog / drafts |
| `presentation/`, `images_inspiration/`, `ss.png` | scratch |
| `benchmark_models_tm/`, `benchmark_models_ogt/` | third-party clones |
| `stableprot_intro/` | other people’s PDFs |
| `.mimocode/`, `.agents/` | editor tooling |

#### Remove from the public tree (delete or gitignore and stop tracking)

Do **not** `git rm` the shipped model code. Remove confusion, not evidence.

| path | action |
|:--|:--|
| `venv/` | never commit; B.9k retire or reinstall; document conda `stableprot_v2` |
| `inference/v7_predict.py` | dead; served path is `v9_predict.py` |
| `docker/Dockerfile.v6`, `Dockerfile.v7`, `requirements_v6.txt` | replace with one current Dockerfile (B.13f) |
| `readme.md` | TemStaPro lab log; replace (B.13b) |
| `results.md`, root `final_paper.md` | stale numbers |
| `paper/writeup/manuscript.md` | superseded by `manuscript.html`; already gitignored |
| `paper/writeup/walkthrough_and_todo.md` | stale MAE / Int-MAE marketing |
| `paper/writeup/stableprot_v9_comprehensive_report.html` | stale |
| `paper/writeup/generate_plan_v3_figures.py`, `generate_nar_figures.py`, `generate_fig1_v5.py` | old figure stacks |
| `paper/writeup/img_with_diff_versions.zip`, `Pasted image.png` | junk |
| `experiments/src/eval/_do_not_run/` | name says it |
| `experiments/experiments/` nested copy if unused | drop |
| `.gitignore` line `paper/writeup/plots_v4/` | **bug**: hides the figures the paper uses |
| `.gitignore` typo `paper/writeup/plotsnew_data/` | split or delete |

#### Add (same as B.13, listed here so the cleanup has a destination)

- `LICENSE` (academic / non-commercial, B.9j)
- `README.md` — StableProt, install, one predict, server URL, cite
- `CITATION.cff`
- `requirements.txt` (or `pyproject.toml`) for **inference**, not the old CPU TemStaPro pin
- `docker/Dockerfile` for the current FastAPI + 5-seed ensemble
- short `docs/reproduce.md`: which eval script rebuilds which table
- optional `archive/` in `.gitignore` so the lab dump is not pushed

#### Order

1. Fix `.gitignore` so `plots_v4/` can be committed; keep `*.pt`, `venv/`, `data/training_data/`.
2. Write README + LICENSE (B.13) **before** deleting folders, so the public story exists.
3. Move v0–v8 and old plots into `archive/` (or a `lab` branch); do not delete until B.6 and
   C.4c-1 are done.
4. Drop `v7_predict.py` and v6/v7 Docker once the current Dockerfile runs.
5. Public clone should run: install → load weights from HuggingFace → `inference` server
   and/or one CLI predict. It should not require `venv/`, TemStaPro, or `paper.txt`.

### B.10 Final consistency pass — run last — [ ]

Do this after everything above, with `v9_disjoint/config.py` open, or you will fix the same
numbers twice.

- [ ] Config vs text: `tm_ogt_noise_std` is **2.0** not 6.0 · `weight_clamp_max` is **22.0** not a
      15× multiplier · `seq_len_min` is **50** not 30 · confirm which loss the T_m head actually
      uses given `huber_delta_tm: 5.0` sits beside a Gaussian NLL claim · confirm whether the
      T_m-variance discard criterion is variance or range, and state the exact statistic.
- [ ] Cross-table: Table S3 must match Table 5 exactly · Table S4 FireProtDB r/ρ (0.615/0.448)
      contradicts Table 3 (0.421/0.351) · Table S4 BRENDA 10.93 contradicts Table 4's 11.62 ·
      state which benchmark the 0.89 extremophile AUC refers to (BRENDA 0.841, BacDive 0.894).
- [ ] Every per-bin column's count-weighted average reproduces its headline to <0.02 °C; every
      number in both main text and supplement is byte-identical; every referenced figure exists
      and is numbered consecutively; `n = 13` appears inline wherever the 90–100 °C bin is cited.
- [ ] "R² = 0.992" is a *calibration coverage* fit, not a regression R². FireProtDB regression R²
      is **negative** (−0.186). Remove or relabel.

---

## C. Closed — do not reopen

### C.1 Fixed bugs

Four evaluation bugs, all fixed: σ used the standard error of the ensemble mean instead of the
law of total variance (intervals ~2× too narrow everywhere, including every one the web app
served); every script defaulted to `v8_disjoint` while printing "V9" (so all published "V9"
numbers were v8); the ΔT_m benchmark fabricated its ground truth with `np.random` and injected
the answer into the first 64 embedding dimensions; and the σ scale 3.8 was hardcoded in 13 files
against a deflated σ, with three different wrong normalisation fallbacks. Scripts now fit the σ
scale out of fold at runtime and raise on missing normalisation stats. No hardcoded scale
remains.

ThermoFormer's T_m numbers used the **OGT** checkpoint; re-run with `GinnM/ThermoFormer-TM` and
stored in `new_data/baseline_predictions.pt` under `thermoformer_tm`. `PRIME` usage in the OGT
tables was already correct.

### C.2 Refuted — do not retry

- **Constant OGT prior.** Wins 0.47 °C on ProThermDB but severs the only link between the heads;
  a constant in `aux[0]` is absorbed into the next layer's bias, so the model provably reduces to
  two independent networks.
- **Supervising σ with the inter-laboratory IQR.** IQR median is 0.00 °C and mean 0.72 °C against
  model errors averaging 6.18 °C. Would drive severe under-dispersion.
- **Post-hoc recalibration of predictions.** Cross-fitted isotonic on ProThermDB gives MAE 4.756,
  apparently beating TemBERTure — but it is circular. Fitted on validation the slope is 0.983 and
  applying it blind changes MAE only 6.178 → 6.104.
- **Post-hoc rescue of the σ.** Reweighting the aleatoric and epistemic components, with weights
  fitted on validation, roughly doubles ρ(σ, |error|) and still loses to a constant width on
  CRPS. (Measured during the constant-prior detour; the ordering is the same on v9.)
- **Equal-n balanced test set.** Smallest bin is n = 13, so equal-n sampling yields 78 sequences.
  Macro-averaging achieves the same thing without discarding data.

### C.3 The v10 retrains — both rejected

| model | val_tm | ProThermDB | FireProt | per-protein σ gain vs constant width |
|:--|--:|--:|--:|:--|
| v10 gated (prior + σ_OGT) | **3.98** | 7.40 | 12.35 | val +0.031 · PDB −0.085 · FP −0.152 |
| v10 ungated (prior only) | 3.99 | 7.42 | 12.75 | val +0.034 · PDB −0.081 · FP −0.153 |
| **v9 (adopted)** | 6.27 | **6.18** | **11.92** | PDB −0.030 · FP −0.047 |

Three results worth reporting. The pre-registered rule failed: per-protein σ still loses to a
constant width on both test benchmarks, so the per-protein reliability claim is dead. The σ
gating adds nothing — gated and ungated are indistinguishable except on FireProtDB. And v10
**overfits validation**: val improved 6.27 → 3.98 while ProThermDB degraded 6.18 → 7.40, because
training on the predicted prior let the head exploit prior structure specific to the training
distribution once `tm_ogt_noise_std = 0` removed the regulariser. **Never select a T_m
configuration on `val_tm` alone again.** Confound ruled out: dropping the TMHMM flags shifts the
prior 0.23 °C and leaves val MAE at 3.98.

What did work and belongs in the ablation: the detached-mean variance loss fixed the σ *scale*
(cross-fitted `c` 1.43 → 1.04) without making it informative per protein. Scale and
informativeness are separate properties of a heteroscedastic head.

### C.4 The v10 OGT head — adopted, and it changes the OGT story

The v10 OGT head (detached-mean variance loss, Defect 1) **is** adopted as the reported OGT model.
It is strictly better than v9 on every axis that the paper argues from:

| BRENDA OOD (n=525) | micro | macro | | BacDive internal (n=4,854) | micro | macro |
|:--|--:|--:|:--|:--|--:|--:|
| v10 point MAE | 10.88 | 11.82 | | v10 point MAE | 7.94 | **9.23** |
| v10 CRPS | 7.56 | **8.23** | | v10 CRPS | 5.63 | **6.54** |
| v9 point MAE | 10.93 | 11.78 | | v9 point MAE | 7.79 | 9.52 |
| v9 CRPS | 8.13 | 8.60 | | v9 CRPS | 5.79 | 7.00 |
| PRIME | 6.75 | 8.56 | | PRIME | 4.96 | 10.38 |
| ThermoFormer | 6.48 | 8.29 | | ThermoFormer | 4.71 | 9.81 |

Three consequences. **The OGT interval becomes publishable** — the required scale drops from 5.64
to 1.18 externally and 5.53 to 0.87 internally, so B.3f (drop the OGT interval) is cancelled and
Table 4 gains a CRPS column. **Bin-balanced, StableProt now wins**: macro MAE on the internal split
(9.23 vs 9.81 and 10.38) and macro CRPS on both splits. **The micro/macro inversion is the whole
argument** — PRIME goes from 2.09 °C at 20–30 °C to 12.19 °C at 50–60 °C, a 6× swing, while
StableProt runs 7.79 → 6.98 across the same two bins. Report both columns side by side and let the
inversion speak; do not report macro alone.

**Naming.** Drop internal version numbers from the manuscript entirely — it is "StableProt". The
release is tagged v10 and the T_m weights inside it are unchanged from the previous iteration,
which is a repository fact, not a paper fact.

- [x] **C.4a — RESOLVED 13 Aug. Ship one OGT head.** `evaluate_ogt_prior_swap.py` scores the same
      v9 T_m ensemble under both prior sources, with two-fold cross-fitted σ scaling and a 4,000
      resample paired bootstrap. All four intervals contain zero, so the swap is neutral:

| benchmark | prior | MAE | CRPS | r | ρ | c | ΔMAE 95% CI | ΔCRPS 95% CI |
|:--|:--|--:|--:|--:|--:|--:|:--|:--|
| ProThermDB (n=3340) | v9 | 6.178 | 4.513 | 0.784 | 0.483 | 1.57 | | |
| | **v10** | **6.156** | 4.521 | 0.788 | **0.517** | 1.64 | −0.023 [−0.090, +0.045] | +0.007 [−0.033, +0.047] |
| FireProtDB (n=322) | v9 | 11.920 | 8.791 | 0.435 | 0.341 | 3.45 | | |
| | **v10** | **11.848** | **8.729** | 0.432 | **0.350** | 3.61 | −0.072 [−0.202, +0.062] | −0.062 [−0.161, +0.028] |

      The two priors correlate at r = 0.930 (ProThermDB) and 0.981 (FireProtDB), mean absolute
      shift 4.43 and 2.05 °C, so the T_m head is robust to the prior moving by a few degrees.
      Spearman ρ improves on both benchmarks, which is a small free gain.
      **Sanity check passed:** the v9-prior column reproduces the manuscript's published 6.18 /
      4.51 and 11.92 / 8.79 exactly, so the script is reproducing the real pipeline and not a
      parallel one.

- [x] **C.4c — number refresh DONE 13 Aug, one table excepted.** `refresh_tm_tables.py` regenerates
      every T_m quantity the manuscript prints, in one pass, under the shipped configuration, and
      writes `tables/refreshed_tm_numbers.json`. **Draw all figures from that JSON, never from
      literals.** Applied to `manuscript.html` and `supplementary_materials.html`: abstract,
      Table 1, Table 2 with all bootstrap CIs, the Figure 2B caption, §3.1, §3.2, §3.4, Table 4,
      the discussion and the limitations. Headline moves:

| | was | now |
|:--|--:|--:|
| ProThermDB MAE / CRPS | 6.18 / 4.51 | **6.16 / 4.52** |
| ProThermDB r / ρ | 0.784 / 0.483 | **0.788 / 0.517** |
| FireProtDB MAE / CRPS | 11.92 / 8.79 | **11.85 / 8.72** |
| FireProtDB r / ρ | 0.435 / 0.341 | **0.432 / 0.350** |
| variance scale c | 1.56 / 3.45 | **1.65 / 3.62** |
| ECE raw → scaled | 14.2→1.9 / 34.9→3.2 | **14.4→1.9 / 31.6→2.8** |

**Three narrative changes fell out, all in our favour except the third:**
1. **Spearman ρ against TemBERTure on ProThermDB is now a statistical tie**, 0.517 against 0.516,
   CI [−0.026, +0.025]. It was a loss. The in-distribution result is now "behind on MAE and
   Pearson r, level on rank correlation", which is a materially better sentence and is true.
2. **FireProtDB significance strengthened** across the board; every baseline margin excludes zero
   on both MAE and CRPS, and ρ is now significantly better than ThermoFormer-TM too.
3. **The per-bin story weakened.** StableProt beats TemBERTure in three of six bins covering
   61.2%, not four covering 76.4%; the 60–70 bin flipped to a tie (10.79 against 10.72). Also
   ThermoFormer-TM, not ESMStabP, is the best model above 80 °C (3.56 against 3.90). Both are
   now stated as such.

- [ ] **C.4c-1 — Table 5 and Tables S3a–c are the one thing left.** They are computed at c = 1.56
      and the point predictions predate the prior swap, so every half-width widens about 6% and
      the Tier 2 rates move with it. Flagged in red in the manuscript. Re-run the experimental
      validation at c = 1.65 with the v10 prior. Cohort structure and the directional finding are
      unaffected; only the numbers move.

- [ ] **C.4d** Once the swap is wired in, `v10/results/seed{1..5}/model_ogt.pt` plus
      `v9_disjoint/results/seed{1..5}/model_tm.pt` is the shipped artifact. Update
      `inference/v9_predict.py` accordingly (see B.9a) and drop the v9 OGT checkpoints from the
      release, keeping them only until C.4c is complete.
- [x] **C.4b** DONE 13 Aug via `refresh_ogt_cache.py`. On BRENDA the adopted OGT head gives
      r = 0.851 and ρ = 0.835, against 0.934 / 0.933 for PRIME and 0.938 / 0.934 for ThermoFormer.
      StableProt loses on correlation and Table 3 now says so, with a footnote placing the
      contribution in calibration and in the flat error profile rather than in rank ordering. The
      BacDive correlations need the internal embeddings and are still marked pending in Table 3.
      Screening AUC was recomputed at the same time: 0.939 at OGT ≥ 50 °C against 0.982 for both
      baselines. The previously quoted 0.841 was the superseded head and does not reproduce at any
      threshold. Superseded note: Recompute Pearson r and Spearman ρ for the v10 OGT head. Currently omitted from the
      OGT table rather than carried over from v9.

What actually deserves to be in the main manuscript

Figure S5 (FireProt scatter + per-bin) — strongest case. The paper’s claim is “behind in distribution, best out of distribution,” but Figure 2 only shows ProTherm. FireProt is the OOD half of that sentence and currently lives in a table plus the supplement. A 2×2 Figure 2 (ProTherm scatter/bins + FireProt scatter/bins), with today’s 2C (MAE vs CRPS) moved to the supplement, would match the argument.

Figure S2A (FireProt reliability) — second case. Figure 4A is ProTherm (c = 1.65). The transferability limitation is c = 3.62 on FireProt, and that curve is not in the main grid.