# V7 Implementation Plan (Revised)

## Accepted Criticisms & Corrections

| DeepSeek's Critique | Response |
|--------------------|---------| 
| ThermoFormer comparison misleading | ✅ **Accepted.** Our Stage 1 = lightweight ThermoFormer. Will cite as direct prior work. |
| Missing SPURS/PRIME/HotProtein | ⚠️ **Partially valid.** SPURS predicts ΔΔG (mutations), NOT absolute Tm — different task. PRIME = zero-shot mutation scoring, not regression. HotProtein = ICLR 2023, not NeurIPS 2025 (DeepSeek hallucinated the venue). Will cite all as related work, benchmark where applicable. |
| Foldseek cost undocumented | ✅ **Accepted.** Budget added below. |
| MC-Dropout suboptimal | ⚠️ **Already using 5-seed ensembles.** MC-Dropout is additional, not replacement. |
| ΔT infeasible on OOD | ✅ **Accepted.** Already flagged this. Dropped from main paper. |
| "No shortcuts" overclaimed | ⚠️ **Disagree.** 10.5% IS low. Adversarial training targets exactly this fraction. Cost/benefit doesn't justify it. |
| FLIP external validation | ✅ **Accepted.** Will add. |

## Key Insight: Split Into Two Phases

**Experiments A-C need ZERO structure tokens.** We can get ESM-2 transfer results in 3-4 days while Foldseek processes in parallel.

---

## Phase I: ESM-2 Transfer Experiments (Days 1-4)

No Foldseek. No SaProt. Just ESM-2 Layer 30 embeddings (already have them).

### Experiment A: Baseline (V6 Current)
- Already measured: **OOD MAE=12.91, PCC=0.44, R²=-0.22**

### Experiment B: OGT Transfer → Tm Fine-tune
```
Stage 1: Train MLP (2560→512→512→256→1) on 943K OGT
  - 5 epochs, LR=1e-4, Huber loss
  - Save backbone weights

Stage 2: Load backbone. NEW head: Linear(256→1) 
  - Train on 43K Tm, 20 epochs
  - Backbone LR=1e-5, Tm head LR=1e-4
  - Each Tm sample seen 1× per epoch
  - Early stopping patience=10
```

### Experiment C: B + OGT-as-Feature
```
Same as B, but Stage 2 input = [backbone_features(256) || OGT_pred(1)]
OGT_pred from Stage 1's OGT head (frozen)
```

### Experiment C2: OGT-as-Feature Only (No Transfer)
```
Train MLP directly on Tm with OGT_pred as extra feature
No OGT pre-training. Tests if transfer adds value beyond the scalar.
```

### Architecture (Shared)
```python
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return x + self.net(x)

class StableProtV7(nn.Module):
    def __init__(self, emb_dim=2560, use_ogt_feature=False):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            ResidualBlock(512, dropout=0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.ogt_head = nn.Linear(256, 1)
        
        tm_input = 257 if use_ogt_feature else 256
        self.tm_head = nn.Sequential(
            nn.Linear(tm_input, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        self.use_ogt_feature = use_ogt_feature
```

### Day-by-Day (Phase I)
| Day | AM | PM |
|-----|----|----|
| 1 | Implement model + Stage 1 training | Run Stage 1 OGT pre-training (3 seeds, ~2h) |
| 2 | Implement Stage 2 + evaluation | Run Exp B (3 seeds) + Exp C (3 seeds) |
| 3 | Run Exp C2 ablation. Re-run baselines on 324-target holdout | Evaluate all. Generate comparison table |
| 4 | **Decision point.** Did transfer help OOD? | Start Phase II setup if proceeding |

### Decision Gate After Phase I

| If... | Then... |
|-------|---------|
| Exp B OOD MAE < 11°C, PCC > 0.50 | Transfer works. Phase II will improve further. Proceed. |
| Exp B ≈ Exp A (no improvement) | Transfer doesn't help. ESM-2 features lack temperature signal. Skip to SaProt directly. |
| Exp C > Exp B | OGT-as-feature adds value. Keep in V7. |
| Exp C2 ≈ Exp C | Transfer doesn't matter, only OGT scalar matters. Simplify. |

---

## Phase II: SaProt Structure Experiments (Days 5-12)

### Foldseek Computational Budget

| Dataset | Proteins | AlphaFold DB Coverage | ESMFold Needed | Time |
|---------|----------|-----------------------|----------------|------|
| OGT (A0A* accessions) | 943K | ~98% (929K are A0A = TrEMBL, covered by AFDB) | ~19K | ~4h for ESMFold |
| Tm (P/Q/O accessions) | 29K | ~95% (Swiss-Prot, well covered) | ~1.5K | ~30min |
| FireProt OOD | 324 | ~80% | ~65 | ~2min |
| **Total** | **973K** | **~97%** | **~21K** | **~5h ESMFold** |

**Bottleneck: Downloading AFDB structures (~943K PDB files) + running Foldseek.**
- AFDB download: ~2-3h (bulk API)
- Foldseek tokenization: ~0.5s/protein × 973K = **~135 hours** on 1 CPU core
- **Parallelized (8 cores): ~17 hours**
- SaProt embedding extraction: ~2s/protein × 973K = **~540 GPU-hours** on RTX 6000

> [!WARNING]
> SaProt embedding extraction for 943K OGT is the real bottleneck: **~540 GPU-hours = ~22 days on one GPU.** This is why we MUST test ESM-2 transfer first (Phase I). If transfer doesn't help, SaProt won't either (the architecture is the problem, not the features).

### SaProt Fallback: Tm-Only + Small OGT Subset
If 22 days is too long, alternative:
- Extract SaProt embeddings for **Tm (29K) + OOD test (324) only** = ~16 GPU-hours
- Use a **random 50K subset** of OGT for Stage 1 = ~28 GPU-hours
- Total: **~44 GPU-hours = ~2 days**. Feasible.

### Experiments D-F (SaProt)
Same architecture as A-C but with SaProt 650M embeddings (dim=1280).

| Exp | Backbone | OGT Transfer | OGT Feature | Structure |
|-----|----------|-------------|-------------|-----------|
| D | SaProt 650M | ❌ | ❌ | ✅ |
| E | SaProt 650M | ✅ | ❌ | ✅ |
| F | SaProt 650M | ✅ | ✅ | ✅ |

---

## Related Work (Corrected)

### Direct Prior Work (Must Cite)
| Model | Relevance | Available? |
|-------|-----------|-----------|
| **ThermoFormer** | Same approach (OGT pre-train → Tm), larger scale | Paper yes, code unclear |
| **EsmTemp** | ESM-2 transfer for Tm (R²=0.70, MAE=4.3°C in-dist) | ✅ [GitHub](https://github.com/SanoScience/esm_temp) |
| **ESMStabP** | OGT as feature + RF | ✅ Available |
| **TemBERTure** | protBERT-based Tm regression | ✅ Available |

### Tangential Work (Cite, Don't Compete Directly)
| Model | Why Tangential |
|-------|---------------|
| **SPURS** | Predicts **ΔΔG from mutations**, not absolute Tm. Different task. |
| **PRIME** | Zero-shot **mutation effect** scoring via MLM. Not regression. |
| **HotProtein** | ICLR 2023 (not 2025). OGT prediction + editing. Relevant but older. |

### Our Contribution vs Prior Work
```
ThermoFormer: OGT pre-train (96M seq) → Tm fine-tune
EsmTemp:      ESM-2 transfer → Tm
ESMStabP:     OGT scalar + ESM-2 embeddings → RF → Tm

Ours (V7):    OGT pre-train (943K) → Structure-aware backbone (SaProt)
              → OGT-as-feature → Tm fine-tune → OOD benchmark
              
              Novel: First to combine all three (transfer + structure + OGT feature)
              in a single framework, benchmarked on strict OOD (<40% identity)
```

---

## External Validation

### FLIP Meltome Benchmark
- Download FLIP thermostability split
- Run V7 on it, report Spearman ρ
- Compare against published SaProt (0.697) and ESM-2 (0.610-0.670)

### FireProtDB OOD (Primary)
- 324 targets, <40% identity to training (CD-HIT-2D)
- Metrics: MAE, PCC, R², AUC at 40/50/60/70/80°C

---

## Dropped From Main Paper

| Dropped | Reason |
|---------|--------|
| ΔT prediction | No OGT in OOD test set. Error compounds. |
| Contrastive OGT alignment | More complex than regression pre-training, unproven benefit |
| Adversarial domain-invariance | Clusters explain only 10.5% — not enough signal to remove |
| Gradient surgery (PiKE/POMSI) | Band-aid on wrong architecture |
| Multi-head simultaneous training | Gradient conflict at 22:1 ratio |

---

## Timeline (12 Days Total)

| Day | Task | Blocking? |
|-----|------|-----------|
| **1** | Implement V7 model + Stage 1 OGT training | No |
| **2** | Run Exp B+C+C2 (ESM-2 transfer). Re-run baselines | No |
| **3** | Evaluate Phase I. Decision gate. | **DECISION** |
| **4** | Start AFDB download + Foldseek setup | Parallel |
| **5-6** | SaProt embedding extraction (Tm + 50K OGT subset) | GPU-bound |
| **7-8** | Run Exp D+E+F (SaProt transfer) | GPU-bound |
| **9** | FLIP Meltome external validation | No |
| **10** | Statistical tests. Ablation analysis | No |
| **11** | Publication figures. Results table | No |
| **12** | Write results section | No |
