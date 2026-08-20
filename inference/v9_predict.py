import os
import sys
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import EsmTokenizer, EsmModel

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
v9_dir = os.path.join(root_dir, "experiments/src/training/v9_disjoint")
if v9_dir not in sys.path:
    sys.path.insert(0, v9_dir)

from train import MultiHeadSaProtV8, enrich_inputs

try:
    from inference.v7_predict import load_saprot_model, get_saprot_embedding
except ModuleNotFoundError:
    from v7_predict import load_saprot_model, get_saprot_embedding

STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")

# Global scale applied to the raw predictive sigma so that intervals attain nominal coverage.
# Refitted by minimising ECE after the variance-aggregation fix; the old 3.8 was fitted against a
# sigma that omitted the aleatoric term and now over-inflates intervals by ~2x.
#
# The scale does not transfer across distributions: 1.56 on in-distribution ProThermDB, 3.29 on
# the FireProtDB out-of-distribution holdout. Submitted sequences are novel by definition, so the
# server uses the out-of-distribution value; under-covering a user's protein is the worse error.
TM_SIGMA_SCALE = 3.29
OGT_SIGMA_SCALE = 3.29  # not independently fitted — no held-out OGT set with stored sigmas yet

# Chou-Fasman & GOR secondary structure propensities
CHOU_FASMAN_HELIX = {'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.70, 'E': 1.51, 'Q': 1.11, 'G': 0.57, 'H': 1.00, 'I': 1.08, 'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13, 'P': 0.57, 'S': 0.77, 'T': 0.83, 'W': 1.08, 'Y': 0.69, 'V': 1.06}
CHOU_FASMAN_SHEET = {'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19, 'E': 0.37, 'Q': 1.10, 'G': 0.75, 'H': 0.87, 'I': 1.60, 'L': 1.30, 'K': 0.74, 'M': 1.05, 'F': 1.38, 'P': 0.55, 'S': 0.75, 'T': 1.19, 'W': 1.37, 'Y': 1.47, 'V': 1.70}
CHOU_FASMAN_LOOP  = {'A': 0.66, 'R': 0.99, 'N': 1.56, 'D': 1.46, 'C': 1.19, 'E': 0.74, 'Q': 0.98, 'G': 1.56, 'H': 0.95, 'I': 0.47, 'L': 0.59, 'K': 1.01, 'M': 0.60, 'F': 0.60, 'P': 1.52, 'S': 1.43, 'T': 0.96, 'W': 0.76, 'Y': 1.14, 'V': 0.50}

def sanitize_sequence(raw_input: str) -> str:
    """Strip FASTA headers, whitespace, digits, punctuation. Return uppercase AA string."""
    lines = raw_input.strip().splitlines()
    seq_lines = [l for l in lines if not l.strip().startswith('>') and not l.strip().startswith('##')]
    raw = ''.join(seq_lines)
    cleaned = re.sub(r'[^A-Za-z]', '', raw)
    return cleaned.upper()

def classify_tm_tier(tm_pred: float) -> str:
    """Classify predicted melting temperature into stability tiers."""
    if tm_pred < 45.0:
        return "LOW"
    elif tm_pred < 65.0:
        return "MEDIUM"
    else:
        return "HIGH"

def predict_secondary_structure(raw_input: str, window_size: int = 5) -> dict:
    """Instant sequence-based secondary structure classifier (<50ms).
    Assigns per-residue states H (Helix), E (Sheet), L (Loop/Coil) and groups contiguous loop regions.
    """
    seq = sanitize_sequence(raw_input)
    L_len = len(seq)
    if L_len == 0:
        return {"sequence": "", "length": 0, "ss_str": "", "loops": []}

    half_w = window_size // 2
    h_scores, e_scores, l_scores = [], [], []

    for i in range(L_len):
        win_seq = seq[max(0, i - half_w): min(L_len, i + half_w + 1)]
        n = len(win_seq)
        h_val = sum(CHOU_FASMAN_HELIX.get(aa, 1.0) for aa in win_seq) / n
        e_val = sum(CHOU_FASMAN_SHEET.get(aa, 1.0) for aa in win_seq) / n
        l_val = sum(CHOU_FASMAN_LOOP.get(aa, 1.0) for aa in win_seq) / n
        h_scores.append(h_val)
        e_scores.append(e_val)
        l_scores.append(l_val)

    ss_chars = []
    for i in range(L_len):
        h, e, l = h_scores[i], e_scores[i], l_scores[i]
        if l >= h and l >= e:
            ss_chars.append('L')
        elif h >= e:
            ss_chars.append('H')
        else:
            ss_chars.append('E')

    ss_str = "".join(ss_chars)

    # Group contiguous 'L' regions of length >= 3
    loops = []
    in_loop = False
    start_idx = 0
    loop_count = 0

    for i in range(L_len):
        if ss_chars[i] == 'L':
            if not in_loop:
                in_loop = True
                start_idx = i
        else:
            if in_loop:
                in_loop = False
                end_idx = i - 1
                loop_len = end_idx - start_idx + 1
                if loop_len >= 3:
                    loop_count += 1
                    subseq = seq[start_idx:end_idx + 1]
                    flex = sum(l_scores[start_idx:end_idx + 1]) / loop_len
                    loops.append({
                        "id": loop_count,
                        "label": f"Loop {loop_count}",
                        "start": start_idx + 1,  # 1-indexed for display
                        "end": end_idx + 1,      # 1-indexed
                        "sequence": subseq,
                        "length": loop_len,
                        "avg_flexibility": round(flex, 2),
                        "positions": list(range(start_idx, end_idx + 1))  # 0-indexed internal
                    })

    # Catch loop extending to end of sequence
    if in_loop:
        end_idx = L_len - 1
        loop_len = end_idx - start_idx + 1
        if loop_len >= 3:
            loop_count += 1
            subseq = seq[start_idx:end_idx + 1]
            flex = sum(l_scores[start_idx:end_idx + 1]) / loop_len
            loops.append({
                "id": loop_count,
                "label": f"Loop {loop_count}",
                "start": start_idx + 1,
                "end": end_idx + 1,
                "sequence": subseq,
                "length": loop_len,
                "avg_flexibility": round(flex, 2),
                "positions": list(range(start_idx, end_idx + 1))
            })

    return {
        "sequence": seq,
        "length": L_len,
        "ss_str": ss_str,
        "loops": loops
    }

class V9Predictor:
    def __init__(self, models_dir: str, device="cuda"):
        self.device = device
        self.embed_model, self.tokenizer = load_saprot_model(device=device)
        self.models_tm = []
        self.models_ogt = []
        
        # Load normalization stats
        norm_path = os.path.join(models_dir, "normalization_stats.pt")
        if os.path.exists(norm_path):
            norms = torch.load(norm_path, map_location='cpu', weights_only=False)
            self.tm_mean = norms['tm_mean']
            self.tm_std = norms['tm_std']
            self.ogt_mean = norms['ogt_mean']
            self.ogt_std = norms['ogt_std']
            print(f"Loaded normalization stats: Tm=N({self.tm_mean:.2f}, {self.tm_std:.2f}), OGT=N({self.ogt_mean:.2f}, {self.ogt_std:.2f})")
        else:
            # Silent fallbacks here previously used v8 statistics (52.88/16.50 against v9's
            # 51.74/11.49), which mis-scales every prediction by ~40% in the standard deviation.
            raise FileNotFoundError(
                f"normalization_stats.pt not found at {norm_path}. Refusing to fall back to "
                "hardcoded statistics — they differ per model version and silently corrupt "
                "every prediction."
            )
            
        for s in range(1, 6):
            pt_tm = os.path.join(models_dir, f"seed{s}/model_tm.pt")
            pt_ogt = os.path.join(models_dir, f"seed{s}/model_ogt.pt")
            if os.path.exists(pt_tm) and os.path.exists(pt_ogt):
                m_t = MultiHeadSaProtV8().to(device)
                m_t.load_state_dict(torch.load(pt_tm, map_location=device, weights_only=False))
                m_t.eval()
                self.models_tm.append(m_t)
                
                m_o = MultiHeadSaProtV8().to(device)
                m_o.load_state_dict(torch.load(pt_ogt, map_location=device, weights_only=False))
                m_o.eval()
                self.models_ogt.append(m_o)
                
        if not self.models_tm:
            raise FileNotFoundError(f"No V9 seed models found in {models_dir}")
        print(f"Loaded {len(self.models_tm)} seed models for V9 ensemble.")

    def predict(self, sequence: str):
        emb = get_saprot_embedding(self.embed_model, self.tokenizer, sequence, device=self.device)
        emb = emb.float()
        
        with torch.no_grad():
            # 1. Stage 1: Predict OGT across 5 seeds
            emb_o, aux_o = enrich_inputs(emb.cpu(), [sequence], tmhmm_flags=None, ogt_priors=None)
            ogt_preds = []
            for m_o in self.models_ogt:
                pred_z = m_o(emb_o.to(self.device), aux_o.to(self.device), head='ogt')
                ogt_preds.append(pred_z.item())
                
            ogt_preds = np.array(ogt_preds)
            ogt_mu_z = np.mean(ogt_preds)
            ogt_sigma_z = np.std(ogt_preds) if len(ogt_preds) > 1 else 0.5  # Epistemic uncertainty across seeds
            
            # Denormalize OGT
            ogt_val = ogt_mu_z * self.ogt_std + self.ogt_mean
            ogt_conf = ogt_sigma_z * self.ogt_std
            
            # 2. Stage 2: Predict Tm using predicted OGT prior
            emb_t, aux_t = enrich_inputs(emb.cpu(), [sequence], tmhmm_flags=None, ogt_priors=np.array([ogt_val]))
            mus = []
            vars_list = []
            for m_t in self.models_tm:
                z_mu, z_lv = m_t(emb_t.to(self.device), aux_t.to(self.device), head='tm')
                pred_mu = z_mu.cpu() * self.tm_std + self.tm_mean
                pred_var = z_lv.cpu() * (self.tm_std ** 2)
                mus.append(pred_mu)
                vars_list.append(pred_var)
                
            mus_stack = torch.stack(mus, dim=0)
            vars_stack = torch.stack(vars_list, dim=0)
            
            # Confidence-weighted ensemble: weight by inverse variance
            weights = 1.0 / (vars_stack + 1e-6)
            tm_val = (mus_stack * weights).sum(dim=0) / weights.sum(dim=0)
            # Predictive variance for a new measurement = mean aleatoric variance + spread of
            # the seed means. 1/weights.sum() alone is the standard error of the ensemble mean,
            # which shrinks with seed count and is not a predictive interval.
            total_var = vars_stack.mean(dim=0) + ((mus_stack - tm_val) ** 2).mean(dim=0)
            tm_conf = torch.sqrt(total_var) * TM_SIGMA_SCALE
            
        return tm_val.item(), tm_conf.item(), ogt_val, ogt_conf * OGT_SIGMA_SCALE

    def predict_single(self, raw_input: str) -> dict:
        """Sanitize raw input, apply length guardrails, and return structured prediction dict."""
        seq = sanitize_sequence(raw_input)
        if len(seq) < 50:
            return {"status": "ERROR_TOO_SHORT", "sequence": seq, "length": len(seq)}
        
        truncated = False
        if len(seq) > 2048:
            seq = seq[:2048]
            truncated = True
        
        tm_val, tm_conf, ogt_val, ogt_conf = self.predict(seq)
        
        return {
            "status": "WARNING_TRUNCATED" if truncated else "OK",
            "sequence": seq,
            "length": len(seq),
            "tm_pred": round(tm_val, 2),
            "tm_conf": round(tm_conf, 2),
            "tm_tier": classify_tm_tier(tm_val),
            "ogt_pred": round(ogt_val, 2),
            "ogt_conf": round(ogt_conf, 2),
        }

    def predict_batch(self, fasta_input: str, is_filepath=False) -> pd.DataFrame:
        """Parse multi-record FASTA string or file and return structured pandas DataFrame."""
        if is_filepath:
            with open(fasta_input, 'r', encoding='utf-8') as f:
                fasta_input = f.read()
        
        entries = []
        current_header = None
        current_seq_lines = []
        for line in fasta_input.strip().splitlines():
            if line.startswith('>'):
                if current_header is not None:
                    entries.append((current_header, '\n'.join(current_seq_lines)))
                current_header = line[1:].strip()
                current_seq_lines = []
            else:
                current_seq_lines.append(line)
        if current_header is not None:
            entries.append((current_header, '\n'.join(current_seq_lines)))
        
        if not entries:
            entries = [("unnamed", fasta_input)]
        
        results = []
        for header, raw_seq in entries:
            res = self.predict_single(raw_seq)
            res["header"] = header
            results.append(res)
        
        return pd.DataFrame(results)

    def predict_mutants(self, sequence: str, positions=None, min_delta_tm=0.5) -> pd.DataFrame:
        """Single-point saturation mutagenesis scan across specified or all sequence positions."""
        seq = sanitize_sequence(sequence)
        wt_result = self.predict_single(seq)
        if wt_result.get("status") == "ERROR_TOO_SHORT":
            return pd.DataFrame()
        
        tm_wt = wt_result["tm_pred"]
        conf_wt = wt_result["tm_conf"]
        
        if positions is None:
            positions = list(range(len(seq)))
        
        rows = []
        for pos in positions:
            if pos < 0 or pos >= len(seq):
                continue
            wt_aa = seq[pos]
            for mut_aa in STANDARD_AAS:
                if mut_aa == wt_aa:
                    continue
                mutant_seq = seq[:pos] + mut_aa + seq[pos+1:]
                tm_mut, tm_conf_mut, _, _ = self.predict(mutant_seq)
                delta_tm = tm_mut - tm_wt
                rows.append({
                    "position": pos + 1,  # 1-indexed for user readability
                    "wt_aa": wt_aa,
                    "mut_aa": mut_aa,
                    "tm_wt": round(tm_wt, 2),
                    "tm_mut": round(tm_mut, 2),
                    "delta_tm": round(delta_tm, 2),
                    "tm_conf_mut": round(tm_conf_mut, 2),
                    "confident": abs(delta_tm) > tm_conf_mut,
                    "tier_mut": classify_tm_tier(tm_mut),
                })
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("delta_tm", ascending=False).reset_index(drop=True)
        return df

    def evaluate_design_edit(self, wt_sequence: str, mutant_sequence: str) -> dict:
        """Evaluate a specific manual or design edit against the wild-type sequence."""
        wt = sanitize_sequence(wt_sequence)
        mut = sanitize_sequence(mutant_sequence)

        if len(wt) != len(mut):
            return {
                "status": "ERROR_LENGTH_MISMATCH",
                "error": f"Sequence length mismatch: WT is {len(wt)} aa, Mutant is {len(mut)} aa."
            }

        res_wt = self.predict_single(wt)
        res_mut = self.predict_single(mut)

        if res_wt.get("status") == "ERROR_TOO_SHORT" or res_mut.get("status") == "ERROR_TOO_SHORT":
            return {"status": "ERROR_TOO_SHORT", "error": "Sequence too short (<50 aa)"}

        # Find mutated positions
        mutations = []
        for i in range(len(wt)):
            if wt[i] != mut[i]:
                mutations.append({
                    "position": i + 1,  # 1-indexed
                    "wt_aa": wt[i],
                    "mut_aa": mut[i],
                    "label": f"{wt[i]}{i+1}{mut[i]}"
                })

        delta_tm = round(res_mut["tm_pred"] - res_wt["tm_pred"], 2)
        is_stab = delta_tm > 0
        effect = "Stabilizing" if delta_tm > 0 else ("Destabilizing" if delta_tm < 0 else "Neutral")
        
        # Secondary structure check
        ss_res = predict_secondary_structure(wt)
        loop_positions = set()
        for l in ss_res["loops"]:
            loop_positions.update(l["positions"])

        in_loop = any((m["position"] - 1) in loop_positions for m in mutations)

        return {
            "status": "OK",
            "wt_sequence": wt,
            "mutant_sequence": mut,
            "wt_tm": res_wt["tm_pred"],
            "wt_conf": res_wt["tm_conf"],
            "wt_tier": res_wt["tm_tier"],
            "mut_tm": res_mut["tm_pred"],
            "mut_conf": res_mut["tm_conf"],
            "mut_tier": res_mut["tm_tier"],
            "delta_tm": delta_tm,
            "is_stabilizing": is_stab,
            "thermal_effect": effect,
            "mutations": mutations,
            "n_mutations": len(mutations),
            "in_loop_region": in_loop
        }


    def generate_random_loop_mutation(self, wt_sequence: str, loop_positions=None, loop_length=None) -> dict:
        """Pick random positions in identified loop regions and generate random non-WT mutations scaled to loop length."""
        seq = sanitize_sequence(wt_sequence)
        
        if not loop_positions:
            ss = predict_secondary_structure(seq)
            loop_pos_set = set()
            for l in ss["loops"]:
                loop_pos_set.update(l["positions"])
            loop_positions = sorted(list(loop_pos_set))

        if not loop_positions:
            # Fallback to any random position if no loop found
            loop_positions = list(range(len(seq)))

        # Determine mutation count k based on actual loop length (e.g. 1 to 3 mutations)
        L = loop_length if loop_length and loop_length > 0 else len(loop_positions)
        k = max(1, min(3, L // 3))
        
        chosen_positions = random.sample(loop_positions, min(k, len(loop_positions)))
        chosen_positions.sort()

        mutant_list = list(seq)
        mutation_labels = []

        for pos_idx in chosen_positions:
            wt_aa = seq[pos_idx]
            candidates = [aa for aa in STANDARD_AAS if aa != wt_aa]
            if random.random() < 0.4 and 'P' in candidates:
                mut_aa = 'P'
            else:
                mut_aa = random.choice(candidates)

            mutant_list[pos_idx] = mut_aa
            mutation_labels.append(f"{wt_aa}{pos_idx+1}{mut_aa}")

        mutant_seq = "".join(mutant_list)

        return {
            "positions": [p + 1 for p in chosen_positions],
            "mutation_label": ", ".join(mutation_labels),
            "mutant_sequence": mutant_seq
        }

    def iterative_evolution(self, sequence: str, n_rounds=5, top_k=5,
                            positions=None, min_delta_tm=0.5,
                            convergence_threshold=0.3,
                            mode="guided",
                            force_all_rounds=False) -> dict:
        """Multi-round directed evolution via greedy single-point mutation stacking."""
        seq = sanitize_sequence(sequence)
        wt_result = self.predict_single(seq)
        if wt_result.get("status") == "ERROR_TOO_SHORT":
            return {"status": "ERROR_TOO_SHORT", "sequence": seq}

        current_seq = seq
        trajectory = []
        accumulated_mutations = []

        if mode == "blind" and positions is None:
            print(f"BLIND MODE: scanning all {len(seq)} positions × 19 AAs = {len(seq)*19} mutants/round")
            print(f"Estimated time: ~{len(seq)*19*n_rounds*0.1/60:.0f} minutes")

        scan_positions = positions  # None = all positions (blind mode)

        for r in range(1, n_rounds + 1):
            print(f"  Round {r}/{n_rounds}...")
            scan_df = self.predict_mutants(current_seq, positions=scan_positions, min_delta_tm=min_delta_tm)

            if scan_df.empty:
                trajectory.append({
                    "round": r, "mutation": None,
                    "tm": self.predict_single(current_seq)["tm_pred"],
                    "delta_tm": 0.0, "tier": self.predict_single(current_seq)["tm_tier"],
                    "note": "NO_MUTATIVE_SCAN_RESULTS"
                })
                break

            candidates = scan_df[(scan_df["delta_tm"] > 0) & (scan_df["confident"])].head(top_k)

            if not force_all_rounds:
                if candidates.empty or candidates.iloc[0]["delta_tm"] < convergence_threshold:
                    trajectory.append({
                        "round": r, "mutation": None,
                        "tm": self.predict_single(current_seq)["tm_pred"],
                        "delta_tm": 0.0,
                        "tier": self.predict_single(current_seq)["tm_tier"],
                        "note": "CONVERGED"
                    })
                    break

            if candidates.empty:
                candidates = scan_df[scan_df["delta_tm"] > 0].head(top_k)
                if candidates.empty:
                    trajectory.append({
                        "round": r, "mutation": None,
                        "tm": self.predict_single(current_seq)["tm_pred"],
                        "delta_tm": 0.0, "tier": self.predict_single(current_seq)["tm_tier"],
                        "note": "NO_STABILIZING_CANDIDATES"
                    })
                    break

            best = candidates.iloc[0]
            pos_0 = int(best["position"]) - 1
            mutation_label = f"{best['wt_aa']}{best['position']}{best['mut_aa']}"
            current_seq = current_seq[:pos_0] + best["mut_aa"] + current_seq[pos_0+1:]
            accumulated_mutations.append(mutation_label)

            new_result = self.predict_single(current_seq)
            seq_identity = sum(a == b for a, b in zip(seq, current_seq)) / len(seq)
            trajectory.append({
                "round": r, "mutation": mutation_label,
                "tm": new_result["tm_pred"], "delta_tm": round(best["delta_tm"], 2),
                "tier": new_result["tm_tier"],
                "seq_identity_to_wt": round(seq_identity, 4),
            })

        final_result = self.predict_single(current_seq)
        return {
            "wt_sequence": seq, "wt_tm": wt_result["tm_pred"], "wt_tier": wt_result["tm_tier"],
            "final_sequence": current_seq, "final_tm": final_result["tm_pred"], "final_tier": final_result["tm_tier"],
            "total_delta_tm": round(final_result["tm_pred"] - wt_result["tm_pred"], 2),
            "n_rounds_completed": len(trajectory),
            "converged": trajectory[-1].get("note") == "CONVERGED" if trajectory else False,
            "mode": mode,
            "positions_scanned": positions if positions else "all",
            "evolution_trajectory": trajectory,
            "accumulated_mutations": accumulated_mutations,
        }
