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

try:
    from .models import MultiHeadSaProtV8, enrich_inputs
except ImportError:
    from models import MultiHeadSaProtV8, enrich_inputs

STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")

# Global scale applied to predictive sigma for calibrated prediction intervals
# Tm: 3.29 (out-of-distribution FireProtDB holdout)
# OGT: 1.18 (out-of-distribution BRENDA holdout with heteroscedastic loss)
TM_SIGMA_SCALE = 3.29
OGT_SIGMA_SCALE = 1.18

# Chou-Fasman secondary structure propensities
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
    """Classify predicted melting temperature into stability regimes."""
    if tm_pred < 40.0:
        return "Psychrophilic (<40°C)"
    elif tm_pred < 60.0:
        return "Mesophilic (40–60°C)"
    elif tm_pred < 80.0:
        return "Thermophilic (60–80°C)"
    else:
        return "Hyperthermophilic (>80°C)"

def load_saprot_model(model_name="westlake-repl/SaProt_650M_AF2", device="cuda"):
    """Load pretrained SaProt structure-aware protein language model and tokenizer."""
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer

def mask_sequence_for_saprot(seq: str) -> str:
    """Format amino acid sequence with mask tokens for SaProt tokenization."""
    return "".join(f"{aa}#" for aa in seq)

def get_saprot_embedding(model, tokenizer, sequence: str, device="cuda") -> torch.Tensor:
    """Generate mean-pooled 1280-dimensional SaProt embedding for an input sequence."""
    saprot_seq = mask_sequence_for_saprot(sequence)
    inputs = tokenizer(
        [saprot_seq],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(device)

    is_cuda = "cuda" in str(device)
    with torch.no_grad(), torch.amp.autocast('cuda' if is_cuda else 'cpu', enabled=is_cuda):
        outputs = model(**inputs)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        hidden = outputs.last_hidden_state
        masked_hidden = hidden * attention_mask
        embeddings = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)

    return embeddings

def predict_secondary_structure(raw_input: str, window_size: int = 5) -> dict:
    """Sequence-based secondary structure classifier.
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
                        "start": start_idx + 1,
                        "end": end_idx + 1,
                        "sequence": subseq,
                        "length": loop_len,
                        "avg_flexibility": round(flex, 2),
                        "positions": list(range(start_idx, end_idx + 1))
                    })

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

class StableProtPredictor:
    """Production predictor for StableProt dual-head ensemble (Tm & OGT)."""
    def __init__(self, models_dir: str, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[StableProt] Initializing on device: {self.device}")
        print("[StableProt] Loading base SaProt_650M_AF2 model & tokenizer...")
        self.embed_model, self.tokenizer = load_saprot_model(device=self.device)
        self.models_tm = []
        self.models_ogt = []
        
        norm_path = os.path.join(models_dir, "normalization_stats.pt")
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"Normalization statistics missing at {norm_path}")
            
        norms = torch.load(norm_path, map_location='cpu', weights_only=False)
        self.tm_mean = float(norms['tm_mean'])
        self.tm_std = float(norms['tm_std'])
        self.ogt_mean = float(norms['ogt_mean'])
        self.ogt_std = float(norms['ogt_std'])
        print(f"[StableProt] Normalization: Tm=N({self.tm_mean:.2f}, {self.tm_std:.2f}), OGT=N({self.ogt_mean:.2f}, {self.ogt_std:.2f})")

        for s in range(1, 6):
            pt_tm = os.path.join(models_dir, f"seed{s}/model_tm.pt")
            pt_ogt = os.path.join(models_dir, f"seed{s}/model_ogt.pt")
            if os.path.exists(pt_tm) and os.path.exists(pt_ogt):
                m_t = MultiHeadSaProtV8().to(self.device)
                m_t.load_state_dict(torch.load(pt_tm, map_location=self.device, weights_only=False))
                m_t.eval()
                self.models_tm.append(m_t)
                
                sd_ogt = torch.load(pt_ogt, map_location=self.device, weights_only=False)
                is_hetero = sd_ogt.get("head_ogt.weight", torch.empty(1)).shape[0] == 2
                m_o = MultiHeadSaProtV8(ogt_heteroscedastic=is_hetero).to(self.device)
                m_o.load_state_dict(sd_ogt)
                m_o.eval()
                self.models_ogt.append(m_o)
                
        if not self.models_tm:
            raise FileNotFoundError(f"No seed models found in {models_dir}")
        print(f"[StableProt] Successfully loaded {len(self.models_tm)}-seed ensemble.")

    def predict(self, sequence: str):
        """Execute 2-stage decoupled inference:
        Stage 1: Predict OGT across 5 seeds.
        Stage 2: Condition Tm prediction on predicted OGT prior across 5 seeds.
        """
        emb = get_saprot_embedding(self.embed_model, self.tokenizer, sequence, device=self.device)
        emb = emb.float()
        
        with torch.no_grad():
            # 1. Stage 1: Predict OGT across ensemble
            emb_o, aux_o = enrich_inputs(emb.cpu(), [sequence], tmhmm_flags=None, ogt_priors=None)
            emb_o, aux_o = emb_o.to(self.device), aux_o.to(self.device)
            ogt_mus, ogt_vars = [], []
            for m_o in self.models_ogt:
                res_o = m_o(emb_o, aux_o, head='ogt')
                if isinstance(res_o, tuple):
                    z_mu, z_var = res_o
                    ogt_mus.append(z_mu.item() * self.ogt_std + self.ogt_mean)
                    ogt_vars.append(z_var.item() * (self.ogt_std ** 2))
                else:
                    ogt_mus.append(res_o.item() * self.ogt_std + self.ogt_mean)
                    ogt_vars.append(0.0)
                
            ogt_mus = np.array(ogt_mus)
            ogt_vars = np.array(ogt_vars)
            if np.all(ogt_vars == 0.0):
                ogt_val = float(np.mean(ogt_mus))
                ogt_sigma = float(np.std(ogt_mus)) if len(ogt_mus) > 1 else 0.5
                ogt_conf = ogt_sigma * TM_SIGMA_SCALE
            else:
                w_o = 1.0 / (ogt_vars + 1e-6)
                ogt_val = float(np.sum(ogt_mus * w_o) / np.sum(w_o))
                total_ogt_var = float(np.mean(ogt_vars) + np.var(ogt_mus))
                ogt_conf = float(np.sqrt(total_ogt_var) * OGT_SIGMA_SCALE)
            
            # 2. Stage 2: Predict Tm conditioned on OGT prior
            emb_t, aux_t = enrich_inputs(emb.cpu(), [sequence], tmhmm_flags=None, ogt_priors=np.array([ogt_val]))
            emb_t, aux_t = emb_t.to(self.device), aux_t.to(self.device)
            mus, vars_list = [], []
            for m_t in self.models_tm:
                z_mu, z_lv = m_t(emb_t, aux_t, head='tm')
                pred_mu = z_mu.cpu() * self.tm_std + self.tm_mean
                pred_var = z_lv.cpu() * (self.tm_std ** 2)
                mus.append(pred_mu)
                vars_list.append(pred_var)
                
            mus_stack = torch.stack(mus, dim=0)
            vars_stack = torch.stack(vars_list, dim=0)
            
            # Confidence-weighted ensemble: weight by inverse variance
            weights = 1.0 / (vars_stack + 1e-6)
            tm_val = (mus_stack * weights).sum(dim=0) / weights.sum(dim=0)
            total_var = vars_stack.mean(dim=0) + ((mus_stack - tm_val) ** 2).mean(dim=0)
            tm_conf = torch.sqrt(total_var) * TM_SIGMA_SCALE
            
        return tm_val.item(), tm_conf.item(), ogt_val, ogt_conf

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

    def evaluate_design_edit(self, wt_sequence: str, mutant_sequence: str) -> dict:
        """Evaluate a specific manual or design edit against wild-type sequence."""
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

        mutations = []
        for i in range(len(wt)):
            if wt[i] != mut[i]:
                mutations.append({
                    "position": i + 1,
                    "wt_aa": wt[i],
                    "mut_aa": mut[i],
                    "label": f"{wt[i]}{i+1}{mut[i]}"
                })

        delta_tm = round(res_mut["tm_pred"] - res_wt["tm_pred"], 2)
        is_stab = delta_tm > 0
        effect = "Stabilizing" if delta_tm > 0 else ("Destabilizing" if delta_tm < 0 else "Neutral")
        
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
        """Pick random positions in identified loop regions and generate random non-WT mutations."""
        seq = sanitize_sequence(wt_sequence)
        
        if not loop_positions:
            ss = predict_secondary_structure(seq)
            loop_pos_set = set()
            for l in ss["loops"]:
                loop_pos_set.update(l["positions"])
            loop_positions = sorted(list(loop_pos_set))

        if not loop_positions:
            loop_positions = list(range(len(seq)))

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
