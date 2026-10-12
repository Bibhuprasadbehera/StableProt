import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_aa_composition(sequences):
    """Calculate 6 standard amino acid composition features:
    charged, hydrophobic, proline, cysteine, aromatic, aliphatic index.
    """
    comp_list = []
    for seq in sequences:
        s_clean = "".join([c for c in str(seq).upper() if c.isupper() and c.isalpha()])
        L = max(len(s_clean), 1)
        f_charged = (s_clean.count("D") + s_clean.count("E") + s_clean.count("K") + s_clean.count("R")) / L
        f_hydro = (s_clean.count("A") + s_clean.count("I") + s_clean.count("L") + s_clean.count("M") + s_clean.count("F") + s_clean.count("W") + s_clean.count("V")) / L
        f_proline = s_clean.count("P") / L
        f_cysteine = s_clean.count("C") / L
        f_aromatic = (s_clean.count("F") + s_clean.count("W") + s_clean.count("Y")) / L
        v = s_clean.count("V")
        il = s_clean.count("I") + s_clean.count("L")
        aliph_raw = (s_clean.count("A") + 2.9 * v + 3.9 * il) / (L * 100.0)
        aliph_norm = min(aliph_raw, 1.0)
        comp_list.append([
            min(f_charged, 1.0),
            min(f_hydro, 1.0),
            min(f_proline, 1.0),
            min(f_cysteine, 1.0),
            min(f_aromatic, 1.0),
            aliph_norm
        ])
    return torch.tensor(comp_list, dtype=torch.float32)

def enrich_inputs(embeddings, sequences, tmhmm_flags=None, ogt_priors=None, ogt_sigmas=None):
    """Enrich 1280-dim SaProt embeddings with length, transmembrane flag, AA composition,
    and optional OGT prior / uncertainty.
    """
    N = embeddings.shape[0]
    lengths = []
    for seq in sequences:
        s = str(seq).upper()
        s_clean = "".join([c for c in s if c.isupper() and c.isalpha()])
        lengths.append(min(len(s_clean) / 2048.0, 1.0))
    lengths_t = torch.tensor(lengths, dtype=torch.float32).unsqueeze(-1)
    
    if tmhmm_flags is not None:
        tm_t = torch.tensor([float(f) for f in tmhmm_flags], dtype=torch.float32).unsqueeze(-1)
    else:
        tm_t = torch.zeros((N, 1), dtype=torch.float32)
        
    aa_comp = compute_aa_composition(sequences)
    
    if ogt_priors is not None:
        ogt_t = torch.tensor([float(o) / 100.0 for o in ogt_priors], dtype=torch.float32).unsqueeze(-1)
        if ogt_sigmas is not None:
            sig_t = torch.tensor([float(s) / 100.0 for s in ogt_sigmas], dtype=torch.float32).unsqueeze(-1)
            aux = torch.cat([ogt_t, sig_t, tm_t, lengths_t, aa_comp], dim=-1)
        else:
            aux = torch.cat([ogt_t, tm_t, lengths_t, aa_comp], dim=-1)
    else:
        aux = torch.cat([tm_t, lengths_t, aa_comp], dim=-1)
    return embeddings.float(), aux

class MultiHeadSaProtV8(nn.Module):
    """Decoupled dual-head prediction architecture for Tm and OGT.
    Supports bottleneck projection of auxiliary features and heteroscedastic uncertainty heads.
    """
    def __init__(self, emb_dim=1280, aux_dim_tm=9, aux_dim_ogt=8, proj_dim=64,
                 hidden1=512, hidden2=256, dropout1=0.3, dropout2=0.2,
                 use_residuals=True, ogt_heteroscedastic=False):
        super().__init__()
        self.proj_dim = proj_dim
        self.use_residuals = use_residuals
        self.ogt_heteroscedastic = ogt_heteroscedastic

        if proj_dim is not None:
            self.aux_proj_tm = nn.Sequential(nn.Linear(aux_dim_tm, proj_dim), nn.GELU(), nn.LayerNorm(proj_dim))
            self.aux_proj_ogt = nn.Sequential(nn.Linear(aux_dim_ogt, proj_dim), nn.GELU(), nn.LayerNorm(proj_dim))
            in_dim_tm = emb_dim + proj_dim
            in_dim_ogt = emb_dim + proj_dim
        else:
            self.aux_proj_tm = None
            self.aux_proj_ogt = None
            in_dim_tm = emb_dim + aux_dim_tm
            in_dim_ogt = emb_dim + aux_dim_ogt
            
        # Disjoint Tm pathway
        self.fc1_tm = nn.Linear(in_dim_tm, hidden1)
        self.ln1_tm = nn.LayerNorm(hidden1)
        self.fc2_tm = nn.Linear(hidden1, hidden2)
        self.ln2_tm = nn.LayerNorm(hidden2)
        if use_residuals:
            self.res_tm = nn.Linear(hidden1, hidden2)
        else:
            self.register_parameter('res_tm', None)
        self.head_tm = nn.Linear(hidden2, 2)
        
        # Disjoint OGT pathway
        self.fc1_ogt = nn.Linear(in_dim_ogt, hidden1)
        self.ln1_ogt = nn.LayerNorm(hidden1)
        self.fc2_ogt = nn.Linear(hidden1, hidden2)
        self.ln2_ogt = nn.LayerNorm(hidden2)
        if use_residuals:
            self.res_ogt = nn.Linear(hidden1, hidden2)
        else:
            self.register_parameter('res_ogt', None)
        self.head_ogt = nn.Linear(hidden2, 2 if ogt_heteroscedastic else 1)
        
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout1)
        self.drop2 = nn.Dropout(dropout2)
        
    def forward(self, x_emb, x_aux=None, head='tm'):
        if x_aux is None:
            if head == 'tm':
                x_aux = x_emb[:, -self.aux_proj_tm[0].in_features:] if self.aux_proj_tm is not None else x_emb[:, 1280:]
                x_emb = x_emb[:, :1280]
            else:
                x_aux = x_emb[:, -8:]
                x_emb = x_emb[:, :1280]
                
        if head == 'tm':
            h_aux = self.aux_proj_tm(x_aux) if self.aux_proj_tm is not None else x_aux
            h = torch.cat([x_emb, h_aux], dim=-1)
            h1 = self.drop1(self.act(self.ln1_tm(self.fc1_tm(h))))
            if self.res_tm is not None:
                h2 = self.drop2(self.act(self.ln2_tm(self.fc2_tm(h1)) + self.res_tm(h1)))
            else:
                h2 = self.drop2(self.act(self.ln2_tm(self.fc2_tm(h1))))
            out = self.head_tm(h2)
            z_mean = out[:, 0]
            z_var = F.softplus(out[:, 1]) + 1e-4
            return z_mean, z_var
        else:
            h_aux = self.aux_proj_ogt(x_aux) if self.aux_proj_ogt is not None else x_aux
            h = torch.cat([x_emb, h_aux], dim=-1)
            h1 = self.drop1(self.act(self.ln1_ogt(self.fc1_ogt(h))))
            if self.res_ogt is not None:
                h2 = self.drop2(self.act(self.ln2_ogt(self.fc2_ogt(h1)) + self.res_ogt(h1)))
            else:
                h2 = self.drop2(self.act(self.ln2_ogt(self.fc2_ogt(h1))))
            out = self.head_ogt(h2)
            if self.ogt_heteroscedastic:
                return out[:, 0], F.softplus(out[:, 1]) + 1e-4
            return out.squeeze(-1)
