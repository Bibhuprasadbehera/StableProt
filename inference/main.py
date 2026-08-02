from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import os
import torch
from .v9_predict import V9Predictor, predict_secondary_structure, sanitize_sequence

app = FastAPI(title="StableProt V9 Predictor & Loop Design Suite")

# Setup templates
templates = Jinja2Templates(directory="inference/templates")

# Initialize model predictor (Singleton pattern)
predictor = None

@app.on_event("startup")
async def startup_event():
    global predictor
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(root_dir, "experiments/src/training/v9_disjoint/results")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading V9 5-seed ensemble from {models_dir} onto {device}...")
    try:
        predictor = V9Predictor(models_dir=models_dir, device=device)
        print("V9 Ensemble loaded successfully.")
    except Exception as e:
        print(f"Failed to load V9 ensemble: {e}")

class PredictRequest(BaseModel):
    sequence: str

class LoopIdentifyRequest(BaseModel):
    sequence: str

class EvaluateDesignRequest(BaseModel):
    wt_sequence: str
    mutant_sequence: str

class RandomMutationRequest(BaseModel):
    sequence: str
    loop_positions: Optional[List[int]] = None
    loop_length: Optional[int] = None

def calculate_aa_composition(seq: str) -> dict:
    total = max(1, len(seq))
    hydrophobic = sum(1 for c in seq if c in "AILMFWV")
    polar = sum(1 for c in seq if c in "STCYNQ")
    charged = sum(1 for c in seq if c in "DEKR")
    aromatic = sum(1 for c in seq if c in "FYW")
    return {
        "hydrophobic": round(hydrophobic / total * 100, 1),
        "polar": round(polar / total * 100, 1),
        "charged": round(charged / total * 100, 1),
        "aromatic": round(aromatic / total * 100, 1)
    }

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"tm": None, "tm_conf": None, "ogt": None, "ogt_conf": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict_gui(request: Request, sequence: str = Form(...)):
    global predictor
    if not predictor:
        return templates.TemplateResponse(request=request, name="index.html", context={"error": "Model not loaded", "sequence": sequence})
    
    try:
        seq = sanitize_sequence(sequence)
        if not seq:
            return templates.TemplateResponse(request=request, name="index.html", context={"error": "Empty sequence", "sequence": sequence})
            
        res = predictor.predict_single(seq)
        if res.get("status") == "ERROR_TOO_SHORT":
            return templates.TemplateResponse(request=request, name="index.html", context={"error": "Sequence too short (<50 aa)", "sequence": sequence})

def get_tm_tier(tm: float) -> str:
    if tm < 40.0:
        return "Psychrophilic (<40°C)"
    elif tm < 60.0:
        return "Mesophilic (40–60°C)"
    elif tm < 80.0:
        return "Thermophilic (60–80°C)"
    else:
        return "Hyperthermophilic (>80°C)"

@app.post("/predict", response_class=HTMLResponse)
async def predict_gui(request: Request, sequence: str = Form(...)):
    global predictor
    if not predictor:
        return templates.TemplateResponse(request=request, name="index.html", context={"error": "Model not loaded", "sequence": sequence})
    
    try:
        seq = sanitize_sequence(sequence)
        if not seq:
            return templates.TemplateResponse(request=request, name="index.html", context={"error": "Empty sequence", "sequence": sequence})
            
        res = predictor.predict_single(seq)
        if res.get("status") == "ERROR_TOO_SHORT":
            return templates.TemplateResponse(request=request, name="index.html", context={"error": "Sequence too short (<50 aa)", "sequence": sequence})

        tm_p, tm_c = res['tm_pred'], res['tm_conf']
        ci_l, ci_h = round(tm_p - tm_c, 2), round(tm_p + tm_c, 2)
        tier = get_tm_tier(tm_p)
        comp = calculate_aa_composition(seq)

        return templates.TemplateResponse(request=request, name="index.html", context={
            "tm": f"{tm_p:.2f}",
            "tm_conf": f"{tm_c:.2f}",
            "ci_low": f"{ci_l:.2f}",
            "ci_high": f"{ci_h:.2f}",
            "thermal_tier": tier,
            "ogt": f"{res['ogt_pred']:.2f}",
            "ogt_conf": f"{res['ogt_conf']:.2f}",
            "sequence": sequence,
            "seq_len": len(seq),
            "aa_composition": comp
        })
    except Exception as e:
        return templates.TemplateResponse(request=request, name="index.html", context={"error": str(e), "sequence": sequence})

@app.post("/api/predict")
async def predict_api(req: PredictRequest):
    global predictor
    if not predictor:
        return {"error": "Model not loaded"}
    try:
        seq = sanitize_sequence(req.sequence)
        res = predictor.predict_single(seq)
        if "tm_pred" in res:
            tm_p, tm_c = res['tm_pred'], res['tm_conf']
            res['ci_low'] = round(tm_p - tm_c, 2)
            res['ci_high'] = round(tm_p + tm_c, 2)
            res['thermal_tier'] = get_tm_tier(tm_p)
            res['seq_len'] = len(seq)
            res['aa_composition'] = calculate_aa_composition(seq)
        return res
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/identify_loops")
async def identify_loops_api(req: LoopIdentifyRequest):
    """Instant secondary structure prediction and loop region identification."""
    try:
        res = predict_secondary_structure(req.sequence)
        return res
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/evaluate_design")
async def evaluate_design_api(req: EvaluateDesignRequest):
    """Evaluate manual or designed sequence edits against wild-type baseline."""
    global predictor
    if not predictor:
        return {"error": "Model not loaded"}
    try:
        res = predictor.evaluate_design_edit(req.wt_sequence, req.mutant_sequence)
        return res
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/random_loop_mutation")
async def random_loop_mutation_api(req: RandomMutationRequest):
    """Generate a random mutation constrained to loop positions."""
    global predictor
    if not predictor:
        return {"error": "Model not loaded"}
    try:
        res = predictor.generate_random_loop_mutation(req.sequence, req.loop_positions, req.loop_length)
        return res
    except Exception as e:
        return {"error": str(e)}
