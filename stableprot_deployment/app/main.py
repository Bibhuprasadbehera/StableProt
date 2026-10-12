from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import os
import torch

try:
    from .predictor import (
        StableProtPredictor,
        predict_secondary_structure,
        sanitize_sequence,
        classify_tm_tier,
    )
except ImportError:
    from predictor import (
        StableProtPredictor,
        predict_secondary_structure,
        sanitize_sequence,
        classify_tm_tier,
    )

app = FastAPI(
    title="StableProt Web Suite",
    description="Probabilistic Machine Learning Platform for Protein Stability & Design",
    version="1.0.0",
)

base_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(base_dir, "static")), name="static")

predictor: Optional[StableProtPredictor] = None

@app.on_event("startup")
async def startup_event():
    global predictor
    default_models_dir = os.path.join(os.path.dirname(base_dir), "models")
    models_dir = os.environ.get("MODELS_DIR", default_models_dir)
    device = os.environ.get("DEVICE", "auto")
    
    print(f"[StableProt Startup] Loading models from {models_dir} (device preference: {device})...")
    try:
        predictor = StableProtPredictor(models_dir=models_dir, device=device)
        print("[StableProt Startup] Models loaded and ready for inference.")
    except Exception as e:
        print(f"[StableProt Startup] Warning: Failed to load models on startup: {e}")

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
        "aromatic": round(aromatic / total * 100, 1),
    }

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "device": str(predictor.device) if predictor else "none",
        "cuda_available": torch.cuda.is_available(),
    }

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"tm": None, "tm_conf": None, "ogt": None, "ogt_conf": None}
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict_gui(request: Request, sequence: str = Form(...)):
    global predictor
    if not predictor:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"error": "Model not loaded. Check server logs.", "sequence": sequence}
        )
    
    try:
        seq = sanitize_sequence(sequence)
        if not seq:
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={"error": "Empty sequence provided.", "sequence": sequence}
            )
            
        res = predictor.predict_single(seq)
        if res.get("status") == "ERROR_TOO_SHORT":
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={"error": "Sequence too short (<50 aa). StableProt requires at least 50 residues.", "sequence": sequence}
            )

        tm_p, tm_c = res['tm_pred'], res['tm_conf']
        ci_l, ci_h = round(tm_p - tm_c, 2), round(tm_p + tm_c, 2)
        tier = res.get("tm_tier", classify_tm_tier(tm_p))
        comp = calculate_aa_composition(seq)

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "tm": f"{tm_p:.2f}",
                "tm_conf": f"{tm_c:.2f}",
                "ci_low": f"{ci_l:.2f}",
                "ci_high": f"{ci_h:.2f}",
                "thermal_tier": tier,
                "ogt": f"{res['ogt_pred']:.2f}",
                "ogt_conf": f"{res['ogt_conf']:.2f}",
                "sequence": sequence,
                "seq_len": len(seq),
                "aa_composition": comp,
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"error": f"Inference Error: {str(e)}", "sequence": sequence}
        )

@app.post("/api/predict")
async def predict_api(req: PredictRequest):
    global predictor
    if not predictor:
        return JSONResponse(status_code=503, content={"error": "Model is not loaded."})
    try:
        seq = sanitize_sequence(req.sequence)
        res = predictor.predict_single(seq)
        if "tm_pred" in res:
            tm_p, tm_c = res['tm_pred'], res['tm_conf']
            res['ci_low'] = round(tm_p - tm_c, 2)
            res['ci_high'] = round(tm_p + tm_c, 2)
            res['thermal_tier'] = res.get("tm_tier", classify_tm_tier(tm_p))
            res['seq_len'] = len(seq)
            res['aa_composition'] = calculate_aa_composition(seq)
        return res
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/identify_loops")
async def identify_loops_api(req: LoopIdentifyRequest):
    """Sequence-based secondary structure prediction and loop region identification."""
    try:
        res = predict_secondary_structure(req.sequence)
        return res
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/evaluate_design")
async def evaluate_design_api(req: EvaluateDesignRequest):
    """Evaluate manual or designed sequence edits against wild-type baseline."""
    global predictor
    if not predictor:
        return JSONResponse(status_code=503, content={"error": "Model is not loaded."})
    try:
        res = predictor.evaluate_design_edit(req.wt_sequence, req.mutant_sequence)
        return res
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/random_loop_mutation")
async def random_loop_mutation_api(req: RandomMutationRequest):
    """Generate a random mutation constrained to loop positions."""
    global predictor
    if not predictor:
        return JSONResponse(status_code=503, content={"error": "Model is not loaded."})
    try:
        res = predictor.generate_random_loop_mutation(req.sequence, req.loop_positions, req.loop_length)
        return res
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
