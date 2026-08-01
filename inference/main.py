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

        return templates.TemplateResponse(request=request, name="index.html", context={
            "tm": f"{res['tm_pred']:.2f}",
            "tm_conf": f"{res['tm_conf']:.2f}",
            "ogt": f"{res['ogt_pred']:.2f}",
            "ogt_conf": f"{res['ogt_conf']:.2f}",
            "sequence": sequence
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
