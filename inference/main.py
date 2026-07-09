from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import torch
from .v9_predict import V9Predictor

app = FastAPI(title="StableProt V8 Predictor")

# Setup templates
templates = Jinja2Templates(directory="inference/templates")

# Initialize model predictor (Singleton pattern)
predictor = None

@app.on_event("startup")
async def startup_event():
    global predictor
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(root_dir, "experiments/src/training/v8_disjoint/results")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading V9 5-seed ensemble from {models_dir} onto {device}...")
    try:
        predictor = V9Predictor(models_dir=models_dir, device=device)
        print("V9 Ensemble loaded successfully.")
    except Exception as e:
        print(f"Failed to load V9 ensemble: {e}")

class PredictRequest(BaseModel):
    sequence: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"tm": None, "tm_conf": None, "ogt": None, "ogt_conf": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict_gui(request: Request, sequence: str = Form(...)):
    global predictor
    if not predictor:
        return templates.TemplateResponse(request=request, name="index.html", context={"error": "Model not loaded", "sequence": sequence})
    
    try:
        seq = sequence.strip().upper()
        if not seq:
            return templates.TemplateResponse(request=request, name="index.html", context={"error": "Empty sequence", "sequence": sequence})
            
        tm, tm_conf, ogt, ogt_conf = predictor.predict(seq)
        return templates.TemplateResponse(request=request, name="index.html", context={"tm": f"{tm:.2f}", "tm_conf": f"{tm_conf:.2f}", "ogt": f"{ogt:.2f}", "ogt_conf": f"{ogt_conf:.2f}", "sequence": sequence})
    except Exception as e:
        return templates.TemplateResponse(request=request, name="index.html", context={"error": str(e), "sequence": sequence})

@app.post("/api/predict")
async def predict_api(req: PredictRequest):
    global predictor
    if not predictor:
        return {"error": "Model not loaded"}
    try:
        seq = req.sequence.strip().upper()
        tm, tm_conf, ogt, ogt_conf = predictor.predict(seq)
        return {"tm": tm, "tm_conf": tm_conf, "ogt": ogt, "ogt_conf": ogt_conf}
    except Exception as e:
        return {"error": str(e)}
