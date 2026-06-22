from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import torch
from .v6_predict import V6Predictor

app = FastAPI(title="StableProt V6 Predictor")

# Setup templates
templates = Jinja2Templates(directory="inference/templates")

# Initialize model predictor (Singleton pattern)
predictor = None

@app.on_event("startup")
async def startup_event():
    global predictor
    # Adjust path if running from root vs inference dir
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_path = os.path.join(root_dir, "experiments/src/training/v6_saprot/results/seed1/model.pt")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading V6 model from {weights_path} onto {device}...")
    try:
        predictor = V6Predictor(model_weights_path=weights_path, device=device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")

class PredictRequest(BaseModel):
    sequence: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"prediction": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict_gui(request: Request, sequence: str = Form(...)):
    global predictor
    if not predictor:
        return templates.TemplateResponse(request=request, name="index.html", context={"error": "Model not loaded", "sequence": sequence})
    
    try:
        seq = sequence.strip().upper()
        if not seq:
            return templates.TemplateResponse(request=request, name="index.html", context={"error": "Empty sequence", "sequence": sequence})
            
        tm = predictor.predict_tm(seq)
        return templates.TemplateResponse(request=request, name="index.html", context={"prediction": f"{tm:.2f}", "sequence": sequence})
    except Exception as e:
        return templates.TemplateResponse(request=request, name="index.html", context={"error": str(e), "sequence": sequence})

@app.post("/api/predict")
async def predict_api(req: PredictRequest):
    global predictor
    if not predictor:
        return {"error": "Model not loaded"}
    try:
        seq = req.sequence.strip().upper()
        tm = predictor.predict_tm(seq)
        return {"tm": tm}
    except Exception as e:
        return {"error": str(e)}
