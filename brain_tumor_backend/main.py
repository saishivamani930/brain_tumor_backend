# brain_tumor_backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
from io import BytesIO
from PIL import Image
import os, io, json
from pathlib import Path
from functools import lru_cache

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

BASE_DIR = Path(__file__).resolve().parent

@lru_cache(maxsize=1)
def get_class_names() -> List[str]:
    with open(BASE_DIR / "labels.json", "r") as f:
        return json.load(f)

# EXACT Colab transforms
from .transforms import IM_TRANSFORM

app = FastAPI(title="Neuro Scan Assist API", version="1.0")

# CORS
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Model (unchanged architecture) -----
class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.cnn = models.resnet18(weights=None)  # no downloads on Render
        self.cnn.fc = nn.Identity()
        enc = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(enc, num_layers=2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)          # [B, 512]
        x = x.unsqueeze(1)       # [B, 1, 512]
        x = self.transformer_encoder(x)
        x = x.squeeze(1)         # [B, 512]
        return self.fc(x)

device = torch.device("cpu")
_model = None

STATE_PATH_NEW = BASE_DIR / "model_state_dict.pth"  # Colab-correct weights
STATE_PATH_OLD = BASE_DIR / "model.pth"             # fallback
WEIGHTS_USED: str | None = None                     # <— track which file loaded

def _load_any_state_into(model: nn.Module, path: Path):
    obj = torch.load(path, map_location="cpu")
    # support both state_dict and whole-model saves
    if isinstance(obj, dict) and any("weight" in k for k in obj.keys()):
        model.load_state_dict(obj, strict=True)
    else:
        model.load_state_dict(obj.state_dict(), strict=True)

def get_model():
    global _model, WEIGHTS_USED
    if _model is None:
        class_names = get_class_names()
        m = HybridCNNTransformer(num_classes=len(class_names))
        if STATE_PATH_NEW.exists():
            _load_any_state_into(m, STATE_PATH_NEW)
            WEIGHTS_USED = "model_state_dict.pth"
        elif STATE_PATH_OLD.exists():
            _load_any_state_into(m, STATE_PATH_OLD)
            WEIGHTS_USED = "model.pth"
        else:
            raise FileNotFoundError("No weights found: model_state_dict.pth or model.pth")
        m.to(device).eval()
        _model = m
    return _model

def preprocess(img_bytes: bytes) -> torch.Tensor:
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    x = IM_TRANSFORM(img).unsqueeze(0)
    return x.to(device)

class PredictResponse(BaseModel):
    pred_class: str
    probs: Dict[str, float]

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict", response_model=PredictResponse)
async def predict(image: UploadFile = File(...)):
    if image.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Upload a PNG or JPEG image.")
    raw = await image.read()
    x = preprocess(raw)

    with torch.no_grad():
        logits = get_model()(x)
        probs_t = F.softmax(logits, dim=1)[0].cpu()

    class_names = get_class_names()
    pred_idx = int(torch.argmax(probs_t).item())
    probs = {class_names[i]: float(probs_t[i].item()) for i in range(len(class_names))}

    return PredictResponse(pred_class=class_names[pred_idx], probs=probs)

# --- Debug (remove later) ---
@app.get("/debug/labels")
def debug_labels():
    cn = get_class_names()
    return {"class_names": cn, "num_classes": len(cn)}

@app.get("/debug/checksum")
def debug_checksum():
    import hashlib
    def sha256(p: Path):
        with open(p, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    files = ["labels.json", "transforms.py", "model_state_dict.pth", "model.pth"]
    return {f: (sha256(BASE_DIR / f) if (BASE_DIR / f).exists() else "missing") for f in files}

@app.get("/debug/which-weights")
def which_weights():
    # Verify whether Render actually loaded the Colab weights
    return {"weights": WEIGHTS_USED}

@app.post("/debug/predict")
async def debug_predict(image: UploadFile = File(...)):
    # Returns top-k to compare with Colab numerically
    raw = await image.read()
    x = preprocess(raw)
    with torch.no_grad():
        logits = get_model()(x)
        probs_t = F.softmax(logits, dim=1)[0].cpu()
    class_names = get_class_names()
    top_p, top_i = torch.topk(probs_t, k=len(class_names))
    topk = [(class_names[int(i)], float(p)) for p, i in zip(top_p, top_i)]
    return {"topk": topk}
if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("brain_tumor_backend.main:app", host="0.0.0.0", port=port, reload=False)
