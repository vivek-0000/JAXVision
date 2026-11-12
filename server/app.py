"""FastAPI backend: serves training metrics, benchmarks and live predictions.

Endpoints
    GET  /api/health      service + model status
    GET  /api/metrics     contents of artifacts/metrics.json (training history)
    GET  /api/benchmark   contents of artifacts/benchmark.json
    GET  /api/model       model summary (classes, params, dataset)
    GET  /api/samples     a few labelled test images (PNG, base64) for the playground
    POST /api/predict     multipart image upload -> top-k class probabilities

Run:  uvicorn server.app:app --port 8000
Env:  JAXVISION_ARTIFACTS  path to the artifacts dir (default: ./artifacts)
"""

from __future__ import annotations

import base64
import io
import json
import os
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from jaxvision.checkpoint import checkpoint_exists, load_checkpoint
from jaxvision.data import normalize
from jaxvision.model import build_model, count_params

ARTIFACTS = Path(os.environ.get("JAXVISION_ARTIFACTS", "artifacts"))
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

app = FastAPI(title="JAXVision API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Predictor:
    """Loads the checkpoint once and exposes a jitted forward pass."""

    def __init__(self, artifacts_dir: Path):
        self.variables, self.meta, self.spec = load_checkpoint(str(artifacts_dir))
        model = build_model(self.spec.num_classes, self.meta["widths"], self.meta["dropout"])
        self.params = count_params(self.variables["params"])
        self._forward = jax.jit(
            lambda vs, x: jax.nn.softmax(model.apply(vs, normalize(x, self.spec), train=False), -1)
        )
        h, w, c = self.spec.image_shape
        self._forward(self.variables, jnp.zeros((1, h, w, c), jnp.uint8))  # warm up / compile

    def predict(self, image: Image.Image, top_k: int = 5) -> list[dict]:
        h, w, c = self.spec.image_shape
        image = image.convert("L" if c == 1 else "RGB").resize((w, h), Image.LANCZOS)
        arr = np.asarray(image, dtype=np.uint8).reshape(1, h, w, c)
        probs = np.asarray(self._forward(self.variables, jnp.asarray(arr)))[0]
        order = np.argsort(-probs)[:top_k]
        return [{"label": self.spec.class_names[i], "probability": float(probs[i])} for i in order]


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor(ARTIFACTS)


def _read_json(name: str):
    path = ARTIFACTS / name
    if not path.exists():
        raise HTTPException(404, f"{name} not found. Train first: python -m jaxvision.train")
    return json.loads(path.read_text())


@app.get("/api/health")
def health():
    return {"status": "ok", "model_ready": checkpoint_exists(str(ARTIFACTS)),
            "backend": jax.default_backend(), "devices": jax.local_device_count()}


@app.get("/api/metrics")
def metrics():
    return _read_json("metrics.json")


@app.get("/api/benchmark")
def benchmark():
    return _read_json("benchmark.json")


@app.get("/api/model")
def model_info():
    if not checkpoint_exists(str(ARTIFACTS)):
        raise HTTPException(404, "No trained model yet. Run: python -m jaxvision.train")
    p = get_predictor()
    return {"dataset": p.spec.name, "classes": list(p.spec.class_names),
            "image_shape": list(p.spec.image_shape), "parameters": p.params,
            "widths": p.meta["widths"]}


@app.get("/api/samples")
def samples(n: int = 8):
    """Labelled test images so the UI playground works without any upload."""
    if not checkpoint_exists(str(ARTIFACTS)):
        raise HTTPException(404, "No trained model yet.")
    p = get_predictor()
    path = ARTIFACTS / "samples.npz"  # written by jaxvision.train
    if not path.exists():
        raise HTTPException(404, "No sample images saved. Re-run training to create them.")
    data = np.load(path)
    x, y = data["x"], data["y"]
    out = []
    for i in range(min(n, len(x))):
        img = Image.fromarray(x[i].squeeze(), "L" if x[i].shape[-1] == 1 else "RGB")
        img = img.resize((128, 128), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append({"image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                    "label": p.spec.class_names[int(y[i])]})
    return out


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), top_k: int = 5):
    if not checkpoint_exists(str(ARTIFACTS)):
        raise HTTPException(404, "No trained model yet. Run: python -m jaxvision.train")
    try:
        image = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "Could not read that file as an image.")
    return {"predictions": get_predictor().predict(image, max(1, min(top_k, 10)))}


# In production, serve the built React app from the same server.
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        target = FRONTEND_DIST / full_path
        return FileResponse(target if target.is_file() else FRONTEND_DIST / "index.html")
