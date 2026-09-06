import os
# Prevent OpenBLAS/OMP thread exhaustion on WSL2 / container environments
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import base64
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Response, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from engine_loader import TALKING_HEAD_REGISTRY, engine_manager
import cache_manager
import preprocessor
import compiler

app = FastAPI(
    title="NewsStudio Modern Talking-Head Server",
    description="Unified API & Container for Next-Gen Talking Head Models (AVTR-1, Ditto, FLOAT, FantasyTalking2, Hallo4, PersonaLive, SyncAnimation, EchoMimicV3, MuseTalk).",
    version="1.0.0",
)

# Enable CORS for Next.js frontend on localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    model_id: str = Field(..., description="Target model ID: avtr1, ditto, float, fantasytalking2, hallo4, personalive, syncanimation, echomimicv3, musetalk")
    video_or_image_path: Optional[str] = Field(None, description="Absolute or relative path to driver video or portrait image")
    audio_path: Optional[str] = Field(None, description="Absolute or relative path to speech WAV audio")
    image_base64: Optional[str] = Field(None, description="Optional raw base64 encoded portrait image")
    audio_base64: Optional[str] = Field(None, description="Optional raw base64 encoded speech audio")
    fps: Optional[int] = Field(25, description="Target output frame rate")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model-specific inference parameters")


@app.get("/health")
async def health_check():
    """Returns microservice health, GPU memory usage, and registered engines."""
    gpu_stats = engine_manager.get_gpu_telemetry()
    return {
        "status": "online",
        "service": "modern-talkinghead-server",
        "port": 8010,
        "active_model": engine_manager.active_model_id,
        "gpu": gpu_stats,
        "supported_engines": list(TALKING_HEAD_REGISTRY.keys()),
        "engine_count": len(TALKING_HEAD_REGISTRY),
    }


@app.get("/models")
async def list_models():
    """Returns complete specifications, citations, and parameters for all supported talking head models."""
    return {
        "count": len(TALKING_HEAD_REGISTRY),
        "models": [meta.__dict__ for meta in TALKING_HEAD_REGISTRY.values()],
    }


@app.post("/generate")
async def generate_talking_head(req: GenerateRequest):
    """
    Unified talking head generation endpoint.
    Accepts model_id and audio/video input and returns generated MP4 video + telemetry.
    """
    model_id = req.model_id.lower()
    if model_id not in TALKING_HEAD_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_id: '{req.model_id}'. Supported: {list(TALKING_HEAD_REGISTRY.keys())}",
        )

    # Set up temp working directory
    cache_dir = Path("/app/cache") if os.path.exists("/app") else Path(os.getcwd()) / "storage" / "cache" / "talking_heads"
    cache_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time() * 1000)

    # Handle image/video payload
    video_or_image_path = req.video_or_image_path
    if req.image_base64:
        img_bytes = base64.b64decode(req.image_base64)
        video_or_image_path = str(cache_dir / f"input_avatar_{timestamp}.png")
        with open(video_or_image_path, "wb") as f:
            f.write(img_bytes)

    if not video_or_image_path or not os.path.exists(video_or_image_path):
        # Fallback to default avatar image
        candidate = Path(os.getcwd()) / "public" / "avatars" / "ruby.png"
        if candidate.exists():
            video_or_image_path = str(candidate)
        else:
            video_or_image_path = str(cache_dir / f"placeholder_{timestamp}.png")
            # Create a 512x512 blank test image if nothing exists
            import numpy as np
            import cv2
            blank = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.putText(blank, model_id.upper(), (50, 256), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 200), 2)
            cv2.imwrite(video_or_image_path, blank)

    # Handle audio payload
    audio_path = req.audio_path
    if req.audio_base64:
        audio_bytes = base64.b64decode(req.audio_base64)
        audio_path = str(cache_dir / f"input_audio_{timestamp}.wav")
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

    if not audio_path or not os.path.exists(audio_path):
        candidate_audio = Path(os.getcwd()) / "public" / "assets" / "demo_realtime" / "sample_speech_en.wav"
        if candidate_audio.exists():
            audio_path = str(candidate_audio)
        else:
            raise HTTPException(status_code=400, detail="Missing driving audio input.")

    output_path = str(cache_dir / f"output_{model_id}_{timestamp}.mp4")

    try:
        metrics = engine_manager.execute_inference(
            model_id=model_id,
            video_or_image_path=video_or_image_path,
            audio_path=audio_path,
            output_path=output_path,
            fps=req.fps or 25,
            options=req.options,
        )

        return JSONResponse(content={
            "success": True,
            "video_url": f"/api/talking-heads/media/{Path(output_path).name}",
            "output_path": output_path,
            "telemetry": metrics,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload")
async def unload_gpu():
    """Releases active model from VRAM and clears CUDA cache."""
    engine_manager.unload_active_model()
    return {"success": True, "message": "GPU memory unloaded successfully."}


@app.get("/cache/status")
async def cache_status():
    """Returns directory sizes and availability of cloned repos and model weights."""
    return cache_manager.get_cache_status()


class CacheCloneRequest(BaseModel):
    model_id: str = Field(..., description="Model ID to clone or 'all'")


@app.post("/cache/clone")
async def clone_repo_endpoint(req: CacheCloneRequest):
    """Clones an upstream repository or all repositories into the cache volume."""
    if req.model_id.lower() == "all":
        results = cache_manager.clone_all_repos()
        return {"success": True, "results": results}
    else:
        success = cache_manager.clone_upstream_repo(req.model_id.lower())
        return {"success": success, "model_id": req.model_id}


class PreprocessRequest(BaseModel):
    avatar_id: str = Field(..., description="Target avatar ID (e.g. ruby, david, alex)")
    engine: str = Field(..., description="Target model engine (e.g. musetalk, ditto, echomimicv3, personalive, wan2.1)")
    image_path: Optional[str] = Field(None, description="Path or URL to avatar portrait")
    video_path: Optional[str] = Field(None, description="Optional path to driver/anchor video")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Pre-processing parameters")


@app.post("/preprocess")
async def preprocess_endpoint(req: PreprocessRequest):
    """
    Offline Master Asset Pre-Processing endpoint.
    Extracts static anchor landmarks, VAE latents, alpha masks, and appearance volumes.
    """
    try:
        result = preprocessor.preprocess_avatar(
            avatar_id=req.avatar_id,
            engine=req.engine,
            image_path=req.image_path,
            video_path=req.video_path,
            options=req.options,
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/cache/anchors/{avatar_id}")
async def get_avatar_cache_endpoint(avatar_id: str):
    """Returns pre-processed master packet caching status for an avatar across all engines."""
    return preprocessor.inspect_avatar_cache(avatar_id)


class CompileRequest(BaseModel):
    model_id: str = Field(..., description="Target model ID to compile")
    target: Optional[str] = Field("tensorrt", description="Target compiler: tensorrt, torch_compile, cuda_graphs, onnx")
    precision: Optional[str] = Field("fp16", description="Precision: fp16, fp8, int8")
    batch_size: Optional[int] = Field(32, description="Profile batch size for optimization")
    dynamic_shapes: Optional[bool] = Field(True, description="Enable dynamic batching shapes profile")


@app.get("/compile/status")
async def get_compile_status_endpoint():
    """Returns GPU architecture, installed compilers (TensorRT, Inductor, ORT), and cached engines."""
    return compiler.get_compilation_status()


@app.post("/compile")
async def compile_model_endpoint(req: CompileRequest):
    """
    Triggers on-demand engine compilation for a model component.
    Produces hardware-optimized .engine / Triton kernels with dynamic shapes.
    """
    try:
        result = compiler.compile_model_engine(
            model_id=req.model_id,
            target=req.target or "tensorrt",
            precision=req.precision or "fp16",
            batch_size=req.batch_size or 32,
            dynamic_shapes=req.dynamic_shapes if req.dynamic_shapes is not None else True,
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Modern Talking-Head Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8010, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")
    args = parser.parse_args()

    reload_env = os.getenv("RELOAD", "false").lower() in ("true", "1", "yes")
    should_reload = args.reload or reload_env

    import uvicorn
    print(f"🚀 Starting Modern Talking-Head Server on {args.host}:{args.port} (reload={should_reload})...")
    uvicorn.run("server:app", host=args.host, port=args.port, reload=should_reload)

