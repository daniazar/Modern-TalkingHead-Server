# Modern-TalkingHead-Server

Unified GPU API microservice and container for next-generation talking head & avatar animation models, designed for the NewsStudio ecosystem.

Modeled directly on `Modern-TTS-Server`, this single container runs on **Port 8010** and orchestrates inference, dynamic VRAM offloading, and streaming delivery for:

1. **AVTR-1** (Avaturn duplex flow-matching interactive avatar)
2. **Ditto** (Ant Group motion-space diffusion)
3. **FLOAT** (DeepBrain AI generative motion latent flow matching)
4. **FantasyTalking2** (Alibaba AMAP CV Lab TLPO preference-optimized avatar)
5. **Hallo4** (Fudan Generative Vision DPO 4K talking portrait)
6. **PersonaLive** (GVC Lab real-time infinite-length portrait diffusion)
7. **SyncAnimation** (IJCAI 2025 NeRF audio-driven human pose & talking head)
8. **EchoMimicV3** (Ant Group 1.3B multi-modal portrait & semi-body animation)
9. **MuseTalk** (Reference baseline 350+ FPS real-time lip sync)

---

## Architecture & Features

- **Port:** `8010` (HTTP & WebSocket streaming)
- **Runtime:** PyTorch 2.5.1 + CUDA 12.4 + cuDNN 9
- **Dynamic VRAM Management:**
  - Configured with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6`
  - Lazy loading: Models are loaded on-demand. When switching between models, previous weights are unloaded via `torch.cuda.empty_cache()` and garbage collection, keeping GPU memory within budget.
- **REST Endpoints:**
  - `GET /health` — Microservice health, GPU telemetry (VRAM allocated/reserved), and active engine.
  - `GET /models` — Full metadata, architectures, resolutions, and citations for all 9 models.
  - `POST /generate` — Unified inference taking audio (WAV / base64) and portrait image/video driver.
  - `POST /unload` — Explicitly purge VRAM and reset GPU memory.

---

## Quickstart with Docker

```bash
# Build and start container with NVIDIA GPU acceleration
docker compose up -d --build

# Check health
curl http://localhost:8010/health

# List supported model engines
curl http://localhost:8010/models
```

---

## Upstream Model References

- AVTR-1: `https://github.com/avaturn-live/avtr-1.git`
- Ditto: `https://github.com/antgroup/ditto-talkinghead.git`
- FLOAT: `https://github.com/deepbrainai-research/float.git`
- FantasyTalking2: `https://github.com/Fantasy-AMAP/fantasy-talking2.git`
- Hallo4: `https://github.com/fudan-generative-vision/hallo4.git`
- PersonaLive: `https://github.com/GVCLab/PersonaLive.git`
- SyncAnimation: `https://github.com/syncanimation/syncanimation.git`
- EchoMimicV3: `https://github.com/antgroup/echomimic_v3.git`
- MuseTalk: `https://github.com/daniazar/MuseTalk`
