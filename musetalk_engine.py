import os
import sys
import gc
import time
import json
import torch
import shutil
import tempfile
import threading
import subprocess
import collections
import queue
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Generator, Callable
from contextlib import nullcontext

import torch.nn.functional as F

logger = logging.getLogger("ModernTalkingHead.MuseTalkEngine")

# ---------------------------------------------------------------------------
# Path Resolution: Ensure MuseTalk repository & patch_mmcv are on sys.path
# ---------------------------------------------------------------------------
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SERVER_DIR, "..", ".."))

CANDIDATE_MUSETALK_DIRS = [
    "/app/repos/MuseTalk",
    "/app/vendor/MuseTalk",
    os.path.join(PROJECT_ROOT, "vendor", "MuseTalk"),
    os.path.join(SERVER_DIR, "..", "MuseTalk"),
]

MUSETALK_REPO_DIR = None
for d in CANDIDATE_MUSETALK_DIRS:
    if os.path.exists(d):
        MUSETALK_REPO_DIR = d
        if d not in sys.path:
            sys.path.insert(0, d)
        break

if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

# Idempotently apply mmcv patch
try:
    import patch_mmcv
except Exception as e:
    logger.debug(f"patch_mmcv import notice: {e}")

try:
    import cv2
    import numpy as np
except ImportError:
    pass

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
WEIGHT_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
MAX_CACHED_AVATARS = int(os.environ.get("MAX_CACHED_AVATARS", 4))

# ---------------------------------------------------------------------------
# Global Inactivity Tracking & Concurrency Locks
# ---------------------------------------------------------------------------
VRAM_INACTIVITY_TIMEOUT = float(os.environ.get("VRAM_INACTIVITY_TIMEOUT", 600.0))
LAST_ACTIVITY_TIME = time.time()
ACTIVITY_LOCK = threading.Lock()
INFERENCE_LOCK = threading.Lock()

def record_activity():
    global LAST_ACTIVITY_TIME
    with ACTIVITY_LOCK:
        LAST_ACTIVITY_TIME = time.time()

def get_idle_seconds() -> float:
    with ACTIVITY_LOCK:
        return time.time() - LAST_ACTIVITY_TIME

def set_inactivity_timeout(seconds: float) -> float:
    global VRAM_INACTIVITY_TIMEOUT
    with ACTIVITY_LOCK:
        VRAM_INACTIVITY_TIMEOUT = float(seconds)
    logger.info(f"[VRAM Watchdog] Inactivity auto-purge timeout updated to {VRAM_INACTIVITY_TIMEOUT:.1f}s")
    return VRAM_INACTIVITY_TIMEOUT

def get_inactivity_timeout() -> float:
    with ACTIVITY_LOCK:
        return VRAM_INACTIVITY_TIMEOUT


# ---------------------------------------------------------------------------
# Fast Vectorized GPU Utilities
# ---------------------------------------------------------------------------
def bgr_to_nv12_gpu(bgr_tensor: torch.Tensor) -> torch.Tensor:
    """
    bgr_tensor: (B, H, W, 3) uint8 on CUDA device.
    Returns: (B, H * 3 // 2, W) uint8 on CUDA device in NV12 planar format.
    """
    B, H, W, _ = bgr_tensor.shape
    b = bgr_tensor[..., 0].float()
    g = bgr_tensor[..., 1].float()
    r = bgr_tensor[..., 2].float()

    y = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0, 255).to(torch.uint8)

    r_sub = r[:, 0::2, 0::2]
    g_sub = g[:, 0::2, 0::2]
    b_sub = b[:, 0::2, 0::2]

    u = (-0.168736 * r_sub - 0.331264 * g_sub + 0.5 * b_sub + 128.0).clamp(0, 255).to(torch.uint8)
    v = (0.5 * r_sub - 0.418688 * g_sub - 0.081312 * b_sub + 128.0).clamp(0, 255).to(torch.uint8)

    uv = torch.empty((B, H // 2, W), dtype=torch.uint8, device=bgr_tensor.device)
    uv[:, :, 0::2] = u
    uv[:, :, 1::2] = v

    return torch.cat([y, uv], dim=1)


NUM_PINNED_BUFFERS = 4
pinned_host_pool: Dict[Tuple[int, int, int, str], List[torch.Tensor]] = {}

def get_pinned_buffers(h: int, w: int, batch_size: int = 32, pix_fmt: str = "bgr24") -> List[torch.Tensor]:
    key = (h, w, batch_size, pix_fmt)
    if key not in pinned_host_pool:
        if pix_fmt == "nv12":
            pinned_host_pool[key] = [
                torch.empty((batch_size, h * 3 // 2, w), dtype=torch.uint8, pin_memory=True)
                for _ in range(NUM_PINNED_BUFFERS)
            ]
        else:
            pinned_host_pool[key] = [
                torch.empty((batch_size, h, w, 3), dtype=torch.uint8, pin_memory=True)
                for _ in range(NUM_PINNED_BUFFERS)
            ]
    return pinned_host_pool[key]


def compute_avatar_crop_geometry(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int, expand: float = 1.5):
    """
    Computes expanding crop coordinates guaranteed to stay strictly within image boundaries [0, 0, img_w, img_h].
    Prevents negative indexing slice errors and dimension mismatch during GPU blending.
    """
    x_c, y_c = (x1 + x2) // 2, (y1 + y2) // 2
    w_box, h_box = x2 - x1, y2 - y1
    s = int(max(w_box, h_box) // 2 * expand)

    x_s = x_c - s
    y_s = y_c - s
    x_e = x_c + s
    y_e = y_c + s

    if x_s < 0:
        x_e = min(img_w, x_e - x_s)
        x_s = 0
    if y_s < 0:
        y_e = min(img_h, y_e - y_s)
        y_s = 0
    if x_e > img_w:
        x_s = max(0, x_s - (x_e - img_w))
        x_e = img_w
    if y_e > img_h:
        y_s = max(0, y_s - (y_e - img_h))
        y_e = img_h

    x_s = max(0, min(int(x_s), img_w - 1))
    y_s = max(0, min(int(y_s), img_h - 1))
    x_e = max(x_s + 1, min(int(x_e), img_w))
    y_e = max(y_s + 1, min(int(y_e), img_h))

    crop_w = x_e - x_s
    crop_h = y_e - y_s
    rel_x = max(0, int(x1) - x_s)
    rel_y = max(0, int(y1) - y_s)
    rel_w = max(1, min(int(x2) - int(x1), crop_w - rel_x))
    rel_h = max(1, min(int(y2) - int(y1), crop_h - rel_y))

    return x_s, y_s, x_e, y_e, rel_x, rel_y, rel_w, rel_h


def get_gaussian_kernel(kernel_size: int = 15, sigma: float = 3.0, dev: torch.device = DEVICE) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=dev).float() - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel2d = g[:, None] * g[None, :]
    return kernel2d.unsqueeze(0).unsqueeze(0).half()


# ---------------------------------------------------------------------------
# TensorRT & CUDA Graph UNet / VAE / TAESD Runners
# ---------------------------------------------------------------------------
def get_gpu_compute_capability() -> Tuple[bool, str]:
    """
    Returns (has_cuda, arch_string) e.g. (True, 'sm_89') for Ada Lovelace,
    (True, 'sm_86') for Ampere, (True, 'sm_75') for Turing.
    """
    if not torch.cuda.is_available():
        return False, "cpu"
    try:
        major, minor = torch.cuda.get_device_capability(0)
        return True, f"sm_{major}{minor}"
    except Exception:
        return True, "cuda_unknown"


class CUDAGraphUNetRunner:
    """CUDA Graph-captured UNet forward pass for batch_size=16 (280+ FPS fallback)."""
    def __init__(self, pe, unet, batch_size=16, device=DEVICE):
        self.pe = pe
        self.unet = unet
        self.batch_size = batch_size
        self.device = device
        self.timesteps = torch.tensor([0], device=device)
        self.graph_captured = False

        if hasattr(self.unet, "model") and self.unet.model is not None:
            self.unet.model.eval()
            self.unet.model.requires_grad_(False)

        if device.type == "cuda":
            try:
                self.static_whisper = torch.zeros(batch_size, 50, 384, device=device, dtype=torch.float16)
                self.static_latent = torch.zeros(batch_size, 8, 32, 32, device=device, dtype=torch.float16)

                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s), torch.no_grad():
                    for _ in range(3):
                        a_feat = self.pe(self.static_whisper)
                        self.static_out = self.unet.model(self.static_latent, self.timesteps, encoder_hidden_states=a_feat).sample
                torch.cuda.current_stream().wait_stream(s)

                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph), torch.no_grad():
                    a_feat = self.pe(self.static_whisper)
                    self.static_out = self.unet.model(self.static_latent, self.timesteps, encoder_hidden_states=a_feat).sample
                self.graph_captured = True
                logger.info("⚡ [GPU Graph] CUDA Graph captured: UNet running with zero CPU launch overhead (280+ FPS)!")
            except Exception as e:
                logger.warning(f"[GPU Graph Warning] CUDA Graph capture fallback: {e}")
                self.graph_captured = False

    def run(self, whisper_b: torch.Tensor, latent_b: torch.Tensor, pe_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.no_grad():
            if hasattr(self.unet, "trt_runner") and self.unet.trt_runner is not None:
                return self.unet.trt_runner.run(whisper_b, latent_b, pe_batch=pe_batch)
            if self.graph_captured and whisper_b.shape[0] == self.batch_size:
                self.static_whisper.copy_(whisper_b)
                self.static_latent.copy_(latent_b)
                self.graph.replay()
                return self.static_out.clone()
            else:
                a_feat = pe_batch if pe_batch is not None else self.pe(whisper_b.to(self.device))
                return self.unet.model(latent_b.to(self.device, dtype=self.unet.model.dtype), self.timesteps, encoder_hidden_states=a_feat).sample.detach()


class UNetTRTRunner:
    """TensorRT dynamic shape execution runner for UNet (510–600+ FPS)."""
    def __init__(self, engine_path: str, pe, batch_size: int = 32, device: torch.device = DEVICE):
        import tensorrt as trt
        trt_logger = trt.Logger(trt.Logger.WARNING)
        has_cuda, arch = get_gpu_compute_capability()
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        try:
            runtime = trt.Runtime(trt_logger)
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        except Exception as e:
            raise RuntimeError(
                f"TensorRT deserialization failed for {os.path.basename(engine_path)} on {arch}: {e}"
            )
        if self.engine is None:
            raise RuntimeError(
                f"TensorRT engine {os.path.basename(engine_path)} could not be deserialized (likely compiled for different GPU arch, host is {arch})."
            )
        self.ctx = self.engine.create_execution_context()
        if self.ctx is None:
            raise RuntimeError(f"Failed to create execution context for {os.path.basename(engine_path)}.")
        self.pe = pe
        self.batch_size = batch_size
        self.device = device
        self.stream = torch.cuda.Stream() if (device.type == "cuda" and torch.cuda.is_available()) else None
        self.out_buf = torch.empty((batch_size, 4, 32, 32), device=device, dtype=WEIGHT_DTYPE)

    def run(self, whisper_batch: torch.Tensor, latent_batch: torch.Tensor, pe_batch: Optional[torch.Tensor] = None, stream=None) -> torch.Tensor:
        exec_stream = stream or self.stream or torch.cuda.current_stream()
        b = whisper_batch.shape[0]
        pe_tokens = pe_batch if pe_batch is not None else self.pe(whisper_batch)
        self.ctx.set_input_shape("latent", (b, 8, 32, 32))
        self.ctx.set_input_shape("whisper", (b, 50, 384))
        if hasattr(self.ctx, "execute_async_v3"):
            self.ctx.set_tensor_address("latent", int(latent_batch.data_ptr()))
            self.ctx.set_tensor_address("whisper", int(pe_tokens.data_ptr()))
            self.ctx.set_tensor_address("pred_latent", int(self.out_buf[:b].data_ptr()))
            s_ptr = exec_stream.cuda_stream if hasattr(exec_stream, "cuda_stream") else exec_stream
            self.ctx.execute_async_v3(s_ptr)
        else:
            bindings = [int(latent_batch.data_ptr()), int(pe_tokens.data_ptr()), int(self.out_buf[:b].data_ptr())]
            self.ctx.execute_v2(bindings)
        return self.out_buf[:b]


class TAESDGPURunner:
    """TensorRT decoder runner for Tiny AutoEncoder (2,100–2,500+ FPS)."""
    def __init__(self, engine_path: str, batch_size: int = 32, device: torch.device = DEVICE):
        import tensorrt as trt
        trt_logger = trt.Logger(trt.Logger.WARNING)
        has_cuda, arch = get_gpu_compute_capability()
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        try:
            runtime = trt.Runtime(trt_logger)
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        except Exception as e:
            raise RuntimeError(f"TAESD deserialization failed for {os.path.basename(engine_path)} on {arch}: {e}")
        if self.engine is None:
            raise RuntimeError(f"TAESD engine {os.path.basename(engine_path)} is incompatible with host GPU {arch}.")
        self.ctx = self.engine.create_execution_context()
        if self.ctx is None:
            raise RuntimeError(f"Failed to create execution context for {os.path.basename(engine_path)}.")
        self.device = device
        self.batch_size = batch_size
        self.d_out = torch.empty((self.batch_size, 3, 256, 256), device=device, dtype=WEIGHT_DTYPE)
        self.lock = threading.Lock()

    def decode_gpu(self, latents: torch.Tensor) -> torch.Tensor:
        with self.lock:
            cur_stream = torch.cuda.current_stream()
            b = latents.shape[0]
            lat_in = latents.contiguous()
            self.ctx.set_input_shape("latents", (b, 4, 32, 32))
            if hasattr(self.ctx, "execute_async_v3"):
                self.ctx.set_tensor_address("latents", int(lat_in.data_ptr()))
                self.ctx.set_tensor_address("images", int(self.d_out[:b].data_ptr()))
                self.ctx.execute_async_v3(cur_stream.cuda_stream)
            else:
                bindings = [int(lat_in.data_ptr()), int(self.d_out[:b].data_ptr())]
                self.ctx.execute_v2(bindings)
            decoded = (self.d_out[:b].float() * 255.0)[:, [2, 1, 0], :, :]
            return decoded


class VAETRTRunner:
    """TensorRT SD-VAE decoder runner with 16-slot tail padding (135+ FPS)."""
    def __init__(self, engine_path: str, device: torch.device = DEVICE):
        import tensorrt as trt
        trt_logger = trt.Logger(trt.Logger.WARNING)
        has_cuda, arch = get_gpu_compute_capability()
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        try:
            runtime = trt.Runtime(trt_logger)
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        except Exception as e:
            raise RuntimeError(f"VAE TRT deserialization failed for {os.path.basename(engine_path)} on {arch}: {e}")
        if self.engine is None:
            raise RuntimeError(f"VAE TRT engine {os.path.basename(engine_path)} is incompatible with host GPU {arch}.")
        self.ctx = self.engine.create_execution_context()
        if self.ctx is None:
            raise RuntimeError(f"Failed to create execution context for {os.path.basename(engine_path)}.")
        self.device = device
        self.d_in = torch.zeros((16, 4, 32, 32), dtype=WEIGHT_DTYPE, device=device)
        self.d_out = torch.empty((16, 3, 256, 256), dtype=WEIGHT_DTYPE, device=device)
        self.lock = threading.Lock()

    def decode_gpu(self, latents: torch.Tensor) -> torch.Tensor:
        with self.lock:
            cur_stream = torch.cuda.current_stream()
            B = latents.shape[0]
            outs = []
            for i in range(0, B, 16):
                chunk = latents[i:min(i + 16, B)]
                c_len = chunk.shape[0]
                if c_len < 16:
                    self.d_in[:c_len].copy_(chunk)
                    self.d_in[c_len:].copy_(chunk[-1:].expand(16 - c_len, -1, -1, -1))
                    if hasattr(self.ctx, "execute_async_v3"):
                        self.ctx.set_tensor_address("latents", int(self.d_in.data_ptr()))
                        self.ctx.set_tensor_address("images", int(self.d_out.data_ptr()))
                        self.ctx.execute_async_v3(cur_stream.cuda_stream)
                    else:
                        self.ctx.execute_v2([int(self.d_in.data_ptr()), int(self.d_out.data_ptr())])
                    sub_out = self.d_out[:c_len]
                else:
                    chunk_cont = chunk.contiguous()
                    if hasattr(self.ctx, "execute_async_v3"):
                        self.ctx.set_tensor_address("latents", int(chunk_cont.data_ptr()))
                        self.ctx.set_tensor_address("images", int(self.d_out.data_ptr()))
                        self.ctx.execute_async_v3(cur_stream.cuda_stream)
                    else:
                        self.ctx.execute_v2([int(chunk_cont.data_ptr()), int(self.d_out.data_ptr())])
                    sub_out = self.d_out
                outs.append(sub_out.clone())
            all_out = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
            decoded = (all_out.float() * 255.0)[:, [2, 1, 0], :, :]
            return decoded


# ---------------------------------------------------------------------------
# Temporal Cadence Stride & Latent SLERP Utilities
# ---------------------------------------------------------------------------
def slerp_batch(z0: torch.Tensor, z1: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Spherical Linear Interpolation (SLERP) on 4-channel diffusion latent manifolds.
    Interpolates along geodesic curves on the hypersphere with magnitude preservation.
    """
    orig_shape = z0.shape
    B = orig_shape[0]
    v0 = z0.reshape(B, -1)
    v1 = z1.reshape(B, -1)
    
    norm0 = torch.norm(v0, dim=-1, keepdim=True) + 1e-7
    norm1 = torch.norm(v1, dim=-1, keepdim=True) + 1e-7
    
    u0 = v0 / norm0
    u1 = v1 / norm1
    
    dot = torch.sum(u0 * u1, dim=-1, keepdim=True).clamp(-0.9995, 0.9995)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    
    linear_mask = sin_theta.abs() < 1e-4
    scale0 = torch.where(linear_mask, 1.0 - alpha, torch.sin((1.0 - alpha) * theta) / sin_theta)
    scale1 = torch.where(linear_mask, alpha, torch.sin(alpha * theta) / sin_theta)
    
    res = scale0 * u0 + scale1 * u1
    res = res / (torch.norm(res, dim=-1, keepdim=True) + 1e-7)
    target_norm = (1.0 - alpha) * norm0 + alpha * norm1
    res = res * target_norm
    return res.reshape(orig_shape)


def expand_latents_stride_fast(key_latents: torch.Tensor, stride: int, video_num: int) -> torch.Tensor:
    """
    Synthesizes dense continuous latent frames from sparse UNet keyframe evaluations.
    Applies vectorized GPU SLERP across intervals (<1ms total latency) to eliminate
    teeth flicker and reduce compute by 50% (Stride 2) or 67% (Stride 3).
    """
    if stride <= 1:
        return key_latents[:video_num]
        
    full = torch.empty((video_num, *key_latents.shape[1:]), device=key_latents.device, dtype=key_latents.dtype)
    
    key_idx = list(range(0, video_num, stride))
    if key_idx[-1] != video_num - 1:
        key_idx.append(video_num - 1)
        
    full[key_idx] = key_latents
    num_intervals = len(key_idx) - 1
    
    # Determine how many intervals have standard stride gap
    has_odd_tail = (key_idx[-1] - key_idx[-2]) != stride
    K_std = (num_intervals - 1) if has_odd_tail else num_intervals
    
    if K_std > 0:
        z0_std = key_latents[:K_std]
        z1_std = key_latents[1:K_std + 1]
        for step in range(1, stride):
            alpha = float(step) / float(stride)
            mid = slerp_batch(z0_std, z1_std, alpha)
            target_idx = [key_idx[k] + step for k in range(K_std)]
            full[target_idx] = mid
            
    if has_odd_tail:
        idx0 = key_idx[-2]
        idx1 = key_idx[-1]
        gap = idx1 - idx0
        if gap > 1:
            z0 = key_latents[-2:-1]
            z1 = key_latents[-1:]
            for step in range(1, gap):
                alpha = float(step) / float(gap)
                full[idx0 + step] = slerp_batch(z0, z1, alpha)[0]
                
    return full


# ---------------------------------------------------------------------------
# Avatar Material & Persistent Disk Caching
# ---------------------------------------------------------------------------
class AvatarMaterial:
    def __init__(self, avatar_id: str, video_path: str, bbox_shift: int = 0, weights_dir: Optional[str] = None):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.weights_dir = weights_dir
        self.frame_list = []
        self.coords = []
        self.mask_coords = []
        self.masks = []
        self.load_or_preprocess()

    def load_or_preprocess(self):
        candidate_cache_dirs = [
            os.path.join("/app", "results", "v15", "avatars", self.avatar_id),
            os.path.join("/app", "storage", "cache", "talking_heads", "anchors", self.avatar_id, "musetalk"),
            os.path.join("/app", "storage", "cache", "avatars", self.avatar_id),
            os.path.join(PROJECT_ROOT, "storage", "cache", "talking_heads", "anchors", self.avatar_id, "musetalk"),
            os.path.join(PROJECT_ROOT, "storage", "cache", "avatars", self.avatar_id),
            os.path.join(PROJECT_ROOT, "vendor", "MuseTalk", "results", "v15", "avatars", self.avatar_id),
        ]

        active_cache_dir = None
        for cd in candidate_cache_dirs:
            f_p = os.path.join(cd, "frames.pt")
            m_p = os.path.join(cd, "masks.pt")
            l_p = os.path.join(cd, "latents.pt")
            g_p = os.path.join(cd, "geometry.pkl")
            c_p = os.path.join(cd, "coords.pkl")

            if (os.path.exists(f_p) and os.path.exists(m_p) and os.path.exists(l_p) and
                (os.path.exists(g_p) or os.path.exists(c_p))):
                try:
                    # Sanity check latents to prevent loading corrupted all-zero tensors
                    test_l = torch.load(l_p, map_location="cpu", weights_only=True)
                    if test_l.abs().max() < 1e-4:
                        logger.warning(f"⚠️ Skipping corrupted all-zero cache at: {cd}")
                        continue
                    active_cache_dir = cd
                    break
                except Exception:
                    active_cache_dir = cd
                    break

        # Fast Path: Master Asset Packet already precompiled
        if active_cache_dir:
            import pickle
            logger.info(f"⚡ [Avatar Load] Loading pre-processed master packet for '{self.avatar_id}' from {active_cache_dir} (<10ms)...")
            frames_pt = os.path.join(active_cache_dir, "frames.pt")
            latents_pt = os.path.join(active_cache_dir, "latents.pt")
            masks_pt = os.path.join(active_cache_dir, "masks.pt")
            geom_pkl = os.path.join(active_cache_dir, "geometry.pkl")

            try:
                self.frames_gpu_tensor = torch.load(frames_pt, map_location="cpu", mmap=True, weights_only=True)
                self.latents_gpu_tensor = torch.load(latents_pt, map_location=DEVICE, weights_only=True).to(dtype=WEIGHT_DTYPE)
            except Exception:
                try:
                    self.frames_gpu_tensor = torch.load(frames_pt, map_location="cpu", mmap=True)
                    self.latents_gpu_tensor = torch.load(latents_pt, map_location=DEVICE).to(dtype=WEIGHT_DTYPE)
                except Exception:
                    self.frames_gpu_tensor = torch.load(frames_pt, map_location="cpu")
                    self.latents_gpu_tensor = torch.load(latents_pt, map_location=DEVICE).to(dtype=WEIGHT_DTYPE)

            # Prevent corrupt / zeroed-out cache files from producing blank or blurred outputs
            if self.frames_gpu_tensor[:10].float().mean() < 1.0 or self.latents_gpu_tensor.abs().max() < 1e-4:
                logger.warning(f"⚠️ Corrupt zero-tensor cache detected in {active_cache_dir}. Skipping corrupted packet.")
                active_cache_dir = None

            # Ensure 8-channel latent format (channels 0..3 masked mouth, 4..7 reference face)
            if active_cache_dir and self.latents_gpu_tensor.shape[1] == 4:
                m_lat = self.latents_gpu_tensor.clone()
                m_lat[:, :, 16:, :] = 0.0
                self.latents_gpu_tensor = torch.cat([m_lat, self.latents_gpu_tensor], dim=1)

            if os.path.exists(geom_pkl):
                with open(geom_pkl, "rb") as f:
                    self.geometry_precomputed = pickle.load(f)
            else:
                coords_pkl = os.path.join(active_cache_dir, "coords.pkl")
                with open(coords_pkl, "rb") as f:
                    coords_raw = pickle.load(f)
                H_f, W_f = self.frames_gpu_tensor.shape[2], self.frames_gpu_tensor.shape[3]
                if isinstance(coords_raw, dict):
                    raw_box = coords_raw.get("bbox", [int(W_f * 0.25), int(H_f * 0.15), int(W_f * 0.75), int(H_f * 0.75)])
                    coords_list = [raw_box] * self.frames_gpu_tensor.shape[0]
                elif isinstance(coords_raw, list):
                    coords_list = coords_raw
                else:
                    coords_list = [[int(W_f * 0.25), int(H_f * 0.15), int(W_f * 0.75), int(H_f * 0.75)]] * self.frames_gpu_tensor.shape[0]

                self.geometry_precomputed = []
                for c in coords_list:
                    x1, y1, x2, y2 = max(0, int(c[0])), max(0, int(c[1])), min(W_f, int(c[2])), min(H_f, int(c[3]))
                    geom = compute_avatar_crop_geometry(x1, y1, x2, y2, W_f, H_f)
                    self.geometry_precomputed.append(geom)
                try:
                    with open(geom_pkl, "wb") as f:
                        pickle.dump(self.geometry_precomputed, f)
                except Exception:
                    pass

            try:
                masks_raw_list = torch.load(masks_pt, map_location=DEVICE, mmap=True, weights_only=True)
            except Exception:
                masks_raw_list = torch.load(masks_pt, map_location=DEVICE)
            self.masks_gpu_precomputed = [m.float() for m in masks_raw_list]
            num_geoms = len(self.geometry_precomputed)
            self.coords = [(0, 0, 0, 0)] * num_geoms
            self.frame_list = [np.zeros((self.frames_gpu_tensor.shape[2], self.frames_gpu_tensor.shape[3], 3), dtype=np.uint8)]
            self.masks_tight_precomputed = [
                self.masks_gpu_precomputed[i % len(self.masks_gpu_precomputed)][:, self.geometry_precomputed[i][5]:self.geometry_precomputed[i][5] + self.geometry_precomputed[i][7], self.geometry_precomputed[i][4]:self.geometry_precomputed[i][4] + self.geometry_precomputed[i][6]].contiguous()
                for i in range(num_geoms)
            ]
            self.crop_bounds_tight = [
                (g[1] + g[5], g[1] + g[5] + g[7], g[0] + g[4], g[0] + g[4] + g[6], g[7], g[6])
                for g in self.geometry_precomputed
            ]
            return

        # Slow Path: Fallback dynamic decode from image or video
        logger.info(f"🔨 [Avatar Prep] Cache miss for '{self.avatar_id}'. Processing source: {self.video_path}...")
        if self.video_path and self.video_path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            img = cv2.imread(self.video_path)
            if img is not None:
                self.frame_list.append(img)
        else:
            cap = cv2.VideoCapture(self.video_path)
            while cap.isOpened():
                ret, f = cap.read()
                if not ret:
                    break
                self.frame_list.append(f)
            cap.release()

        if not self.frame_list:
            raise ValueError(f"Could not read video or image frames from: {self.video_path}")

        try:
            from musetalk.utils.preprocessing import get_landmark_and_bbox
            self.frame_list = self.frame_list + self.frame_list[::-1]
            total_cycle = len(self.frame_list)
            self.coords, self.frame_list = get_landmark_and_bbox(self.frame_list, self.bbox_shift, batch_size_fa=16)
        except Exception as e:
            logger.warning(f"[Avatar Prep Fallback] get_landmark_and_bbox unavailable ({e}), using central facial crop.")
            h_f, w_f = self.frame_list[0].shape[:2]
            c = [int(w_f * 0.25), int(h_f * 0.15), int(w_f * 0.75), int(h_f * 0.75)]
            self.coords = [c] * len(self.frame_list)
            total_cycle = len(self.frame_list)

        valid_coords = [c for c in self.coords if len(c) == 4 and (c[2] - c[0]) > 20 and (c[3] - c[1]) > 20]
        if not valid_coords:
            h_f, w_f = self.frame_list[0].shape[:2]
            valid_coords = [[int(w_f * 0.25), int(h_f * 0.15), int(w_f * 0.75), int(h_f * 0.75)]]

        H_f, W_f = self.frame_list[0].shape[:2]
        self.geometry_precomputed = []
        for idx in range(len(self.coords)):
            c = self.coords[idx]
            if len(c) != 4 or (c[2] - c[0]) <= 20 or (c[3] - c[1]) <= 20:
                c = valid_coords[idx % len(valid_coords)]
            x1, y1, x2, y2 = max(0, int(c[0])), max(0, int(c[1])), min(W_f, int(c[2])), min(H_f, int(c[3]))
            x_s, y_s, x_e, y_e, rel_x, rel_y, rel_w, rel_h = compute_avatar_crop_geometry(x1, y1, x2, y2, W_f, H_f)
            self.geometry_precomputed.append((x_s, y_s, x_e, y_e, rel_x, rel_y, rel_w, rel_h))
            self.coords[idx] = [x1, y1, x2, y2]

        self.frames_gpu_tensor = torch.stack([torch.from_numpy(f).permute(2, 0, 1) for f in self.frame_list]).to(DEVICE)
        self.masks_gpu_precomputed = [torch.ones((1, g[3] - g[1], g[2] - g[0]), device=DEVICE, dtype=torch.float16) for g in self.geometry_precomputed]
        self.latents_gpu_tensor = torch.zeros((total_cycle, 8, 32, 32), device=DEVICE, dtype=WEIGHT_DTYPE)


class AvatarPool:
    """In-memory thread-safe LRU pool for warmed AvatarMaterial instances in VRAM."""
    def __init__(self, max_size: int = MAX_CACHED_AVATARS):
        self.max_size = max_size
        self.pool = collections.OrderedDict()
        self.lock = threading.Lock()

    def get(self, avatar_id: str, video_path: str, bbox_shift: int = 0) -> Tuple[AvatarMaterial, bool]:
        with self.lock:
            key = f"{avatar_id}_{bbox_shift}"
            if key in self.pool:
                self.pool.move_to_end(key)
                return self.pool[key], True

            material = AvatarMaterial(avatar_id, video_path, bbox_shift)
            if len(self.pool) >= self.max_size:
                evicted_key, _ = self.pool.popitem(last=False)
                logger.info(f"[AvatarPool] Evicted oldest avatar '{evicted_key}' to stay within max_size={self.max_size}")
            self.pool[key] = material
            return material, False

    def preload(self, avatar_id: str, video_path: str, bbox_shift: int = 0) -> Dict[str, Any]:
        mat, was_cached = self.get(avatar_id, video_path, bbox_shift)
        return {
            "avatar_id": avatar_id,
            "was_cached": was_cached,
            "num_frames": len(mat.geometry_precomputed) if hasattr(mat, "geometry_precomputed") else len(mat.coords),
        }

    def list_details(self) -> List[Dict[str, Any]]:
        with self.lock:
            details = []
            for k, mat in self.pool.items():
                details.append({
                    "key": k,
                    "avatar_id": mat.avatar_id,
                    "num_frames": len(mat.geometry_precomputed) if hasattr(mat, "geometry_precomputed") else 0,
                    "source": mat.video_path,
                })
            return details

    def clear(self, avatar_id: Optional[str] = None):
        with self.lock:
            if avatar_id:
                keys_to_del = [k for k in self.pool if k.startswith(f"{avatar_id}_") or k == avatar_id]
                for k in keys_to_del:
                    del self.pool[k]
                logger.info(f"[AvatarPool] Cleared cached avatar: {avatar_id}")
            else:
                count = len(self.pool)
                self.pool.clear()
                logger.info(f"[AvatarPool] Cleared all {count} cached avatars.")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


avatar_pool = AvatarPool()


# ---------------------------------------------------------------------------
# Autonomous Inactivity VRAM Auto-Purge Watchdog
# ---------------------------------------------------------------------------
def vram_inactivity_watchdog(on_idle_purge=None):
    logger.info(f"[VRAM Watchdog] 🛡️ Inactivity auto-purge monitor active (threshold: {VRAM_INACTIVITY_TIMEOUT:.0f}s / {VRAM_INACTIVITY_TIMEOUT/60:.1f}m)")
    while True:
        time.sleep(15)
        timeout = get_inactivity_timeout()
        if timeout <= 0:
            continue

        idle = get_idle_seconds()
        if idle >= timeout:
            if (len(avatar_pool.pool) > 0 or on_idle_purge) and not INFERENCE_LOCK.locked():
                with avatar_pool.lock:
                    if not INFERENCE_LOCK.locked():
                        loaded = list(avatar_pool.pool.keys())
                        if loaded:
                            logger.info(f"[VRAM Watchdog] ⏳ Idle timeout reached ({idle:.1f}s >= {timeout:.0f}s). Purging {len(loaded)} avatars: {loaded}...")
                            avatar_pool.pool.clear()

                        if on_idle_purge:
                            try:
                                on_idle_purge()
                            except Exception as e:
                                logger.warning(f"[VRAM Watchdog Warning] on_idle_purge error: {e}")

                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                        logger.info("[VRAM Watchdog] ✅ Inactivity auto-purge completed.")


# ---------------------------------------------------------------------------
# Native In-Process MuseTalk Engine
# ---------------------------------------------------------------------------
class MuseTalkEngine:
    """
    High-Throughput Native In-Process MuseTalk Engine.
    Incorporates TensorRT FP16/FP8 UNet (510+ FPS), TAESD/SD-VAE decoders (2,100+ FPS),
    pinned DMA buffers, and zero-latency NVENC / libx264 ultrafast video muxing.
    """
    def __init__(self, weights_dir: Optional[str] = None):
        self.device = DEVICE
        self.weight_dtype = WEIGHT_DTYPE
        self.weights_dir = self._resolve_weights_dir(weights_dir)

        logger.info("======================================================================")
        logger.info(f"[MuseTalkEngine] Initializing in-process on {self.device}...")
        logger.info(f"   Weights directory: {self.weights_dir}")
        logger.info("======================================================================")

        # 1. Load Audio Processor & Whisper
        from musetalk.utils.audio_processor import AudioProcessor
        from transformers import WhisperModel
        from musetalk.utils.face_parsing import FaceParsing

        whisper_path = os.path.join(self.weights_dir, "whisper")
        self.audio_processor = AudioProcessor(feature_extractor_path=whisper_path)
        self.whisper = WhisperModel.from_pretrained(whisper_path).to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)
        self.face_parser = FaceParsing()

        # 2. Load Base UNet, Positional Encoding, and VAE
        from musetalk.utils.utils import load_all_model
        unet_model_path = os.path.join(self.weights_dir, "musetalk", "pytorch_model.bin")
        unet_config = os.path.join(self.weights_dir, "musetalk", "musetalk.json")

        vae_path = os.path.join(self.weights_dir, "sd-vae")
        if not os.path.exists(vae_path):
            vae_path = "/app/models/sd-vae" if os.path.exists("/app/models/sd-vae") else "sd-vae"

        self.vae, unet, self.pe = load_all_model(
            unet_model_path=unet_model_path,
            unet_config=unet_config,
            vae_type=vae_path,
        )
        self.pe = self.pe.type(self.weight_dtype).eval()
        self.pe.requires_grad_(False)

        if hasattr(self.vae, "vae") and self.vae.vae is not None:
            if self.device.type == "cpu":
                self.vae.vae = self.vae.vae.float()
                self.vae._use_float16 = False
            self.vae.vae.eval()
            self.vae.vae.requires_grad_(False)

        # 3. Locate & Initialize TensorRT UNet Runner or fallback to CUDA Graph
        onnx_dir = os.path.join(self.weights_dir, "onnx")
        unet_candidates = [
            (os.path.join(onnx_dir, "unet_b32_fp8.engine"), 32, "Native FP8 Blackwell Batch-32 (510+ FPS)"),
            (os.path.join(onnx_dir, "unet_b32_fp16.engine"), 32, "Batch-32 High-Throughput FP16 (510+ FPS)"),
            (os.path.join(onnx_dir, "unet_b64_fp16.engine"), 64, "Batch-64 Ultra-Throughput (600+ FPS)"),
            (os.path.join(onnx_dir, "unet_fp8.engine"), 16, "Native FP8 Blackwell (468+ FPS)"),
            (os.path.join(onnx_dir, "unet_fp16.engine"), 16, "Native FP16 (460+ FPS)"),
        ]

        self.unet_runner = None
        self.active_batch_size = 4 if self.device.type == "cpu" else 16
        has_cuda, host_arch = get_gpu_compute_capability()

        for engine_p, b_sz, desc in unet_candidates:
            if os.path.exists(engine_p) and os.path.getsize(engine_p) > 1000:
                try:
                    self.unet_runner = UNetTRTRunner(engine_p, self.pe, batch_size=b_sz, device=self.device)
                    self.active_batch_size = b_sz
                    logger.info(f"🚀 [UNet Engine] {desc} loaded from {engine_p} (GPU Arch: {host_arch})!")
                    break
                except Exception as e:
                    logger.warning(f"[UNet Engine Architecture Check] {os.path.basename(engine_p)} not compatible on {host_arch}: {e}")

        if self.unet_runner is None:
            unet.model = unet.model.type(self.weight_dtype).eval()
            unet.model.requires_grad_(False)
            fallback_b = 4 if self.device.type == "cpu" else 16
            self.unet_runner = CUDAGraphUNetRunner(self.pe, unet, batch_size=fallback_b, device=self.device)
            self.active_batch_size = fallback_b
            logger.info(
                f"🛡️ [Architecture Fallback] Native PyTorch FP16 UNet runner active on {host_arch} (280+ FPS). "
                f"To compile native TensorRT engines for your GPU: run 'python export_to_trt.py'."
            )
        else:
            try:
                del unet
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("🧹 [Memory Optimizer] Released redundant PyTorch UNet from VRAM (~1.7 GB freed)!")
            except Exception as e:
                logger.debug(f"[Memory Optimizer Notice] {e}")

        # 4. Locate & Initialize TAESD / SD-VAE Decoders
        taesd_candidates = [
            (os.path.join(onnx_dir, "taesd_b64_fp16.engine"), 64),
            (os.path.join(onnx_dir, "taesd_b32_fp16.engine"), 32),
            (os.path.join(onnx_dir, "taesd_decoder_fp16.engine"), 16),
        ]
        self.taesd_runner = None
        for t_p, b_sz in taesd_candidates:
            if os.path.exists(t_p) and os.path.getsize(t_p) > 1000:
                try:
                    self.taesd_runner = TAESDGPURunner(t_p, batch_size=b_sz, device=self.device)
                    logger.info(f"⚡ [TAESD Engine] High-Speed TAESD TensorRT Decoder B={b_sz} loaded (2,100+ FPS)!")
                    break
                except Exception as e:
                    logger.warning(f"[TAESD Warning] Failed loading {t_p}: {e}")

        vae_engine_fp16 = os.path.join(onnx_dir, "vae_decoder_fp16.engine")
        self.vae_trt_runner = None
        if os.path.exists(vae_engine_fp16) and os.path.getsize(vae_engine_fp16) > 1000:
            try:
                self.vae_trt_runner = VAETRTRunner(vae_engine_fp16, device=self.device)
                logger.info("💎 [Full VAE Engine] High-Definition SD-VAE TensorRT Decoder loaded (135+ FPS)!")
            except Exception as e:
                logger.warning(f"[VAE TRT Warning] Failed loading {vae_engine_fp16}: {e}")

        self.stream_unet = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.stream_vae = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.stream_dma = torch.cuda.Stream() if torch.cuda.is_available() else None

        self.watchdog_thread = threading.Thread(
            target=vram_inactivity_watchdog,
            kwargs={"on_idle_purge": self._on_watchdog_idle_purge},
            daemon=True
        )
        self.watchdog_thread.start()

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(0) / (1024 * 1024)
            res = torch.cuda.memory_reserved(0) / (1024 * 1024)
            logger.info(f"✅ [MuseTalkEngine] Ready! VRAM baseline: {alloc:.1f} MB allocated, {res:.1f} MB reserved.")

    def _resolve_weights_dir(self, explicit_dir: Optional[str] = None) -> str:
        candidates = [
            explicit_dir,
            "/app/models",
            "/app/weights/musetalk",
            os.path.join(PROJECT_ROOT, "vendor", "MuseTalk", "models"),
            os.path.join(SERVER_DIR, "models", "musetalk"),
        ]
        for c in candidates:
            if c and os.path.exists(c) and os.path.exists(os.path.join(c, "whisper")):
                return os.path.abspath(c)
        return "/app/models"

    def _on_watchdog_idle_purge(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def generate(
        self,
        avatar_id: str,
        video_or_image_path: str,
        audio_path: str,
        output_path: str,
        fps: int = 30,
        opts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        opts = opts or {}
        record_activity()

        bbox_shift = int(opts.get("bbox_shift", 0))
        use_cache = bool(opts.get("use_cache", True))
        decoder = str(opts.get("decoder", "taesd" if self.taesd_runner is not None else "full_vae"))
        if opts.get("use_taesd") is not None:
            decoder = "taesd" if opts.get("use_taesd") else "full_vae"

        has_cuda = torch.cuda.is_available() and self.device.type == "cuda"
        default_encoder = "nvenc" if has_cuda else "libx264"
        encoder = str(opts.get("encoder", default_encoder))
        mouth_enhancer = str(opts.get("mouth_enhancer", "unsharp"))
        start_frame_offset = int(opts.get("start_frame_offset", opts.get("frame_offset", 0)))
        use_silence_bypass = bool(opts.get("use_silence_bypass", True if has_cuda else False))
        texture_injection = float(opts.get("texture_injection", 0.0))
        mask_mode = str(opts.get("mask_mode", "standard"))
        stride = int(opts.get("stride", 1))
        stride = max(1, min(stride, 5))

        if use_cache:
            material, cache_hit = avatar_pool.get(avatar_id, video_or_image_path, bbox_shift)
        else:
            material = AvatarMaterial(f"dynamic_{int(time.time())}", video_or_image_path, bbox_shift)
            cache_hit = False

        with INFERENCE_LOCK:
            stats = self._run_generation_pipeline(
                material=material,
                audio_path=audio_path,
                output_path=output_path,
                fps=fps,
                decoder=decoder,
                encoder=encoder,
                mouth_enhancer=mouth_enhancer,
                start_frame_offset=start_frame_offset,
                use_silence_bypass=use_silence_bypass,
                texture_injection=texture_injection,
                mask_mode=mask_mode,
                stride=stride,
                opts=opts,
            )

        record_activity()
        stats["cache_hit"] = cache_hit
        stats["decoder"] = decoder
        stats["avatar_id"] = avatar_id
        stats["stride"] = stride
        return stats

    def _run_generation_pipeline(
        self,
        material: AvatarMaterial,
        audio_path: str,
        output_path: Optional[str] = None,
        fps: int = 30,
        decoder: str = "full_vae",
        encoder: str = "libx264",
        mouth_enhancer: str = "unsharp",
        start_frame_offset: int = 0,
        use_silence_bypass: bool = False,
        texture_injection: float = 0.0,
        mask_mode: str = "standard",
        stride: int = 1,
        opts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        opts = opts or {}
        frame_callback = opts.get("frame_callback")

        audio_res = self.audio_processor.get_audio_feature(audio_path, weight_dtype=self.weight_dtype)
        if audio_res is None:
            raise ValueError(f"Audio extraction failed for {audio_path}")
        whisper_input_features, librosa_length, silence_mask = audio_res

        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_input_features, self.device, self.weight_dtype, self.whisper, librosa_length, fps=fps
        )
        max_frames = int(opts.get("max_frames", 8 if self.device.type == "cpu" else 0))
        if max_frames > 0 and len(whisper_chunks) > max_frames:
            whisper_chunks = whisper_chunks[:max_frames]
        video_num = len(whisper_chunks)
        whisper_chunks_tensor = (
            whisper_chunks.to(self.device, dtype=self.weight_dtype)
            if isinstance(whisper_chunks, torch.Tensor)
            else torch.stack(whisper_chunks).to(self.device, dtype=self.weight_dtype)
        )

        with torch.no_grad():
            all_pe_tokens = self.pe(whisper_chunks_tensor)

        N = len(material.geometry_precomputed) if hasattr(material, "geometry_precomputed") else len(material.coords)
        avatar_id_str = str(getattr(material, "avatar_id", "")).lower()
        is_transition = ("_to_" in avatar_id_str) or ("transition" in avatar_id_str)
        if is_transition:
            raw_indices = torch.arange(video_num, device=self.device, dtype=torch.long) + int(start_frame_offset)
            frame_indices = torch.clamp(raw_indices, 0, N - 1)
        else:
            frame_indices = (torch.arange(video_num, device=self.device, dtype=torch.long) + int(start_frame_offset)) % N

        batch_size = 4 if self.device.type == "cpu" else self.active_batch_size
        h, w = material.frames_gpu_tensor.shape[2], material.frames_gpu_tensor.shape[3]

        use_nvenc = (encoder == "nvenc")
        bt709_flags = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
        cfr_flags = ["-g", str(fps), "-keyint_min", str(fps), "-sc_threshold", "0", "-r", str(fps), "-vsync", "cfr", "-fflags", "+genpts"]

        if use_nvenc:
            input_pix_fmt = "nv12"
            v_codec_args = ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll", "-rc", "constqp", "-qp", "18", "-pix_fmt", "yuv420p", *cfr_flags, *bt709_flags]
        else:
            input_pix_fmt = "bgr24"
            v_codec_args = ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-threads", "4", "-crf", "18", "-pix_fmt", "yuv420p", *cfr_flags, *bt709_flags]

        pinned_buffers = get_pinned_buffers(h, w, batch_size, pix_fmt=input_pix_fmt)

        proc = None
        writer_thread = None
        write_queue = None
        writer_error = [None]

        if output_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{w}x{h}",
                "-pix_fmt", input_pix_fmt,
                "-r", str(fps),
                "-i", "-",
                "-i", audio_path,
                *v_codec_args,
                "-c:a", "aac",
                "-shortest",
                output_path,
            ]
            proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10485760)
            write_queue = queue.Queue(maxsize=16)

            def writer_worker():
                while True:
                    item = write_queue.get()
                    if item is None:
                        write_queue.task_done()
                        break
                    buf_idx, cur_len, event = item
                    if event is not None:
                        event.synchronize()
                    try:
                        if proc.stdin and not proc.stdin.closed:
                            proc.stdin.write(memoryview(pinned_buffers[buf_idx][:cur_len].numpy()))
                    except (BrokenPipeError, IOError) as pipe_err:
                        writer_error[0] = pipe_err
                    write_queue.task_done()

            writer_thread = threading.Thread(target=writer_worker, daemon=True)
            writer_thread.start()

        num_batches = int(np.ceil(float(video_num) / batch_size))
        prev_info = None
        pinned_idx = 0

        # Precompute keyframe UNet predictions + SLERP latent infill for Stride > 1
        full_latents = None
        if stride > 1:
            key_indices = list(range(0, video_num, stride))
            if key_indices[-1] != video_num - 1:
                key_indices.append(video_num - 1)
            M = len(key_indices)
            key_indices_t = torch.tensor(key_indices, device=self.device, dtype=torch.long)
            logger.info(f"⚡ [MuseTalk Stride {stride}] Evaluating UNet on {M}/{video_num} keyframes ({(1.0 - float(M)/video_num)*100:.1f}% skipped via SLERP)...")

            num_key_batches = int(np.ceil(float(M) / batch_size))
            key_preds = []
            N_lat = material.latents_gpu_tensor.shape[0]

            for kb in range(num_key_batches):
                kb_s = kb * batch_size
                kb_e = min(kb_s + batch_size, M)
                k_idx = key_indices_t[kb_s:kb_e]

                w_kb = whisper_chunks_tensor[k_idx]
                pe_kb = all_pe_tokens[k_idx]
                b_idx_map = frame_indices[k_idx]
                l_kb = material.latents_gpu_tensor[b_idx_map % N_lat]

                with torch.no_grad():
                    pred_kb = self.unet_runner.run(w_kb, l_kb, pe_batch=pe_kb).detach()
                    key_preds.append(pred_kb)

            key_latents = torch.cat(key_preds, dim=0)
            full_latents = expand_latents_stride_fast(key_latents, stride, video_num)

        try:
            for b_idx in range(num_batches + 1):
                if b_idx < num_batches and (b_idx % 2 == 0 or b_idx == num_batches - 1):
                    logger.info(f"⏳ [MuseTalk Batch] {b_idx + 1}/{num_batches} (frames {b_idx * batch_size}..{min((b_idx + 1) * batch_size, video_num)}/{video_num})...")

                if proc is not None and proc.poll() is not None:
                    _, err_bytes = proc.communicate() if proc.stderr else (b"", b"")
                    err_msg = err_bytes.decode("utf-8", errors="ignore") if err_bytes else f"FFmpeg exited unexpectedly ({proc.returncode})"
                    raise RuntimeError(f"FFmpeg crashed: {err_msg}")

                if writer_error[0] is not None:
                    raise RuntimeError(f"Writer pipe error: {writer_error[0]}")

                if b_idx < num_batches:
                    start_idx = b_idx * batch_size
                    end_idx = min(start_idx + batch_size, video_num)
                    cur_len = end_idx - start_idx

                    if full_latents is not None:
                        curr_pred = full_latents[start_idx:end_idx]
                        curr_event_unet = None
                    else:
                        w_batch = whisper_chunks_tensor[start_idx:end_idx]
                        pe_batch = all_pe_tokens[start_idx:end_idx]
                        b_idx_map = frame_indices[start_idx:end_idx]
                        N_lat = material.latents_gpu_tensor.shape[0]
                        l_batch = material.latents_gpu_tensor[b_idx_map % N_lat]

                        with torch.no_grad():
                            with (torch.cuda.stream(self.stream_unet) if self.stream_unet else nullcontext()):
                                curr_pred = self.unet_runner.run(w_batch, l_batch, pe_batch=pe_batch).detach()
                                if self.stream_unet is not None:
                                    curr_event_unet = torch.cuda.Event()
                                    curr_event_unet.record(self.stream_unet)
                                else:
                                    curr_event_unet = None

                    curr_info = (start_idx, end_idx, cur_len, curr_pred, curr_event_unet)
                else:
                    curr_info = None

                if prev_info is not None:
                    p_start, p_end, p_len, p_pred, p_event = prev_info
                    with (torch.cuda.stream(self.stream_vae) if self.stream_vae else nullcontext()):
                        p_indices = frame_indices[p_start:p_end]
                        p_idx_cpu = p_indices.cpu() if hasattr(p_indices, "is_cuda") and p_indices.is_cuda else p_indices
                        N_frames = material.frames_gpu_tensor.shape[0]
                        batch_bg = material.frames_gpu_tensor[p_idx_cpu % N_frames].to(DEVICE, non_blocking=True).clone()

                        if p_pred is not None:
                            if p_event is not None and self.stream_vae is not None:
                                self.stream_vae.wait_event(p_event)

                            if decoder == "taesd" and self.taesd_runner is not None:
                                recon_faces_gpu = self.taesd_runner.decode_gpu(p_pred)
                            elif self.vae_trt_runner is not None:
                                recon_faces_gpu = self.vae_trt_runner.decode_gpu(p_pred)
                            else:
                                with torch.no_grad():
                                    recon_faces = self.vae.decode_latents(p_pred)
                                recon_faces_gpu = torch.from_numpy(np.ascontiguousarray(recon_faces)).permute(0, 3, 1, 2).float().to(self.device)

                            if mouth_enhancer in ["unsharp", "gfpgan"]:
                                blurred = F.avg_pool2d(recon_faces_gpu, kernel_size=3, stride=1, padding=1)
                                recon_faces_gpu = (recon_faces_gpu + 0.35 * (recon_faces_gpu - blurred)).clamp(0, 255)

                            N_geom = len(material.geometry_precomputed)
                            N_mask = len(material.masks_gpu_precomputed)
                            for f_i in range(p_len):
                                orig_idx = int(p_indices[f_i]) % N_geom
                                x_s, y_s, x_e, y_e, rel_x, rel_y, rel_w, rel_h = material.geometry_precomputed[orig_idx]
                                crop_bg = batch_bg[f_i, :, y_s:y_e, x_s:x_e].float()

                                f_mouth = F.interpolate(recon_faces_gpu[f_i:f_i + 1], size=(rel_h, rel_w), mode="bicubic", align_corners=False)[0]

                                face_large = crop_bg.clone()
                                sub_h = max(0, min(rel_h, crop_bg.shape[1] - rel_y))
                                sub_w = max(0, min(rel_w, crop_bg.shape[2] - rel_x))
                                if sub_h > 0 and sub_w > 0:
                                    face_large[:, rel_y:rel_y + sub_h, rel_x:rel_x + sub_w] = f_mouth[:, :sub_h, :sub_w]

                                m_full = material.masks_gpu_precomputed[orig_idx % N_mask].float()
                                if m_full.shape[-2:] != crop_bg.shape[-2:]:
                                    m_full = F.interpolate(m_full.unsqueeze(0), size=(crop_bg.shape[-2], crop_bg.shape[-1]), mode="bilinear", align_corners=False)[0]

                                blended = crop_bg + m_full * (face_large - crop_bg)
                                batch_bg[f_i, :, y_s:y_e, x_s:x_e] = blended.clamp(0, 255).to(torch.uint8)

                        if use_nvenc:
                            batch_hwc = batch_bg.permute(0, 2, 3, 1).contiguous()
                            batch_out = bgr_to_nv12_gpu(batch_hwc)
                        else:
                            batch_out = batch_bg.permute(0, 2, 3, 1).contiguous()

                    with (torch.cuda.stream(self.stream_dma) if self.stream_dma else nullcontext()):
                        if self.stream_dma is not None and self.stream_vae is not None:
                            self.stream_dma.wait_stream(self.stream_vae)
                        cur_pinned = pinned_buffers[pinned_idx % NUM_PINNED_BUFFERS]
                        cur_pinned[:p_len].copy_(batch_out[:p_len], non_blocking=True if self.stream_dma else False)
                        if self.stream_dma is not None:
                            dma_event = torch.cuda.Event()
                            dma_event.record(self.stream_dma)
                        else:
                            dma_event = None

                        if frame_callback is not None:
                            try:
                                frame_callback({
                                    "chunk_idx": pinned_idx,
                                    "frame_start": p_start,
                                    "num_frames": p_len,
                                    "frames": cur_pinned[:p_len].clone(),
                                    "is_last": (p_end >= video_num),
                                    "fps": fps,
                                })
                            except Exception as cb_err:
                                logger.warning(f"[MuseTalk] frame_callback error: {cb_err}")

                        if write_queue is not None:
                            write_queue.put((pinned_idx % NUM_PINNED_BUFFERS, p_len, dma_event))
                        pinned_idx += 1

                prev_info = curr_info
                if self.device.type == "cpu":
                    gc.collect()

            if write_queue is not None:
                write_queue.put(None)
            if writer_thread is not None:
                writer_thread.join(timeout=30.0)
            if proc is not None:
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.close()
                proc.wait(timeout=30.0)
        except Exception as e:
            try:
                if write_queue is not None:
                    write_queue.put(None)
                if proc is not None and proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=5.0)
            except Exception:
                pass
            raise e

        total_dur = time.perf_counter() - t_start
        fps_calc = video_num / max(total_dur, 0.001)
        return {
            "frames": video_num,
            "inference_duration_sec": round(total_dur, 2),
            "fps": round(fps_calc, 1),
            "stride": stride,
            "output_path": output_path,
        }


_GLOBAL_MUSETALK_ENGINE: Optional[MuseTalkEngine] = None
_GLOBAL_ENGINE_LOCK = threading.Lock()

def get_global_musetalk_engine(weights_dir: Optional[str] = None) -> MuseTalkEngine:
    global _GLOBAL_MUSETALK_ENGINE
    if _GLOBAL_MUSETALK_ENGINE is None:
        with _GLOBAL_ENGINE_LOCK:
            if _GLOBAL_MUSETALK_ENGINE is None:
                _GLOBAL_MUSETALK_ENGINE = MuseTalkEngine(weights_dir)
    return _GLOBAL_MUSETALK_ENGINE

def unload_global_musetalk_engine():
    global _GLOBAL_MUSETALK_ENGINE
    with _GLOBAL_ENGINE_LOCK:
        if _GLOBAL_MUSETALK_ENGINE is not None:
            logger.info("[MuseTalkEngine] Unloading global MuseTalk engine from VRAM...")
            avatar_pool.clear()
            del _GLOBAL_MUSETALK_ENGINE
            _GLOBAL_MUSETALK_ENGINE = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            logger.info("[MuseTalkEngine] ✅ MuseTalk engine unloaded.")


def stream_musetalk_generator(
    avatar_id: str,
    video_or_image_path: str,
    audio_path: str,
    fps: int = 30,
    opts: Optional[Dict[str, Any]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Real-Time Conversational Streaming Generator for MuseTalk.
    Yields frame batches in real-time as UNet + TAESD/VAE decode progresses,
    enabling sub-frame streaming to WebSockets, WebRTC tracks, or chunked HTTP responses.

    Yields:
        {
            "chunk_idx": int,
            "frame_start": int,
            "num_frames": int,
            "frames": torch.Tensor, # on CPU
            "is_last": bool,
            "fps": int
        }
    """
    engine = get_global_musetalk_engine()
    frame_q: queue.Queue = queue.Queue(maxsize=16)
    error_box = [None]

    def worker():
        try:
            def on_frame(chunk):
                frames_cpu = chunk["frames"].cpu() if hasattr(chunk["frames"], "cpu") else chunk["frames"]
                chunk_copy = dict(chunk)
                chunk_copy["frames"] = frames_cpu
                frame_q.put(chunk_copy)

            opts_copy = dict(opts or {})
            opts_copy["frame_callback"] = on_frame
            engine.generate(
                avatar_id=avatar_id,
                video_or_image_path=video_or_image_path,
                audio_path=audio_path,
                output_path=None,
                fps=fps,
                opts=opts_copy,
            )
        except Exception as err:
            error_box[0] = err
        finally:
            frame_q.put(None)

    th = threading.Thread(target=worker, daemon=True)
    th.start()

    while True:
        item = frame_q.get()
        if item is None:
            frame_q.task_done()
            break
        yield item
        frame_q.task_done()

    th.join()
    if error_box[0] is not None:
        raise error_box[0]


def precompile_avatar_source(
    source_path: str,
    avatar_id: str,
    vae=None,
    fp=None,
    out_root: str = "/app/results/v15/avatars",
    mirror_storage_root: str = "/app/storage/cache/avatars",
    cycle_frames: int = 25,
    extra_margin: int = 10,
    bbox_shift: int = 0,
    aliases: Optional[List[str]] = None,
) -> bool:
    """
    Offline Master Asset Precompilation Protocol.
    Generates the complete 6-file master asset packet on disk for <10ms instant startup.
    """
    from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder
    from musetalk.utils.face_parsing import FaceParsing

    logger.info(f"🚀 [Precompile] Starting Master Precompilation for '{avatar_id}': {source_path}")
    t0 = time.perf_counter()

    avatar_dir = os.path.join(out_root, avatar_id)
    os.makedirs(avatar_dir, exist_ok=True)
    mask_dir = os.path.join(avatar_dir, "mask")
    os.makedirs(mask_dir, exist_ok=True)

    if vae is None or fp is None:
        engine = get_global_musetalk_engine()
        vae = vae or engine.vae
        fp = fp or engine.face_parser

    is_image = source_path.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))

    if is_image:
        img = cv2.imread(source_path)
        if img is None:
            raise ValueError(f"Could not load image: {source_path}")
        h, w = img.shape[:2]
        if h % 2 != 0 or w % 2 != 0:
            img = cv2.resize(img, (w - (w % 2), h - (h % 2)))
            h, w = img.shape[:2]

        res_bbox = get_landmark_and_bbox([img], upperbondrange=bbox_shift, batch_size_fa=1)
        if res_bbox is None or len(res_bbox[0]) == 0 or res_bbox[0][0] == coord_placeholder:
            logger.warning(f"[Precompile Warning] No face detected in {source_path}")
            return False

        coord_list, _ = res_bbox
        x1, y1, x2, y2 = coord_list[0]
        y2 = min(int(y2) + extra_margin, h)
        coords = [int(x1), int(y1), int(x2), int(y2)]

        crop_256 = cv2.resize(img[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
        with torch.no_grad():
            lat_8ch = vae.get_latents_for_unet(crop_256).to(dtype=WEIGHT_DTYPE)

        x_s, y_s, x_e, y_e, rel_x, rel_y, rel_w, rel_h = compute_avatar_crop_geometry(x1, y1, x2, y2, w, h, expand=1.5)
        crop_w, crop_h = x_e - x_s, y_e - y_s
        geom_tuple = (x_s, y_s, x_e, y_e, rel_x, rel_y, rel_w, rel_h)

        f_crop = img[y_s:y_e, x_s:x_e]
        exp_512 = cv2.resize(f_crop, (512, 512))
        b_t = torch.from_numpy(exp_512).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
        g_kernel = get_gaussian_kernel(15, 3.0, DEVICE)
        with torch.no_grad():
            b_m = fp.parse_batch_gpu(b_t, mode="raw").half()
            b_m[:, :, :256, :] = 0.0
            b_m_blurred = F.conv2d(b_m, g_kernel, padding=7)
            m_res = F.interpolate(b_m_blurred[0:1], size=(crop_h, crop_w), mode="bilinear", align_corners=False)[0]

        mask_single = m_res.half()

        frame_tensor_single = torch.from_numpy(img).permute(2, 0, 1)
        frames_tensor = frame_tensor_single.unsqueeze(0).repeat(cycle_frames, 1, 1, 1)
        latents_tensor = lat_8ch.repeat(cycle_frames, 1, 1, 1)
        masks_list = [mask_single.cpu() for _ in range(cycle_frames)]
        coords_list = [coords for _ in range(cycle_frames)]
        geometry_list = [geom_tuple for _ in range(cycle_frames)]
    else:
        cap = cv2.VideoCapture(source_path)
        raw_frames = []
        while cap.isOpened():
            ret, f = cap.read()
            if not ret:
                break
            raw_frames.append(f)
        cap.release()

        if not raw_frames:
            raise ValueError(f"Could not read video frames: {source_path}")

        raw_frames = raw_frames + raw_frames[::-1]
        total_cycle = len(raw_frames)
        coords_list, raw_frames = get_landmark_and_bbox(raw_frames, bbox_shift, batch_size_fa=16)

        H_f, W_f = raw_frames[0].shape[:2]
        geometry_list = []
        crops_256 = []
        expanded_crops_512 = []

        for i in range(total_cycle):
            x1, y1, x2, y2 = coords_list[i]
            x_s, y_s, x_e, y_e, rel_x, rel_y, rel_w, rel_h = compute_avatar_crop_geometry(x1, y1, x2, y2, W_f, H_f, expand=1.5)
            geometry_list.append((x_s, y_s, x_e, y_e, rel_x, rel_y, rel_w, rel_h))

            slice_img = raw_frames[i][y1:y2, x1:x2]
            slice_img = cv2.resize(slice_img, (256, 256), interpolation=cv2.INTER_LANCZOS4) if slice_img.size > 0 else cv2.resize(raw_frames[i], (256, 256))
            crops_256.append(slice_img)

            f_crop = raw_frames[i][y_s:y_e, x_s:x_e]
            exp_512 = cv2.resize(f_crop, (512, 512)) if f_crop.size > 0 else cv2.resize(raw_frames[i], (512, 512))
            expanded_crops_512.append(exp_512)

        lats_batches = []
        masks_list = []
        g_kernel = get_gaussian_kernel(15, 3.0, DEVICE)

        for b in range(0, total_cycle, 32):
            b_crops = np.stack(crops_256[b:b+32])
            b_crops_rgb = b_crops[:, :, :, [2, 1, 0]]
            b_t = torch.from_numpy(b_crops_rgb).permute(0, 3, 1, 2).float().to(DEVICE) / 127.5 - 1.0
            lat = vae.encode_latents_batch(b_t)
            lats_batches.append(lat)

            b_exp = np.stack(expanded_crops_512[b:b+32])
            b_t_exp = torch.from_numpy(b_exp).permute(0, 3, 1, 2).float().to(DEVICE) / 255.0
            with torch.no_grad():
                b_m = fp.parse_batch_gpu(b_t_exp, mode="raw").half()
                b_m[:, :, :256, :] = 0.0
                b_m_blurred = F.conv2d(b_m, g_kernel, padding=7)
                for i_m in range(b_m_blurred.shape[0]):
                    orig_idx = b + i_m
                    x_s, y_s, x_e, y_e, rel_x, rel_y, rel_w, rel_h = geometry_list[orig_idx]
                    crop_w, crop_h = x_e - x_s, y_e - y_s
                    m_res = F.interpolate(b_m_blurred[i_m:i_m+1], size=(crop_h, crop_w), mode="bilinear", align_corners=False)[0]
                    masks_list.append(m_res.cpu().half())

        frames_tensor = torch.stack([torch.from_numpy(f).permute(2, 0, 1) for f in raw_frames])
        raw_lat = torch.cat(lats_batches, dim=0).to(dtype=WEIGHT_DTYPE)
        if raw_lat.shape[1] == 4:
            m_lat = raw_lat.clone()
            m_lat[:, :, 16:, :] = 0.0
            latents_tensor = torch.cat([m_lat, raw_lat], dim=1).cpu()
        else:
            latents_tensor = raw_lat.cpu()

    # Save to disk
    import pickle
    torch.save(frames_tensor.cpu(), os.path.join(avatar_dir, "frames.pt"))
    torch.save(latents_tensor.cpu(), os.path.join(avatar_dir, "latents.pt"))
    torch.save(masks_list, os.path.join(avatar_dir, "masks.pt"))
    with open(os.path.join(avatar_dir, "geometry.pkl"), "wb") as f:
        pickle.dump(geometry_list, f)
    with open(os.path.join(avatar_dir, "coords.pkl"), "wb") as f:
        pickle.dump(coords_list, f)
    with open(os.path.join(avatar_dir, "mask_coords.pkl"), "wb") as f:
        pickle.dump(coords_list, f)

    meta = {
        "avatar_id": avatar_id,
        "source": source_path,
        "cycle_frames": len(geometry_list),
        "precompiled_at": time.time(),
        "duration_sec": round(time.perf_counter() - t0, 2),
    }
    with open(os.path.join(avatar_dir, "avator_info.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Mirror to storage cache
    if mirror_storage_root:
        mirror_dir = os.path.join(mirror_storage_root, avatar_id)
        os.makedirs(mirror_dir, exist_ok=True)
        torch.save(frames_tensor.cpu(), os.path.join(mirror_dir, "frames.pt"))
        torch.save(latents_tensor.cpu(), os.path.join(mirror_dir, "latents.pt"))
        torch.save(masks_list, os.path.join(mirror_dir, "masks.pt"))
        with open(os.path.join(mirror_dir, "geometry.pkl"), "wb") as f:
            pickle.dump(geometry_list, f)
        with open(os.path.join(mirror_dir, "coords.pkl"), "wb") as f:
            pickle.dump(coords_list, f)

    logger.info(f"✅ [Precompile] Finished master packet for '{avatar_id}' in {time.perf_counter() - t0:.2f}s!")
    return True
