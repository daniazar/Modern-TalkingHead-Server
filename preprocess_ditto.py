"""
vendor/Modern-TalkingHead-Server/preprocess_ditto.py

Offline Master Asset Preprocessor for Ditto Talking Head Engine.
Consolidates the complete 2-step offline preprocessing protocol for NewsStudio:
Pre-computes all 3D volumetric appearance features, 3DMM facial keypoints,
normalized GPU affine sampling grids, warped alpha masks, and background video frames.

Outputs the 6-File Master Asset Packet to:
  /app/cache/avatars/ditto/{avatar_id}/ (or storage/cache/avatars/ditto/{avatar_id}/)

Artifacts generated:
  1. f_s.pt         - (N, 32, 16, 64, 64) float16 appearance feature volume
  2. x_s_info.pt    - Dict of stacked 3DMM landmarks (kp, pitch, yaw, roll, t, exp)
  3. grids.pt       - (N, H, W, 2) float16 normalized GPU affine sampling grids
  4. masks.pt       - (N, 1, H, W) float16 pre-warped alpha blending masks
  5. frames.pt      - (N, 3, H, W) uint8 uncompressed background frames
  6. ditto_info.json- Metadata manifest (dimensions, frame count, FPS, eye baseline, ROI)
"""

import os
import sys
import gc
import json
import time
import argparse
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger("ModernTalkingHead.PreprocessDitto")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

DITTO_REPO = "/app/repos/Ditto" if os.path.exists("/app/repos/Ditto") else os.path.join(os.getcwd(), "repos", "Ditto")
if os.path.exists(DITTO_REPO) and DITTO_REPO not in sys.path:
    sys.path.insert(0, DITTO_REPO)

DITTO_WEIGHTS = "/app/weights/ditto" if os.path.exists("/app/weights/ditto") else os.path.join(os.getcwd(), "weights", "ditto")


def compute_roi_box(M_c2o_lst, H: int, W: int, src_h: int = 512, src_w: int = 512, pad: int = 32) -> Tuple[int, int, int, int]:
    """
    Computes a tight bounding box envelope around all warped face crops across all frames.
    Forces mod-2 alignment for clean 4:2:0 subsampling with zero phase offset.
    """
    corners_src = np.array([
        [0, 0, 1],
        [src_w, 0, 1],
        [src_w, src_h, 1],
        [0, src_h, 1]
    ], dtype=np.float32).T  # (3, 4)

    min_x, min_y = 1e9, 1e9
    max_x, max_y = -1e9, -1e9

    for M_c2o in M_c2o_lst:
        pts = M_c2o @ corners_src  # (2, 4)
        min_x = min(min_x, np.min(pts[0]))
        max_x = max(max_x, np.max(pts[0]))
        min_y = min(min_y, np.min(pts[1]))
        max_y = max(max_y, np.max(pts[1]))

    # Add margin and clamp to video frame dimensions
    x_min = max(0, int(np.floor(min_x)) - pad)
    y_min = max(0, int(np.floor(min_y)) - pad)
    x_max = min(W, int(np.ceil(max_x)) + pad)
    y_max = min(H, int(np.ceil(max_y)) + pad)

    # Force mod-2 alignment for 4:2:0 YUV chroma consistency
    x_min = x_min & ~1
    y_min = y_min & ~1
    x_max = (x_max + 1) & ~1
    y_max = (y_max + 1) & ~1

    return (y_min, y_max, x_min, x_max)


def preprocess_avatar_ditto(
    video_path: str,
    avatar_id: str,
    cache_pkl: Optional[str] = None,
    output_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Executes offline master asset pre-processing for a Ditto avatar loop.
    Extracts and compiles all 6 master packet files.
    """
    t_start = time.perf_counter()
    clean_id = avatar_id.lower().strip().replace(" ", "_")

    if output_dir is None:
        if os.path.exists("/app/cache"):
            target_dir = Path("/app/cache/avatars/ditto") / clean_id
        else:
            target_dir = Path(os.getcwd()) / "storage" / "cache" / "avatars" / "ditto" / clean_id
    else:
        target_dir = Path(output_dir) / clean_id

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"🎬 [Ditto Preprocessor] Starting offline preprocessing for avatar '{clean_id}'...")
    logger.info(f"   Target directory: {target_dir}")
    logger.info(f"   Video input:      {video_path}")

    # Determine execution device
    use_cuda = (device.startswith("cuda") and torch.cuda.is_available())
    calc_device = "cuda:0" if use_cuda else "cpu"

    # 1. Acquire source_info
    source_info = None
    if cache_pkl and os.path.exists(cache_pkl):
        logger.info(f"   [1/6] Loading source_info from {cache_pkl}...")
        with open(cache_pkl, "rb") as f:
            source_info = pickle.load(f)
    else:
        logger.info(f"   [1/6] Running AvatarRegistrar on {video_path}...")
        t0 = time.perf_counter()
        from core.atomic_components.avatar_registrar import AvatarRegistrar
        from core.atomic_components.cfg import parse_cfg
        cfg_pkl = os.path.join(DITTO_WEIGHTS, "ditto_cfg", "v0.4_hubert_cfg_pytorch.pkl")
        data_root = os.path.join(DITTO_WEIGHTS, "ditto_pytorch")
        [avatar_registrar_cfg, _, _, _, _, _, _, _] = parse_cfg(cfg_pkl, data_root, {})
        registrar = AvatarRegistrar(**avatar_registrar_cfg)
        source_info = registrar(video_path)
        logger.info(f"   AvatarRegistrar completed in {time.perf_counter() - t0:.2f}s")

    N = len(source_info["x_s_info_lst"])
    M_c2o_lst = source_info["M_c2o_lst"]
    x_s_info_lst = source_info["x_s_info_lst"]
    logger.info(f"   Total video loop frames: {N}")

    # 2. Extract 3D Appearance Feature Volume (f_s.pt)
    f_s_path = target_dir / "f_s.pt"
    if f_s_path.exists() and f_s_path.stat().st_size > 0:
        logger.info(f"   [2/6] f_s.pt already exists, skipping...")
    else:
        logger.info(f"   [2/6] Compiling 3D Appearance Feature Volume (f_s.pt)...")
        t0 = time.perf_counter()
        f_s_stacked = np.concatenate(source_info["f_s_lst"], axis=0)  # (N, 32, 16, 64, 64)
        f_s_tensor = torch.from_numpy(f_s_stacked).to(dtype=torch.float16)
        torch.save(f_s_tensor, str(f_s_path), _use_new_zipfile_serialization=True)
        del f_s_stacked, f_s_tensor
        gc.collect()
        logger.info(f"   ✅ Saved f_s.pt ({f_s_path.stat().st_size / (1024 * 1024):.1f} MB in {time.perf_counter() - t0:.2f}s)")

    # 3. Stacked 3DMM Keypoints (x_s_info.pt)
    x_s_path = target_dir / "x_s_info.pt"
    if x_s_path.exists() and x_s_path.stat().st_size > 0:
        logger.info(f"   [3/6] x_s_info.pt already exists, skipping...")
    else:
        logger.info(f"   [3/6] Compiling Stacked 3DMM Facial Keypoints (x_s_info.pt)...")
        t0 = time.perf_counter()
        x_s_dict = {
            "pitch": torch.from_numpy(np.concatenate([item["pitch"] for item in x_s_info_lst], axis=0)).to(torch.float32),
            "yaw": torch.from_numpy(np.concatenate([item["yaw"] for item in x_s_info_lst], axis=0)).to(torch.float32),
            "roll": torch.from_numpy(np.concatenate([item["roll"] for item in x_s_info_lst], axis=0)).to(torch.float32),
            "t": torch.from_numpy(np.concatenate([item["t"] for item in x_s_info_lst], axis=0)).to(torch.float32),
            "exp": torch.from_numpy(np.concatenate([item["exp"] for item in x_s_info_lst], axis=0)).to(torch.float32),
            "scale": torch.from_numpy(np.concatenate([item["scale"] for item in x_s_info_lst], axis=0)).to(torch.float32),
            "kp": torch.from_numpy(np.concatenate([item["kp"] for item in x_s_info_lst], axis=0)).to(torch.float32),
            "raw_list": x_s_info_lst,
            "M_c2o_lst": source_info.get("M_c2o_lst"),
            "sc": source_info.get("sc"),
            "eye_open_lst": source_info.get("eye_open_lst"),
            "eye_ball_lst": source_info.get("eye_ball_lst"),
        }
        torch.save(x_s_dict, str(x_s_path), _use_new_zipfile_serialization=True)
        del x_s_dict
        gc.collect()
        logger.info(f"   ✅ Saved x_s_info.pt in {time.perf_counter() - t0:.2f}s")

    # Release raw f_s_lst from source_info to reclaim RAM
    if "f_s_lst" in source_info:
        del source_info["f_s_lst"]
    gc.collect()

    # 4. Extract Background Video Frames (frames.pt)
    frames_path = target_dir / "frames.pt"
    is_image = video_path.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    if is_image:
        img_bgr = cv2.imread(video_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read input image from {video_path}")
        H, W = img_bgr.shape[:2]
        fps = 25.0
        cap = None
    else:
        cap = cv2.VideoCapture(video_path)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if frames_path.exists() and frames_path.stat().st_size > 0:
        logger.info(f"   [4/6] frames.pt already exists, skipping extraction...")
        if cap is not None:
            cap.release()
    else:
        logger.info(f"   [4/6] Extracting Background Frames (frames.pt, is_image={is_image})...")
        t0 = time.perf_counter()
        if is_image:
            frame_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            frames_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0)
        else:
            frames_tensor = torch.empty((N, 3, H, W), dtype=torch.uint8)
            idx = 0
            while idx < N:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames_tensor[idx] = torch.from_numpy(frame_rgb).permute(2, 0, 1)
                idx += 1
            cap.release()
        torch.save(frames_tensor, str(frames_path), _use_new_zipfile_serialization=True)
        size_frames_mb = frames_path.stat().st_size / (1024 * 1024)
        del frames_tensor
        gc.collect()
        logger.info(f"   ✅ Saved frames.pt: ({size_frames_mb:.1f} MB in {time.perf_counter() - t0:.2f}s)")

    # 5. Pre-compute Affine Sampling Grids (grids.pt) & Warped Alpha Masks (masks.pt)
    grids_path = target_dir / "grids.pt"
    masks_path = target_dir / "masks.pt"
    src_h, src_w = 512, 512

    # Calculate ROI envelope across all frames
    roi_box = compute_roi_box(M_c2o_lst, H, W, src_h, src_w)
    logger.info(f"   Computed Face ROI Envelope: y=[{roi_box[0]}, {roi_box[1]}], x=[{roi_box[2]}, {roi_box[3]}]")

    if grids_path.exists() and masks_path.exists() and grids_path.stat().st_size > 0:
        logger.info(f"   [5/6] grids.pt and masks.pt already exist, skipping...")
    else:
        logger.info(f"   [5/6] Pre-computing GPU Affine Grids & Alpha Masks...")
        t0 = time.perf_counter()

        y_dst, x_dst = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing='ij')
        dst_pts = np.stack([x_dst, y_dst, np.ones_like(x_dst)], axis=-1)  # (H, W, 3)
        dst_pts_gpu = torch.from_numpy(dst_pts).to(device=calc_device, dtype=torch.float32)

        from core.atomic_components.putback import get_mask
        mask_raw = get_mask(src_h, src_w, 0.9, 0.9).astype(np.float32)
        if mask_raw.ndim == 3:
            mask_raw = mask_raw[:, :, 0]
        mask_tensor = torch.from_numpy(mask_raw)[None, None].to(device=calc_device, dtype=torch.float16)

        grids_tensor = torch.empty((N, H, W, 2), dtype=torch.float16)
        masks_tensor = torch.empty((N, 1, H, W), dtype=torch.float16)

        for i, M_c2o in enumerate(M_c2o_lst):
            M_3x3 = np.eye(3, dtype=np.float32)
            M_3x3[:2, :] = M_c2o[:2, :]
            M_inv = torch.from_numpy(np.linalg.inv(M_3x3)[:2, :].T).to(device=calc_device, dtype=torch.float32)

            src_pts = dst_pts_gpu @ M_inv
            x_norm = ((2.0 * src_pts[:, :, 0] + 1.0) / float(src_w) - 1.0).to(torch.float16)
            y_norm = ((2.0 * src_pts[:, :, 1] + 1.0) / float(src_h) - 1.0).to(torch.float16)
            grid = torch.stack([x_norm, y_norm], dim=-1)[None]

            with torch.no_grad():
                mask_w = torch.nn.functional.grid_sample(
                    mask_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=False
                )

            grids_tensor[i] = grid[0].cpu()
            masks_tensor[i] = mask_w[0].cpu()

        del dst_pts_gpu, mask_tensor
        gc.collect()

        torch.save(grids_tensor, str(grids_path), _use_new_zipfile_serialization=True)
        torch.save(masks_tensor, str(masks_path), _use_new_zipfile_serialization=True)
        del grids_tensor, masks_tensor
        gc.collect()
        logger.info(f"   ✅ Saved grids.pt ({grids_path.stat().st_size / (1024 * 1024):.1f} MB) and masks.pt in {time.perf_counter() - t0:.2f}s")

    # 6. Save ditto_info.json Manifest
    manifest_path = target_dir / "ditto_info.json"
    manifest = {
        "avatar_id": clean_id,
        "source_video": video_path,
        "is_image": is_image,
        "total_frames": N,
        "width": W,
        "height": H,
        "fps": float(fps),
        "src_resolution": [src_w, src_h],
        "roi_box": list(roi_box),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "eye_baseline": {
            "eye_f0_mode": True,
            "delta_eye_open_n": -1,
            "calm_blink_enabled": False,
        },
        "files": {
            "f_s": "f_s.pt",
            "x_s_info": "x_s_info.pt",
            "grids": "grids.pt",
            "masks": "masks.pt",
            "frames": "frames.pt",
            "manifest": "ditto_info.json",
        }
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"   ✅ Saved ditto_info.json manifest")

    total_dur = time.perf_counter() - t_start
    logger.info(f"🎉 [Ditto Preprocessor] Complete in {total_dur:.2f}s for avatar '{clean_id}'")

    return {
        "avatar_id": clean_id,
        "engine": "ditto",
        "cached": True,
        "cache_hit": False,
        "cache_dir": str(target_dir),
        "total_frames": N,
        "fps": float(fps),
        "resolution": f"{W}x{H}",
        "roi_box": list(roi_box),
        "manifest": manifest,
        "duration_seconds": round(total_dur, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ditto Master Avatar Preprocessor")
    parser.add_argument("--video_path", type=str, required=True, help="Input video loop path")
    parser.add_argument("--avatar_id", type=str, required=True, help="Avatar ID")
    parser.add_argument("--cache_pkl", type=str, default=None, help="Pre-computed source_info.pkl")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    args = parser.parse_args()

    res = preprocess_avatar_ditto(
        video_path=args.video_path,
        avatar_id=args.avatar_id,
        cache_pkl=args.cache_pkl,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(json.dumps(res, indent=2))
