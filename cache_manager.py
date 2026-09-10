import os
import sys
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TalkingHeadCacheManager")

# Base directories resolution (supports container environment and host development)
CONTAINER_MODE = os.path.exists("/app")
BASE_DIR = Path("/app") if CONTAINER_MODE else Path(__file__).resolve().parent.parent.parent
REPOS_DIR = Path(os.getenv("REPOS_DIR", "/app/repos" if CONTAINER_MODE else BASE_DIR / "storage" / "models" / "talking_heads" / "repos"))
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", "/app/weights" if CONTAINER_MODE else BASE_DIR / "storage" / "models" / "talking_heads" / "weights"))
HF_CACHE_DIR = Path(os.getenv("HF_HOME", "/app/hf_cache" if CONTAINER_MODE else BASE_DIR / "storage" / "models" / "talking_heads" / "hf_cache"))

UPSTREAM_REPOS: Dict[str, Dict[str, Any]] = {
    "musetalk": {
        "name": "MuseTalk",
        "url": "https://github.com/TMElyralab/MuseTalk.git",
        "folder": "MuseTalk",
        "hf_model_id": "TMElyralab/MuseTalk",
    },
    "avtr1": {
        "name": "AVTR-1",
        "url": "https://github.com/avaturn-live/avtr-1.git",
        "folder": "AVTR-1",
        "hf_model_id": "bentay85/avtr-1-tensorrt-ada",
    },
    "ditto": {
        "name": "Ditto-TalkingHead",
        "url": "https://github.com/PKU-YuanGroup/Ditto-TalkingHead.git",
        "folder": "Ditto",
        "hf_model_id": "digital-avatar/ditto-talkinghead",
    },
    "float": {
        "name": "FLOAT",
        "url": "https://github.com/deepbrainai-research/float.git",
        "folder": "FLOAT",
        "hf_model_id": "yuvraj108c/float",
    },
    "fantasytalking2": {
        "name": "FantasyTalking2",
        "url": "https://github.com/Fantasy-AMAP/fantasy-talking.git",
        "folder": "FantasyTalking2",
        "hf_model_id": "acvlab/FantasyTalking",
    },
    "hallo4": {
        "name": "Hallo4",
        "url": "https://github.com/fudan-generative-vision/hallo.git",
        "folder": "Hallo4",
        "hf_model_id": "fudan-generative-vision/hallo",
    },
    "personalive": {
        "name": "PersonaLive",
        "url": "https://github.com/GVCLab/PersonaLive.git",
        "folder": "PersonaLive",
        "hf_model_id": "huaichang/PersonaLive",
    },
    "syncanimation": {
        "name": "SyncAnimation",
        "url": "https://github.com/ZiqiaoPeng/SyncTalk.git",
        "folder": "SyncAnimation",
        "hf_model_id": "camenduru/SyncTalk",
    },
    "echomimicv3": {
        "name": "EchoMimicV3",
        "url": "https://github.com/antgroup/echomimic_v3.git",
        "folder": "EchoMimicV3",
        "hf_model_id": "BadToBest/EchoMimicV3",
    },
}


def get_dir_size_mb(path: Path) -> float:
    """Calculates directory size in Megabytes."""
    if not path.exists():
        return 0.0
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception:
        pass
    return round(total / (1024 * 1024), 2)


def get_cache_status() -> Dict[str, Any]:
    """Inspects repos and weights cache status across directories."""
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    status: Dict[str, Any] = {
        "storage": {
            "repos_dir": str(REPOS_DIR),
            "weights_dir": str(WEIGHTS_DIR),
            "hf_cache_dir": str(HF_CACHE_DIR),
            "total_repos_mb": get_dir_size_mb(REPOS_DIR),
            "total_weights_mb": get_dir_size_mb(WEIGHTS_DIR),
            "total_hf_cache_mb": get_dir_size_mb(HF_CACHE_DIR),
        },
        "models": {},
    }

    for model_id, info in UPSTREAM_REPOS.items():
        repo_path = REPOS_DIR / info["folder"]
        alt_repo_path = BASE_DIR / "vendor" / info["folder"]
        weights_path = WEIGHTS_DIR / model_id

        is_cloned = (repo_path.exists() and any(repo_path.iterdir())) or (
            alt_repo_path.exists() and any(alt_repo_path.iterdir())
        )
        active_repo = str(repo_path if repo_path.exists() else alt_repo_path)

        weights_files = [f for f in weights_path.rglob("*") if f.is_file() and f.suffix.lower() in (".safetensors", ".pth", ".pt", ".bin", ".onnx", ".engine", ".ckpt")] if weights_path.exists() else []
        has_weights = len(weights_files) > 0

        if model_id == "musetalk" and not has_weights:
            alt_musetalk = Path("/app/models")
            if alt_musetalk.exists() and any(alt_musetalk.iterdir()):
                weights_path = alt_musetalk
                has_weights = True
            else:
                host_musetalk = BASE_DIR / "vendor" / "MuseTalk" / "models"
                if host_musetalk.exists() and any(host_musetalk.iterdir()):
                    weights_path = host_musetalk
                    has_weights = True

        status["models"][model_id] = {
            "name": info["name"],
            "url": info["url"],
            "folder": info["folder"],
            "repo_cloned": is_cloned,
            "repo_path": active_repo if is_cloned else None,
            "repo_size_mb": get_dir_size_mb(Path(active_repo)) if is_cloned else 0.0,
            "weights_cached": has_weights,
            "weights_path": str(weights_path) if has_weights else None,
            "weights_size_mb": get_dir_size_mb(weights_path) if has_weights else 0.0,
            "hf_model_id": info.get("hf_model_id"),
        }

    return status


def clone_upstream_repo(model_id: str, depth: int = 1) -> bool:
    """Clones an upstream repository if not already cached."""
    if model_id not in UPSTREAM_REPOS:
        logger.error(f"Unknown model_id: {model_id}")
        return False

    info = UPSTREAM_REPOS[model_id]
    target_path = REPOS_DIR / info["folder"]
    alt_path = BASE_DIR / "vendor" / info["folder"]

    if (target_path.exists() and any(target_path.iterdir())) or (alt_path.exists() and any(alt_path.iterdir())):
        logger.info(f"✓ Repo '{info['name']}' already cached at {target_path if target_path.exists() else alt_path}")
        return True

    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"⬇ Cloning {info['name']} from {info['url']} (shallow depth={depth})...")

    cmd = ["git", "clone", "--depth", str(depth), info["url"], str(target_path)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if res.returncode == 0:
            logger.info(f"✓ Successfully cached {info['name']} into {target_path}")
            return True
        else:
            logger.warning(f"Git clone failed for {info['name']}: {res.stderr.strip()}")
            return False
    except Exception as e:
        logger.warning(f"Failed to clone {info['name']}: {e}")
        return False


def clone_all_repos(depth: int = 1) -> Dict[str, bool]:
    """Clones all registered upstream repos into the cache directory."""
    results = {}
    for model_id in UPSTREAM_REPOS:
        results[model_id] = clone_upstream_repo(model_id, depth=depth)
    return results


def download_huggingface_weights(model_id: str) -> bool:
    """Downloads model weights via huggingface_hub into the persistent HF_CACHE_DIR or WEIGHTS_DIR."""
    if model_id not in UPSTREAM_REPOS:
        logger.error(f"Unknown model_id: {model_id}")
        return False

    info = UPSTREAM_REPOS[model_id]
    hf_id = info.get("hf_model_id")
    if not hf_id:
        logger.info(f"No default HuggingFace ID for {model_id}.")
        return False

    target_weights = WEIGHTS_DIR / model_id
    target_weights.mkdir(parents=True, exist_ok=True)

    # Guard: Stop downloading if free disk space drops below 50 GB
    try:
        usage = shutil.disk_usage(WEIGHTS_DIR)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < 50.0:
            logger.error(
                f"🛑 DISK SAFEGUARD TRIGGERED: Free space is {free_gb:.1f} GB (< 50 GB threshold). "
                f"Halting download of '{model_id}' to protect disk volume."
            )
            return False
    except Exception as de:
        logger.warning(f"Could not verify disk space: {de}")

    try:
        from huggingface_hub import snapshot_download
        logger.info(f"⬇ Downloading HuggingFace weights for {info['name']} ({hf_id})...")
        token = os.getenv("HF_TOKEN")
        snapshot_download(
            repo_id=hf_id,
            local_dir=str(target_weights),
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4,
            token=token,
        )
        logger.info(f"✓ Model weights downloaded to {target_weights}")

        # Check if any .zip files were downloaded (e.g. SyncTalk) and extract them
        import zipfile
        for zf in target_weights.glob("*.zip"):
            logger.info(f"Extracting archive {zf.name}...")
            try:
                with zipfile.ZipFile(zf, "r") as zip_ref:
                    zip_ref.extractall(target_weights)
                logger.info(f"✓ Successfully extracted {zf.name}")
            except Exception as ze:
                logger.warning(f"Failed to extract {zf.name}: {ze}")

        return True
    except ImportError:
        logger.warning("huggingface_hub library not installed.")
        return False
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "gated" in err_msg.lower() or "restricted" in err_msg.lower():
            logger.warning(f"Repository {hf_id} for '{model_id}' is gated. Set HF_TOKEN environment variable with approved access to download.")
        else:
            logger.warning(f"Failed to download weights for {model_id}: {e}")
        return False


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Talking Head Repo & Model Cache Manager")
    parser.add_argument("--status", action="store_true", help="Print current cache status")
    parser.add_argument("--clone", type=str, help="Clone specific repo or 'all'")
    parser.add_argument("--download", type=str, help="Download weights for specific model or 'all'")
    args = parser.parse_args()

    if args.status or (not args.clone and not args.download):
        status = get_cache_status()
        print(json.dumps(status, indent=2))

    if args.clone:
        if args.clone.lower() == "all":
            clone_all_repos()
        else:
            clone_upstream_repo(args.clone.lower())

    if args.download:
        if args.download.lower() == "all":
            for m in UPSTREAM_REPOS:
                download_huggingface_weights(m)
        else:
            download_huggingface_weights(args.download.lower())
