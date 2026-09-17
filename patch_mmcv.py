import site
import os
import torch
import numpy.core.multiarray

# Patch PyTorch 2.6 weights_only=False default for legacy model checkpoints
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Self-test CUDA kernel execution on startup (detects architecture mismatch like sm_120 on sm_90 builds)
_orig_is_available = torch.cuda.is_available
_cuda_functional = False
if _orig_is_available():
    try:
        _t = torch.zeros(1, device="cuda")
        del _t
        _cuda_functional = True
    except Exception as _e:
        import logging
        logging.getLogger("ModernTalkingHead").warning(
            f"⚠️ [Architecture Self-Test] GPU is present ({torch.cuda.get_device_name(0)}) but CUDA kernel execution failed ({_e}). "
            f"Likely NVIDIA Blackwell sm_120 vs PyTorch sm_90 binary mismatch. "
            f"Redirecting torch.cuda.is_available() to False for graceful CPU fallback."
        )
        _cuda_functional = False

def _safe_cuda_is_available():
    return _cuda_functional

torch.cuda.is_available = _safe_cuda_is_available

# Patch mmcv.utils.ext_loader to safely mock missing mmcv._ext in mmcv-lite
for sp in site.getsitepackages():
    ext_loader_path = os.path.join(sp, 'mmcv', 'utils', 'ext_loader.py')
    if os.path.exists(ext_loader_path):
        try:
            with open(ext_loader_path, 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                if "ext = importlib.import_module('mmcv.' + name)" in line:
                    indent = line[:len(line) - len(line.lstrip())]
                    new_lines.append(f"{indent}try:\n")
                    new_lines.append(f"{indent}    ext = importlib.import_module('mmcv.' + name)\n")
                    new_lines.append(f"{indent}except ModuleNotFoundError:\n")
                    new_lines.append(f"{indent}    class DummyExt:\n")
                    new_lines.append(f"{indent}        def __getattr__(self, name):\n")
                    new_lines.append(f"{indent}            return lambda *args, **kwargs: None\n")
                    new_lines.append(f"{indent}    ext = DummyExt()\n")
                else:
                    new_lines.append(line)

            with open(ext_loader_path, 'w') as f:
                f.writelines(new_lines)
            print(f"[Mock] Cleanly patched mmcv.utils.ext_loader at: {ext_loader_path}")
        except Exception as e:
            print(f"[Mock Warning] Could not patch ext_loader at {ext_loader_path}: {e}")

