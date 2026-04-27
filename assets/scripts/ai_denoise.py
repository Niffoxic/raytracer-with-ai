from __future__ import annotations

import gc
import sys
import threading

try:
    from __main__ import emit_status, log
except ImportError:
    def emit_status(phase: str, progress: float = 0.0, message: str = "") -> None:
        sys.stderr.write(f"[status] {phase} {progress*100:.0f}% {message}\n")

    def log(msg: str) -> None:
        sys.stderr.write(f"[ai_denoise] {msg}\n")

REQUIRES = ["deepinv"]
TILE_SIZE    = 256
TILE_OVERLAP = 16

_model:  object | None = None
_torch:  object | None = None
_np:     object | None = None
_Image:  object | None = None
_device: str           = "cpu"
_lock = threading.Lock()


class _CancelDenoise(Exception):
    pass

def _install_tqdm_hook() -> bool:
    try:
        import tqdm
        import tqdm.auto as _auto
    except ImportError:
        return False

    base = _auto.tqdm

    class _StatusTqdm(base):
        def _emit(self) -> None:
            total = int(getattr(self, "total", 0) or 0)
            pos   = int(getattr(self, "n", 0) or 0)
            desc  = (getattr(self, "desc", "") or "file").split("/")[-1]
            if total > 0:
                mb_done = pos / 1_048_576.0
                mb_tot  = total / 1_048_576.0
                emit_status(
                    "downloading",
                    max(0.0, min(1.0, pos / total)),
                    f"{desc}  {mb_done:.1f} / {mb_tot:.1f} MB")
            elif pos > 0:
                emit_status("downloading", 0.0,
                            f"{desc}  {pos / 1_048_576.0:.1f} MB")

        def update(self, n=1):
            r = super().update(n)
            try: self._emit()
            except Exception: pass
            return r

        def close(self):
            try: self._emit()
            except Exception: pass
            return super().close()

    import tqdm as _tqdm_mod
    _auto.tqdm = _StatusTqdm
    _tqdm_mod.tqdm = _StatusTqdm
    try:
        import tqdm.std as _std
        _std.tqdm = _StatusTqdm
    except Exception:
        pass
    for mod_name in (
        "huggingface_hub.utils.tqdm",
        "huggingface_hub.utils._tqdm",
    ):
        try:
            import importlib
            m = importlib.import_module(mod_name)
            if hasattr(m, "tqdm"):    m.tqdm    = _StatusTqdm
            if hasattr(m, "tqdm_hf"): m.tqdm_hf = _StatusTqdm
        except Exception:
            pass
    return True


def _tile_origins(extent: int, tile: int, overlap: int) -> list[int]:
    if extent <= tile:
        return [0]
    stride = max(1, tile - 2 * overlap)
    origins: list[int] = []
    x = 0
    while x + tile < extent:
        origins.append(x)
        x += stride
    origins.append(extent - tile)
    seen: set[int] = set()
    out: list[int] = []
    for o in origins:
        if o in seen:
            continue
        seen.add(o)
        out.append(o)
    return out


def load():
    global _model, _torch, _np, _Image, _device

    emit_status("preparing", 0.0, "checking dependencies")
    try:
        import numpy as np
        import torch
        from PIL import Image
        import deepinv
    except ImportError as ex:
        raise ImportError(
            f"{ex}. Missing a shared AI dep; run 'AI Models -> Install "
            "AI deps' (torch / Pillow / huggingface_hub / numpy / tqdm). "
            "Script-private deps are auto-installed."
        ) from ex

    _torch = torch
    _np    = np
    _Image = Image

    _install_tqdm_hook()

    cached = False
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(
            "deepinv/Restormer", "real_denoising.pth") is not None
    except Exception:
        pass

    if cached:
        emit_status("loading", 0.0,
                    "using cached Restormer real_denoising")
    else:
        emit_status(
            "downloading", 0.0,
            "fetching Restormer real_denoising")

    model = deepinv.models.Restormer(pretrained="denoising_real")

    emit_status("loading", 0.6, "moving model to device")
    if torch.cuda.is_available():
        _device = "cuda"
        model = model.to("cuda")
    else:
        _device = "cpu"
    model.eval()

    with _lock:
        _model = model

    emit_status("ready", 1.0,
                f"model=Restormer (denoising_real) device={_device}")
    log(f"ai_denoise ready (device={_device})")
    return {
        "name": "AI: Denoise (Restormer)",
        "version": "1.0",
        "expected_input": "rgb8",
        "supports_streaming": False,
    }


def _denoise_tile(tile):
    torch = _torch
    _, _, th, tw = tile.shape
    pad_h = (8 - th % 8) % 8
    pad_w = (8 - tw % 8) % 8
    if pad_h or pad_w:
        tile = torch.nn.functional.pad(
            tile, (0, pad_w, 0, pad_h), mode="reflect")
    out = _model(tile)
    if pad_h or pad_w:
        out = out[..., :th, :tw]
    return out


def apply(rgb, w, h, emit_partial, should_cancel):
    assert _model is not None, "apply() called before load()"
    torch = _torch
    np    = _np
    Image = _Image

    emit_status("preparing", 0.0, f"{w}x{h} RGB input")

    src = Image.frombytes("RGB", (w, h), bytes(rgb))
    arr = np.asarray(src, dtype=np.float32) / 255.0
    inp = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    inp = inp.to(_device)

    out = torch.zeros_like(inp)
    weight = torch.zeros((1, 1, h, w), device=_device, dtype=inp.dtype)

    ys = _tile_origins(h, TILE_SIZE, TILE_OVERLAP)
    xs = _tile_origins(w, TILE_SIZE, TILE_OVERLAP)
    total = len(ys) * len(xs)
    done = 0
    log(f"tiling {w}x{h} into {total} tile(s) of {TILE_SIZE}px")

    try:
        with torch.inference_mode():
            for y in ys:
                y_end = min(y + TILE_SIZE, h)
                for x in xs:
                    if should_cancel():
                        raise _CancelDenoise()
                    x_end = min(x + TILE_SIZE, w)
                    tile = inp[..., y:y_end, x:x_end]
                    tile_out = _denoise_tile(tile)
                    out[..., y:y_end, x:x_end] += tile_out
                    weight[..., y:y_end, x:x_end] += 1.0
                    done += 1
                    emit_status(
                        "denoising", done / total,
                        f"tile {done}/{total}")
    except _CancelDenoise:
        emit_status("cancelled", 0.0)
        return bytes(rgb)

    out = (out / weight.clamp(min=1.0)).clamp(0.0, 1.0)

    emit_status("finishing", 1.0, "assembling output")
    result = (out[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0)
    result = result.round().clip(0, 255).astype(np.uint8)
    out_bytes = result.tobytes()
    if len(out_bytes) != w * h * 3:
        raise ValueError(
            f"internal: output length {len(out_bytes)} != expected {w*h*3}")
    emit_status("done", 1.0)
    return out_bytes


def unload():
    global _model
    with _lock:
        m = _model
        _model = None
    del m
    gc.collect()
    if _torch is not None:
        try:
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass
    emit_status("unloaded", 0.0)
    log("ai_denoise unloaded")
