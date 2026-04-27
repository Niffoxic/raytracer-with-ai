from __future__ import annotations

import io
import os
import sys
import threading
from pathlib import Path

try:
    from __main__ import emit_status, log
except ImportError:
    def emit_status(phase: str, progress: float = 0.0, message: str = "") -> None:
        sys.stderr.write(f"[status] {phase} {progress*100:.0f}% {message}\n")

    def log(msg: str) -> None:
        sys.stderr.write(f"[ai_hf_api] {msg}\n")

MODEL_ID           = "black-forest-labs/FLUX.2-dev"
PROVIDER           = "auto"
DEFAULT_PROMPT     = ("photorealistic, cinematic lighting, 4k, highly detailed, "
                      "sharp focus, dramatic, professional photography")
DEFAULT_NEG_PROMPT = ""
STRENGTH           = 0.6
GUIDANCE_SCALE: float | None = None
INFERENCE_STEPS: int | None  = None
MAX_UPLOAD_SIDE    = 1024
REQUEST_TIMEOUT_S  = 180


_client:   object | None = None
_Image:    object | None = None
_lock = threading.Lock()


def _token_from_file() -> str | None:
    path = Path(__file__).resolve().parent / ".hf_token"
    if not path.exists():
        return None
    try:
        tok = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return tok or None


def _resolve_token() -> str:
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        t = os.environ.get(var)
        if t:
            return t.strip()
    t = _token_from_file()
    if t:
        return t
    raise RuntimeError(
        "Hugging Face API token not found. Set HF_TOKEN in the environment "
        "or place the token in assets/scripts/.hf_token. Create a token at "
        "https://huggingface.co/settings/tokens with the 'Inference Providers' "
        "permission enabled.")


def _fit_upload(src, Image, max_side: int):
    w, h = src.size
    longest = max(w, h)
    if longest <= max_side:
        nw, nh = w, h
    else:
        scale = max_side / float(longest)
        nw = max(16, int(w * scale))
        nh = max(16, int(h * scale))
    nw -= nw % 16
    nh -= nh % 16
    nw = max(16, nw)
    nh = max(16, nh)
    if (nw, nh) == (w, h):
        return src
    return src.resize((nw, nh), Image.LANCZOS)


def load():
    global _client, _Image

    emit_status("preparing", 0.0, "checking dependencies")
    try:
        from huggingface_hub import InferenceClient
        from PIL import Image
    except ImportError as ex:
        raise ImportError(
            f"{ex}. Missing a shared AI dep; run 'AI Models -> Install AI "
            "deps' (which installs huggingface_hub + Pillow + friends)."
        ) from ex

    emit_status("preparing", 0.3, "resolving token")
    token = _resolve_token()

    emit_status("preparing", 0.7, f"connecting via provider='{PROVIDER}'")
    client = InferenceClient(
        model=MODEL_ID,
        token=token,
        provider=PROVIDER,
        timeout=REQUEST_TIMEOUT_S,
    )

    with _lock:
        _client = client
        _Image = Image

    emit_status("ready", 1.0, f"model={MODEL_ID} provider={PROVIDER}")
    log(f"ai_hf_api ready (model={MODEL_ID}, provider={PROVIDER})")
    return {
        "name": f"AI: HuggingFace API ({MODEL_ID.split('/')[-1]})",
        "version": "1.0",
        "expected_input": "rgb8",
        "supports_streaming": False,
    }


def _call_image_to_image(client, src_img, prompt: str):
    kwargs: dict = {"prompt": prompt, "model": MODEL_ID}
    if DEFAULT_NEG_PROMPT:
        kwargs["negative_prompt"] = DEFAULT_NEG_PROMPT
    if STRENGTH is not None:
        kwargs["strength"] = float(STRENGTH)
    if GUIDANCE_SCALE is not None:
        kwargs["guidance_scale"] = float(GUIDANCE_SCALE)
    if INFERENCE_STEPS is not None:
        kwargs["num_inference_steps"] = int(INFERENCE_STEPS)
    return client.image_to_image(src_img, **kwargs)

def apply(rgb, w, h, emit_partial, should_cancel):
    assert _client is not None and _Image is not None, "apply() called before load()"
    Image = _Image

    if should_cancel():
        emit_status("cancelled", 0.0)
        return bytes(rgb)

    emit_status("preparing", 0.0, f"{w}x{h} -> fitting for upload")
    src = Image.frombytes("RGB", (w, h), bytes(rgb))
    upload = _fit_upload(src, Image, MAX_UPLOAD_SIDE)
    up_w, up_h = upload.size

    buf = io.BytesIO()
    upload.save(buf, format="PNG")
    payload = buf.getvalue()
    emit_status("submitting", 0.0,
                f"{up_w}x{up_h} PNG {len(payload) / 1024:.1f} KB -> {MODEL_ID}")

    if should_cancel():
        emit_status("cancelled", 0.0)
        return bytes(rgb)

    try:
        emit_status("generating", 0.0, "remote inference (no local streaming)")
        with _lock:
            result_img = _call_image_to_image(_client, payload, DEFAULT_PROMPT)
    except Exception as ex:
        emit_status("error", 0.0, str(ex)[:200])
        raise

    if should_cancel():
        emit_status("cancelled", 0.0)
        return bytes(rgb)

    emit_status("finishing", 0.9, f"resizing result back to {w}x{h}")
    result_img = result_img.convert("RGB")
    if result_img.size != (w, h):
        result_img = result_img.resize((w, h), Image.LANCZOS)
    out_bytes = result_img.tobytes()
    if len(out_bytes) != w * h * 3:
        raise ValueError(
            f"internal: output length {len(out_bytes)} != expected {w*h*3}")
    emit_status("done", 1.0)
    return out_bytes


def unload():
    global _client, _Image
    with _lock:
        _client = None
        _Image = None
    emit_status("unloaded", 0.0)
    log("ai_hf_api unloaded")
