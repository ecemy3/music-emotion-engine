"""Hugging Face FLUX image generation helpers with local PNG cache.

This module keeps image generation isolated from UI code and provides
production-safe error handling for common network and API failures.
"""

from __future__ import annotations

import hashlib
import os

import requests

DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
DEFAULT_API_URL = f"https://router.huggingface.co/hf-inference/models/{DEFAULT_MODEL_ID}"
LEGACY_API_URL = f"https://api-inference.huggingface.co/models/{DEFAULT_MODEL_ID}"
DEFAULT_CACHE_DIR = os.path.join("cache", "images")
DEFAULT_TIMEOUT_SEC = 120.0


class FluxImageGenerationError(Exception):
    """Base exception for FLUX image generation failures."""


class FluxRateLimitError(FluxImageGenerationError):
    """Raised when Hugging Face returns HTTP 429."""


class FluxServerError(FluxImageGenerationError):
    """Raised when Hugging Face returns 5xx errors."""


class FluxTimeoutError(FluxImageGenerationError):
    """Raised when request exceeds timeout."""


class FluxConnectionError(FluxImageGenerationError):
    """Raised when network connection to API fails."""


class FluxInvalidResponseError(FluxImageGenerationError):
    """Raised when response body is not an image."""


def _ensure_cache_dir(cache_dir: str = DEFAULT_CACHE_DIR) -> None:
    os.makedirs(cache_dir, exist_ok=True)


def _normalize_prompt(prompt: str) -> str:
    text = str(prompt or "").strip()
    if not text:
        raise ValueError("FLUX gorsel uretimi icin prompt gereklidir.")
    return text


def compute_prompt_hash(prompt: str) -> str:
    """Return SHA256 hash for deterministic prompt cache keys."""
    prompt_text = _normalize_prompt(prompt)
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def get_flux_cache_path(prompt: str, cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    """Return cache path of the PNG file for a prompt."""
    _ensure_cache_dir(cache_dir)
    return os.path.join(cache_dir, f"{compute_prompt_hash(prompt)}.png")


def load_flux_cached_image(prompt: str, cache_dir: str = DEFAULT_CACHE_DIR) -> bytes | None:
    """Load cached PNG bytes for the prompt if available."""
    cache_path = get_flux_cache_path(prompt=prompt, cache_dir=cache_dir)
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "rb") as file:
            payload = file.read()
        if payload:
            return payload
    except OSError:
        return None

    return None


def save_flux_cached_image(
    prompt: str,
    image_bytes: bytes,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> str:
    """Persist generated image as PNG at cache/images/<sha256>.png."""
    cache_path = get_flux_cache_path(prompt=prompt, cache_dir=cache_dir)
    with open(cache_path, "wb") as file:
        file.write(image_bytes)
    return cache_path


def _format_http_error(response: requests.Response) -> str:
    body = response.text if response.text is not None else ""
    return f"HTTP {response.status_code}: {body}"


def _looks_like_image(content_type: str, payload: bytes) -> bool:
    low_ct = (content_type or "").casefold()
    if low_ct.startswith("image/"):
        return True

    png_signature = b"\x89PNG\r\n\x1a\n"
    jpeg_signature = b"\xff\xd8\xff"
    webp_signature = b"RIFF"

    return (
        payload.startswith(png_signature)
        or payload.startswith(jpeg_signature)
        or payload.startswith(webp_signature)
    )


def _request_flux_image(
    api_key: str,
    prompt: str,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> bytes:
    """Call Hugging Face Inference API and return raw image bytes."""
    token = str(api_key or "").strip()
    if not token:
        raise ValueError("Hugging Face API key gereklidir.")

    endpoint_candidates = [
        f"https://router.huggingface.co/hf-inference/models/{model_id}",
        f"https://api-inference.huggingface.co/models/{model_id}",
    ]
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "image/png",
        "Content-Type": "application/json",
    }
    prompt_text = _normalize_prompt(prompt)

    connection_errors: list[str] = []

    for url in endpoint_candidates:
        try:
            response = requests.post(
                url,
                headers=headers,
                json={"inputs": prompt_text},
                timeout=120,
            )
        except requests.Timeout as exc:
            raise FluxTimeoutError("Hugging Face API zaman asimina ugradi.") from exc
        except requests.ConnectionError as exc:
            connection_errors.append(f"{url} -> {exc}")
            continue
        except requests.RequestException as exc:
            raise FluxImageGenerationError(
                "Hugging Face API istegi sirasinda beklenmeyen bir hata olustu."
            ) from exc

        if response.status_code == 200:
            payload = response.content or b""
            content_type = response.headers.get("content-type", "")
            if not payload or not _looks_like_image(content_type=content_type, payload=payload):
                raise FluxInvalidResponseError(_format_http_error(response))
            return payload

        error_text = _format_http_error(response)

        if response.status_code == 429:
            raise FluxRateLimitError(error_text)

        if response.status_code >= 500:
            raise FluxServerError(error_text)

        raise FluxImageGenerationError(error_text)

    joined_errors = " | ".join(connection_errors) if connection_errors else "unknown"
    raise FluxConnectionError(
        "Hugging Face API baglanti hatasi olustu (DNS/erisim sorunu). "
        f"Detay: {joined_errors}"
    )


def generate_flux_image(
    api_key: str,
    prompt: str,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> bytes:
    """Generate image with FLUX.1-schnell and return image bytes."""
    prompt_text = _normalize_prompt(prompt)

    cached_bytes = load_flux_cached_image(prompt_text, cache_dir=cache_dir)
    if cached_bytes is not None:
        return cached_bytes

    image_bytes = _request_flux_image(
        api_key=api_key,
        prompt=prompt_text,
        model_id=model_id,
        timeout_sec=timeout_sec,
    )
    save_flux_cached_image(prompt_text, image_bytes=image_bytes, cache_dir=cache_dir)
    return image_bytes
