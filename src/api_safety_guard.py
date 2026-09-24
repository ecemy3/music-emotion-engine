"""API usage and cost safety guard helpers for image generation.

Bu güvenlik mekanizmasının amacı yanlışlıkla yüzlerce API çağrısı yapılmasını
önlemek ve gereksiz maliyet oluşmasını engellemektir.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import date
from typing import Any, Callable, Dict, Tuple

DEFAULT_CONFIG_PATH = os.path.join("config", "api_limits.json")
DEFAULT_USAGE_LOG_PATH = os.path.join("logs", "api_usage.json")
DEFAULT_CACHE_DIR = os.path.join("cache", "images")

DEFAULT_LIMITS: Dict[str, Any] = {
    "api_enabled": True,
    "max_image_requests_per_day": 20,
    "max_retry_per_request": 2,
    "max_requests_per_session": 10,
}

JSON_WRITE_LOCK = threading.Lock()
PERMISSION_RETRY_DELAYS_SEC = (0.10, 0.25, 0.50)


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _try_get_streamlit() -> Any | None:
    try:
        import streamlit as st

        return st
    except Exception:
        return None


def _get_streamlit_run_token() -> str | None:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx(suppress_warning=True)
        if ctx is None:
            return None

        # Streamlit her rerun'da bu koleksiyonlari sifirlayip yeniden olusturur.
        # Bu nedenle object id'si, ayni session icin rerun token gibi kullanilabilir.
        widget_marker = getattr(ctx, "widget_ids_this_run", None)
        form_marker = getattr(ctx, "form_ids_this_run", None)
        session_id = str(getattr(ctx, "session_id", "unknown"))

        if widget_marker is not None or form_marker is not None:
            return f"{session_id}:{id(widget_marker)}:{id(form_marker)}"

        return f"{session_id}:ctx-{id(ctx)}"
    except Exception:
        return None


def _warn_usage_log_write_failed(exc: Exception | None = None) -> None:
    message = "⚠️ Kullanım logu güncellenemedi."
    if exc is not None:
        print(f"{message} Detay: {exc}")
    else:
        print(message)

    st = _try_get_streamlit()
    if st is None:
        return

    run_token = _get_streamlit_run_token()
    warning_token_key = "_usage_log_warning_token"

    # Ayni rerun'da tekrarlayan warning spam'ini engelle.
    try:
        if run_token and st.session_state.get(warning_token_key) == run_token:
            return

        if run_token:
            st.session_state[warning_token_key] = run_token
    except Exception:
        pass

    try:
        st.warning(message)
    except Exception:
        pass


def _should_skip_increment_this_rerun() -> bool:
    st = _try_get_streamlit()
    if st is None:
        return False

    run_token = _get_streamlit_run_token()
    if run_token is None:
        return False

    run_key = "_usage_incremented_run_token"
    try:
        if st.session_state.get(run_key) != run_token:
            st.session_state[run_key] = run_token
            st.session_state["usage_incremented"] = False

        return bool(st.session_state.get("usage_incremented", False))
    except Exception:
        return False


def _mark_increment_this_rerun() -> None:
    st = _try_get_streamlit()
    if st is None:
        return

    run_token = _get_streamlit_run_token()
    if run_token is None:
        return

    try:
        st.session_state["_usage_incremented_run_token"] = run_token
        st.session_state["usage_incremented"] = True
    except Exception:
        pass


def _open_json_for_update(path: str):
    try:
        return open(path, "r+", encoding="utf-8")
    except FileNotFoundError:
        return open(path, "w+", encoding="utf-8")


def _read_dict_from_open_json(file) -> Dict[str, Any]:
    file.seek(0)
    raw = file.read()
    if not raw.strip():
        return {}

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {}


def _update_json_dict_in_place(
    path: str,
    updater: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """Read-update-truncate-write-flush JSON in place with retry and lock."""
    _ensure_parent_dir(path)
    last_permission_error: PermissionError | None = None

    for retry_index in range(len(PERMISSION_RETRY_DELAYS_SEC) + 1):
        try:
            with JSON_WRITE_LOCK:
                with _open_json_for_update(path) as file:
                    current_payload = _read_dict_from_open_json(file)
                    updated_payload = updater(dict(current_payload))
                    if not isinstance(updated_payload, dict):
                        updated_payload = {}

                    # Islenen akiş: read -> update -> truncate -> write -> flush -> close
                    file.seek(0)
                    file.truncate()
                    json.dump(updated_payload, file, ensure_ascii=False, indent=2)
                    file.flush()
                    try:
                        os.fsync(file.fileno())
                    except OSError:
                        pass

                    return updated_payload
        except PermissionError as exc:
            last_permission_error = exc
            if retry_index >= len(PERMISSION_RETRY_DELAYS_SEC):
                break
            time.sleep(PERMISSION_RETRY_DELAYS_SEC[retry_index])

    if last_permission_error is not None:
        raise last_permission_error

    raise PermissionError(f"JSON dosyasi guncellenemedi: {path}")


def _write_json_dict_in_place(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return _update_json_dict_in_place(path, lambda _existing: dict(payload))


def _read_json_dict(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    return {}


def _to_non_negative_int(value: Any, fallback: int) -> int:
    try:
        parsed = int(value)
        if parsed >= 0:
            return parsed
    except Exception:
        pass
    return int(fallback)


def _to_positive_int(value: Any, fallback: int) -> int:
    try:
        parsed = int(value)
        if parsed > 0:
            return parsed
    except Exception:
        pass
    return int(fallback)


def _to_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False

    return bool(fallback)


def ensure_guard_runtime_paths(
    config_path: str = DEFAULT_CONFIG_PATH,
    usage_log_path: str = DEFAULT_USAGE_LOG_PATH,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> None:
    """Ensure runtime directories/files used by safety guard exist."""
    _ensure_parent_dir(config_path)
    _ensure_parent_dir(usage_log_path)
    os.makedirs(cache_dir, exist_ok=True)


def load_api_limits(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load limits from config file and apply safe defaults."""
    ensure_guard_runtime_paths(config_path=config_path)

    if not os.path.exists(config_path):
        try:
            _write_json_dict_in_place(config_path, DEFAULT_LIMITS)
        except Exception:
            pass

    raw = _read_json_dict(config_path)

    merged = dict(DEFAULT_LIMITS)
    merged.update(raw)

    return {
        "api_enabled": _to_bool(merged.get("api_enabled"), DEFAULT_LIMITS["api_enabled"]),
        "max_image_requests_per_day": _to_positive_int(
            merged.get("max_image_requests_per_day"),
            DEFAULT_LIMITS["max_image_requests_per_day"],
        ),
        "max_retry_per_request": _to_positive_int(
            merged.get("max_retry_per_request"),
            DEFAULT_LIMITS["max_retry_per_request"],
        ),
        "max_requests_per_session": _to_positive_int(
            merged.get("max_requests_per_session"),
            DEFAULT_LIMITS["max_requests_per_session"],
        ),
    }


def _today_iso() -> str:
    return date.today().isoformat()


def get_daily_usage_record(usage_log_path: str = DEFAULT_USAGE_LOG_PATH) -> Dict[str, Any]:
    """Return normalized usage record for current date."""
    ensure_guard_runtime_paths(usage_log_path=usage_log_path)

    current_day = _today_iso()

    def normalize_record(raw: Dict[str, Any]) -> Dict[str, Any]:
        raw_date = str(raw.get("date", "")).strip()
        raw_count = _to_non_negative_int(raw.get("image_requests", 0), 0)

        if raw_date != current_day:
            return {
                "date": current_day,
                "image_requests": 0,
            }

        return {
            "date": current_day,
            "image_requests": raw_count,
        }

    try:
        return _update_json_dict_in_place(usage_log_path, normalize_record)
    except Exception as exc:
        _warn_usage_log_write_failed(exc)
        return normalize_record(_read_json_dict(usage_log_path))


def increment_daily_usage(
    amount: int = 1,
    usage_log_path: str = DEFAULT_USAGE_LOG_PATH,
) -> Dict[str, Any]:
    """Increment daily API request counter and return updated record."""
    ensure_guard_runtime_paths(usage_log_path=usage_log_path)

    increment = max(0, int(amount))
    if increment <= 0:
        return get_daily_usage_record(usage_log_path=usage_log_path)

    if _should_skip_increment_this_rerun():
        return get_daily_usage_record(usage_log_path=usage_log_path)

    current_day = _today_iso()

    def updater(raw: Dict[str, Any]) -> Dict[str, Any]:
        raw_date = str(raw.get("date", "")).strip()
        raw_count = _to_non_negative_int(raw.get("image_requests", 0), 0)

        if raw_date != current_day:
            raw_count = 0

        return {
            "date": current_day,
            "image_requests": raw_count + increment,
        }

    try:
        updated = _update_json_dict_in_place(usage_log_path, updater)
        _mark_increment_this_rerun()
        return updated
    except Exception as exc:
        _mark_increment_this_rerun()
        _warn_usage_log_write_failed(exc)
        return get_daily_usage_record(usage_log_path=usage_log_path)


def compute_prompt_hash(prompt: str) -> str:
    """Generate deterministic hash key for cache lookup."""
    prompt_text = (prompt or "").strip()
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def get_cache_paths(
    prompt_hash: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Tuple[str, str]:
    """Return cache image and metadata paths for a prompt hash."""
    ensure_guard_runtime_paths(cache_dir=cache_dir)
    image_path = os.path.join(cache_dir, f"{prompt_hash}.img")
    meta_path = os.path.join(cache_dir, f"{prompt_hash}.json")
    return image_path, meta_path


def load_cached_image(
    prompt_hash: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Tuple[bytes | None, Dict[str, Any] | None]:
    """Load cached image bytes and metadata if available."""
    image_path, meta_path = get_cache_paths(prompt_hash, cache_dir=cache_dir)

    if not os.path.exists(image_path):
        return None, None

    try:
        with open(image_path, "rb") as file:
            image_bytes = file.read()
    except Exception:
        return None, None

    metadata = _read_json_dict(meta_path)
    if not metadata:
        metadata = None

    return image_bytes, metadata


def save_cached_image(
    prompt_hash: str,
    image_bytes: bytes,
    metadata: Dict[str, Any] | None = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Tuple[str, str]:
    """Persist generated image bytes and metadata into cache."""
    image_path, meta_path = get_cache_paths(prompt_hash, cache_dir=cache_dir)

    with open(image_path, "wb") as file:
        file.write(image_bytes)

    payload = dict(metadata or {})
    payload.setdefault("prompt_hash", prompt_hash)
    payload.setdefault("cached_date", _today_iso())
    _write_json_dict_in_place(meta_path, payload)

    return image_path, meta_path


def log_image_request(
    *,
    date_text: str,
    prompt_hash: str,
    cache_hit: bool,
    api_called: bool,
) -> None:
    """Write deterministic request diagnostics to console."""
    print("Image Request")
    print(f"Date: {date_text}")
    print(f"Prompt Hash: {prompt_hash}")
    print(f"Cache: {'YES' if cache_hit else 'NO'}")
    print(f"API Called: {'YES' if api_called else 'NO'}")
