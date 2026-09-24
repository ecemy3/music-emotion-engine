"""Dynamic image generation helpers for Google GenAI.

This module is capability-aware and model-type aware:
- discovers models from the active API key,
- filters out non-image families,
- chooses API method by model type,
- extracts image bytes from multiple SDK response formats,
- logs full errors with traceback.
"""

from __future__ import annotations

import base64
import traceback
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Tuple


NO_ACCESS_MESSAGE = (
    "Bu API Key ile erişilebilen bir görsel üretim modeli bulunamadı.\n"
    "Lütfen farklı bir Google AI Studio veya Vertex AI API Key kullanın."
)

EXCLUDED_MODEL_KEYWORDS = (
    "gemma",
    "robotics",
    "embedding",
    "aqa",
    "veo",
    "lyria",
    "deep-research",
    "tts",
    "computer-use",
)

MODEL_TYPE_NANO_BANANA = "nano-banana"
MODEL_TYPE_IMAGEN = "imagen"
MODEL_TYPE_GEMINI_IMAGE = "gemini-image"
MODEL_TYPE_GEMINI_IMAGE_PREVIEW = "gemini-image-preview"

MODEL_PRIORITY = {
    MODEL_TYPE_NANO_BANANA: 0,
    MODEL_TYPE_IMAGEN: 1,
    MODEL_TYPE_GEMINI_IMAGE: 2,
    MODEL_TYPE_GEMINI_IMAGE_PREVIEW: 3,
}


def _try_get_streamlit() -> Any | None:
    """Return streamlit module if available, otherwise None."""
    try:
        import streamlit as st

        return st
    except Exception:
        return None


def _log(message: str, debug: bool = False) -> None:
    """Write logs to console and, in debug mode, to Streamlit."""
    print(message)

    if not debug:
        return

    st = _try_get_streamlit()
    if st is None:
        return

    try:
        st.write(message)
    except Exception as e:
        print(f"[image_generator] Streamlit write failed: {e}")


def _log_exception(prefix: str, exc: Exception, debug: bool = False) -> str:
    """Log full exception and traceback to console and Streamlit."""
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    full = f"{prefix}: {exc}"

    print(full)
    print(tb)

    st = _try_get_streamlit()
    if debug and st is not None:
        try:
            st.error(full)
            st.code(tb, language="text")
        except Exception as streamlit_error:
            print(f"[image_generator] Streamlit exception log failed: {streamlit_error}")

    if debug:
        _log(full, debug=True)

    return tb


def _short_repr(value: Any, max_len: int = 4000) -> str:
    """Return truncated repr for debug readability."""
    text = repr(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 20] + "... <truncated>"


def _to_bytes(data: Any) -> bytes | None:
    """Convert common binary payload representations to bytes."""
    if isinstance(data, bytes):
        return data

    if isinstance(data, bytearray):
        return bytes(data)

    if isinstance(data, memoryview):
        return bytes(data)

    if isinstance(data, str):
        try:
            return base64.b64decode(data)
        except Exception:
            return None

    return None


def _model_to_dict(value: Any) -> Dict[str, Any]:
    """Best-effort object->dict conversion for SDK responses/models."""
    try:
        if hasattr(value, "model_dump"):
            dumped = value.model_dump()
            if isinstance(dumped, dict):
                return dumped
    except Exception:
        pass

    try:
        if hasattr(value, "to_dict"):
            dumped = value.to_dict()
            if isinstance(dumped, dict):
                return dumped
    except Exception:
        pass

    try:
        if is_dataclass(value):
            dumped = asdict(value)
            if isinstance(dumped, dict):
                return dumped
    except Exception:
        pass

    raw = getattr(value, "__dict__", None)
    if isinstance(raw, dict):
        return raw

    return {}


def _normalize_action(action: str) -> str:
    """Normalize action names (for stable capability checks)."""
    return "".join(ch for ch in str(action).casefold() if ch.isalnum())


def _is_not_found_error(exc: Exception) -> bool:
    """Return True when exception resembles a 404/not found issue."""
    text = str(exc).casefold()
    status_code = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    return (
        status_code == 404
        or code == 404
        or "404" in text
        or "not found" in text
        or "notfound" in text
    )


def _is_permission_error(exc: Exception) -> bool:
    """Return True when exception resembles a permission/access issue."""
    text = str(exc).casefold()
    status_code = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    return (
        status_code == 403
        or code == 403
        or "403" in text
        or "permission" in text
        or "forbidden" in text
        or "access denied" in text
    )


def _build_client(api_key: str) -> Any:
    """Create Google GenAI client for the provided key."""
    if not api_key or not str(api_key).strip():
        raise ValueError("Gemini API key gerekli.")

    import google.genai as genai

    return genai.Client(api_key=api_key)


def _get_available_models(client: Any, debug: bool = False) -> List[Any]:
    """Fetch all models available to the current API key."""
    try:
        models = list(client.models.list())
    except Exception as e:
        _log_exception("Model listesi alınamadı", e, debug=debug)
        if _is_not_found_error(e) or _is_permission_error(e):
            raise RuntimeError(NO_ACCESS_MESSAGE) from e
        raise

    _log("Detected models:", debug=debug)
    for model in models:
        model_name = getattr(model, "name", "<unknown-model>")
        _log(f"- {model_name}", debug=debug)

    return models


def _extract_supported_actions(model: Any) -> List[str]:
    """Extract supported_actions robustly across SDK schema differences."""
    actions: List[str] = []

    for attr in (
        "supported_actions",
        "supported_generation_methods",
        "supported_methods",
        "methods",
    ):
        value = getattr(model, attr, None)
        if isinstance(value, (list, tuple, set)):
            actions.extend(str(item).strip() for item in value if str(item).strip())

    model_dict = _model_to_dict(model)
    for key, value in model_dict.items():
        key_low = str(key).casefold()
        if (
            "supported_action" in key_low
            or "supported_generation_method" in key_low
            or "supported_method" in key_low
        ) and isinstance(value, (list, tuple, set)):
            actions.extend(str(item).strip() for item in value if str(item).strip())

    deduped: List[str] = []
    seen = set()
    for action in actions:
        normalized = _normalize_action(action)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(action)

    return deduped


def _is_excluded_model_name(model_name: str) -> bool:
    """Return True for model families that must never be tested."""
    low = model_name.casefold()
    return any(keyword in low for keyword in EXCLUDED_MODEL_KEYWORDS)


def _classify_model_type(model_name: str) -> str | None:
    """Classify model by name into supported image model families."""
    short = model_name.split("/")[-1].casefold()

    if "nano-banana" in short:
        return MODEL_TYPE_NANO_BANANA

    if short.startswith("imagen") or "imagen-" in short:
        return MODEL_TYPE_IMAGEN

    if "gemini" in short and "image" in short and "preview" in short:
        return MODEL_TYPE_GEMINI_IMAGE_PREVIEW

    if "gemini" in short and "image" in short:
        return MODEL_TYPE_GEMINI_IMAGE

    return None


def _select_method_for_type(
    model_type: str,
    supports_generate_images: bool,
    supports_generate_content: bool,
) -> str | None:
    """Select API method based on model type and supported actions."""
    if model_type == MODEL_TYPE_IMAGEN:
        return "generate_images" if supports_generate_images else None

    if model_type in (
        MODEL_TYPE_NANO_BANANA,
        MODEL_TYPE_GEMINI_IMAGE,
        MODEL_TYPE_GEMINI_IMAGE_PREVIEW,
    ):
        return "generate_content" if supports_generate_content else None

    return None


def _filter_image_models(models: List[Any], debug: bool = False) -> List[Dict[str, Any]]:
    """Return strictly filtered image model candidates in priority order."""
    candidates: List[Dict[str, Any]] = []

    for model in models:
        model_name = str(getattr(model, "name", "")).strip()
        if not model_name:
            continue

        if _is_excluded_model_name(model_name):
            if debug:
                _log(f"Skipping excluded model: {model_name}", debug=debug)
            continue

        model_type = _classify_model_type(model_name)
        if model_type is None:
            if debug:
                _log(f"Skipping unsupported model family: {model_name}", debug=debug)
            continue

        supported_actions = _extract_supported_actions(model)
        normalized = [_normalize_action(action) for action in supported_actions]

        supports_generate_images = any("generateimages" in action for action in normalized)
        supports_generate_content = any("generatecontent" in action for action in normalized)

        method = _select_method_for_type(
            model_type=model_type,
            supports_generate_images=supports_generate_images,
            supports_generate_content=supports_generate_content,
        )

        if method is None:
            if debug:
                _log(
                    "Skipping model due to action mismatch: "
                    f"{model_name} | type={model_type} | actions={supported_actions}",
                    debug=debug,
                )
            continue

        candidates.append(
            {
                "name": model_name,
                "model_type": model_type,
                "method": method,
                "supported_actions": supported_actions,
                "priority": MODEL_PRIORITY.get(model_type, 999),
                "short_name": model_name.split("/")[-1].casefold(),
            }
        )

    candidates.sort(key=lambda x: (x["priority"], x["short_name"]))

    if debug:
        _log("Filtered candidates:", debug=debug)
        for candidate in candidates:
            _log(
                f"- {candidate['name']} | type={candidate['model_type']} | "
                f"method={candidate['method']} | actions={candidate['supported_actions']}",
                debug=debug,
            )

    return candidates


def _debug_response_structure(response: Any, debug: bool = False) -> None:
    """Log response type/dir/model_dump/repr in debug mode."""
    if not debug:
        return

    _log(f"Response type: {type(response)}", debug=debug)

    try:
        _log(f"Response dir: {dir(response)}", debug=debug)
    except Exception as e:
        _log(f"Response dir unavailable: {e}", debug=debug)

    try:
        if hasattr(response, "model_dump"):
            _log(f"Response model_dump: {_short_repr(response.model_dump())}", debug=debug)
            return
    except Exception as e:
        _log(f"Response model_dump unavailable: {e}", debug=debug)

    _log(f"Response repr: {_short_repr(response)}", debug=debug)


def _iter_children(node: Any) -> Iterable[Any]:
    """Yield children for recursive traversal on mixed dict/object structures."""
    if node is None:
        return []

    if isinstance(node, dict):
        return node.values()

    if isinstance(node, (list, tuple, set)):
        return node

    mapped = _model_to_dict(node)
    if mapped:
        return mapped.values()

    return []


def _extract_image_bytes(response: Any) -> bytes | None:
    """Extract image bytes from multiple Google GenAI SDK response formats."""
    visited: set[int] = set()

    def _extract_from_node(node: Any, depth: int = 0) -> bytes | None:
        if node is None or depth > 10:
            return None

        node_id = id(node)
        if node_id in visited:
            return None
        visited.add(node_id)

        # Direct object access paths first.
        if not isinstance(node, dict):
            for attr in ("image_bytes", "bytes", "data"):
                raw = getattr(node, attr, None)
                data_bytes = _to_bytes(raw)
                if data_bytes and attr == "image_bytes":
                    return data_bytes

            inline_data = getattr(node, "inline_data", None)
            if inline_data is not None:
                mime_type = str(getattr(inline_data, "mime_type", "")).casefold()
                data_bytes = _to_bytes(getattr(inline_data, "data", None))
                if data_bytes and "image" in mime_type:
                    return data_bytes

        # Dict-like paths.
        payload = node if isinstance(node, dict) else _model_to_dict(node)
        if payload:
            mime_type = str(
                payload.get("mime_type")
                or payload.get("mimeType")
                or payload.get("mime")
                or ""
            ).casefold()

            for key in ("image_bytes", "imageBytes", "bytes", "data", "b64_data", "b64Data"):
                if key not in payload:
                    continue
                data_bytes = _to_bytes(payload.get(key))
                if not data_bytes:
                    continue
                if "image" in mime_type or "image" in key.casefold() or key in ("image_bytes", "imageBytes"):
                    return data_bytes

        # Known container fields.
        container_fields = (
            "generated_images",
            "images",
            "candidates",
            "content",
            "parts",
            "inline_data",
            "image",
            "output",
        )

        for field in container_fields:
            child = getattr(node, field, None)
            if child is None and isinstance(payload, dict):
                child = payload.get(field)
            if child is None:
                continue

            found = _extract_from_node(child, depth + 1)
            if found:
                return found

        # Generic recursive fallback.
        for child in _iter_children(node):
            found = _extract_from_node(child, depth + 1)
            if found:
                return found

        return None

    return _extract_from_node(response)


def _try_generate(
    client: Any,
    candidate: Dict[str, Any],
    prompt: str,
    aspect_ratio: str = "1:1",
    debug: bool = False,
) -> Tuple[bytes | None, str, bool]:
    """Try generation on one model and return (image, reason, access_issue)."""
    model_name = candidate["name"]
    method = candidate["method"]

    try:
        if method == "generate_images":
            from google.genai import types

            config = types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio=aspect_ratio,
            )
            response = client.models.generate_images(
                model=model_name,
                prompt=prompt,
                config=config,
            )
        elif method == "generate_content":
            from google.genai import types

            config = types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                temperature=0.9,
            )
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
        else:
            return None, f"Desteklenmeyen method: {method}", False

        _debug_response_structure(response, debug=debug)
        image_bytes = _extract_image_bytes(response)
        if image_bytes:
            return image_bytes, "Image üretimi başarılı.", False

        if method == "generate_content":
            return None, "Bu model text üretti ancak image üretmedi.", False

        return None, "Model response içinde image bulunamadı.", False

    except Exception as e:
        tb = _log_exception(f"{model_name} {method} hatası", e, debug=debug)
        access_issue = _is_not_found_error(e) or _is_permission_error(e)
        return None, f"{e}\n{tb}", access_issue


def generate_abstract_image(
    api_key: str,
    prompt: str,
    aspect_ratio: str = "1:1",
    debug: bool = False,
) -> Tuple[bytes, str]:
    """Generate image and return (image_bytes, selected_model_name)."""
    prompt_text = (prompt or "").strip()
    if not prompt_text:
        raise ValueError("Görsel oluşturmak için geçerli bir prompt gerekiyor.")

    client = _build_client(api_key)
    models = _get_available_models(client, debug=debug)
    candidates = _filter_image_models(models, debug=debug)

    if not candidates:
        raise RuntimeError(NO_ACCESS_MESSAGE)

    reasons: List[str] = []
    non_access_failure_seen = False

    for candidate in candidates:
        model_name = candidate["name"]
        model_type = candidate["model_type"]
        method = candidate["method"]

        _log("Trying:", debug=debug)
        _log(model_name, debug=debug)
        _log("Model type:", debug=debug)
        _log(model_type, debug=debug)
        _log("Method:", debug=debug)
        _log(method, debug=debug)

        image_bytes, reason, is_access_issue = _try_generate(
            client=client,
            candidate=candidate,
            prompt=prompt_text,
            aspect_ratio=aspect_ratio,
            debug=debug,
        )

        if image_bytes:
            _log("Image found:", debug=debug)
            _log("YES", debug=debug)
            _log("Reason:", debug=debug)
            _log(reason, debug=debug)
            return image_bytes, model_name

        _log("Image found:", debug=debug)
        _log("NO", debug=debug)
        _log("Reason:", debug=debug)
        _log(reason, debug=debug)

        reasons.append(f"{model_name} ({method}): {reason}")
        if not is_access_issue:
            non_access_failure_seen = True

    if reasons and not non_access_failure_seen:
        raise RuntimeError(NO_ACCESS_MESSAGE)

    raise RuntimeError(
        "Görsel üretimi başarısız. Denenen modellerde alınan nedenler:\n"
        + "\n".join(reasons)
    )


def generate_abstract_image_from_prompt(
    api_key: str,
    prompt: str,
    aspect_ratio: str = "1:1",
    debug: bool = False,
) -> Tuple[bytes, str]:
    """Backward-compatible wrapper for existing app imports."""
    return generate_abstract_image(
        api_key=api_key,
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        debug=debug,
    )
