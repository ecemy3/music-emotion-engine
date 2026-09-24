"""
Song-level emotional atmosphere analysis helpers.

This module transforms aggregated survey answers into a structured
Gemini JSON analysis payload that can be used for abstract art generation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.survey_utils import survey_row_to_va


EMOTION_COLUMNS = {
    "peaceful": "Bu müzik beni huzurlu ve sakin hissettirdi. ",
    "energetic": "Bu müzik beni neşeli ve enerjik hissettirdi. ",
    "nostalgic": "Bu müzik bana nostaljik bir his verdi. ",
    "melancholic": "Bu müzik beni üzgün veya melankolik hissettirdi. ",
    "admiration": "Bu müzik bana hayranlık ve şaşkınlık hisleri uyandırdı. ",
    "anxious": "Bu müzik beni gergin veya huzursuz hissettirdi. ",
    "motivated": "Bu müzik beni güçlü, motive olmuş hissettirdi. ",
}

EMOTION_LABELS_TR = {
    "peaceful": "huzurlu",
    "energetic": "enerjik",
    "nostalgic": "nostaljik",
    "melancholic": "melankolik",
    "admiration": "hayranlık",
    "anxious": "kaygılı",
    "motivated": "motive",
}

REQUIRED_ANALYSIS_FIELDS = [
    "title",
    "summary",
    "emotion_start",
    "emotion_end",
    "transition",
    "nano_prompt",
]


def _coerce_likert(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").clip(lower=1, upper=5)


def _distribution(frame: pd.DataFrame, column_name: str) -> Dict[str, float]:
    if column_name not in frame.columns:
        return {}

    values = (
        frame[column_name]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
    )
    if values.empty:
        return {}

    dist = (values.value_counts(normalize=True) * 100.0).head(6)
    return {str(k): round(float(v), 2) for k, v in dist.items()}


def build_song_emotion_profile(song_df: pd.DataFrame) -> Dict[str, Any]:
    """Build a compact and deterministic profile from selected-song survey rows."""
    if song_df is None or song_df.empty:
        raise ValueError("Secilen sarki icin anket verisi bulunamadi.")

    missing_cols = [c for c in EMOTION_COLUMNS.values() if c not in song_df.columns]
    if missing_cols:
        raise ValueError(f"Duygu kolonlari eksik: {missing_cols}")

    emotion_means: Dict[str, float] = {}
    emotion_stds: Dict[str, float] = {}

    for key, col in EMOTION_COLUMNS.items():
        values = _coerce_likert(song_df[col])
        emotion_means[key] = round(float(values.mean(skipna=True)), 4)
        emotion_stds[key] = round(float(values.std(skipna=True)), 4)

    top_emotions = sorted(
        emotion_means.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    va_scores: List[Any] = []
    for _, row in song_df.iterrows():
        try:
            v, a = survey_row_to_va(row)
        except Exception:
            continue
        if pd.notna(v) and pd.notna(a):
            va_scores.append((float(v), float(a)))

    if va_scores:
        valence_values = np.array([v for v, _ in va_scores], dtype=float)
        arousal_values = np.array([a for _, a in va_scores], dtype=float)
        valence_mean = round(float(np.mean(valence_values)), 4)
        arousal_mean = round(float(np.mean(arousal_values)), 4)
        valence_std = round(float(np.std(valence_values)), 4)
        arousal_std = round(float(np.std(arousal_values)), 4)
    else:
        valence_mean = 5.0
        arousal_mean = 5.0
        valence_std = 0.0
        arousal_std = 0.0

    before_dist = _distribution(song_df, "Müziği dinlemeden önceki duygu durumunuz")
    after_dist = _distribution(song_df, "Müziği dinledikten sonraki duygu durumunuz.")

    return {
        "response_count": int(len(song_df)),
        "emotion_mean_likert": {
            EMOTION_LABELS_TR[k]: round(v, 4) for k, v in emotion_means.items()
        },
        "emotion_std_likert": {
            EMOTION_LABELS_TR[k]: round(emotion_stds[k], 4) for k in emotion_means.keys()
        },
        "top_emotions": [
            {
                "emotion": EMOTION_LABELS_TR[key],
                "mean_likert": round(score, 4),
            }
            for key, score in top_emotions[:5]
        ],
        "valence_mean": valence_mean,
        "arousal_mean": arousal_mean,
        "valence_std": valence_std,
        "arousal_std": arousal_std,
        "before_mood_distribution": before_dist,
        "after_mood_distribution": after_dist,
    }


def build_profile_signature(profile: Dict[str, Any]) -> str:
    """Generate a stable signature used to invalidate stale UI state."""
    payload = {
        "response_count": profile.get("response_count", 0),
        "emotion_mean_likert": profile.get("emotion_mean_likert", {}),
        "valence_mean": profile.get("valence_mean", 0.0),
        "arousal_mean": profile.get("arousal_mean", 0.0),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _extract_json_block(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Gemini yanitinda JSON bulunamadi.")

    return json.loads(cleaned[start : end + 1])


def normalize_analysis_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce Gemini response to the required schema."""
    normalized: Dict[str, Any] = {}

    for field in REQUIRED_ANALYSIS_FIELDS:
        value = payload.get(field)
        normalized[field] = "" if value is None else str(value).strip()

    if not normalized["title"]:
        normalized["title"] = "Emotional Atmosphere"
    if not normalized["summary"]:
        normalized["summary"] = "Collective emotional atmosphere extracted from survey responses."
    if not normalized["emotion_start"]:
        normalized["emotion_start"] = "uncertainty"
    if not normalized["emotion_end"]:
        normalized["emotion_end"] = "clarity"
    if not normalized["transition"]:
        normalized["transition"] = "subtle emotional evolution"

    return normalized


def _build_prompt(song_name: str, profile: Dict[str, Any]) -> str:
    schema = {
        "title": "",
        "summary": "",
        "emotion_start": "",
        "emotion_end": "",
        "transition": "",
        "nano_prompt": "",
    }

    return f"""
Sen bir muzik psikolojisi analisti ve soyut sanat yonetmenisin.

Gorev:
- Asagidaki toplu anket verisini analiz et.
- Secili sarkinin dinleyicilerde olusturdugu ORTAK duygusal atmosferi cikar.
- Ciktini yalnizca JSON olarak ver. Aciklama, markdown, kod blogu veya ek metin yazma.

Mutlak kurallar:
- JSON disinda hicbir sey donme.
- nano_prompt sadece soyut sanat uretimi icin uygun olmali.
- nano_prompt yuksek kaliteli, modern, sinematik ve estetik olmalidir.
- nano_prompt kesinlikle insan yuzu, insan figuru, yazi, tipografi, logo, watermark icermemelidir.
- Tamamen non-figurative abstract art olmalidir.

JSON semasi (alan isimlerini aynen kullan):
{json.dumps(schema, ensure_ascii=False, indent=2)}

Secili sarki: {song_name}
Anket ozeti:
{json.dumps(profile, ensure_ascii=False, indent=2)}
""".strip()


def _response_text_from_legacy(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    candidates = getattr(response, "candidates", None) or []
    collected: List[str] = []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                collected.append(part_text)

    return "\n".join(collected).strip()


def _call_gemini_legacy(api_key: str, model_name: str, prompt: str) -> str:
    import google.generativeai as legacy_genai

    legacy_genai.configure(api_key=api_key)
    model = legacy_genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    text = _response_text_from_legacy(response)
    if not text:
        raise RuntimeError("Legacy Gemini bos yanit dondurdu.")
    return text


def _response_text_from_modern(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    candidates = getattr(response, "candidates", None) or []
    collected: List[str] = []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                collected.append(part_text)

    return "\n".join(collected).strip()


def _call_gemini_modern(api_key: str, model_name: str, prompt: str) -> str:
    import google.genai as modern_genai

    client = modern_genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    text = _response_text_from_modern(response)
    if not text:
        raise RuntimeError("Modern Gemini bos yanit dondurdu.")
    return text


def analyze_song_emotional_atmosphere(
    song_name: str,
    profile: Dict[str, Any],
    api_key: str,
    model_name: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    """Run Gemini analysis and return normalized JSON payload."""
    if not api_key or not str(api_key).strip():
        raise ValueError("Gemini API key gerekli.")

    prompt = _build_prompt(song_name=song_name, profile=profile)

    model_candidates = [model_name, "gemini-2.5-flash", "gemini-1.5-flash"]
    model_candidates = list(dict.fromkeys(model_candidates))

    errors: List[str] = []

    for candidate_model in model_candidates:
        for caller in (_call_gemini_legacy, _call_gemini_modern):
            try:
                raw_text = caller(api_key=api_key, model_name=candidate_model, prompt=prompt)
                payload = _extract_json_block(raw_text)
                payload = normalize_analysis_payload(payload)
                return payload
            except Exception as exc:
                errors.append(f"{candidate_model}/{caller.__name__}: {exc}")

    raise RuntimeError(
        "Gemini JSON analizi basarisiz oldu. "
        + " | ".join(errors[-4:])
    )
