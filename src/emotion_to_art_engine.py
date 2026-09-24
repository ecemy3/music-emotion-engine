"""Deterministic Emotion-to-Art engine for abstract artwork generation.

This module transforms survey-derived psychometric signals into an explainable
art-direction report and a production-ready prompt for FLUX image generation.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.survey_utils import survey_row_to_va

ACADEMIC_RATIONALE = (
    "Bu sistem yalnızca baskın duyguları saymamaktadır. "
    "Anketlerden elde edilen psikometrik göstergeler (Valence, Arousal, duygu "
    "çeşitliliği, entropy ve duygu yoğunluğu), renk psikolojisi, kompozisyon "
    "teorisi ve sanat kuralları ile birleştirilerek çok katmanlı bir "
    "Emotion-to-Art Engine oluşturulmuştur. Böylece büyük dil modeli "
    "kullanılmadan deterministik ve açıklanabilir bir soyut sanat üretim "
    "sistemi geliştirilmiştir."
)

EMOTION_COLUMNS: Dict[str, str] = {
    "calm": "Bu müzik beni huzurlu ve sakin hissettirdi. ",
    "happy": "Bu müzik beni neşeli ve enerjik hissettirdi. ",
    "nostalgic": "Bu müzik bana nostaljik bir his verdi. ",
    "sad": "Bu müzik beni üzgün veya melankolik hissettirdi. ",
    "admiration": "Bu müzik bana hayranlık ve şaşkınlık hisleri uyandırdı. ",
    "tense": "Bu müzik beni gergin veya huzursuz hissettirdi. ",
    "motivated": "Bu müzik beni güçlü, motive olmuş hissettirdi. ",
}

TARGET_EMOTIONS: Tuple[str, ...] = (
    "huzurlu",
    "nostaljik",
    "mutlu",
    "uzgun",
    "motivasyon",
    "heyecan",
    "romantik",
    "yalniz",
    "ozgur",
    "melankolik",
    "kaygili",
)

DISPLAY_LABELS: Dict[str, str] = {
    "huzurlu": "Huzurlu",
    "nostaljik": "Nostaljik",
    "mutlu": "Mutlu",
    "uzgun": "Üzgün",
    "motivasyon": "Motivasyon",
    "heyecan": "Heyecan",
    "romantik": "Romantik",
    "yalniz": "Yalnız",
    "ozgur": "Özgür",
    "melankolik": "Melankolik",
    "kaygili": "Kaygılı",
}

COLUMN_TO_EMOTION_WEIGHTS: Dict[str, Dict[str, float]] = {
    EMOTION_COLUMNS["calm"]: {"huzurlu": 1.0},
    EMOTION_COLUMNS["happy"]: {"mutlu": 0.6, "heyecan": 0.4},
    EMOTION_COLUMNS["nostalgic"]: {"nostaljik": 0.7, "melankolik": 0.3},
    EMOTION_COLUMNS["sad"]: {"uzgun": 0.6, "melankolik": 0.4},
    EMOTION_COLUMNS["admiration"]: {
        "heyecan": 0.5,
        "ozgur": 0.3,
        "motivasyon": 0.2,
    },
    EMOTION_COLUMNS["tense"]: {"kaygili": 0.85, "melankolik": 0.15},
    EMOTION_COLUMNS["motivated"]: {"motivasyon": 0.7, "ozgur": 0.3},
}

MOOD_COLUMNS_CANDIDATES: Tuple[str, ...] = (
    "Müziği dinlemeden önceki duygu durumunuz",
    "Müziği dinledikten sonraki duygu durumunuz.",
)

TEXT_EMOTION_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "huzurlu": ("huzur", "sakin", "dingin", "rahat"),
    "nostaljik": ("nostal", "gecmis"),
    "mutlu": ("mutlu", "neseli", "pozitif", "keyif"),
    "uzgun": ("uzgun", "huzun", "keder"),
    "motivasyon": ("motive", "motivasyon", "guclu", "azimli"),
    "heyecan": ("heyecan", "cosku", "enerjik", "dinamik"),
    "romantik": ("romantik", "ask", "sevgi"),
    "yalniz": ("yalniz", "yalnizlik", "issiz", "tek basin"),
    "ozgur": ("ozgur", "ozgurluk", "ferah", "acik ufuk"),
    "melankolik": ("melankol", "huzunlu"),
    "kaygili": ("kaygi", "gergin", "huzursuz", "anksiy"),
}

COLOR_PSYCHOLOGY: Dict[str, Tuple[str, ...]] = {
    "huzurlu": ("Soft Blue", "White", "Silver"),
    "nostaljik": ("Sepia", "Gold", "Brown"),
    "mutlu": ("Yellow", "Orange", "Cream"),
    "motivasyon": ("Orange", "Amber", "Gold"),
    "uzgun": ("Navy", "Gray", "Dark Blue"),
    "melankolik": ("Indigo", "Fog Gray", "Violet"),
    "romantik": ("Rose", "Purple", "Pink"),
    "ozgur": ("Cyan", "Sky Blue", "White"),
    "heyecan": ("Amber", "Orange", "Scarlet"),
    "kaygili": ("Graphite", "Deep Purple", "Steel Blue"),
    "yalniz": ("Smoke Gray", "Dust Blue", "Pale White"),
}

ATMOSPHERE_DEFAULT_COLORS: Dict[str, Tuple[str, ...]] = {
    "Peaceful": ("Soft Blue", "White", "Silver"),
    "Energetic": ("Orange", "Amber", "Gold"),
    "Melancholic": ("Indigo", "Fog Gray", "Violet"),
    "Chaotic": ("Crimson", "Graphite", "Electric Blue"),
    "Minimal": ("Off White", "Soft Gray", "Light Taupe"),
    "Experimental": ("Neon Cyan", "Magenta", "Ultraviolet"),
}

LIGHTING_RULES: Dict[str, str] = {
    "Peaceful": "Soft Cinematic Light",
    "Energetic": "High Contrast Dramatic Light",
    "Melancholic": "Diffused Foggy Light",
    "Chaotic": "Strong Directional Light",
    "Minimal": "Soft Ambient Gallery Light",
    "Experimental": "Hybrid Neon Diffused Light",
}

COMPOSITION_RULES: Dict[str, str] = {
    "Peaceful": "Flowing, Organic, Minimal",
    "Energetic": "Dynamic, Explosive, Radial Motion",
    "Melancholic": "Mist, Soft Gradients, Layered Textures",
    "Chaotic": "Fractals, Broken Geometry, Sharp Contrasts",
    "Minimal": "Negative Space, Balanced Geometry, Clean Layers",
    "Experimental": "Asymmetric Structures, Deconstructed Layers, Motion Fields",
}

BRUSH_STYLE_RULES: Dict[str, str] = {
    "Peaceful": "Watercolor, Soft Brush",
    "Energetic": "Oil Paint, Bold Strokes",
    "Melancholic": "Textured Canvas",
    "Chaotic": "Palette Knife, Aggressive Mixed Strokes",
    "Minimal": "Soft Acrylic Wash",
    "Experimental": "Mixed Media",
}


def _normalize_text(value: Any) -> str:
    text = str(value or "").casefold().strip()
    replacements = {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _coerce_likert(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").clip(lower=1, upper=5)


def _likert_to_unit(values: pd.Series) -> pd.Series:
    return ((values - 1.0) / 4.0).fillna(0.0)


def _collect_emotion_scores(song_df: pd.DataFrame) -> Dict[str, float]:
    scores = {emotion: 0.0 for emotion in TARGET_EMOTIONS}

    for column_name, emotion_weights in COLUMN_TO_EMOTION_WEIGHTS.items():
        if column_name not in song_df.columns:
            continue

        values = _coerce_likert(song_df[column_name])
        weighted_signal = float(_likert_to_unit(values).sum())

        for emotion, weight in emotion_weights.items():
            scores[emotion] += weighted_signal * float(weight)

    for column_name in MOOD_COLUMNS_CANDIDATES:
        if column_name not in song_df.columns:
            continue

        for raw_value in song_df[column_name].dropna():
            text = _normalize_text(raw_value)
            for emotion, keywords in TEXT_EMOTION_KEYWORDS.items():
                if any(keyword in text for keyword in keywords):
                    scores[emotion] += 1.0

    if sum(scores.values()) <= 0.0:
        scores["huzurlu"] = 1.0

    return scores


def _to_distribution(scores: Dict[str, float]) -> List[Dict[str, Any]]:
    total_score = float(sum(scores.values()))
    items: List[Dict[str, Any]] = []

    for emotion, raw_score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        ratio = (raw_score / total_score) if total_score > 0 else 0.0
        items.append(
            {
                "emotion_key": emotion,
                "emotion": DISPLAY_LABELS.get(emotion, emotion),
                "count": int(round(raw_score)),
                "ratio": round(ratio, 6),
                "percentage": round(ratio * 100.0, 2),
            }
        )

    return items


def _valence_level(value_01: float) -> str:
    if value_01 < 0.25:
        return "Çok Negatif"
    if value_01 < 0.45:
        return "Negatif"
    if value_01 < 0.55:
        return "Nötr"
    if value_01 < 0.75:
        return "Pozitif"
    return "Çok Pozitif"


def _arousal_level(value_01: float) -> str:
    if value_01 < 0.25:
        return "Çok Sakin"
    if value_01 < 0.50:
        return "Sakin"
    if value_01 < 0.70:
        return "Orta Enerji"
    return "Yüksek Enerji"


def _valence_polarity(level: str) -> str:
    if level in {"Pozitif", "Çok Pozitif"}:
        return "positive"
    if level in {"Negatif", "Çok Negatif"}:
        return "negative"
    return "neutral"


def _arousal_polarity(level: str) -> str:
    if level in {"Çok Sakin", "Sakin"}:
        return "low"
    if level == "Yüksek Enerji":
        return "high"
    return "mid"


def _determine_atmosphere(valence_level: str, arousal_level: str) -> str:
    valence_group = _valence_polarity(valence_level)
    arousal_group = _arousal_polarity(arousal_level)

    if valence_group == "positive" and arousal_group == "low":
        return "Peaceful"
    if valence_group == "positive" and arousal_group in {"mid", "high"}:
        return "Energetic"
    if valence_group == "negative" and arousal_group in {"low", "mid"}:
        return "Melancholic"
    if valence_group == "negative" and arousal_group == "high":
        return "Chaotic"
    if valence_group == "neutral" and arousal_group == "low":
        return "Minimal"
    return "Experimental"


def _normalized_entropy(probabilities: Iterable[float]) -> float:
    p = np.array([float(x) for x in probabilities if float(x) > 0.0], dtype=float)
    if p.size == 0:
        return 0.0

    entropy_value = float(-np.sum(p * np.log2(p)))
    base = math.log2(float(len(probabilities))) if len(tuple(probabilities)) > 1 else 1.0
    if base <= 0.0:
        return 0.0

    return max(0.0, min(1.0, entropy_value / base))


def _intensity_label(entropy_value: float) -> str:
    if entropy_value >= 0.66:
        return "Karmaşık Atmosfer"
    if entropy_value <= 0.33:
        return "Tek Baskın Atmosfer"
    return "Dengeli Çok Katmanlı Atmosfer"


def _build_color_palette(dominant_emotions: List[str], atmosphere: str) -> List[str]:
    colors: List[str] = []

    for emotion_key in dominant_emotions:
        for color in COLOR_PSYCHOLOGY.get(emotion_key, ()):  # pragma: no branch
            if color not in colors:
                colors.append(color)

    for fallback_color in ATMOSPHERE_DEFAULT_COLORS.get(atmosphere, ()):  # pragma: no branch
        if len(colors) >= 6:
            break
        if fallback_color not in colors:
            colors.append(fallback_color)

    if len(colors) < 3:
        for fallback_color in ("Soft Blue", "Amber", "Fog Gray"):
            if fallback_color not in colors:
                colors.append(fallback_color)
            if len(colors) >= 3:
                break

    return colors[:6]


def _build_symbolism(
    dominant_emotions: List[str],
    atmosphere: str,
    valence_value: float,
) -> List[str]:
    symbols: List[str] = []

    if "motivasyon" in dominant_emotions or valence_value >= 0.55:
        symbols.append("Rising Light")
    if "ozgur" in dominant_emotions:
        symbols.append("Open Horizon")
    if "yalniz" in dominant_emotions:
        symbols.append("Empty Space")
    if "nostaljik" in dominant_emotions:
        symbols.append("Fading Layers")
    if "huzurlu" in dominant_emotions or atmosphere == "Peaceful":
        symbols.append("Flowing Water")
    if "heyecan" in dominant_emotions or atmosphere == "Energetic":
        symbols.append("Spirals")

    if not symbols:
        if atmosphere in {"Minimal", "Melancholic"}:
            symbols = ["Empty Space", "Fading Layers"]
        elif atmosphere == "Chaotic":
            symbols = ["Spirals", "Rising Light"]
        else:
            symbols = ["Flowing Water", "Open Horizon"]

    deduped: List[str] = []
    seen = set()
    for symbol in symbols:
        if symbol in seen:
            continue
        seen.add(symbol)
        deduped.append(symbol)

    return deduped


def _build_prompt(
    dominant_items: List[Dict[str, Any]],
    atmosphere: str,
    intensity_label: str,
    entropy_value: float,
    diversity_value: float,
    color_palette: List[str],
    composition: str,
    lighting: str,
    brush_style: str,
    symbolism: List[str],
) -> str:
    dominant_lines = []
    for item in dominant_items:
        dominant_lines.append(f"- {item['emotion']} ({item['percentage']:.2f}%)")

    prompt = (
        "Create a museum-quality contemporary abstract artwork.\n\n"
        "Dominant emotions:\n"
        + "\n".join(dominant_lines)
        + "\n\n"
        f"Atmosphere:\n{atmosphere}\n\n"
        "Emotional intensity:\n"
        f"{intensity_label} (entropy: {entropy_value:.3f}, diversity: {diversity_value:.3f})\n\n"
        "Color palette:\n"
        f"{', '.join(color_palette)}\n\n"
        "Composition:\n"
        f"{composition}\n\n"
        "Lighting:\n"
        f"{lighting}\n\n"
        "Brush style:\n"
        f"{brush_style}\n\n"
        "Symbolism:\n"
        f"{', '.join(symbolism)}\n\n"
        "Highly detailed.\n\n"
        "Emotional.\n\n"
        "Contemporary fine art.\n\n"
        "Textured canvas.\n\n"
        "Cinematic.\n\n"
        "No people.\n\n"
        "No text.\n\n"
        "No logos.\n\n"
        "No watermark.\n\n"
        "8K quality."
    )

    return "\n".join(line.rstrip() for line in prompt.splitlines()).strip()


def build_emotion_profile_signature(song_df: pd.DataFrame) -> str:
    """Create deterministic signature for cache/session invalidation."""
    payload: Dict[str, Any] = {
        "rows": int(len(song_df)),
        "means": {},
    }

    for column_name in EMOTION_COLUMNS.values():
        if column_name not in song_df.columns:
            continue
        values = _coerce_likert(song_df[column_name])
        payload["means"][column_name] = round(float(values.mean(skipna=True)), 6)

    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def analyze_emotion_to_art(song_name: str, song_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze survey responses and return explainable Emotion-to-Art payload."""
    if song_df is None or song_df.empty:
        raise ValueError("Secilen sarki icin anket verisi bulunamadi.")

    missing_cols = [
        column_name for column_name in EMOTION_COLUMNS.values() if column_name not in song_df.columns
    ]
    if missing_cols:
        raise ValueError(f"Duygu kolonlari eksik: {missing_cols}")

    emotion_scores = _collect_emotion_scores(song_df)
    distribution = _to_distribution(emotion_scores)
    dominant_distribution = distribution[:5]
    dominant_emotion_keys = [item["emotion_key"] for item in dominant_distribution]

    va_values: List[Tuple[float, float]] = []
    for _, row in song_df.iterrows():
        try:
            valence_1_9, arousal_1_9 = survey_row_to_va(row)
        except Exception:
            continue

        if pd.isna(valence_1_9) or pd.isna(arousal_1_9):
            continue

        valence_01 = max(0.0, min(1.0, (float(valence_1_9) - 1.0) / 8.0))
        arousal_01 = max(0.0, min(1.0, (float(arousal_1_9) - 1.0) / 8.0))
        va_values.append((valence_01, arousal_01))

    if va_values:
        valence_mean_01 = float(np.mean([item[0] for item in va_values]))
        arousal_mean_01 = float(np.mean([item[1] for item in va_values]))
    else:
        valence_mean_01 = 0.5
        arousal_mean_01 = 0.5

    valence_category = _valence_level(valence_mean_01)
    arousal_category = _arousal_level(arousal_mean_01)
    atmosphere = _determine_atmosphere(valence_category, arousal_category)

    probability_vector = [item["ratio"] for item in distribution]
    entropy_value = _normalized_entropy(probability_vector)
    variance_value = float(np.var(probability_vector)) if probability_vector else 0.0
    std_dev_value = float(np.std(probability_vector)) if probability_vector else 0.0
    diversity_value = float(
        sum(1 for value in probability_vector if value >= 0.03) / max(1, len(probability_vector))
    )
    intensity_label = _intensity_label(entropy_value)

    color_palette = _build_color_palette(dominant_emotion_keys, atmosphere)
    lighting = LIGHTING_RULES.get(atmosphere, "Soft Cinematic Light")
    composition = COMPOSITION_RULES.get(atmosphere, "Flowing, Organic, Minimal")
    brush_style = BRUSH_STYLE_RULES.get(atmosphere, "Mixed Media")
    symbolism = _build_symbolism(dominant_emotion_keys, atmosphere, valence_mean_01)

    generated_prompt = _build_prompt(
        dominant_items=dominant_distribution,
        atmosphere=atmosphere,
        intensity_label=intensity_label,
        entropy_value=entropy_value,
        diversity_value=diversity_value,
        color_palette=color_palette,
        composition=composition,
        lighting=lighting,
        brush_style=brush_style,
        symbolism=symbolism,
    )

    return {
        "song_name": str(song_name),
        "response_count": int(len(song_df)),
        "emotion_distribution": distribution,
        "dominant_emotions": dominant_distribution,
        "valence_mean": round(valence_mean_01, 4),
        "valence_category": valence_category,
        "arousal_mean": round(arousal_mean_01, 4),
        "arousal_category": arousal_category,
        "atmosphere": atmosphere,
        "emotion_intensity": intensity_label,
        "std_dev": round(std_dev_value, 6),
        "variance": round(variance_value, 6),
        "entropy": round(entropy_value, 6),
        "diversity": round(diversity_value, 6),
        "color_palette": color_palette,
        "composition": composition,
        "lighting": lighting,
        "art_style": brush_style,
        "brush_style": brush_style,
        "symbolism": symbolism,
        "generated_prompt": generated_prompt,
        "profile_signature": build_emotion_profile_signature(song_df),
        "academic_rationale": ACADEMIC_RATIONALE,
    }
