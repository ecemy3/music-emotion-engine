"""Deterministic Art DNA Engine for music-driven abstract artwork.

The engine converts survey-derived emotion signals into a reproducible,
multi-layer art identity. The same song and survey profile always produce the
same Art DNA and prompt, while different song identifiers perturb style,
composition, color balance, motion, texture, lighting, and symbolism.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.survey_utils import survey_row_to_va

ACADEMIC_RATIONALE = (
    "Art DNA Engine, yalnizca ilk bes baskin duyguyu kullanmak yerine tum duygu "
    "dagilimini, valence-arousal konumunu, entropy, cesitlilik, yogunluk ve "
    "pozitif/negatif denge metriklerini birlikte degerlendirir. Deterministik "
    "seed mekanizmasi ayni sarki icin ayni sanatsal kimligi korurken farkli "
    "sarkilarin renk oranlari, stil harmanlari, kompozisyon, hareket, doku, "
    "isik ve sembolizm katmanlarinda ayrismasini saglar."
)

EMOTION_COLUMNS: Dict[str, str] = {
    "calm": "Bu muzik beni huzurlu ve sakin hissettirdi. ",
    "happy": "Bu muzik beni neseli ve enerjik hissettirdi. ",
    "nostalgic": "Bu muzik bana nostaljik bir his verdi. ",
    "sad": "Bu muzik beni uzgun veya melankolik hissettirdi. ",
    "admiration": "Bu muzik bana hayranlik ve saskinlik hisleri uyandirdi. ",
    "tense": "Bu muzik beni gergin veya huzursuz hissettirdi. ",
    "motivated": "Bu muzik beni guclu, motive olmus hissettirdi. ",
}

ALT_EMOTION_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "calm": (
        "Bu muzik beni huzurlu ve sakin hissettirdi. ",
        "Bu m\u00fczik beni huzurlu ve sakin hissettirdi. ",
        "Bu mÃ¼zik beni huzurlu ve sakin hissettirdi. ",
    ),
    "happy": (
        "Bu muzik beni neseli ve enerjik hissettirdi. ",
        "Bu m\u00fczik beni ne\u015feli ve enerjik hissettirdi. ",
        "Bu mÃ¼zik beni neÅŸeli ve enerjik hissettirdi. ",
    ),
    "nostalgic": (
        "Bu muzik bana nostaljik bir his verdi. ",
        "Bu m\u00fczik bana nostaljik bir his verdi. ",
        "Bu mÃ¼zik bana nostaljik bir his verdi. ",
    ),
    "sad": (
        "Bu muzik beni uzgun veya melankolik hissettirdi. ",
        "Bu m\u00fczik beni \u00fczg\u00fcn veya melankolik hissettirdi. ",
        "Bu mÃ¼zik beni Ã¼zgÃ¼n veya melankolik hissettirdi. ",
    ),
    "admiration": (
        "Bu muzik bana hayranlik ve saskinlik hisleri uyandirdi. ",
        "Bu m\u00fczik bana hayranl\u0131k ve \u015fa\u015fk\u0131nl\u0131k hisleri uyand\u0131rd\u0131. ",
        "Bu mÃ¼zik bana hayranlÄ±k ve ÅŸaÅŸkÄ±nlÄ±k hisleri uyandÄ±rdÄ±. ",
    ),
    "tense": (
        "Bu muzik beni gergin veya huzursuz hissettirdi. ",
        "Bu m\u00fczik beni gergin veya huzursuz hissettirdi. ",
        "Bu mÃ¼zik beni gergin veya huzursuz hissettirdi. ",
    ),
    "motivated": (
        "Bu muzik beni guclu, motive olmus hissettirdi. ",
        "Bu m\u00fczik beni g\u00fc\u00e7l\u00fc, motive olmu\u015f hissettirdi. ",
        "Bu mÃ¼zik beni gÃ¼Ã§lÃ¼, motive olmuÅŸ hissettirdi. ",
    ),
}

TARGET_EMOTIONS: Tuple[str, ...] = (
    "peace",
    "nostalgia",
    "joy",
    "sadness",
    "motivation",
    "excitement",
    "romance",
    "loneliness",
    "freedom",
    "melancholy",
    "anxiety",
)

DISPLAY_LABELS: Dict[str, str] = {
    "peace": "Peace",
    "nostalgia": "Nostalgia",
    "joy": "Joy",
    "sadness": "Sadness",
    "motivation": "Motivation",
    "excitement": "Excitement",
    "romance": "Romance",
    "loneliness": "Loneliness",
    "freedom": "Freedom",
    "melancholy": "Melancholy",
    "anxiety": "Anxiety",
}

COLUMN_TO_EMOTION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "calm": {"peace": 1.0},
    "happy": {"joy": 0.6, "excitement": 0.4},
    "nostalgic": {"nostalgia": 0.7, "melancholy": 0.3},
    "sad": {"sadness": 0.6, "melancholy": 0.4},
    "admiration": {"excitement": 0.5, "freedom": 0.3, "motivation": 0.2},
    "tense": {"anxiety": 0.85, "melancholy": 0.15},
    "motivated": {"motivation": 0.7, "freedom": 0.3},
}

MOOD_COLUMNS_CANDIDATES: Tuple[str, ...] = (
    "Muzigi dinlemeden onceki duygu durumunuz",
    "Muzigi dinledikten sonraki duygu durumunuz.",
    "M\u00fczi\u011fi dinlemeden \u00f6nceki duygu durumunuz",
    "M\u00fczi\u011fi dinledikten sonraki duygu durumunuz.",
    "MÃ¼ziÄŸi dinlemeden Ã¶nceki duygu durumunuz",
    "MÃ¼ziÄŸi dinledikten sonraki duygu durumunuz.",
)

TEXT_EMOTION_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "peace": ("huzur", "sakin", "dingin", "rahat", "peace", "calm"),
    "nostalgia": ("nostal", "gecmis", "memory", "past"),
    "joy": ("mutlu", "neseli", "pozitif", "keyif", "happy", "joy"),
    "sadness": ("uzgun", "huzun", "keder", "sad"),
    "motivation": ("motive", "motivasyon", "guclu", "azimli", "power"),
    "excitement": ("heyecan", "cosku", "enerjik", "dinamik", "excite"),
    "romance": ("romantik", "ask", "sevgi", "love"),
    "loneliness": ("yalniz", "yalnizlik", "issiz", "alone"),
    "freedom": ("ozgur", "ozgurluk", "ferah", "acik ufuk", "free"),
    "melancholy": ("melankol", "huzunlu", "melanchol"),
    "anxiety": ("kaygi", "gergin", "huzursuz", "anksiy", "tense"),
}

POSITIVE_EMOTIONS = {"peace", "joy", "motivation", "excitement", "romance", "freedom"}
NEGATIVE_EMOTIONS = {"sadness", "loneliness", "melancholy", "anxiety"}

EMOTION_COLOR_BANK: Dict[str, Tuple[str, ...]] = {
    "peace": ("Soft Blue", "Ivory", "Silver", "Sage"),
    "nostalgia": ("Sepia", "Copper", "Dusty Rose", "Old Gold"),
    "joy": ("Sun Yellow", "Warm Orange", "Cream", "Coral"),
    "sadness": ("Navy", "Ash Gray", "Rain Blue", "Muted Violet"),
    "motivation": ("Amber", "Burnt Orange", "Gold", "Vermilion"),
    "excitement": ("Scarlet", "Electric Orange", "Amber", "Hot Coral"),
    "romance": ("Rose", "Plum", "Soft Pink", "Wine"),
    "loneliness": ("Pale White", "Smoke Gray", "Dust Blue", "Faded Cyan"),
    "freedom": ("Open Sky Blue", "Cyan", "White", "Turquoise"),
    "melancholy": ("Indigo", "Fog Gray", "Violet", "Deep Blue"),
    "anxiety": ("Graphite", "Steel Blue", "Deep Purple", "Acid Green"),
}

ATMOSPHERE_STYLES: Dict[str, Tuple[str, ...]] = {
    "Peaceful": ("Japanese Ink Wash", "Minimalism", "Organic Abstract"),
    "Energetic": ("Abstract Expressionism", "Action Painting", "Dynamic Modernism"),
    "Melancholic": ("Symbolism", "Atmospheric Painting", "Dreamscape"),
    "Chaotic": ("Cubism", "Deconstructivism", "Generative Geometry"),
    "Minimal": ("Minimalism", "Color Field Painting", "Organic Abstract"),
    "Experimental": ("Digital Surrealism", "Algorithmic Art", "Contemporary Mixed Media"),
}

COMPOSITION_OPTIONS = (
    "Golden Ratio",
    "Radial Composition",
    "Spiral Motion",
    "Asymmetric Balance",
    "Floating Geometry",
    "Fragmented Layers",
    "Negative Space",
    "Organic Flow",
    "Diagonal Movement",
    "Circular Harmony",
    "Concentric Waves",
    "Vertical Energy",
    "Horizontal Calm",
)

MOTION_OPTIONS = (
    "Calm Flow",
    "Upward Motion",
    "Clockwise Spiral",
    "Counter Clockwise Motion",
    "Explosive Burst",
    "Wave Rhythm",
    "Fractal Expansion",
    "Floating Drift",
    "Pulsating Energy",
)

TEXTURE_OPTIONS = (
    "Heavy Oil Impasto",
    "Watercolor Bloom",
    "Ink Diffusion",
    "Acrylic Wash",
    "Canvas Grain",
    "Marble Texture",
    "Metallic Pigment",
    "Dry Brush",
    "Palette Knife",
    "Mixed Media",
)

LIGHTING_OPTIONS = (
    "Soft Morning Light",
    "Golden Hour",
    "Museum Gallery Light",
    "Diffused Fog",
    "Dramatic Contrast",
    "Neon Reflection",
    "Ambient Glow",
    "Volumetric Light",
)

SPATIAL_OPTIONS = (
    "Center Focus",
    "Left Weighted",
    "Right Weighted",
    "Bottom Heavy",
    "Floating Islands",
    "Layered Depth",
    "Infinite Space",
    "Fractal Space",
)

SYMBOLISM_MAP: Dict[str, Tuple[str, ...]] = {
    "peace": ("Flowing Water", "Still Air", "Soft Horizon"),
    "joy": ("Rising Light", "Radiant Fields", "Bright Openings"),
    "nostalgia": ("Fading Memories", "Sepia Echoes", "Distant Windows"),
    "loneliness": ("Empty Horizon", "Isolated Island", "Silent Distance"),
    "excitement": ("Spiral Energy", "Ignition Marks", "Bursting Sparks"),
    "motivation": ("Ascending Geometry", "Rising Columns", "Forward Vectors"),
    "melancholy": ("Dissolving Mist", "Falling Veils", "Dim Reflections"),
    "freedom": ("Open Sky", "Unbound Lines", "Wide Horizon"),
    "sadness": ("Rain Veils", "Submerged Light", "Soft Collapse"),
    "romance": ("Interwoven Curves", "Rose Glow", "Tender Orbit"),
    "anxiety": ("Tension Lines", "Compressed Space", "Fractured Signals"),
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
        "Ã§": "c",
        "ÄŸ": "g",
        "Ä±": "i",
        "Ã¶": "o",
        "ÅŸ": "s",
        "Ã¼": "u",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _find_column(song_df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    normalized_columns = {_normalize_text(column): column for column in song_df.columns}
    for candidate in candidates:
        match = normalized_columns.get(_normalize_text(candidate))
        if match is not None:
            return match
    return None


def _coerce_likert(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").clip(lower=1, upper=5)


def _likert_to_unit(values: pd.Series) -> pd.Series:
    return ((values - 1.0) / 4.0).fillna(0.0)


def _collect_emotion_scores(song_df: pd.DataFrame) -> Dict[str, float]:
    scores = {emotion: 0.0 for emotion in TARGET_EMOTIONS}

    for signal_key, emotion_weights in COLUMN_TO_EMOTION_WEIGHTS.items():
        column_name = _find_column(song_df, ALT_EMOTION_COLUMNS[signal_key])
        if column_name is None:
            continue

        weighted_signal = float(_likert_to_unit(_coerce_likert(song_df[column_name])).sum())
        for emotion, weight in emotion_weights.items():
            scores[emotion] += weighted_signal * float(weight)

    for column_name in song_df.columns:
        normalized_name = _normalize_text(column_name)
        if not any(_normalize_text(candidate) == normalized_name for candidate in MOOD_COLUMNS_CANDIDATES):
            continue
        for raw_value in song_df[column_name].dropna():
            text = _normalize_text(raw_value)
            for emotion, keywords in TEXT_EMOTION_KEYWORDS.items():
                if any(keyword in text for keyword in keywords):
                    scores[emotion] += 1.0

    if sum(scores.values()) <= 0.0:
        scores["peace"] = 1.0
    return scores


def _to_distribution(scores: Dict[str, float]) -> List[Dict[str, Any]]:
    total_score = float(sum(scores.values()))
    distribution = []
    for emotion, raw_score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        ratio = (raw_score / total_score) if total_score > 0.0 else 0.0
        distribution.append(
            {
                "emotion_key": emotion,
                "emotion": DISPLAY_LABELS.get(emotion, emotion),
                "score": round(float(raw_score), 6),
                "count": int(round(raw_score)),
                "ratio": round(ratio, 6),
                "percentage": round(ratio * 100.0, 2),
            }
        )
    return distribution


def _normalized_entropy(probabilities: Iterable[float]) -> float:
    values = [float(value) for value in probabilities if float(value) > 0.0]
    if not values:
        return 0.0
    entropy_value = float(-sum(value * math.log2(value) for value in values))
    max_entropy = math.log2(len(TARGET_EMOTIONS))
    return max(0.0, min(1.0, entropy_value / max_entropy))


def _level(value: float, labels: Tuple[str, str, str, str, str]) -> str:
    if value < 0.2:
        return labels[0]
    if value < 0.4:
        return labels[1]
    if value < 0.6:
        return labels[2]
    if value < 0.8:
        return labels[3]
    return labels[4]


def _determine_atmosphere(valence: float, arousal: float, entropy: float) -> str:
    if entropy >= 0.86 and arousal >= 0.48:
        return "Experimental"
    if valence >= 0.55 and arousal < 0.48:
        return "Peaceful"
    if valence >= 0.55 and arousal >= 0.48:
        return "Energetic"
    if valence < 0.45 and arousal >= 0.65:
        return "Chaotic"
    if valence < 0.45:
        return "Melancholic"
    if arousal < 0.38:
        return "Minimal"
    return "Experimental"


def _profile_signature(song_df: pd.DataFrame) -> str:
    payload: Dict[str, Any] = {"rows": int(len(song_df)), "means": {}}
    for candidates in ALT_EMOTION_COLUMNS.values():
        column_name = _find_column(song_df, candidates)
        if column_name is None:
            continue
        values = _coerce_likert(song_df[column_name])
        payload["means"][column_name] = round(float(values.mean(skipna=True)), 6)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _make_seed(song_name: str, profile_signature: str, distribution: List[Dict[str, Any]]) -> int:
    compact_distribution = [
        (item["emotion_key"], round(float(item["ratio"]), 4)) for item in distribution
    ]
    seed_payload = json.dumps(
        {
            "song_name": str(song_name),
            "profile_signature": profile_signature,
            "distribution": compact_distribution,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(seed_payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _weighted_style_blend(rng: random.Random, atmosphere: str) -> List[Dict[str, Any]]:
    styles = list(ATMOSPHERE_STYLES.get(atmosphere, ATMOSPHERE_STYLES["Experimental"]))
    rng.shuffle(styles)
    chosen = styles[: rng.choice((2, 3))]
    raw_weights = [rng.uniform(0.25, 0.9) for _ in chosen]
    total = sum(raw_weights)
    weights = [max(5, int(round(value / total * 100.0))) for value in raw_weights]
    difference = 100 - sum(weights)
    weights[0] += difference
    return [{"style": style, "weight": weight} for style, weight in zip(chosen, weights)]


def _color_dna(
    rng: random.Random,
    distribution: List[Dict[str, Any]],
    atmosphere: str,
) -> List[Dict[str, Any]]:
    colors: List[str] = []
    for item in distribution:
        emotion_key = str(item["emotion_key"])
        for color in EMOTION_COLOR_BANK.get(emotion_key, ()):
            if color not in colors:
                colors.append(color)
            if len(colors) >= 8:
                break
        if len(colors) >= 8:
            break

    atmosphere_defaults = {
        "Peaceful": ("Ivory", "Soft Blue", "Sage"),
        "Energetic": ("Amber", "Scarlet", "Gold"),
        "Melancholic": ("Indigo", "Fog Gray", "Violet"),
        "Chaotic": ("Crimson", "Graphite", "Electric Blue"),
        "Minimal": ("Off White", "Soft Gray", "Pale Silver"),
        "Experimental": ("Neon Cyan", "Magenta", "Ultraviolet"),
    }
    for color in atmosphere_defaults.get(atmosphere, ()):
        if color not in colors:
            colors.append(color)

    rng.shuffle(colors)
    selected = colors[:4]
    roles = ("Primary", "Secondary", "Accent", "Highlight")
    primary = rng.randint(46, 70)
    secondary = rng.randint(16, min(32, 88 - primary))
    accent = rng.randint(7, min(18, 96 - primary - secondary))
    highlight = 100 - primary - secondary - accent
    weights = [primary, secondary, accent, highlight]
    return [
        {"role": role, "percentage": percentage, "color": color}
        for role, percentage, color in zip(roles, weights, selected)
    ]


def _choose_features(
    rng: random.Random,
    options: Sequence[str],
    count_range: Tuple[int, int],
    preferred: Sequence[str] = (),
) -> List[str]:
    count = rng.randint(count_range[0], count_range[1])
    pool = list(dict.fromkeys([*preferred, *options]))
    chosen = list(preferred[:count])
    remaining = [item for item in pool if item not in chosen]
    rng.shuffle(remaining)
    chosen.extend(remaining[: max(0, count - len(chosen))])
    return chosen[:count]


def _build_symbolism(
    rng: random.Random,
    distribution: List[Dict[str, Any]],
    valence: float,
    arousal: float,
) -> List[Dict[str, Any]]:
    symbols: List[Dict[str, Any]] = []
    for item in distribution[:5]:
        emotion_key = str(item["emotion_key"])
        candidates = list(SYMBOLISM_MAP.get(emotion_key, ()))
        if not candidates:
            continue
        symbol = rng.choice(candidates)
        symbols.append(
            {
                "emotion": item["emotion"],
                "symbol": symbol,
                "weight": round(float(item["ratio"]), 4),
            }
        )

    if valence >= 0.62 and not any(entry["symbol"] == "Rising Light" for entry in symbols):
        symbols.append({"emotion": "Valence", "symbol": "Rising Light", "weight": round(valence, 4)})
    if arousal >= 0.68 and not any("Spiral" in entry["symbol"] for entry in symbols):
        symbols.append({"emotion": "Arousal", "symbol": "Spiral Energy", "weight": round(arousal, 4)})

    return symbols[:6]


def _format_style(style_blend: List[Dict[str, Any]]) -> str:
    return ", ".join(f"{item['weight']}% {item['style']}" for item in style_blend)


def _format_color_dna(color_dna: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"{item['role']}: {item['percentage']}% {item['color']}" for item in color_dna
    )


def _format_emotion_profile(profile: Dict[str, Any]) -> str:
    dominant = ", ".join(
        f"{item['emotion']} {float(item['percentage']):.2f}%"
        for item in profile["dominant_emotions"]
    )
    return (
        f"Dominant emotions: {dominant}\n"
        f"Valence: {profile['valence']:.3f} ({profile['valence_level']})\n"
        f"Arousal: {profile['arousal']:.3f} ({profile['arousal_level']})\n"
        f"Entropy: {profile['entropy']:.3f}\n"
        f"Diversity: {profile['diversity']:.3f}\n"
        f"Emotional intensity: {profile['emotional_intensity']:.3f} "
        f"({profile['intensity_label']})\n"
        f"Positive ratio: {profile['positive_ratio']:.3f}\n"
        f"Negative ratio: {profile['negative_ratio']:.3f}\n"
        f"Emotional balance: {profile['emotional_balance']:.3f}"
    )


def _build_prompt(art_dna: Dict[str, Any]) -> str:
    symbolism_lines = [
        f"- {item['emotion']} -> {item['symbol']} (weight {item['weight']:.3f})"
        for item in art_dna["symbolism"]
    ]
    prompt = f"""
Create a completely unique museum-quality contemporary abstract artwork.

This artwork must have its own visual identity and must not resemble common abstract paintings.

Art Style:
{_format_style(art_dna["art_style"])}

Emotion Profile:
{_format_emotion_profile(art_dna["emotion_profile"])}

Atmosphere:
{art_dna["atmosphere"]}

Color DNA:
{_format_color_dna(art_dna["color_dna"])}

Composition DNA:
{", ".join(art_dna["composition_dna"])}

Motion DNA:
{", ".join(art_dna["motion_dna"])}

Texture DNA:
{", ".join(art_dna["texture_dna"])}

Lighting DNA:
{", ".join(art_dna["lighting_dna"])}

Spatial Layout:
{", ".join(art_dna["spatial_layout"])}

Symbolism:
{chr(10).join(symbolism_lines)}

Highly expressive.

Original.

Complex.

Layered.

Textured.

Contemporary fine art.

Cinematic.

Museum quality.

Ultra detailed.

8K.

No people.

No faces.

No text.

No typography.

No logos.

No watermark.

Generate a visually distinctive composition that is clearly different from previous artworks.

Avoid repetitive layouts, repetitive color balance, repetitive geometry and repetitive symbolism.
"""
    return "\n".join(line.rstrip() for line in prompt.splitlines()).strip()


def build_emotion_profile_signature(song_df: pd.DataFrame) -> str:
    """Create deterministic signature for cache/session invalidation."""
    return _profile_signature(song_df)


def analyze_art_dna(song_name: str, song_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze survey responses and return deterministic Art DNA payload."""
    if song_df is None or song_df.empty:
        raise ValueError("Secilen sarki icin anket verisi bulunamadi.")

    found_columns = [
        _find_column(song_df, ALT_EMOTION_COLUMNS[signal_key])
        for signal_key in COLUMN_TO_EMOTION_WEIGHTS
    ]
    missing_signals = [
        signal_key
        for signal_key, column_name in zip(COLUMN_TO_EMOTION_WEIGHTS, found_columns)
        if column_name is None
    ]
    if missing_signals:
        raise ValueError(f"Duygu sinyal kolonlari eksik: {missing_signals}")

    scores = _collect_emotion_scores(song_df)
    distribution = _to_distribution(scores)
    probabilities = [float(item["ratio"]) for item in distribution]

    va_values: List[Tuple[float, float]] = []
    for _, row in song_df.iterrows():
        try:
            valence_1_9, arousal_1_9 = survey_row_to_va(row)
        except Exception:
            continue
        if pd.isna(valence_1_9) or pd.isna(arousal_1_9):
            continue
        va_values.append(
            (
                max(0.0, min(1.0, (float(valence_1_9) - 1.0) / 8.0)),
                max(0.0, min(1.0, (float(arousal_1_9) - 1.0) / 8.0)),
            )
        )

    valence = float(np.mean([item[0] for item in va_values])) if va_values else 0.5
    arousal = float(np.mean([item[1] for item in va_values])) if va_values else 0.5
    entropy = _normalized_entropy(probabilities)
    diversity = float(sum(1 for value in probabilities if value >= 0.03) / len(TARGET_EMOTIONS))
    emotional_intensity = float(max(probabilities) * 0.55 + arousal * 0.30 + (1.0 - entropy) * 0.15)
    positive_ratio = float(sum(item["ratio"] for item in distribution if item["emotion_key"] in POSITIVE_EMOTIONS))
    negative_ratio = float(sum(item["ratio"] for item in distribution if item["emotion_key"] in NEGATIVE_EMOTIONS))
    emotional_balance = float(positive_ratio - negative_ratio)
    atmosphere = _determine_atmosphere(valence=valence, arousal=arousal, entropy=entropy)

    profile_signature = _profile_signature(song_df)
    seed = _make_seed(song_name=song_name, profile_signature=profile_signature, distribution=distribution)
    rng = random.Random(seed)

    composition_preferred = {
        "Peaceful": ("Organic Flow", "Horizontal Calm", "Negative Space"),
        "Energetic": ("Diagonal Movement", "Radial Composition", "Vertical Energy"),
        "Melancholic": ("Layered Depth", "Negative Space", "Floating Geometry"),
        "Chaotic": ("Fragmented Layers", "Asymmetric Balance", "Spiral Motion"),
        "Minimal": ("Negative Space", "Golden Ratio", "Horizontal Calm"),
        "Experimental": ("Floating Geometry", "Asymmetric Balance", "Concentric Waves"),
    }
    motion_preferred = {
        "Peaceful": ("Calm Flow", "Floating Drift", "Wave Rhythm"),
        "Energetic": ("Upward Motion", "Explosive Burst", "Pulsating Energy"),
        "Melancholic": ("Floating Drift", "Wave Rhythm", "Counter Clockwise Motion"),
        "Chaotic": ("Fractal Expansion", "Explosive Burst", "Clockwise Spiral"),
        "Minimal": ("Calm Flow", "Floating Drift"),
        "Experimental": ("Fractal Expansion", "Pulsating Energy", "Clockwise Spiral"),
    }
    texture_preferred = {
        "Peaceful": ("Watercolor Bloom", "Ink Diffusion", "Acrylic Wash"),
        "Energetic": ("Heavy Oil Impasto", "Palette Knife", "Metallic Pigment"),
        "Melancholic": ("Canvas Grain", "Dry Brush", "Watercolor Bloom"),
        "Chaotic": ("Palette Knife", "Mixed Media", "Heavy Oil Impasto"),
        "Minimal": ("Acrylic Wash", "Canvas Grain", "Ink Diffusion"),
        "Experimental": ("Mixed Media", "Metallic Pigment", "Marble Texture"),
    }
    lighting_preferred = {
        "Peaceful": ("Soft Morning Light", "Ambient Glow", "Museum Gallery Light"),
        "Energetic": ("Golden Hour", "Dramatic Contrast", "Volumetric Light"),
        "Melancholic": ("Diffused Fog", "Ambient Glow", "Museum Gallery Light"),
        "Chaotic": ("Dramatic Contrast", "Neon Reflection", "Volumetric Light"),
        "Minimal": ("Museum Gallery Light", "Soft Morning Light", "Ambient Glow"),
        "Experimental": ("Neon Reflection", "Volumetric Light", "Dramatic Contrast"),
    }
    spatial_preferred = {
        "Peaceful": ("Center Focus", "Layered Depth", "Floating Islands"),
        "Energetic": ("Right Weighted", "Floating Islands", "Layered Depth"),
        "Melancholic": ("Bottom Heavy", "Infinite Space", "Left Weighted"),
        "Chaotic": ("Fractal Space", "Right Weighted", "Floating Islands"),
        "Minimal": ("Center Focus", "Infinite Space", "Left Weighted"),
        "Experimental": ("Fractal Space", "Infinite Space", "Floating Islands"),
    }

    emotion_profile = {
        "dominant_emotions": distribution[:5],
        "valence": round(valence, 6),
        "valence_level": _level(valence, ("Very Negative", "Negative", "Neutral", "Positive", "Very Positive")),
        "arousal": round(arousal, 6),
        "arousal_level": _level(arousal, ("Very Calm", "Calm", "Moderate Energy", "High Energy", "Extreme Energy")),
        "entropy": round(entropy, 6),
        "diversity": round(diversity, 6),
        "emotional_intensity": round(emotional_intensity, 6),
        "intensity_label": _level(emotional_intensity, ("Subtle", "Soft", "Balanced", "Intense", "Extreme")),
        "positive_ratio": round(positive_ratio, 6),
        "negative_ratio": round(negative_ratio, 6),
        "emotional_balance": round(emotional_balance, 6),
    }

    art_dna = {
        "song_name": str(song_name),
        "seed": seed,
        "profile_signature": profile_signature,
        "response_count": int(len(song_df)),
        "emotion_distribution": distribution,
        "emotion_profile": emotion_profile,
        "dominant_emotions": distribution[:5],
        "valence_mean": round(valence, 4),
        "valence_category": emotion_profile["valence_level"],
        "arousal_mean": round(arousal, 4),
        "arousal_category": emotion_profile["arousal_level"],
        "entropy": round(entropy, 6),
        "diversity": round(diversity, 6),
        "emotion_intensity": emotion_profile["intensity_label"],
        "positive_ratio": round(positive_ratio, 6),
        "negative_ratio": round(negative_ratio, 6),
        "emotional_balance": round(emotional_balance, 6),
        "atmosphere": atmosphere,
        "art_style": _weighted_style_blend(rng, atmosphere),
        "color_dna": _color_dna(rng, distribution, atmosphere),
        "composition_dna": _choose_features(
            rng, COMPOSITION_OPTIONS, (2, 4), composition_preferred[atmosphere]
        ),
        "motion_dna": _choose_features(rng, MOTION_OPTIONS, (1, 3), motion_preferred[atmosphere]),
        "texture_dna": _choose_features(
            rng, TEXTURE_OPTIONS, (2, 3), texture_preferred[atmosphere]
        ),
        "lighting_dna": _choose_features(
            rng, LIGHTING_OPTIONS, (1, 2), lighting_preferred[atmosphere]
        ),
        "spatial_layout": _choose_features(
            rng, SPATIAL_OPTIONS, (1, 3), spatial_preferred[atmosphere]
        ),
        "symbolism": _build_symbolism(rng, distribution, valence, arousal),
        "academic_rationale": ACADEMIC_RATIONALE,
    }
    art_dna["generated_prompt"] = _build_prompt(art_dna)

    # Backward-compatible aliases for older UI/reporting code.
    art_dna["color_palette"] = [
        f"{item['role']}: {item['percentage']}% {item['color']}" for item in art_dna["color_dna"]
    ]
    art_dna["composition"] = ", ".join(art_dna["composition_dna"])
    art_dna["lighting"] = ", ".join(art_dna["lighting_dna"])
    art_dna["brush_style"] = ", ".join(art_dna["texture_dna"])

    return art_dna


def analyze_emotion_to_art(song_name: str, song_df: pd.DataFrame) -> Dict[str, Any]:
    """Backward-compatible entry point for the former Emotion-to-Art API."""
    return analyze_art_dna(song_name=song_name, song_df=song_df)
