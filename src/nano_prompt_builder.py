"""Nano prompt quality and safety helpers for abstract art generation."""

from __future__ import annotations

from typing import Any, Dict

QUALITY_SUFFIX = (
    "Ultra-detailed high quality abstract artwork, modern cinematic mood, "
    "aesthetic composition, rich textures, atmospheric depth, 8k visual fidelity."
)

SAFETY_SUFFIX = (
    "Non-figurative abstract art only. No human, no face, no body, no character, "
    "no text, no letters, no typography, no logo, no watermark."
)


def build_fallback_nano_prompt(analysis: Dict[str, Any]) -> str:
    """Build a deterministic fallback prompt when Gemini nano_prompt is missing."""
    emotion_start = str(analysis.get("emotion_start", "melancholy")).strip() or "melancholy"
    emotion_end = str(analysis.get("emotion_end", "inner peace")).strip() or "inner peace"
    transition = str(analysis.get("transition", "dark to light")).strip() or "dark to light"
    art_style = str(analysis.get("art_style", "abstract expressionism")).strip() or "abstract expressionism"
    lighting = str(analysis.get("lighting", "soft cinematic")).strip() or "soft cinematic"
    composition = str(analysis.get("composition", "balanced minimalist")).strip() or "balanced minimalist"

    colors = analysis.get("dominant_colors", []) or []
    if isinstance(colors, str):
        colors = [c.strip() for c in colors.split(",") if c.strip()]
    color_text = ", ".join(str(c).strip() for c in colors if str(c).strip()) or "deep blue, amber, soft teal"

    elements = analysis.get("visual_elements", []) or []
    if isinstance(elements, str):
        elements = [e.strip() for e in elements.split(",") if e.strip()]
    elements_text = ", ".join(str(e).strip() for e in elements if str(e).strip()) or "mist, fluid waves, light particles"

    return (
        "Create an abstract emotional artwork inspired by a musical journey from "
        f"{emotion_start} to {emotion_end}, with a {transition} transition. "
        f"Use dominant colors: {color_text}. Include visual cues: {elements_text}. "
        f"Style: {art_style}. Lighting: {lighting}. Composition: {composition}. "
        f"{QUALITY_SUFFIX} {SAFETY_SUFFIX}"
    )


def ensure_nano_prompt_quality(nano_prompt: str, analysis: Dict[str, Any]) -> str:
    """Ensure the prompt includes mandatory quality and safety constraints."""
    base = (nano_prompt or "").strip()
    if len(base) < 40:
        base = build_fallback_nano_prompt(analysis)

    lowered = base.lower()

    if "abstract" not in lowered:
        base = f"{base} Abstract art direction."
    if "cinematic" not in lowered:
        base = f"{base} Cinematic lighting and atmosphere."
    if "high quality" not in lowered and "ultra-detailed" not in lowered:
        base = f"{base} {QUALITY_SUFFIX}"

    # Always reinforce hard constraints to keep outputs purely abstract.
    if "no human" not in lowered or "no text" not in lowered or "no logo" not in lowered:
        base = f"{base} {SAFETY_SUFFIX}"

    return " ".join(base.split())
