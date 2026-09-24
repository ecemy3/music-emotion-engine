"""
Birden fazla aday müziği hedeflenen bir duyguya (Valence-Arousal) göre
sıralayan saf fonksiyonlar - Streamlit'ten bağımsız, test edilebilir.

Model tahminleri src/inference.py'nin çoklu pencereli hattı ile üretilir;
bu modül src/audio_preprocessing.py ve src/inference.py'nin davranışını
DEĞİŞTİRMEZ, sadece onların çıktısını sıralama/karşılaştırma metriklerine
dönüştürür.
"""

import math
import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import yaml

from inference import AudioInput, MultiWindowResult, predict_multi_window

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMOTION_TARGETS_PATH = os.path.join(REPO_ROOT, "configs", "emotion_targets.yaml")

# Model hata payı yaklaşık 0.8 RMSE (bkz. scripts/validate_multi_window.py
# doğrulama sonuçları: RMSE_Valence≈0.81, RMSE_Arousal≈0.79). Bu yüzden VA
# uzayındaki uzaklıkları birbirinden bu değerden az farklı olan adaylar
# "istatistiksel olarak ayırt edilemez" sayılır. Sabittir.
INDISTINGUISHABLE_THRESHOLD = 0.5

# Bir pencerenin "hedefe yakın" sayılması için VA uzayındaki Öklid uzaklık sınırı.
CLOSE_WINDOW_THRESHOLD = 1.0


@dataclass
class EmotionTarget:
    valence: float
    arousal: float
    label: Optional[str] = None


def load_emotion_targets(path: str = EMOTION_TARGETS_PATH) -> Dict[str, EmotionTarget]:
    """configs/emotion_targets.yaml içindeki etiket -> VA eşlemesini yükle."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return {
        name: EmotionTarget(valence=float(values["valence"]), arousal=float(values["arousal"]), label=name)
        for name, values in raw.items()
    }


def resolve_target(
    target_label: Optional[str] = None,
    manual_valence: Optional[float] = None,
    manual_arousal: Optional[float] = None,
    targets: Optional[Dict[str, EmotionTarget]] = None,
) -> "tuple[float, float]":
    """Etiket ya da manuel VA değerlerinden hedef (valence, arousal) döndür."""
    if target_label is not None:
        targets = targets if targets is not None else load_emotion_targets()
        if target_label not in targets:
            raise ValueError(
                f"Bilinmeyen hedef etiket: {target_label!r}. "
                f"Geçerli etiketler: {sorted(targets.keys())}"
            )
        t = targets[target_label]
        return t.valence, t.arousal

    if manual_valence is None or manual_arousal is None:
        raise ValueError("target_label ya da (manual_valence, manual_arousal) verilmeli.")
    return float(manual_valence), float(manual_arousal)


def euclidean_distance(v1: float, a1: float, v2: float, a2: float) -> float:
    return math.sqrt((v1 - v2) ** 2 + (a1 - a2) ** 2)


@dataclass
class CandidateRanking:
    name: str
    valence: float
    arousal: float
    valence_std: float
    arousal_std: float
    distance: float
    consistency: float
    pct_windows_close: float
    n_windows_used: int
    rank: int = 0
    group_id: int = 0
    indistinguishable_from_best: bool = False
    multi_window_result: Optional[MultiWindowResult] = field(default=None, repr=False, compare=False)


def _pct_windows_close(
    windows,
    target_v: float,
    target_a: float,
    threshold: float = CLOSE_WINDOW_THRESHOLD,
) -> float:
    """Sessiz olmayan pencerelerin yüzde kaçı hedefe `threshold` birimden yakın."""
    used = [w for w in windows if not w.skipped]
    if not used:
        return 0.0
    close = sum(
        1 for w in used if euclidean_distance(w.valence, w.arousal, target_v, target_a) <= threshold
    )
    return 100.0 * close / len(used)


def score_candidate(name: str, mw: MultiWindowResult, target_v: float, target_a: float) -> CandidateRanking:
    """Bir MultiWindowResult'tan sıralama metriklerini hesapla (henüz rank/group atanmamış)."""
    distance = euclidean_distance(mw.valence, mw.arousal, target_v, target_a)
    # Tutarlılık: valence ve arousal std'lerinin birleşik (Öklid) büyüklüğü.
    # Düşük değer = pencereler arası tahmin daha tutarlı.
    consistency = math.sqrt(mw.valence_std ** 2 + mw.arousal_std ** 2)
    pct_close = _pct_windows_close(mw.windows, target_v, target_a)

    return CandidateRanking(
        name=name,
        valence=mw.valence,
        arousal=mw.arousal,
        valence_std=mw.valence_std,
        arousal_std=mw.arousal_std,
        distance=distance,
        consistency=consistency,
        pct_windows_close=pct_close,
        n_windows_used=mw.n_windows_used,
        multi_window_result=mw,
    )


def rank_candidates(
    candidates: List[CandidateRanking],
    threshold: float = INDISTINGUISHABLE_THRESHOLD,
) -> List[CandidateRanking]:
    """
    Adayları uzaklığa göre sırala; rank, group_id ve
    indistinguishable_from_best alanlarını doldurarak YENİ bir liste döndürür
    (girdi listesi değiştirilmez).

    - rank: 1 = en yakın (en iyi) aday.
    - group_id: sıralı listede art arda gelen adaylar arasındaki uzaklık farkı
      `threshold`'dan küçükse aynı gruba (zincirleme) atanır. Zincirleme
      olduğu için bir grubun uç noktaları arasındaki fark threshold'u
      aşabilir; asıl karar için `indistinguishable_from_best` kullanın.
    - indistinguishable_from_best: adayın uzaklığı ile en iyi adayın
      uzaklığı arasındaki fark `threshold`'dan küçükse True (en iyiyle
      doğrudan karşılaştırma, zincirleme değil).
    """
    ordered = sorted(candidates, key=lambda c: c.distance)
    if not ordered:
        return ordered

    best_distance = ordered[0].distance
    result = []
    group_id = 0
    prev_distance = None

    for i, c in enumerate(ordered):
        if prev_distance is not None and (c.distance - prev_distance) >= threshold:
            group_id += 1
        result.append(
            replace(
                c,
                rank=i + 1,
                group_id=group_id,
                indistinguishable_from_best=(c.distance - best_distance) < threshold,
            )
        )
        prev_distance = c.distance

    return result


def rank_audio_candidates(
    audio_inputs: Dict[str, AudioInput],
    selected_model_runtime: dict,
    target_v: float,
    target_a: float,
    device: str = "cpu",
) -> List[CandidateRanking]:
    """
    Her aday ses için src/inference.py ile çoklu pencere tahmini yapıp
    hedefe göre sıralar.

    audio_inputs: {görünen_ad: dosya_yolu_veya_bytes} eşlemesi.
    selected_model_runtime: src/inference.py'nin beklediği sözleşme -
        {"type": "single", "model": ...} veya
        {"type": "ensemble", "models": {...}, "weights": {...}}
    """
    candidates = [
        score_candidate(name, predict_multi_window(audio_input, selected_model_runtime, device=device), target_v, target_a)
        for name, audio_input in audio_inputs.items()
    ]
    return rank_candidates(candidates)
