"""
Çoklu pencereli (multi-window / sliding-window) tahmin.

src/audio_preprocessing.py'deki tek-pencere hattı (SR/N_FFT/... sabitleri,
`load_audio_waveform`, `wav_to_logmel`) eğitim scriptleri tarafından
kullanıldığı için burada DEĞİŞTİRİLMEDİ - bu modül sadece onun üzerine bir
pencereleme + agregasyon katmanı ekler:

  - Parça baştan sona (kesilmeden) eğitimdeki gibi okunur (22050 Hz, mono).
  - 30 saniyelik pencerelere bölünür (model 30 sn ile eğitildi), pencere
    kayması 15 sn (%50 örtüşme).
  - Her pencerenin ham (normalize edilmemiş) RMS enerjisi hesaplanır; bu
    enerji sabit bir eşiğin altındaysa pencere "sessiz" sayılıp tahmine
    katılmaz. Bu kontrol örnek-bazlı normalizasyondan (wav_to_logmel içindeki
    (m - m.mean())/m.std()) ÖNCE yapılır, çünkü o normalizasyon sessiz bir
    pencereyi de diğerleri gibi std=1'e çekip saf gürültüyü anlamlı bir
    sinyalmiş gibi modele verir.
  - Mel dönüşümü her pencere için mevcut `wav_to_logmel` ile aynen
    uygulanır (pencere uzunluğu = eğitimdeki SAMPLES olduğu için mel şekli
    de eğitimdekiyle birebir aynıdır), pencereler batch halinde modele verilir.
  - Genel Valence/Arousal, sessiz olmayan pencerelerin ortalaması; ayrıca
    pencereler arası standart sapma ve tam zaman çizelgesi döndürülür.
"""

import io
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torchaudio
import soundfile as sf

from audio_preprocessing import SR, DURATION, wav_to_logmel

WINDOW_SEC = float(DURATION)  # 30 sn - modelin eğitildiği pencere uzunluğu
HOP_SEC = 15.0  # %50 örtüşme
WINDOW_SAMPLES = int(WINDOW_SEC * SR)
HOP_SAMPLES = int(HOP_SEC * SR)
MIN_TAIL_SAMPLES = int(HOP_SEC * SR)  # son pencere bundan kısaysa atılır

# Ham dalga formu üzerinde (normalizasyondan önce) RMS enerji eşiği.
# 0.01 ~ -40 dBFS: yaygın kullanılan, mutlak (dosyaya göre değişmeyen) bir
# "duyulabilir sinyal yok" sınırı. Sabittir; deneysel/dosya bazlı ayarlanmaz.
SILENCE_RMS_THRESHOLD = 0.01

AudioInput = Union[str, "os.PathLike[str]", bytes, io.IOBase]


@dataclass
class WindowPrediction:
    start_sec: float
    end_sec: float
    rms: float
    skipped: bool
    valence: Optional[float] = None
    arousal: Optional[float] = None


@dataclass
class MultiWindowResult:
    valence: float
    arousal: float
    valence_std: float
    arousal_std: float
    windows: List[WindowPrediction]
    n_windows_total: int
    n_windows_used: int
    n_windows_skipped: int
    used_all_windows_fallback: bool = False

    def to_records(self) -> List[Dict]:
        """Zaman çizelgesini tablo/DataFrame'e dönüştürmeye uygun kayıt listesi olarak döndür."""
        return [
            {
                "start_sec": w.start_sec,
                "end_sec": w.end_sec,
                "rms": w.rms,
                "skipped": w.skipped,
                "valence": w.valence,
                "arousal": w.arousal,
            }
            for w in self.windows
        ]


def _load_full_waveform(path_or_bytes: AudioInput) -> torch.Tensor:
    """
    Eğitimdeki okuma/mono/resample adımlarıyla birebir aynı, ama pad/trim
    YOK - parçanın tamamı döndürülür. (src/audio_preprocessing.py'nin
    `load_audio_waveform` fonksiyonu kasıtlı olarak değiştirilmedi; bu,
    onun pad/trim'siz hali.)
    """
    if isinstance(path_or_bytes, (bytes, bytearray)):
        source = io.BytesIO(path_or_bytes)
    else:
        source = path_or_bytes

    try:
        wav_np, sr = sf.read(source, dtype="float32")
    except Exception as exc:
        raise RuntimeError(
            "soundfile ses dosyasını okuyamadı. Bu genellikle libsndfile'ın "
            "desteklemediği bir codec yüzünden olur. Dosyayı WAV'a çevirip "
            f"tekrar deneyin. Orijinal hata: {exc!r}"
        ) from exc

    wav = torch.from_numpy(wav_np)
    if wav.ndim == 2:
        wav = torch.mean(wav, dim=1)
    wav = wav.unsqueeze(0)

    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)

    return wav


def _build_windows(wav: torch.Tensor):
    """wav: (1, total_samples). (start_sample, end_sample, pencere_tensörü) listesi döndürür.

    end_sample, pad edilmemiş asıl içerik sınırıdır (pencere tensörü pad
    edilmiş olsa bile zaman çizelgesinde gerçek süre gösterilsin diye).
    """
    total_samples = wav.shape[1]
    windows = []

    if total_samples <= WINDOW_SAMPLES:
        w = wav
        if total_samples < WINDOW_SAMPLES:
            w = torch.nn.functional.pad(w, (0, WINDOW_SAMPLES - total_samples))
        windows.append((0, total_samples, w))
        return windows

    start = 0
    while start < total_samples:
        end = start + WINDOW_SAMPLES
        if end <= total_samples:
            windows.append((start, end, wav[:, start:end]))
            start += HOP_SAMPLES
        else:
            remaining = total_samples - start
            if remaining < MIN_TAIL_SAMPLES:
                break
            w = torch.nn.functional.pad(wav[:, start:], (0, WINDOW_SAMPLES - remaining))
            windows.append((start, total_samples, w))
            break

    return windows


def _rms(wav: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(wav ** 2)).item())


def _forward_ensemble(selected_model_runtime: dict, mel_batch: torch.Tensor):
    """app.py'deki predict_with_selected_model ile aynı ağırlıklandırmayı batch üzerinde uygular."""
    if selected_model_runtime["type"] == "single":
        model = selected_model_runtime["model"]
        model.eval()
        with torch.no_grad():
            out = model(mel_batch).cpu().numpy()
        return out[:, 0], out[:, 1]

    weights = selected_model_runtime["weights"]
    models = selected_model_runtime["models"]
    n = mel_batch.shape[0]
    weighted_v = np.zeros(n, dtype=np.float64)
    weighted_a = np.zeros(n, dtype=np.float64)
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            out = model(mel_batch).cpu().numpy()
        w = weights[name]
        weighted_v += w * out[:, 0]
        weighted_a += w * out[:, 1]
    return weighted_v, weighted_a


def predict_multi_window(
    path_or_bytes: AudioInput,
    selected_model_runtime: dict,
    device: str = "cpu",
) -> MultiWindowResult:
    """
    Parçanın tamamını 30 sn / %50 örtüşmeli pencerelerle analiz eder.

    selected_model_runtime: app.py'deki ile aynı sözleşme -
        {"type": "single", "model": ...} veya
        {"type": "ensemble", "models": {...}, "weights": {...}}
    """
    wav = _load_full_waveform(path_or_bytes)
    raw_windows = _build_windows(wav)

    entries = []
    for start, end, w in raw_windows:
        rms = _rms(w)
        entries.append(
            {
                "start": start,
                "end": end,
                "wav": w,
                "rms": rms,
                "skipped": rms < SILENCE_RMS_THRESHOLD,
            }
        )

    used_fallback = False
    valid_entries = [e for e in entries if not e["skipped"]]
    if not valid_entries:
        # Tüm pencereler sessiz sayıldıysa (ör. tamamen sessiz/bozuk dosya):
        # hiç tahmin üretmemek yerine hepsini kullan, bunu raporda işaretle.
        used_fallback = True
        for e in entries:
            e["skipped"] = False
        valid_entries = entries

    mel_batch = torch.stack([wav_to_logmel(e["wav"]) for e in valid_entries], dim=0).to(device)
    per_window_v, per_window_a = _forward_ensemble(selected_model_runtime, mel_batch)

    for e, v, a in zip(valid_entries, per_window_v, per_window_a):
        e["valence"] = float(v)
        e["arousal"] = float(a)

    windows = [
        WindowPrediction(
            start_sec=e["start"] / SR,
            end_sec=e["end"] / SR,
            rms=e["rms"],
            skipped=e["skipped"],
            valence=e.get("valence"),
            arousal=e.get("arousal"),
        )
        for e in entries
    ]

    overall_v = float(np.mean(per_window_v))
    overall_a = float(np.mean(per_window_a))
    std_v = float(np.std(per_window_v)) if len(per_window_v) > 1 else 0.0
    std_a = float(np.std(per_window_a)) if len(per_window_a) > 1 else 0.0

    return MultiWindowResult(
        valence=overall_v,
        arousal=overall_a,
        valence_std=std_v,
        arousal_std=std_a,
        windows=windows,
        n_windows_total=len(entries),
        n_windows_used=len(valid_entries),
        n_windows_skipped=len(entries) - len(valid_entries),
        used_all_windows_fallback=used_fallback,
    )
