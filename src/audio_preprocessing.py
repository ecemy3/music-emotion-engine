"""
Eğitim (src/train_deam_cnn_va.py, src/train_cnn_experiments.py,
src/train_crnn_experiments.py) ve canlı çıkarım (app.py) için TEK ortak
ön işleme hattı.

Bu modülden önce app.py kendi SR=16000 / N_FFT=1024 tanımlarını kullanıyordu;
bu, modellerin eğitimde hiç görmediği bir girdi üretiyordu. Parametreler
burada eğitim scriptlerindeki değerlerle birebir sabitlenmiştir - hiçbir
yerde override edilmemelidir.

Not: Pad/trim mantığı (ilk 30 saniyeyi al) eğitimdekiyle aynı bırakıldı;
bu davranışın kendisi ayrı bir iyileştirme konusu olarak değerlendirilecek.
"""

import io
import os
from pathlib import Path
from typing import Union

import soundfile as sf
import torch
import torchaudio

# --- Eğitim scriptleriyle birebir aynı sabitler ---
SR = 22050
DURATION = 30
SAMPLES = SR * DURATION

N_FFT = 2048
HOP = 512
N_MELS = 128

# MelSpectrogram/AmplitudeToDB argümanları eğitim kodunda örtük (default)
# bırakılmıştı; burada torchaudio'nun o zamanki varsayılanlarıyla birebir
# aynı değerler açıkça yazılmıştır (torchaudio==2.5.1+cu118 referans alındı).
mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR,
    n_fft=N_FFT,
    win_length=N_FFT,
    hop_length=HOP,
    f_min=0.0,
    f_max=SR / 2,
    pad=0,
    n_mels=N_MELS,
    window_fn=torch.hann_window,
    power=2.0,
    normalized=False,
    center=True,
    pad_mode="reflect",
    norm=None,
    mel_scale="htk",
)
to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=None)

AudioInput = Union[str, "os.PathLike[str]", bytes, io.IOBase]


def load_audio_waveform(path_or_bytes: AudioInput) -> torch.Tensor:
    """
    Eğitim scriptlerindeki `load_audio` ile birebir aynı davranış:
    - soundfile ile dosyanın kendi native sample rate'inde oku (librosa YOK)
    - stereo ise torch.mean ile mono'ya çevir
    - native SR, hedef SR'den farklıysa torchaudio.functional.resample ile çevir
    - SAMPLES'tan kısaysa sona sıfır pad, uzunsa baştan SAMPLES kadar kes

    Args:
        path_or_bytes: dosya yolu (str/Path) ya da ham ses bytes'ı

    Returns:
        torch.Tensor: şekil (1, SAMPLES)

    Raises:
        RuntimeError: soundfile dosyayı okuyamazsa (örn. desteklenmeyen codec).
            Başka bir kütüphaneye sessizce geçilmez.
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
            "desteklemediği bir codec (ör. bazı MP3 kodlamaları) yüzünden olur. "
            "Dosyayı WAV'a çevirip tekrar deneyin; librosa'ya sessizce "
            "geçilmiyor çünkü bu, eğitimdeki ön işlemeden farklı bir hat "
            f"anlamına gelir. Orijinal hata: {exc!r}"
        ) from exc

    wav = torch.from_numpy(wav_np)

    if wav.ndim == 2:
        wav = torch.mean(wav, dim=1)

    wav = wav.unsqueeze(0)

    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)

    if wav.shape[1] < SAMPLES:
        wav = torch.nn.functional.pad(wav, (0, SAMPLES - wav.shape[1]))
    else:
        wav = wav[:, :SAMPLES]

    return wav


def wav_to_logmel(wav: torch.Tensor) -> torch.Tensor:
    """Eğitim scriptlerindeki `wav_to_logmel` ile birebir aynı: log-mel + örnek bazlı normalizasyon."""
    m = mel_spec(wav)
    m = to_db(m)
    m = (m - m.mean()) / (m.std() + 1e-6)
    return m


def load_and_preprocess(path_or_bytes: AudioInput) -> torch.Tensor:
    """
    Dosya yolundan ya da ham bytes'tan, eğitimdeki ön işlemeyle birebir aynı
    şekilde normalize edilmiş log-mel spektrogramı üretir.

    Returns:
        torch.Tensor: şekil (1, N_MELS, T) - batch boyutu eklenmemiştir,
        eğitimdeki Dataset.__getitem__ çıktısıyla aynı konvansiyon.
    """
    wav = load_audio_waveform(path_or_bytes)
    return wav_to_logmel(wav)
