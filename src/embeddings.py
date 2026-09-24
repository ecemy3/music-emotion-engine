"""
DEAM şarkıları için CLAP (laion/clap-htsat-unfused) ses embedding'lerini
çıkarır ve şarkı bazında ortalayıp cache/embeddings/clap_deam.npy olarak
kaydeder.

Pencereleme: 10 sn pencere, 5 sn kayma. src/inference.py'deki mantıkla
aynı şekilde son pencere parçanın sonuna hizalanır (dolgu yok); parça
10 sn'den kısaysa tek pencere 10 sn'ye pad'lenir.

src/audio_preprocessing.py'ye DOKUNULMADI. Burada CLAP'ın kendi beklediği
48 kHz örnekleme hızına özel, tamamen ayrı bir ön işleme hattı var -
eğitim scriptlerinin kullandığı 22050 Hz hattıyla hiçbir ilişkisi yok.

NOT: laion/clap-htsat-unfused checkpoint'i safetensors formatında değil
(sadece pytorch_model.bin var), bu yüzden transformers>=4.50 civarı
sürümler CVE-2025-32434 güvenlik kısıtı yüzünden (torch<2.6 ile) yüklemeyi
reddediyor. requirements.txt'te bu yüzden transformers==4.44.2 pinlendi.

Kullanım:
    python src/embeddings.py
"""

import io
import os
import time
from typing import Union

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from transformers import ClapModel, ClapProcessor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(REPO_ROOT, "data", "deam", "audio")
ANN_PATH = os.path.join(REPO_ROOT, "data", "deam", "annotations.csv")
CACHE_DIR = os.path.join(REPO_ROOT, "cache", "embeddings")
EMBEDDINGS_OUT = os.path.join(CACHE_DIR, "clap_deam.npy")
SONG_IDS_OUT = os.path.join(CACHE_DIR, "clap_deam_song_ids.npy")

CLAP_MODEL_NAME = "laion/clap-htsat-unfused"
CLAP_SR = 48000
WINDOW_SEC = 10.0
HOP_SEC = 5.0
WINDOW_SAMPLES = int(WINDOW_SEC * CLAP_SR)
HOP_SAMPLES = int(HOP_SEC * CLAP_SR)
EMBED_DIM = 512


def load_waveform_48k(path_or_bytes: Union[str, "os.PathLike[str]", bytes, io.IOBase]) -> torch.Tensor:
    """
    CLAP'ın beklediği 48 kHz mono waveform - src/audio_preprocessing.py'den
    tamamen bağımsız, ayrı bir okuma hattı. Kesme/pad burada YAPILMAZ
    (pencereleme _build_windows_no_padding'de ayrı ele alınır).
    """
    source = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
    wav_np, sr = sf.read(source, dtype="float32")
    wav = torch.from_numpy(wav_np)
    if wav.ndim == 2:
        wav = torch.mean(wav, dim=1)
    wav = wav.unsqueeze(0)
    if sr != CLAP_SR:
        wav = torchaudio.functional.resample(wav, sr, CLAP_SR)
    return wav


def build_windows_no_padding(wav: torch.Tensor):
    """
    0, 5, 10, ... sn şeklinde ilerleyen 10 sn'lik pencereler; son normal
    pencere parçanın sonuna ulaşmıyorsa sona hizalanan (örtüşebilen) ek bir
    pencere eklenir - src/inference.py'deki pencereleme mantığıyla aynı,
    dolgu yok. Parça 10 sn'den kısaysa tek pencere 10 sn'ye pad'lenir.
    """
    total = wav.shape[1]
    windows = []

    if total <= WINDOW_SAMPLES:
        pad = WINDOW_SAMPLES - total
        w = torch.nn.functional.pad(wav, (0, pad)) if pad > 0 else wav
        windows.append(w)
        return windows

    start = 0
    last_end = 0
    while start + WINDOW_SAMPLES <= total:
        windows.append(wav[:, start : start + WINDOW_SAMPLES])
        last_end = start + WINDOW_SAMPLES
        start += HOP_SAMPLES

    if last_end < total:
        tail_start = total - WINDOW_SAMPLES
        windows.append(wav[:, tail_start:total])

    return windows


def extract_song_embedding(model, processor, device, wav_path: str) -> np.ndarray:
    """Bir şarkının tüm pencerelerinin CLAP embedding'lerini çıkarıp ortalamasını döndürür (512,)."""
    wav = load_waveform_48k(wav_path)
    windows = build_windows_no_padding(wav)
    arrs = [w.squeeze(0).numpy() for w in windows]

    inputs = processor(audios=arrs, sampling_rate=CLAP_SR, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        window_embeds = model.get_audio_features(**inputs)  # (n_windows, 512)

    return window_embeds.mean(dim=0).cpu().numpy()


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"CLAP modeli yükleniyor: {CLAP_MODEL_NAME} (device={device})")
    processor = ClapProcessor.from_pretrained(CLAP_MODEL_NAME)
    model = ClapModel.from_pretrained(CLAP_MODEL_NAME).to(device)
    model.eval()

    ann = pd.read_csv(ANN_PATH)
    available = {int(f.split(".")[0]) for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")}
    song_ids = sorted(int(sid) for sid in ann["song_id"] if sid in available)
    print(f"İşlenecek şarkı sayısı: {len(song_ids)}")

    embeddings = np.zeros((len(song_ids), EMBED_DIM), dtype=np.float32)
    t0 = time.time()
    for i, sid in enumerate(song_ids):
        wav_path = os.path.join(AUDIO_DIR, f"{sid}.wav")
        embeddings[i] = extract_song_embedding(model, processor, device, wav_path)
        if (i + 1) % 100 == 0 or (i + 1) == len(song_ids):
            elapsed = time.time() - t0
            print(f"  [{i + 1}/{len(song_ids)}] işlendi ({elapsed:.1f} sn, {elapsed / (i + 1):.3f} sn/şarkı)")

    elapsed_total = time.time() - t0
    np.save(EMBEDDINGS_OUT, embeddings)
    np.save(SONG_IDS_OUT, np.array(song_ids, dtype=np.int64))

    print(f"\nToplam süre: {elapsed_total:.1f} sn ({elapsed_total / len(song_ids):.3f} sn/şarkı, device={device})")
    print(f"Embedding'ler kaydedildi: {EMBEDDINGS_OUT} (şekil {embeddings.shape})")
    print(f"Şarkı id eşlemesi: {SONG_IDS_OUT}")


if __name__ == "__main__":
    main()
