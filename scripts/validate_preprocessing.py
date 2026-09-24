"""
configs/fixed_split_seed42.csv test setinden rastgele 20 şarkı seçip:

  1. Yeni ön işleme hattıyla (src/audio_preprocessing.py - eğitimle birebir
     aynı: SR=22050, N_FFT=2048) tahmin üretir ve results/<model>_predictions.csv
     içindeki (eğitim sırasında üretilmiş) referans tahminlerle karşılaştırır.
     Bu, modülü ayrı bir dosyaya çıkarmanın hiçbir sayısal fark yaratmadığını
     doğrular - ortalama mutlak fark ~0 olmalı.

  2. Aynı şarkılar için ESKİ (app.py'nin PR öncesi kullandığı, SR=16000,
     N_FFT=1024, librosa tabanlı) hattı da çalıştırıp aynı referansla
     karşılaştırır. Bu, orijinal hatanın (modelin hiç görmediği bir girdi
     alması) tahminler üzerindeki sayısal etkisini gösterir.

  3. transformer_v2 ve cnn_transformer_v1 modellerinde sabit uzunlukta
     positional embedding olup olmadığını, hem eski (~938 kare) hem yeni
     (~1293 kare) zaman ekseni uzunluğuyla ileri geçiş yaparak doğrular.

Kullanım:
    python scripts/validate_preprocessing.py
"""

import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torchaudio

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from audio_preprocessing import (  # noqa: E402
    DURATION,
    load_audio_waveform,
    wav_to_logmel,
)
from train_cnn_experiments import CNNVA  # noqa: E402
from train_crnn_experiments import AudioTransformerVA, CNNTransformerVA, CRNNVA  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AUDIO_DIR = os.path.join(REPO_ROOT, "data", "deam", "audio")
SPLIT_CSV = os.path.join(REPO_ROOT, "configs", "fixed_split_seed42.csv")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
MODELS_DIR = os.path.join(REPO_ROOT, "models")

SEED = 42
N_SONGS = 20
DIFF_THRESHOLD = 0.01

NO_FIXED_LENGTH_MODELS = {"transformer_v2", "cnn_transformer_v1"}

# --- Eski (buggy) hat: app.py'nin bu düzeltmeden önce kullandığı parametreler.
# Sadece karşılaştırma/etki ölçümü için burada tutulur; canlı koda geri
# eklenmemiştir.
OLD_SR = 16000
OLD_N_FFT = 1024
OLD_HOP = 512
OLD_N_MELS = 128
OLD_SAMPLES = OLD_SR * DURATION

_old_mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=OLD_SR, n_fft=OLD_N_FFT, hop_length=OLD_HOP, n_mels=OLD_N_MELS
)
_old_to_db = torchaudio.transforms.AmplitudeToDB()

MODEL_SPECS = {
    "cnn_baseline": lambda: CNNVA(dropout=0.4, architecture="baseline"),
    "cnn_optimized": lambda: CNNVA(dropout=0.4, architecture="optimized"),
    "crnn_v2": lambda: CRNNVA(hidden_size=128, num_layers=2, bidirectional=True, dropout=0.3),
    "transformer_v2": lambda: AudioTransformerVA(),
    "cnn_transformer_v1": lambda: CNNTransformerVA(),
}


def load_model(name: str):
    model = MODEL_SPECS[name]().to(DEVICE)
    ckpt_path = os.path.join(MODELS_DIR, f"{name}_best.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def old_load_audio(path: str) -> torch.Tensor:
    """app.py'nin PR öncesi kullandığı librosa tabanlı hat (SR=16000, pad/trim)."""
    import librosa

    wav_np, sr = librosa.load(path, sr=OLD_SR, mono=True)
    wav = torch.from_numpy(wav_np).unsqueeze(0)
    if wav.shape[1] < OLD_SAMPLES:
        wav = torch.nn.functional.pad(wav, (0, OLD_SAMPLES - wav.shape[1]))
    else:
        wav = wav[:, :OLD_SAMPLES]
    return wav


def old_wav_to_logmel(wav: torch.Tensor) -> torch.Tensor:
    m = _old_mel_spec(wav)
    m = _old_to_db(m)
    m = (m - m.mean()) / (m.std() + 1e-6)
    return m


def predict(model, mel: torch.Tensor):
    with torch.no_grad():
        v, a = model(mel.unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
    return float(v), float(a)


def main():
    random.seed(SEED)

    split_df = pd.read_csv(SPLIT_CSV)
    test_ids = split_df.loc[split_df["split"] == "test", "song_id"].astype(int).tolist()
    sample_ids = sorted(random.sample(test_ids, min(N_SONGS, len(test_ids))))

    print(f"Test setinden seçilen {len(sample_ids)} şarkı: {sample_ids}\n")

    new_mels, old_mels, missing = {}, {}, []
    old_pipeline_available = True

    for sid in sample_ids:
        wav_path = os.path.join(AUDIO_DIR, f"{sid}.wav")
        if not os.path.exists(wav_path):
            missing.append(sid)
            continue

        new_mels[sid] = wav_to_logmel(load_audio_waveform(wav_path))

        try:
            old_mels[sid] = old_wav_to_logmel(old_load_audio(wav_path))
        except ImportError:
            old_pipeline_available = False
            old_mels[sid] = None

    if missing:
        print(f"UYARI: ses dosyası bulunamayan song_id'ler atlandı: {missing}")
    if not old_pipeline_available:
        print("UYARI: librosa kurulu değil, eski hat karşılaştırması atlanıyor.")

    example_sid = next(iter(new_mels))
    new_shape = tuple(new_mels[example_sid].shape)
    print(f"Yeni hat mel şekli (örnek song {example_sid}): {new_shape} (T={new_shape[-1]})")
    if old_mels.get(example_sid) is not None:
        old_shape = tuple(old_mels[example_sid].shape)
        print(f"Eski hat mel şekli (örnek song {example_sid}): {old_shape} (T={old_shape[-1]})")
    print()

    report_rows = []

    for model_name in MODEL_SPECS:
        print(f"=== {model_name} ===")
        pred_csv = os.path.join(RESULTS_DIR, f"{model_name}_predictions.csv")
        if not os.path.exists(pred_csv):
            print(f"  HATA: {pred_csv} bulunamadı, atlanıyor.")
            continue
        ref_df = pd.read_csv(pred_csv).set_index("sample_id")

        try:
            model = load_model(model_name)
        except Exception as exc:
            print(f"  HATA: model yüklenemedi: {exc}")
            continue

        new_diffs, old_diffs = [], []
        new_forward_ok, old_forward_ok = True, True

        for sid in sample_ids:
            if sid not in new_mels or sid not in ref_df.index:
                continue
            ref_v = float(ref_df.loc[sid, "pred_valence"])
            ref_a = float(ref_df.loc[sid, "pred_arousal"])

            try:
                nv, na = predict(model, new_mels[sid])
                new_diffs.append(abs(nv - ref_v))
                new_diffs.append(abs(na - ref_a))
            except Exception as exc:
                new_forward_ok = False
                print(f"  song {sid}: YENİ hatta forward pass hatası: {exc}")

            if old_mels.get(sid) is not None:
                try:
                    ov, oa = predict(model, old_mels[sid])
                    old_diffs.append(abs(ov - ref_v))
                    old_diffs.append(abs(oa - ref_a))
                except Exception as exc:
                    old_forward_ok = False
                    print(f"  song {sid}: ESKİ hatta forward pass hatası: {exc}")

        mean_new = float(np.mean(new_diffs)) if new_diffs else float("nan")
        mean_old = float(np.mean(old_diffs)) if old_diffs else float("nan")

        print(f"  Yeni hat - ortalama mutlak fark (referansa göre): {mean_new:.4f}")
        print(f"  Eski hat - ortalama mutlak fark (referansa göre): {mean_old:.4f}")

        if mean_new > DIFF_THRESHOLD:
            print(
                f"  UYARI: yeni hattın farkı eşik değeri {DIFF_THRESHOLD}'ı aşıyor, "
                f"nedeni araştırılmalı (parametre uyuşmazlığı, checkpoint/mimari "
                f"eşleşmesi, batch-size'a bağlı BatchNorm/normalizasyon farkı vb.)."
            )

        if model_name in NO_FIXED_LENGTH_MODELS:
            print(
                f"  Sabit-uzunluk bağımlılığı kontrolü: yeni hat (~1293 kare) "
                f"forward pass {'BAŞARILI' if new_forward_ok else 'BAŞARISIZ'}, "
                f"eski hat (~938 kare) forward pass "
                f"{'BAŞARILI' if old_forward_ok else 'BAŞARISIZ'} -> "
                f"{'sabit uzunluk bağımlılığı YOK, model her iki uzunlukla da çalışıyor.' if (new_forward_ok and old_forward_ok) else 'sabit uzunluk bağımlılığı OLABİLİR, incelenmeli.'}"
            )

        report_rows.append(
            {
                "model": model_name,
                "n_compared": len(new_diffs) // 2,
                "mean_abs_diff_new_pipeline": mean_new,
                "mean_abs_diff_old_pipeline": mean_old,
                "new_pipeline_forward_ok": new_forward_ok,
                "old_pipeline_forward_ok": old_forward_ok,
            }
        )
        print()

    report_df = pd.DataFrame(report_rows)
    print("=== Özet ===")
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()
