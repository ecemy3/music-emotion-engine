"""
Çoklu pencereli (multi-window) tahminin doğrulanması.

  1. configs/fixed_split_seed42.csv test setinin TAMAMINDA, eski yöntem
     (sadece ilk 30 sn) ve yeni yöntem (30 sn / %50 örtüşmeli çoklu pencere,
     dolgu yok) için ensemble tahminlerinin data/deam/annotations.csv'deki
     gerçek (anket ortalaması) değerlere göre RMSE ve Pearson korelasyonunu
     (valence ve arousal ayrı ayrı) hesaplar ve tablo olarak gösterir.

     Not: "eski ve yeni sonuçlar neredeyse aynı olmalı" beklentisi yanlıştı
     (bkz. önceki doğrulama) - asıl soru hangi yöntemin gerçek etiketlere
     daha yakın olduğu, bu yüzden doğrulama RMSE/Pearson'a dayanıyor.

  2. data/Adagio-for-strings.mp3 (tam uzunluklu, ~417 sn bir parça) için
     pencere sayısını, pencere başına dolgu oranını (30 sn'den uzun bir
     parça olduğu için hepsi 0 olmalı), genel skoru, eski yöntemle farkını,
     toplam çalışma süresini ve tam zaman çizelgesini raporlar.

Kullanım:
    python scripts/validate_multi_window.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from audio_preprocessing import load_audio_waveform, wav_to_logmel  # noqa: E402
from inference import predict_multi_window  # noqa: E402
from train_cnn_experiments import CNNVA  # noqa: E402
from train_crnn_experiments import CRNNVA  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AUDIO_DIR = os.path.join(REPO_ROOT, "data", "deam", "audio")
SPLIT_CSV = os.path.join(REPO_ROOT, "configs", "fixed_split_seed42.csv")
ANN_PATH = os.path.join(REPO_ROOT, "data", "deam", "annotations.csv")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
ADAGIO_PATH = os.path.join(REPO_ROOT, "data", "Adagio-for-strings.mp3")

# app.py'deki ensemble bileşenleri/ağırlıkları
ENSEMBLE_COMPONENTS = ("cnn_optimized", "cnn_baseline", "crnn_v2")
ENSEMBLE_WEIGHTS = {"cnn_optimized": 0.30, "cnn_baseline": 0.30, "crnn_v2": 0.40}

# Önceki oturumda (fix/preprocessing doğrulaması) app.py ile ölçülen,
# eski tek-pencere (sadece ilk 30 sn) yöntemin Adagio tahmini.
OLD_METHOD_ADAGIO_VALENCE = 3.71
OLD_METHOD_ADAGIO_AROUSAL = 3.63


def load_ensemble_models():
    ctors = {
        "cnn_optimized": lambda: CNNVA(dropout=0.4, architecture="optimized"),
        "cnn_baseline": lambda: CNNVA(dropout=0.4, architecture="baseline"),
        "crnn_v2": lambda: CRNNVA(hidden_size=128, num_layers=2, bidirectional=True, dropout=0.3),
    }
    models = {}
    for name in ENSEMBLE_COMPONENTS:
        model = ctors[name]().to(DEVICE)
        ckpt = torch.load(os.path.join(MODELS_DIR, f"{name}_best.pt"), map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models[name] = model
    return models


def old_single_window_predict(models, wav_path):
    """app.py'nin eski (sadece ilk 30 sn) davranışıyla birebir aynı ensemble tahmini."""
    wav = load_audio_waveform(wav_path)
    mel = wav_to_logmel(wav).unsqueeze(0).to(DEVICE)
    weighted_v, weighted_a = 0.0, 0.0
    for name, model in models.items():
        with torch.no_grad():
            v, a = model(mel).cpu().numpy()[0]
        weighted_v += ENSEMBLE_WEIGHTS[name] * v
        weighted_a += ENSEMBLE_WEIGHTS[name] * a
    return float(weighted_v), float(weighted_a)


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main():
    models = load_ensemble_models()
    runtime = {"type": "ensemble", "models": models, "weights": ENSEMBLE_WEIGHTS}

    split_df = pd.read_csv(SPLIT_CSV)
    ann_df = pd.read_csv(ANN_PATH).set_index("song_id")
    test_ids = split_df.loc[split_df["split"] == "test", "song_id"].astype(int).tolist()

    print(f"=== Test seti ({len(test_ids)} şarkı) - eski (ilk 30 sn) vs yeni (çoklu pencere) ===")
    print("(data/deam/annotations.csv'deki gerçek değerlere göre RMSE/Pearson)\n")

    rows = []
    t0_all = time.time()
    for i, sid in enumerate(test_ids, 1):
        wav_path = os.path.join(AUDIO_DIR, f"{sid}.wav")
        if not os.path.exists(wav_path):
            print(f"  UYARI: {wav_path} bulunamadı, atlanıyor.")
            continue
        if sid not in ann_df.index:
            print(f"  UYARI: song_id {sid} annotations.csv'de yok, atlanıyor.")
            continue

        true_v = float(ann_df.loc[sid, "valence"])
        true_a = float(ann_df.loc[sid, "arousal"])

        old_v, old_a = old_single_window_predict(models, wav_path)
        mw = predict_multi_window(wav_path, runtime, device=DEVICE)
        max_pad_ratio = max((w.padding_ratio for w in mw.windows), default=0.0)

        rows.append(
            {
                "song_id": sid,
                "true_valence": true_v,
                "true_arousal": true_a,
                "old_valence": old_v,
                "old_arousal": old_a,
                "new_valence": mw.valence,
                "new_arousal": mw.arousal,
                "n_windows": mw.n_windows_total,
                "max_padding_ratio": max_pad_ratio,
            }
        )
        print(f"  [{i}/{len(test_ids)}] song {sid} işlendi", end="\r")

    elapsed_all = time.time() - t0_all
    print(f"\n\nİşlenen şarkı sayısı: {len(rows)}  (toplam süre: {elapsed_all:.1f} sn)\n")

    df = pd.DataFrame(rows)

    summary_rows = []
    for method_label, vcol, acol in [
        ("Eski (ilk 30 sn)", "old_valence", "old_arousal"),
        ("Yeni (çoklu pencere)", "new_valence", "new_arousal"),
    ]:
        summary_rows.append(
            {
                "Yöntem": method_label,
                "RMSE (Valence)": rmse(df["true_valence"], df[vcol]),
                "RMSE (Arousal)": rmse(df["true_arousal"], df[acol]),
                "Pearson (Valence)": pearsonr(df["true_valence"], df[vcol])[0],
                "Pearson (Arousal)": pearsonr(df["true_arousal"], df[acol])[0],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    pd.set_option("display.width", 160)
    print(summary_df.to_string(index=False))

    max_pad_overall = df["max_padding_ratio"].max()
    n_padded = int((df["max_padding_ratio"] > 0).sum())
    print(
        f"\nTest setindeki tüm parçalarda en yüksek pencere dolgu oranı: {max_pad_overall:.4f} "
        f"({n_padded} parçada >0). DEAM klipleri 30 sn'den uzun olduğu için 0 olması beklenir."
    )

    print("\n\n=== Adagio for Strings (tam uzunluklu parça) ===")
    if not os.path.exists(ADAGIO_PATH):
        print(f"UYARI: {ADAGIO_PATH} bulunamadı, atlanıyor.")
        return

    t0 = time.time()
    mw = predict_multi_window(ADAGIO_PATH, runtime, device=DEVICE)
    elapsed = time.time() - t0

    print(f"Toplam pencere: {mw.n_windows_total}")
    print(f"Kullanılan pencere: {mw.n_windows_used}")
    print(f"Sessiz kabul edilip atlanan pencere: {mw.n_windows_skipped}")
    max_pad_adagio = max((w.padding_ratio for w in mw.windows), default=0.0)
    print(f"En yüksek dolgu oranı: {max_pad_adagio:.4f} (0 olmalı, parça 30 sn'den çok daha uzun)")
    if mw.used_all_windows_fallback:
        print("UYARI: tüm pencereler sessiz görünüyordu, güvenlik için hepsi kullanıldı.")
    print(
        f"Genel skor: Valence={mw.valence:.4f} (std {mw.valence_std:.4f}), "
        f"Arousal={mw.arousal:.4f} (std {mw.arousal_std:.4f})"
    )
    print(
        f"Eski yöntemle (ilk 30 sn, V={OLD_METHOD_ADAGIO_VALENCE}/A={OLD_METHOD_ADAGIO_AROUSAL}) fark: "
        f"DeltaV={abs(mw.valence - OLD_METHOD_ADAGIO_VALENCE):.4f}, "
        f"DeltaA={abs(mw.arousal - OLD_METHOD_ADAGIO_AROUSAL):.4f}"
    )
    print(f"Toplam çalışma süresi: {elapsed:.2f} sn")

    print("\nZaman çizelgesi (tamamı):")
    timeline_df = pd.DataFrame(mw.to_records())
    print(timeline_df.to_string(index=False))


if __name__ == "__main__":
    main()
