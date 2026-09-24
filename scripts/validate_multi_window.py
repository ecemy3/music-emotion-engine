"""
Çoklu pencereli (multi-window) tahminin doğrulanması.

  1. DEAM test setinden 20 kısa klip (~45 sn) üzerinde eski (sadece ilk
     30 sn) ve yeni (30 sn / %50 örtüşmeli çoklu pencere) yöntemin
     sonuçlarını karşılaştırır. Klipler kısa olduğu için sonuçların
     birbirine yakın olması beklenir - ortalama fark yazdırılır.
  2. data/Adagio-for-strings.mp3 (tam uzunluklu, ~417 sn bir parça) için
     pencere sayısını, atlanan pencereleri, genel skoru, eski yöntemle
     (önceki oturumda ölçülen V=3.71/A=3.63 referansıyla) farkını ve
     toplam çalışma süresini raporlar.

Kullanım:
    python scripts/validate_multi_window.py
"""

import os
import random
import sys
import time

import pandas as pd
import torch

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
MODELS_DIR = os.path.join(REPO_ROOT, "models")
ADAGIO_PATH = os.path.join(REPO_ROOT, "data", "Adagio-for-strings.mp3")

SEED = 42
N_SONGS = 20

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


def main():
    random.seed(SEED)
    models = load_ensemble_models()
    runtime = {"type": "ensemble", "models": models, "weights": ENSEMBLE_WEIGHTS}

    print(f"=== Kısa DEAM klipleri (n={N_SONGS}) - eski (ilk 30 sn) vs yeni (çoklu pencere) ===")
    split_df = pd.read_csv(SPLIT_CSV)
    test_ids = split_df.loc[split_df["split"] == "test", "song_id"].astype(int).tolist()
    sample_ids = sorted(random.sample(test_ids, min(N_SONGS, len(test_ids))))

    rows = []
    for sid in sample_ids:
        wav_path = os.path.join(AUDIO_DIR, f"{sid}.wav")
        if not os.path.exists(wav_path):
            print(f"  UYARI: {wav_path} bulunamadı, atlanıyor.")
            continue

        old_v, old_a = old_single_window_predict(models, wav_path)
        mw = predict_multi_window(wav_path, runtime, device=DEVICE)

        rows.append(
            {
                "song_id": sid,
                "n_windows": mw.n_windows_total,
                "n_skipped": mw.n_windows_skipped,
                "old_valence": old_v,
                "new_valence": mw.valence,
                "diff_v": abs(old_v - mw.valence),
                "old_arousal": old_a,
                "new_arousal": mw.arousal,
                "diff_a": abs(old_a - mw.arousal),
            }
        )

    report_df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(report_df.to_string(index=False))
    print(f"\nOrtalama |fark| Valence: {report_df['diff_v'].mean():.4f}")
    print(f"Ortalama |fark| Arousal: {report_df['diff_a'].mean():.4f}")

    print("\n=== Adagio for Strings (tam uzunluklu parça) ===")
    if not os.path.exists(ADAGIO_PATH):
        print(f"UYARI: {ADAGIO_PATH} bulunamadı, atlanıyor.")
        return

    t0 = time.time()
    mw = predict_multi_window(ADAGIO_PATH, runtime, device=DEVICE)
    elapsed = time.time() - t0

    print(f"Toplam pencere: {mw.n_windows_total}")
    print(f"Kullanılan pencere: {mw.n_windows_used}")
    print(f"Sessiz kabul edilip atlanan pencere: {mw.n_windows_skipped}")
    if mw.used_all_windows_fallback:
        print("UYARI: tüm pencereler sessiz görünüyordu, güvenlik için hepsi kullanıldı.")
    print(
        f"Genel skor: Valence={mw.valence:.4f} (std {mw.valence_std:.4f}), "
        f"Arousal={mw.arousal:.4f} (std {mw.arousal_std:.4f})"
    )
    print(
        f"Eski yöntemle (ilk 30 sn, V={OLD_METHOD_ADAGIO_VALENCE}/A={OLD_METHOD_ADAGIO_AROUSAL}) fark: "
        f"ΔV={abs(mw.valence - OLD_METHOD_ADAGIO_VALENCE):.4f}, "
        f"ΔA={abs(mw.arousal - OLD_METHOD_ADAGIO_AROUSAL):.4f}"
    )
    print(f"Toplam çalışma süresi: {elapsed:.2f} sn")

    print("\nZaman çizelgesi:")
    timeline_df = pd.DataFrame(mw.to_records())
    print(timeline_df.to_string(index=False))


if __name__ == "__main__":
    main()
