"""
CLAP-tabanlı modelden ÖNCEKİ (mevcut CNN/CRNN ensemble) durumun temel
(baseline) ölçümü. src/train_clap_head.py ile eğitilecek modelle
karşılaştırma için referans sayılar üretir. Hiçbir model eğitilmez/yeniden
eğitilmez - sadece mevcut checkpoint'lerle çıkarım yapılır.

1. data/deam/annotations.csv'de gerçek Valence-Arousal Pearson korelasyonu.
2. Mevcut production ensemble'ın (cnn_optimized 0.30, cnn_baseline 0.30,
   crnn_v2 0.40) test setindeki tahmin V-A korelasyonu ve ortalama |V-A|
   farkı - results/*_predictions.csv'den (eğitim sırasında üretildi,
   inference kodundan bağımsız).
3. "Ayırt etme testi": tüm etiketli veri setinde (medyandan >=1 birim uzak)
   yüksek-V/düşük-A ve düşük-V/yüksek-A gruplarında ensemble'ın ortalama
   tahmin (V-A) farkı - canlı çıkarımla (eğitimdeki tek-pencere hattıyla).

Kullanım:
    python scripts/baseline_diagnostics.py
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
from train_cnn_experiments import CNNVA  # noqa: E402
from train_crnn_experiments import CRNNVA  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AUDIO_DIR = os.path.join(REPO_ROOT, "data", "deam", "audio")
ANN_PATH = os.path.join(REPO_ROOT, "data", "deam", "annotations.csv")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
MODELS_DIR = os.path.join(REPO_ROOT, "models")

ENSEMBLE_COMPONENTS = ("cnn_optimized", "cnn_baseline", "crnn_v2")
ENSEMBLE_WEIGHTS = {"cnn_optimized": 0.30, "cnn_baseline": 0.30, "crnn_v2": 0.40}

# "Ayırt etme testi" için medyandan en az bu kadar birim uzak olma şartı.
GROUP_MARGIN = 1.0


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


def ensemble_predict_single_window(models, wav_path):
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
    # --- 1) Gerçek etiketlerde V-A korelasyonu (tüm veri seti) ---
    ann = pd.read_csv(ANN_PATH)
    r_true_all, _ = pearsonr(ann["valence"], ann["arousal"])
    print("=== 1) Gerçek etiketlerde Valence-Arousal korelasyonu ===")
    print(f"Tüm annotations.csv (n={len(ann)}): Pearson(V,A) = {r_true_all:.4f}")
    print()

    # --- 2) Mevcut ensemble'ın test setindeki V-A korelasyonu / |fark| ---
    print("=== 2) Mevcut ensemble - test setinde tahmin V-A korelasyonu ve ortalama |V-A| farkı ===")
    dfs = {name: pd.read_csv(os.path.join(RESULTS_DIR, f"{name}_predictions.csv")).set_index("sample_id") for name in ENSEMBLE_COMPONENTS}
    common_ids = set.intersection(*[set(df.index) for df in dfs.values()])
    common_ids = sorted(common_ids)

    ens_pred_v, ens_pred_a, true_v, true_a = [], [], [], []
    for sid in common_ids:
        wv = sum(ENSEMBLE_WEIGHTS[name] * dfs[name].loc[sid, "pred_valence"] for name in ENSEMBLE_COMPONENTS)
        wa = sum(ENSEMBLE_WEIGHTS[name] * dfs[name].loc[sid, "pred_arousal"] for name in ENSEMBLE_COMPONENTS)
        ens_pred_v.append(wv)
        ens_pred_a.append(wa)
        true_v.append(dfs["cnn_baseline"].loc[sid, "true_valence"])
        true_a.append(dfs["cnn_baseline"].loc[sid, "true_arousal"])

    ens_pred_v = np.array(ens_pred_v)
    ens_pred_a = np.array(ens_pred_a)
    true_v = np.array(true_v)
    true_a = np.array(true_a)

    r_pred, _ = pearsonr(ens_pred_v, ens_pred_a)
    r_true_test, _ = pearsonr(true_v, true_a)
    mean_abs_diff_pred = np.mean(np.abs(ens_pred_v - ens_pred_a))
    mean_abs_diff_true = np.mean(np.abs(true_v - true_a))

    print(f"Test seti (n={len(common_ids)}):")
    print(f"  Pearson(pred_V, pred_A)  = {r_pred:.4f}")
    print(f"  Pearson(true_V, true_A)  = {r_true_test:.4f}")
    print(f"  Ortalama |pred_V - pred_A| = {mean_abs_diff_pred:.4f}")
    print(f"  Ortalama |true_V - true_A| = {mean_abs_diff_true:.4f}")
    print()

    # --- 3) Ayırt etme testi: tüm etiketli veri setinde canlı çıkarım ---
    print("=== 3) Ayırt etme testi (tüm etiketli veri seti, canlı çıkarım) ===")
    available = {int(f.split(".")[0]) for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")}
    ann_avail = ann[ann["song_id"].isin(available)]
    med_v, med_a = ann_avail["valence"].median(), ann_avail["arousal"].median()

    g_high_v_low_a = ann_avail[(ann_avail.valence >= med_v + GROUP_MARGIN) & (ann_avail.arousal <= med_a - GROUP_MARGIN)]
    g_low_v_high_a = ann_avail[(ann_avail.valence <= med_v - GROUP_MARGIN) & (ann_avail.arousal >= med_a + GROUP_MARGIN)]

    print(f"Medyan: V={med_v:.2f}, A={med_a:.2f}")
    print(f"Yüksek-V/Düşük-A grubu: n={len(g_high_v_low_a)}")
    print(f"Düşük-V/Yüksek-A grubu: n={len(g_low_v_high_a)}")

    models = load_ensemble_models()
    t0 = time.time()

    def group_pred_diffs(group_df):
        diffs = []
        for sid in group_df["song_id"]:
            wav_path = os.path.join(AUDIO_DIR, f"{sid}.wav")
            v, a = ensemble_predict_single_window(models, wav_path)
            diffs.append(v - a)
        return diffs

    g1_pred_diffs = group_pred_diffs(g_high_v_low_a)
    g2_pred_diffs = group_pred_diffs(g_low_v_high_a)
    elapsed = time.time() - t0

    g1_true_diff = float((g_high_v_low_a["valence"] - g_high_v_low_a["arousal"]).mean())
    g2_true_diff = float((g_low_v_high_a["valence"] - g_low_v_high_a["arousal"]).mean())
    g1_pred_mean = float(np.mean(g1_pred_diffs))
    g2_pred_mean = float(np.mean(g2_pred_diffs))

    print(f"\nÇalışma süresi: {elapsed:.1f} sn ({DEVICE})")
    print(f"\n{'Grup':<20}{'n':>5}{'gerçek (V-A) ort.':>20}{'tahmin (V-A) ort.':>20}")
    print(f"{'Yüksek-V/Düşük-A':<20}{len(g_high_v_low_a):>5}{g1_true_diff:>20.4f}{g1_pred_mean:>20.4f}")
    print(f"{'Düşük-V/Yüksek-A':<20}{len(g_low_v_high_a):>5}{g2_true_diff:>20.4f}{g2_pred_mean:>20.4f}")
    print(f"\nGerçek grup ayrımı (V-A farkı): {g1_true_diff - g2_true_diff:.4f}")
    print(f"Tahmin edilen grup ayrımı (V-A farkı): {g1_pred_mean - g2_pred_mean:.4f}")
    print(
        f"\n-> Ensemble, gerçekte {g1_true_diff - g2_true_diff:.2f} birim ayrışan iki zıt grubu "
        f"tahminde sadece {g1_pred_mean - g2_pred_mean:.4f} birimle ayırabiliyor."
    )


if __name__ == "__main__":
    main()
