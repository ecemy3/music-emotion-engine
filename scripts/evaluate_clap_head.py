"""
models/clap_head_best.pt'yi test setinde (configs/fixed_split_seed42.csv,
split=='test' - eğitimde HİÇ görülmedi) değerlendirir ve mevcut CNN/CRNN
ensemble'ın baseline sonuçlarıyla (scripts/baseline_diagnostics.py) yan
yana karşılaştırır. Ayrıca lambda taramasındaki her denemenin validation
sonuçlarını özetler.

Hiçbir model eğitilmez - sadece değerlendirme yapılır.

Kullanım:
    python scripts/evaluate_clap_head.py
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from train_clap_head import ClapVAHead, ccc_numpy, compute_quadrant  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CACHE_DIR = os.path.join(REPO_ROOT, "cache", "embeddings")
EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "clap_deam.npy")
SONG_IDS_PATH = os.path.join(CACHE_DIR, "clap_deam_song_ids.npy")
ANN_PATH = os.path.join(REPO_ROOT, "data", "deam", "annotations.csv")
SPLIT_PATH = os.path.join(REPO_ROOT, "configs", "fixed_split_seed42.csv")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
LOGS_DIR = os.path.join(REPO_ROOT, "logs")
BEST_CKPT_PATH = os.path.join(MODELS_DIR, "clap_head_best.pt")

# scripts/baseline_diagnostics.py'nin bu oturumda üretilmiş sonuçları
# (mevcut CNN/CRNN ensemble - CLAP'tan ÖNCEKİ durum). Güncellemek için o
# scripti tekrar çalıştırın.
BASELINE = {
    "true_pearson_va_full_dataset": 0.5700,
    "test_n": 270,
    "test_rmse_v": 0.8239,
    "test_rmse_a": 0.8050,
    "test_pearson_v": 0.7317,
    "test_pearson_a": 0.7927,
    "test_ccc_v": 0.6720,
    "test_ccc_a": 0.7351,
    "test_pred_pearson_va": 0.9990,
    "test_true_pearson_va": 0.6692,
    "test_pred_mean_abs_diff": 0.0696,
    "test_true_mean_abs_diff": 0.8191,
    "test_pred_range_v": (2.93, 6.77),
    "test_pred_range_a": (2.76, 6.98),
    "separability_true_diff": 6.1133,
    "separability_pred_diff": 0.0276,
    "separability_group_high_v_low_a_n": 25,
    "separability_group_low_v_high_a_n": 19,
}

# scripts/baseline_diagnostics.py ile BİREBİR AYNI gruplar (aynı song_id
# kümeleri) - karşılaştırmanın adil olması için.
GROUP_MARGIN = 1.0


def load_checkpoint_model(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    hp = ckpt["hyperparameters"]
    model = ClapVAHead(hidden_dims=tuple(hp["hidden_dims"]), dropout=hp["dropout"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def predict(model, embeddings: np.ndarray, label_std: dict):
    with torch.no_grad():
        emb_t = torch.from_numpy(embeddings).float().to(DEVICE)
        pv, pa, pq = model(emb_t)
    pred_v = pv.cpu().numpy() * label_std["valence_std"] + label_std["valence_mean"]
    pred_a = pa.cpu().numpy() * label_std["arousal_std"] + label_std["arousal_mean"]
    pred_quad = pq.argmax(dim=1).cpu().numpy()
    return pred_v, pred_a, pred_quad


def main():
    if not os.path.exists(BEST_CKPT_PATH):
        print(f"HATA: {BEST_CKPT_PATH} bulunamadı. Önce src/train_clap_head.py --sweep çalıştırılmalı.")
        return

    model, ckpt = load_checkpoint_model(BEST_CKPT_PATH)
    label_std = ckpt["label_standardization"]
    hp = ckpt["hyperparameters"]
    print(f"Yüklenen checkpoint: {BEST_CKPT_PATH}")
    print(f"lambda_quadrant={ckpt['lambda_quadrant']}, epoch={ckpt['epoch']}")
    print(f"Kayıtlı validation metrikleri: {ckpt['val_metrics']}\n")

    embeddings = np.load(EMBEDDINGS_PATH)
    song_ids = np.load(SONG_IDS_PATH)
    id_to_idx = {int(sid): i for i, sid in enumerate(song_ids)}

    ann = pd.read_csv(ANN_PATH).set_index("song_id")
    split_df = pd.read_csv(SPLIT_PATH)

    test_ids = [int(sid) for sid in split_df.loc[split_df["split"] == "test", "song_id"] if int(sid) in id_to_idx]
    test_idxs = [id_to_idx[sid] for sid in test_ids]
    test_emb = embeddings[test_idxs]
    true_v = ann.loc[test_ids, "valence"].to_numpy(dtype=np.float64)
    true_a = ann.loc[test_ids, "arousal"].to_numpy(dtype=np.float64)
    true_quad = compute_quadrant(true_v, true_a, hp["median_valence"], hp["median_arousal"])

    pred_v, pred_a, pred_quad = predict(model, test_emb, label_std)

    rmse_v = float(np.sqrt(mean_squared_error(true_v, pred_v)))
    rmse_a = float(np.sqrt(mean_squared_error(true_a, pred_a)))
    pearson_v = float(pearsonr(pred_v, true_v)[0])
    pearson_a = float(pearsonr(pred_a, true_a)[0])
    ccc_v = ccc_numpy(pred_v, true_v)
    ccc_a = ccc_numpy(pred_a, true_a)
    pred_pearson_va = float(pearsonr(pred_v, pred_a)[0])
    true_pearson_va = float(pearsonr(true_v, true_a)[0])
    quad_acc = float((pred_quad == true_quad).mean())

    print("=" * 78)
    print(f"TEST SETİ (n={len(test_ids)}) - CLAP head vs mevcut CNN/CRNN ensemble (baseline)")
    print("=" * 78)
    comparison_rows = [
        {"Metrik": "RMSE (Valence)", "Baseline (ensemble)": f"{BASELINE['test_rmse_v']:.4f}", "CLAP head": f"{rmse_v:.4f}"},
        {"Metrik": "RMSE (Arousal)", "Baseline (ensemble)": f"{BASELINE['test_rmse_a']:.4f}", "CLAP head": f"{rmse_a:.4f}"},
        {"Metrik": "Pearson (Valence)", "Baseline (ensemble)": f"{BASELINE['test_pearson_v']:.4f}", "CLAP head": f"{pearson_v:.4f}"},
        {"Metrik": "Pearson (Arousal)", "Baseline (ensemble)": f"{BASELINE['test_pearson_a']:.4f}", "CLAP head": f"{pearson_a:.4f}"},
        {"Metrik": "CCC (Valence)", "Baseline (ensemble)": f"{BASELINE['test_ccc_v']:.4f}", "CLAP head": f"{ccc_v:.4f}"},
        {"Metrik": "CCC (Arousal)", "Baseline (ensemble)": f"{BASELINE['test_ccc_a']:.4f}", "CLAP head": f"{ccc_a:.4f}"},
        {"Metrik": "Tahmin Pearson(V,A)", "Baseline (ensemble)": f"{BASELINE['test_pred_pearson_va']:.4f}", "CLAP head": f"{pred_pearson_va:.4f}"},
        {"Metrik": "Gerçek Pearson(V,A)", "Baseline (ensemble)": f"{BASELINE['test_true_pearson_va']:.4f}", "CLAP head": f"{true_pearson_va:.4f}"},
        {"Metrik": "Ortalama |pred_V - pred_A|", "Baseline (ensemble)": f"{BASELINE['test_pred_mean_abs_diff']:.4f}", "CLAP head": f"{np.mean(np.abs(pred_v - pred_a)):.4f}"},
        {"Metrik": "Quadrant doğruluğu (4 sınıf, şans=0.25)", "Baseline (ensemble)": "-", "CLAP head": f"{quad_acc:.4f}"},
    ]
    print(pd.DataFrame(comparison_rows).to_string(index=False))

    bv = BASELINE["test_pred_range_v"]
    ba = BASELINE["test_pred_range_a"]
    print(f"\nTahmin aralığı  -> Baseline V:[{bv[0]:.2f},{bv[1]:.2f}] A:[{ba[0]:.2f},{ba[1]:.2f}]  |  "
          f"CLAP head V:[{pred_v.min():.2f},{pred_v.max():.2f}] A:[{pred_a.min():.2f},{pred_a.max():.2f}]")
    print(f"Gerçek aralık   -> Valence: [{true_v.min():.2f}, {true_v.max():.2f}]  Arousal: [{true_a.min():.2f}, {true_a.max():.2f}]")

    # --- Ayırt etme testi: baseline_diagnostics.py ile BİREBİR AYNI gruplar ---
    print("\n" + "=" * 78)
    print("AYIRT ETME TESTİ (baseline_diagnostics.py ile aynı gruplar)")
    print("=" * 78)
    available_ids = set(int(sid) for sid in song_ids)
    ann_full = pd.read_csv(ANN_PATH)
    ann_avail = ann_full[ann_full["song_id"].isin(available_ids)]
    med_v_full, med_a_full = ann_avail["valence"].median(), ann_avail["arousal"].median()

    g1 = ann_avail[(ann_avail.valence >= med_v_full + GROUP_MARGIN) & (ann_avail.arousal <= med_a_full - GROUP_MARGIN)]
    g2 = ann_avail[(ann_avail.valence <= med_v_full - GROUP_MARGIN) & (ann_avail.arousal >= med_a_full + GROUP_MARGIN)]

    g1_idxs = [id_to_idx[sid] for sid in g1["song_id"] if sid in id_to_idx]
    g2_idxs = [id_to_idx[sid] for sid in g2["song_id"] if sid in id_to_idx]

    g1_pred_v, g1_pred_a, _ = predict(model, embeddings[g1_idxs], label_std)
    g2_pred_v, g2_pred_a, _ = predict(model, embeddings[g2_idxs], label_std)

    g1_pred_diff = float(np.mean(g1_pred_v - g1_pred_a))
    g2_pred_diff = float(np.mean(g2_pred_v - g2_pred_a))

    print(f"Yüksek-V/Düşük-A grubu (n={len(g1)}): CLAP head tahmin (V-A) ort. = {g1_pred_diff:.4f}  [baseline: {BASELINE['separability_pred_diff']:.4f} (iki grup farkı, tek grup değil)]")
    print(f"Düşük-V/Yüksek-A grubu (n={len(g2)}): CLAP head tahmin (V-A) ort. = {g2_pred_diff:.4f}")
    print(f"\nCLAP head grup ayrımı (V-A farkı): {g1_pred_diff - g2_pred_diff:.4f}")
    print(f"Baseline (ensemble) grup ayrımı (V-A farkı): {BASELINE['separability_pred_diff']:.4f}")
    print(f"Gerçek grup ayrımı (V-A farkı): {BASELINE['separability_true_diff']:.4f}")

    # --- Lambda taraması validation özeti ---
    print("\n" + "=" * 78)
    print("LAMBDA TARAMASI - VALIDATION SONUÇLARI ÖZETİ")
    print("=" * 78)
    summary_rows = []
    for hist_path in sorted(glob.glob(os.path.join(LOGS_DIR, "clap_head_lambda*_history.csv"))):
        hist_df = pd.read_csv(hist_path)
        best_row = hist_df.loc[hist_df["val_mean_ccc"].idxmax()]
        lam_str = os.path.basename(hist_path).replace("clap_head_lambda", "").replace("_history.csv", "")
        summary_rows.append(
            {
                "lambda": lam_str,
                "en_iyi_epoch": int(best_row["epoch"]),
                "toplam_epoch": int(hist_df["epoch"].max()),
                "val_RMSE_V": round(best_row["val_rmse_v"], 4),
                "val_RMSE_A": round(best_row["val_rmse_a"], 4),
                "val_CCC_V": round(best_row["val_ccc_v"], 4),
                "val_CCC_A": round(best_row["val_ccc_a"], 4),
                "val_mean_CCC": round(best_row["val_mean_ccc"], 4),
                "val_QuadAcc": round(best_row["val_quad_acc"], 4),
                "val_PredVACorr": round(best_row["val_pred_va_corr"], 4),
            }
        )
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df["secilen_mi"] = summary_df["val_mean_CCC"] == summary_df["val_mean_CCC"].max()
        print(summary_df.to_string(index=False))
    else:
        print("logs/clap_head_lambda*_history.csv bulunamadı.")


if __name__ == "__main__":
    main()
