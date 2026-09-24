"""
DEAM Dataset üzerinde Valence-Arousal tahmini için CNN modeli eğitimi
"""

import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

from audio_preprocessing import (
    load_audio_waveform as load_audio,
    wav_to_logmel,
    SR,
    DURATION,
    SAMPLES,
    N_FFT,
    HOP,
    N_MELS,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "data/deam"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
ANN_PATH = os.path.join(DATA_DIR, "annotations.csv")

MODELS_DIR = "models"
LOGS_DIR = "logs"
RESULTS_DIR = "results"
CONFIGS_DIR = "configs"

BASELINE_NAME = "cnn_baseline"
BEST_MODEL_OUT = os.path.join(MODELS_DIR, f"{BASELINE_NAME}_best.pt")
LAST_MODEL_OUT = os.path.join(MODELS_DIR, f"{BASELINE_NAME}_last.pt")
PREDICTIONS_OUT = os.path.join(RESULTS_DIR, f"{BASELINE_NAME}_predictions.csv")
COMPARISON_OUT = os.path.join(RESULTS_DIR, "model_comparison.csv")
HISTORY_OUT = os.path.join(LOGS_DIR, f"{BASELINE_NAME}_history.csv")
LOSS_PLOT_OUT = os.path.join(RESULTS_DIR, f"{BASELINE_NAME}_loss_curve.png")
SCATTER_V_OUT = os.path.join(RESULTS_DIR, f"{BASELINE_NAME}_scatter_valence.png")
SCATTER_A_OUT = os.path.join(RESULTS_DIR, f"{BASELINE_NAME}_scatter_arousal.png")
ERROR_HIST_OUT = os.path.join(RESULTS_DIR, f"{BASELINE_NAME}_error_hist.png")
CONFIG_OUT = os.path.join(CONFIGS_DIR, f"{BASELINE_NAME}.yaml")
SPLIT_OUT = os.path.join(CONFIGS_DIR, "fixed_split_seed42.csv")
EXPERIMENT_LOG_OUT = os.path.join(LOGS_DIR, "experiment_log.csv")
README_EXPERIMENTS_OUT = "README_experiments.md"

BATCH_SIZE = 16
EPOCHS = 35
LR = 1e-4
SEED = 42
TEST_SIZE = 0.15
VAL_SIZE_TOTAL = 0.15


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs():
    for directory in [MODELS_DIR, LOGS_DIR, RESULTS_DIR, CONFIGS_DIR]:
        os.makedirs(directory, exist_ok=True)

def compute_metrics(y_true, y_pred):
    """RMSE ve Pearson Correlation hesapla (valence ve arousal için ayrı)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse_v = np.sqrt(mean_squared_error(y_true[:,0], y_pred[:,0]))
    rmse_a = np.sqrt(mean_squared_error(y_true[:,1], y_pred[:,1]))

    corr_v = pearsonr(y_true[:,0], y_pred[:,0])[0]
    corr_a = pearsonr(y_true[:,1], y_pred[:,1])[0]

    return rmse_v, rmse_a, corr_v, corr_a


def compute_metrics_dict(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse_v = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
    rmse_a = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
    mae_v = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_a = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    pearson_v = pearsonr(y_true[:, 0], y_pred[:, 0])[0]
    pearson_a = pearsonr(y_true[:, 1], y_pred[:, 1])[0]

    return {
        "rmse_v": float(rmse_v),
        "rmse_a": float(rmse_a),
        "pearson_v": float(pearson_v),
        "pearson_a": float(pearson_a),
        "mae_v": float(mae_v),
        "mae_a": float(mae_a),
        "avg_rmse": float((rmse_v + rmse_a) / 2.0),
        "avg_pearson": float((pearson_v + pearson_a) / 2.0),
    }

def show_mels(df):
    """5 adet örnek Mel-Spectrogram görselleştir"""
    sample_df = df.sample(5)
    plt.figure(figsize=(15,8))
    for i,row in enumerate(sample_df.itertuples()):
        wav = load_audio(os.path.join(AUDIO_DIR,f"{int(row.song_id)}.wav"))
        mel = wav_to_logmel(wav)[0].numpy()  # (n_mels, time_frames) şeklinde 2D array
        plt.subplot(2,3,i+1)
        plt.imshow(mel, origin='lower', aspect='auto', cmap='viridis')
        plt.title(f"Song {int(row.song_id)} - V:{row.valence:.2f} A:{row.arousal:.2f}")
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency')
        plt.colorbar()
    plt.tight_layout()
    plt.savefig('mel_spectrograms_samples.png')
    print("\n✅ Mel-spectrogram örnekleri 'mel_spectrograms_samples.png' olarak kaydedildi\n")

class DEAMDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(AUDIO_DIR, f"{int(row.song_id)}.wav")
        wav = load_audio(path)
        x = wav_to_logmel(wav)
        y = torch.tensor([row.valence, row.arousal], dtype=torch.float32)
        return x, y


class DEAMDatasetWithMeta(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(AUDIO_DIR, f"{int(row.song_id)}.wav")
        wav = load_audio(path)
        x = wav_to_logmel(wav)
        y = torch.tensor([row.valence, row.arousal], dtype=torch.float32)
        sample_id = int(row.song_id)
        return x, y, sample_id

class CNNVA(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64,2)  # Sigmoid kaldırıldı - direkt regression
        )

    def forward(self,x):
        return self.head(self.net(x))


def create_or_load_fixed_split(df: pd.DataFrame):
    if os.path.exists(SPLIT_OUT):
        split_df = pd.read_csv(SPLIT_OUT)
        merged = df.merge(split_df, on="song_id", how="left")
        if merged["split"].isnull().any():
            raise ValueError("Sabit split dosyası ile annotation eşleşmesi eksik.")
        return (
            merged[merged["split"] == "train"].copy(),
            merged[merged["split"] == "val"].copy(),
            merged[merged["split"] == "test"].copy(),
        )

    test_df, train_val_df = train_test_split(
        df,
        test_size=1.0 - TEST_SIZE,
        random_state=SEED,
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=VAL_SIZE_TOTAL / (1.0 - TEST_SIZE),
        random_state=SEED,
    )

    split_df = pd.concat(
        [
            train_df[["song_id"]].assign(split="train"),
            val_df[["song_id"]].assign(split="val"),
            test_df[["song_id"]].assign(split="test"),
        ],
        ignore_index=True,
    )
    split_df.to_csv(SPLIT_OUT, index=False)

    return train_df.copy(), val_df.copy(), test_df.copy()


def save_baseline_config(train_count: int, val_count: int, test_count: int):
    cfg_lines = [
        f"experiment_name: {BASELINE_NAME}",
        f"seed: {SEED}",
        "split:",
        f"  train_count: {train_count}",
        f"  val_count: {val_count}",
        f"  test_count: {test_count}",
        "audio:",
        f"  sample_rate: {SR}",
        f"  duration_sec: {DURATION}",
        f"  n_fft: {N_FFT}",
        f"  hop_length: {HOP}",
        f"  n_mels: {N_MELS}",
        "training:",
        f"  batch_size: {BATCH_SIZE}",
        f"  epochs: {EPOCHS}",
        f"  lr: {LR}",
        "optimizer:",
        "  name: Adam",
        "  scheduler: ReduceLROnPlateau",
    ]
    with open(CONFIG_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(cfg_lines) + "\n")


def append_experiment_log(train_count: int, val_count: int, test_count: int):
    row = pd.DataFrame(
        [
            {
                "experiment_name": BASELINE_NAME,
                "seed": SEED,
                "train_count": train_count,
                "val_count": val_count,
                "test_count": test_count,
                "config_path": CONFIG_OUT,
                "split_path": SPLIT_OUT,
            }
        ]
    )
    if os.path.exists(EXPERIMENT_LOG_OUT):
        old = pd.read_csv(EXPERIMENT_LOG_OUT)
        old = old[old["experiment_name"] != BASELINE_NAME]
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(EXPERIMENT_LOG_OUT, index=False)


def write_experiments_readme():
    content = (
        "# Experiment Tracking\n\n"
        "Bu dosya baseline deneyinin sabitlenmesi için oluşturuldu.\n\n"
        "- Sabit split dosyası: `configs/fixed_split_seed42.csv`\n"
        "- Baseline config: `configs/cnn_baseline.yaml`\n"
        "- Deney logu: `logs/experiment_log.csv`\n"
        "- Baseline en iyi model: `models/cnn_baseline_best.pt`\n"
        "- Baseline son model: `models/cnn_baseline_last.pt`\n"
        "- Baseline tahminler: `results/cnn_baseline_predictions.csv`\n"
        "- Baseline metrikler: `results/model_comparison.csv`\n"
    )
    with open(README_EXPERIMENTS_OUT, "w", encoding="utf-8") as f:
        f.write(content)


def save_history(history_rows):
    pd.DataFrame(history_rows).to_csv(HISTORY_OUT, index=False)


def save_loss_curve(history_rows):
    hist = pd.DataFrame(history_rows)
    plt.figure(figsize=(8, 5))
    plt.plot(hist["epoch"], hist["train_loss"], label="Train Loss")
    plt.plot(hist["epoch"], hist["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Baseline - Train vs Val Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_OUT, dpi=150)
    plt.close()


def save_scatter_plots(pred_df: pd.DataFrame):
    plt.figure(figsize=(6, 6))
    plt.scatter(pred_df["true_valence"], pred_df["pred_valence"], alpha=0.6)
    mn = min(pred_df["true_valence"].min(), pred_df["pred_valence"].min())
    mx = max(pred_df["true_valence"].max(), pred_df["pred_valence"].max())
    plt.plot([mn, mx], [mn, mx], "r--")
    plt.xlabel("True Valence")
    plt.ylabel("Pred Valence")
    plt.title("CNN Baseline - Valence Scatter")
    plt.tight_layout()
    plt.savefig(SCATTER_V_OUT, dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(pred_df["true_arousal"], pred_df["pred_arousal"], alpha=0.6)
    mn = min(pred_df["true_arousal"].min(), pred_df["pred_arousal"].min())
    mx = max(pred_df["true_arousal"].max(), pred_df["pred_arousal"].max())
    plt.plot([mn, mx], [mn, mx], "r--")
    plt.xlabel("True Arousal")
    plt.ylabel("Pred Arousal")
    plt.title("CNN Baseline - Arousal Scatter")
    plt.tight_layout()
    plt.savefig(SCATTER_A_OUT, dpi=150)
    plt.close()


def save_error_hist(pred_df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    plt.hist(pred_df["valence_error"], bins=20, alpha=0.6, label="Valence Error")
    plt.hist(pred_df["arousal_error"], bins=20, alpha=0.6, label="Arousal Error")
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.title("CNN Baseline - Error Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ERROR_HIST_OUT, dpi=150)
    plt.close()


def save_model_comparison(metrics: dict):
    row = {
        "model_name": BASELINE_NAME,
        "rmse_v": metrics["rmse_v"],
        "rmse_a": metrics["rmse_a"],
        "pearson_v": metrics["pearson_v"],
        "pearson_a": metrics["pearson_a"],
        "mae_v": metrics["mae_v"],
        "mae_a": metrics["mae_a"],
        "avg_rmse": metrics["avg_rmse"],
        "avg_pearson": metrics["avg_pearson"],
    }

    row_df = pd.DataFrame([row])
    if os.path.exists(COMPARISON_OUT):
        existing = pd.read_csv(COMPARISON_OUT)
        existing = existing[existing["model_name"] != BASELINE_NAME]
        row_df = pd.concat([existing, row_df], ignore_index=True)
    row_df.to_csv(COMPARISON_OUT, index=False)


def evaluate_on_test(model, test_df):
    test_loader = DataLoader(DEAMDatasetWithMeta(test_df), batch_size=BATCH_SIZE)
    y_true, y_pred = [], []
    records = []
    model.eval()

    with torch.no_grad():
        for x, y, sample_ids in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)

            y_np = y.cpu().numpy()
            out_np = out.cpu().numpy()
            sample_ids_np = sample_ids.numpy()

            y_true.extend(y_np.tolist())
            y_pred.extend(out_np.tolist())

            for i in range(len(sample_ids_np)):
                true_v = float(y_np[i, 0])
                true_a = float(y_np[i, 1])
                pred_v = float(out_np[i, 0])
                pred_a = float(out_np[i, 1])
                records.append(
                    {
                        "sample_id": int(sample_ids_np[i]),
                        "song_name": f"song_{int(sample_ids_np[i])}",
                        "split": "test",
                        "true_valence": true_v,
                        "pred_valence": pred_v,
                        "true_arousal": true_a,
                        "pred_arousal": pred_a,
                        "valence_error": pred_v - true_v,
                        "arousal_error": pred_a - true_a,
                        "human_mean_valence": true_v,
                        "human_mean_arousal": true_a,
                    }
                )

    pred_df = pd.DataFrame(records).sort_values("sample_id").reset_index(drop=True)
    pred_df.to_csv(PREDICTIONS_OUT, index=False)
    metrics = compute_metrics_dict(y_true, y_pred)
    return pred_df, metrics

def main():
    ensure_dirs()
    set_seed(SEED)

    print("=" * 60)
    print("DEAM CNN Valence-Arousal Model Eğitimi")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LR}")
    print(f"Seed: {SEED}")
    print("=" * 60)
    
    df = pd.read_csv(ANN_PATH)
    print(f"\nToplam veri: {len(df)} şarkı")
    
    # Mel-Spectrogram görselleştirmesi
    print("\n5 örnek Mel-Spectrogram görselleştiriliyor...")
    show_mels(df)
    
    train_df, val_df, test_df = create_or_load_fixed_split(df)
    print(f"Eğitim seti: {len(train_df)} şarkı")
    print(f"Validasyon seti: {len(val_df)} şarkı")
    print(f"Test seti: {len(test_df)} şarkı")

    save_baseline_config(len(train_df), len(val_df), len(test_df))
    append_experiment_log(len(train_df), len(val_df), len(test_df))
    write_experiments_readme()

    train_loader = DataLoader(DEAMDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(DEAMDataset(val_df), batch_size=BATCH_SIZE)

    model = CNNVA().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    
    # Learning Rate Scheduler ve Early Stopping
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    PATIENCE = 6
    counter = 0
    history_rows = []

    print(f"\nModel parametreleri: {sum(p.numel() for p in model.parameters()):,}")
    print("\nEğitim başlıyor...\n")

    best = 1e9
    for e in range(EPOCHS):
        model.train()
        train_loss = 0
        for x,y in tqdm(train_loader, desc=f"Epoch {e+1}/{EPOCHS}"):
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)

        # Validation ve metrik hesaplama
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            val_loss = 0
            for x,y in val_loader:
                x,y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                val_loss += loss_fn(out, y).item()
                y_true.extend(y.cpu().numpy())
                y_pred.extend(out.cpu().numpy())
            val_loss /= len(val_loader)
        
        # Metrik hesaplama
        rmse_v, rmse_a, corr_v, corr_a = compute_metrics(y_true, y_pred)
        
        # Learning rate scheduler
        scheduler.step(val_loss)

        history_rows.append(
            {
                "epoch": e + 1,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "lr": float(opt.param_groups[0]["lr"]),
            }
        )

        print(f"Epoch {e+1}/{EPOCHS} | Train {train_loss:.3f} | Val {val_loss:.3f} | "
              f"RMSE V:{rmse_v:.2f} A:{rmse_a:.2f} | "
              f"Corr V:{corr_v:.2f} A:{corr_a:.2f} | LR:{opt.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if val_loss < best:
            best = val_loss
            counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "epoch": e + 1,
                    "val_loss": float(val_loss),
                    "config_path": CONFIG_OUT,
                    "seed": SEED,
                },
                BEST_MODEL_OUT,
            )
            print(f"✅ Model kaydedildi (Best Val Loss: {val_loss:.3f})")
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"\n🛑 Early stopping! (Patience: {PATIENCE})")
                break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "epoch": len(history_rows),
            "val_loss": float(history_rows[-1]["val_loss"] if history_rows else np.nan),
            "config_path": CONFIG_OUT,
            "seed": SEED,
        },
        LAST_MODEL_OUT,
    )

    best_ckpt = torch.load(BEST_MODEL_OUT, map_location=DEVICE)
    model.load_state_dict(best_ckpt["model_state_dict"])

    pred_df, metrics = evaluate_on_test(model, test_df)
    save_history(history_rows)
    save_loss_curve(history_rows)
    save_scatter_plots(pred_df)
    save_error_hist(pred_df)
    save_model_comparison(metrics)

    print("\n" + "=" * 60)
    print("Eğitim tamamlandı!")
    print(f"En iyi validasyon loss: {best:.4f}")
    print(f"Best model: {BEST_MODEL_OUT}")
    print(f"Last model: {LAST_MODEL_OUT}")
    print(f"Predictions: {PREDICTIONS_OUT}")
    print(f"Metrics table: {COMPARISON_OUT}")
    print("=" * 60)

if __name__ == "__main__":
    main()
