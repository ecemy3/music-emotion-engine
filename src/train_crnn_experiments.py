"""
AŞAMA 3 - CRNN deney scripti

Bu script, AŞAMA 1'de sabitlenen split'i (configs/fixed_split_seed42.csv)
kullanarak CRNN modellerini CNN ile aynı metrik/çıktı formatında çalıştırır.
"""

import argparse
import os
import random
import shutil
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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

COMPARISON_OUT = os.path.join(RESULTS_DIR, "model_comparison.csv")
SPLIT_OUT = os.path.join(CONFIGS_DIR, "fixed_split_seed42.csv")
EXPERIMENT_LOG_OUT = os.path.join(LOGS_DIR, "experiment_log.csv")

DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 60
DEFAULT_LR = 1e-3
DEFAULT_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE_TOTAL = 0.15


EXPERIMENT_PRESETS: Dict[str, Dict] = {
    "crnn_v1": {
        "optimizer": "adam",
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "dropout": 0.4,
        "loss": "mse",
        "augmentation": "none",
        "hidden_size": 128,
        "num_layers": 1,
        "bidirectional": False,
        "epochs": 35,
        "early_patience": 6,
        "scheduler_patience": 3,
        "notes": "crnn v1 simple",
    },
    "crnn_v1_bilstm": {
        "optimizer": "adam",
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "dropout": 0.4,
        "loss": "mse",
        "augmentation": "none",
        "hidden_size": 128,
        "num_layers": 1,
        "bidirectional": True,
        "epochs": 35,
        "early_patience": 6,
        "scheduler_patience": 3,
        "notes": "crnn v1 bilstm",
    },
    "crnn_v2": {
        "optimizer": "adam",
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "dropout": 0.3,
        "loss": "mse_ccc",
        "augmentation": "specaugment",
        "hidden_size": 128,
        "num_layers": 2,
        "bidirectional": True,
        "epochs": 60,
        "early_patience": 10,
        "scheduler_patience": 4,
        "notes": "crnn v2: stronger cnn + bilstm(2) + mse_ccc + specaugment",
    },
    "transformer_v1": {
        "optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "dropout": 0.3,
        "loss": "mse_ccc",
        "augmentation": "specaugment_noise",
        "hidden_size": 256,
        "num_layers": 4,
        "bidirectional": True,
        "epochs": 80,
        "early_patience": 12,
        "scheduler_patience": 4,
        "notes": "audio transformer baseline",
    },
    "transformer_v2": {
        "optimizer": "adamw",
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "dropout": 0.4,
        "loss": "mse_ccc",
        "augmentation": "specaugment",
        "hidden_size": 256,
        "num_layers": 4,
        "bidirectional": True,
        "epochs": 80,
        "early_patience": 15,
        "scheduler_patience": 5,
        "notes": "audio transformer v2: lower lr + higher dropout + specaugment",
    },
    "cnn_transformer_v1": {
        "optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "dropout": 0.3,
        "loss": "mse_ccc",
        "augmentation": "specaugment_noise",
        "hidden_size": 256,
        "num_layers": 4,
        "bidirectional": True,
        "epochs": 80,
        "early_patience": 12,
        "scheduler_patience": 4,
        "notes": "cnn + transformer hybrid",
    },
}


@dataclass
class ArtifactPaths:
    best_model: str
    last_model: str
    predictions_csv: str
    history_csv: str
    config_yaml: str
    loss_plot: str
    scatter_valence_plot: str
    scatter_arousal_plot: str
    error_hist_plot: str


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


def build_artifact_paths(experiment_name: str) -> ArtifactPaths:
    return ArtifactPaths(
        best_model=os.path.join(MODELS_DIR, f"{experiment_name}_best.pt"),
        last_model=os.path.join(MODELS_DIR, f"{experiment_name}_last.pt"),
        predictions_csv=os.path.join(RESULTS_DIR, f"{experiment_name}_predictions.csv"),
        history_csv=os.path.join(LOGS_DIR, f"{experiment_name}_history.csv"),
        config_yaml=os.path.join(CONFIGS_DIR, f"{experiment_name}.yaml"),
        loss_plot=os.path.join(RESULTS_DIR, f"{experiment_name}_loss_curve.png"),
        scatter_valence_plot=os.path.join(RESULTS_DIR, f"{experiment_name}_scatter_valence.png"),
        scatter_arousal_plot=os.path.join(RESULTS_DIR, f"{experiment_name}_scatter_arousal.png"),
        error_hist_plot=os.path.join(RESULTS_DIR, f"{experiment_name}_error_hist.png"),
    )


class DEAMDataset(Dataset):
    def __init__(self, df: pd.DataFrame, split: str, augmentation: str = "none"):
        self.df = df.reset_index(drop=True)
        self.split = split
        self.augmentation = augmentation

        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=20)

    def __len__(self):
        return len(self.df)

    def _augment(self, mel: torch.Tensor) -> torch.Tensor:
        if self.split != "train" or self.augmentation == "none":
            return mel

        x = mel.clone()
        if self.augmentation in ["specaugment", "specaugment_noise"]:
            x = self.freq_mask(x)
            x = self.time_mask(x)

        if self.augmentation == "specaugment_noise":
            x = x + torch.randn_like(x) * 0.01

        return x

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = os.path.join(AUDIO_DIR, f"{int(row.song_id)}.wav")
        wav = load_audio(path)
        x = wav_to_logmel(wav)
        x = self._augment(x)
        y = torch.tensor([row.valence, row.arousal], dtype=torch.float32)
        sample_id = int(row.song_id)
        return x, y, sample_id


class CRNNVA(nn.Module):
    def __init__(self, hidden_size=128, num_layers=1, bidirectional=False, dropout=0.4):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )

        lstm_input_size = self._infer_lstm_input_size()

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0 if num_layers == 1 else dropout,
        )

        lstm_out_size = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(lstm_out_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def _infer_lstm_input_size(self) -> int:
        # Infer C*F after CNN so LSTM input stays correct if CNN depth changes.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, N_MELS, 32)
            feat = self.cnn(dummy)
            _, c, f, _ = feat.shape
        return c * f

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        feat = self.cnn(x)  # (B, C, F, T)
        b, c, f, t = feat.shape
        feat = feat.permute(0, 3, 1, 2)  # (B, T, C, F)
        feat = feat.reshape(b, t, c * f)  # (B, T, C*F)

        out, _ = self.lstm(feat)  # (B, T, H)
        pooled = out[:, -1, :]  # last-timestep pooling
        return self.head(pooled)


class AudioTransformerVA(nn.Module):
    def __init__(self, n_mels=128, embed_dim=256, num_heads=8, num_layers=4, dropout=0.3):
        super().__init__()

        # patch embedding
        self.patch_embed = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=(16, 16),
            stride=(16, 16),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        # x: (B,1,n_mels,T)

        x = self.patch_embed(x)

        b, c, f, t = x.shape

        x = x.flatten(2).transpose(1, 2)  # (B,N,C)

        x = self.transformer(x)

        x = x.mean(dim=1)

        return self.head(x)


class CNNTransformerVA(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=4, dropout=0.3):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        # x: (B,1,n_mels,T)

        x = self.cnn(x)

        # (B,C,F,T)
        x = torch.mean(x, dim=2)  # frequency pooling

        # (B,C,T) -> (B,T,C)
        x = x.permute(0, 2, 1)

        x = self.transformer(x)

        x = torch.mean(x, dim=1)

        return self.head(x)


def ccc_score_torch(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    true_mean = torch.mean(y_true)
    pred_mean = torch.mean(y_pred)

    true_var = torch.var(y_true, unbiased=False)
    pred_var = torch.var(y_pred, unbiased=False)
    covariance = torch.mean((y_true - true_mean) * (y_pred - pred_mean))

    ccc = (2 * covariance) / (true_var + pred_var + (true_mean - pred_mean) ** 2 + eps)
    return ccc


def mse_ccc_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.3) -> torch.Tensor:
    mse = nn.functional.mse_loss(pred, target)
    ccc_v = ccc_score_torch(target[:, 0], pred[:, 0])
    ccc_a = ccc_score_torch(target[:, 1], pred[:, 1])
    ccc_term = 1 - (ccc_v + ccc_a) / 2
    return mse + alpha * ccc_term


def get_loss_function(loss_name: str):
    if loss_name == "smoothl1":
        return nn.SmoothL1Loss()
    if loss_name == "mse_ccc":
        return None
    return nn.MSELoss()


def compute_metrics_dict(y_true, y_pred) -> Dict[str, float]:
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


def create_or_load_fixed_split(df: pd.DataFrame, seed: int):
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

    train_val_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=seed)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=VAL_SIZE_TOTAL / (1.0 - TEST_SIZE),
        random_state=seed,
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


def save_config_yaml(args, train_count, val_count, test_count, paths: ArtifactPaths):
    lines = [
        f"experiment_name: {args.experiment_name}",
        f"seed: {args.seed}",
        "split:",
        f"  train_count: {train_count}",
        f"  val_count: {val_count}",
        f"  test_count: {test_count}",
        "training:",
        f"  batch_size: {args.batch_size}",
        f"  epochs: {args.epochs}",
        f"  lr: {args.lr}",
        f"  optimizer: {args.optimizer}",
        f"  weight_decay: {args.weight_decay}",
        f"  dropout: {args.dropout}",
        f"  loss: {args.loss}",
        f"  augmentation: {args.augmentation}",
        "model:",
        f"  hidden_size: {args.hidden_size}",
        f"  num_layers: {args.num_layers}",
        f"  bidirectional: {args.bidirectional}",
        "artifacts:",
        f"  best_model: {paths.best_model}",
        f"  last_model: {paths.last_model}",
        f"  predictions: {paths.predictions_csv}",
        f"  history: {paths.history_csv}",
    ]
    with open(paths.config_yaml, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def append_experiment_log(args, train_count, val_count, test_count, config_path):
    row = pd.DataFrame([
        {
            "experiment_name": args.experiment_name,
            "seed": args.seed,
            "train_count": train_count,
            "val_count": val_count,
            "test_count": test_count,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "loss": args.loss,
            "augmentation": args.augmentation,
            "config_path": config_path,
            "split_path": SPLIT_OUT,
            "notes": args.notes,
        }
    ])

    if os.path.exists(EXPERIMENT_LOG_OUT):
        old = pd.read_csv(EXPERIMENT_LOG_OUT)
        old = old[old["experiment_name"] != args.experiment_name]
        row = pd.concat([old, row], ignore_index=True)

    row.to_csv(EXPERIMENT_LOG_OUT, index=False)


def save_history(history_rows, history_out):
    pd.DataFrame(history_rows).to_csv(history_out, index=False)


def save_loss_curve(history_rows, loss_plot_out, title_prefix):
    hist = pd.DataFrame(history_rows)
    plt.figure(figsize=(8, 5))
    plt.plot(hist["epoch"], hist["train_loss"], label="Train Loss")
    plt.plot(hist["epoch"], hist["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} - Train vs Val Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_plot_out, dpi=150)
    plt.close()


def save_scatter_plots(pred_df, out_v, out_a, title_prefix):
    plt.figure(figsize=(6, 6))
    plt.scatter(pred_df["true_valence"], pred_df["pred_valence"], alpha=0.6)
    mn = min(pred_df["true_valence"].min(), pred_df["pred_valence"].min())
    mx = max(pred_df["true_valence"].max(), pred_df["pred_valence"].max())
    plt.plot([mn, mx], [mn, mx], "r--")
    plt.xlabel("True Valence")
    plt.ylabel("Pred Valence")
    plt.title(f"{title_prefix} - Valence Scatter")
    plt.tight_layout()
    plt.savefig(out_v, dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(pred_df["true_arousal"], pred_df["pred_arousal"], alpha=0.6)
    mn = min(pred_df["true_arousal"].min(), pred_df["pred_arousal"].min())
    mx = max(pred_df["true_arousal"].max(), pred_df["pred_arousal"].max())
    plt.plot([mn, mx], [mn, mx], "r--")
    plt.xlabel("True Arousal")
    plt.ylabel("Pred Arousal")
    plt.title(f"{title_prefix} - Arousal Scatter")
    plt.tight_layout()
    plt.savefig(out_a, dpi=150)
    plt.close()


def save_error_hist(pred_df, out_hist, title_prefix):
    plt.figure(figsize=(8, 5))
    plt.hist(pred_df["valence_error"], bins=20, alpha=0.6, label="Valence Error")
    plt.hist(pred_df["arousal_error"], bins=20, alpha=0.6, label="Arousal Error")
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.title(f"{title_prefix} - Error Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_hist, dpi=150)
    plt.close()


def save_model_comparison(args, metrics):
    row = {
        "model_name": args.experiment_name,
        "rmse_v": metrics["rmse_v"],
        "rmse_a": metrics["rmse_a"],
        "pearson_v": metrics["pearson_v"],
        "pearson_a": metrics["pearson_a"],
        "mae_v": metrics["mae_v"],
        "mae_a": metrics["mae_a"],
        "avg_rmse": metrics["avg_rmse"],
        "avg_pearson": metrics["avg_pearson"],
        "notes": args.notes,
    }

    row_df = pd.DataFrame([row])
    if os.path.exists(COMPARISON_OUT):
        existing = pd.read_csv(COMPARISON_OUT)
        if "notes" not in existing.columns:
            existing["notes"] = ""
        existing = existing[existing["model_name"] != args.experiment_name]
        row_df = pd.concat([existing, row_df], ignore_index=True)

    row_df.to_csv(COMPARISON_OUT, index=False)


def evaluate_on_test(model, test_df, batch_size, predictions_out):
    test_loader = DataLoader(DEAMDataset(test_df, split="test", augmentation="none"), batch_size=batch_size)

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
    pred_df.to_csv(predictions_out, index=False)
    metrics = compute_metrics_dict(y_true, y_pred)
    return pred_df, metrics


def build_optimizer(args, model):
    if args.optimizer.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train_one_experiment(args):
    ensure_dirs()
    set_seed(args.seed)

    print("=" * 60)
    print(f"CRNN Experiment: {args.experiment_name}")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Preset: {args.preset}")

    df = pd.read_csv(ANN_PATH)
    train_df, val_df, test_df = create_or_load_fixed_split(df, seed=args.seed)
    print(f"Split -> train:{len(train_df)} val:{len(val_df)} test:{len(test_df)} | seed:{args.seed}")

    paths = build_artifact_paths(args.experiment_name)
    save_config_yaml(args, len(train_df), len(val_df), len(test_df), paths)
    append_experiment_log(args, len(train_df), len(val_df), len(test_df), paths.config_yaml)

    train_loader = DataLoader(
        DEAMDataset(train_df, split="train", augmentation=args.augmentation),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        DEAMDataset(val_df, split="val", augmentation="none"),
        batch_size=args.batch_size,
        shuffle=False,
    )

    if "cnn_transformer" in args.experiment_name:
        model = CNNTransformerVA().to(DEVICE)
    elif "transformer" in args.experiment_name:
        model = AudioTransformerVA().to(DEVICE)
    else:
        model = CRNNVA(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            bidirectional=args.bidirectional,
            dropout=args.dropout,
        ).to(DEVICE)

    optimizer = build_optimizer(args, model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.scheduler_patience, factor=0.5)

    criterion = None if args.loss == "mse_ccc" else get_loss_function(args.loss)

    best_val_loss = float("inf")
    early_counter = 0
    history_rows = []

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0

        for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            pred = model(x)

            if args.loss == "mse_ccc":
                loss = mse_ccc_loss(pred, y)
            else:
                loss = criterion(pred, y)

            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        train_loss = train_loss_sum / max(len(train_loader), 1)

        model.eval()
        y_true, y_pred = [], []
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                pred = model(x)

                if args.loss == "mse_ccc":
                    val_loss = mse_ccc_loss(pred, y)
                else:
                    val_loss = criterion(pred, y)

                val_loss_sum += val_loss.item()
                y_true.extend(y.cpu().numpy().tolist())
                y_pred.extend(pred.cpu().numpy().tolist())

        val_loss = val_loss_sum / max(len(val_loader), 1)
        scheduler.step(val_loss)

        metrics = compute_metrics_dict(y_true, y_pred)
        history_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
            f"RMSE V:{metrics['rmse_v']:.3f} A:{metrics['rmse_a']:.3f} | "
            f"Pearson V:{metrics['pearson_v']:.3f} A:{metrics['pearson_a']:.3f} | "
            f"LR:{optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": float(val_loss),
                    "config_path": paths.config_yaml,
                    "seed": args.seed,
                    "experiment_name": args.experiment_name,
                },
                paths.best_model,
            )
        else:
            early_counter += 1
            if early_counter >= args.early_patience:
                print(f"Early stopping tetiklendi (patience={args.early_patience})")
                break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": len(history_rows),
            "val_loss": float(history_rows[-1]["val_loss"] if history_rows else np.nan),
            "config_path": paths.config_yaml,
            "seed": args.seed,
            "experiment_name": args.experiment_name,
        },
        paths.last_model,
    )

    best_ckpt = torch.load(paths.best_model, map_location=DEVICE, weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"])

    pred_df, test_metrics = evaluate_on_test(
        model=model,
        test_df=test_df,
        batch_size=args.batch_size,
        predictions_out=paths.predictions_csv,
    )

    save_history(history_rows, paths.history_csv)
    save_loss_curve(history_rows, paths.loss_plot, args.experiment_name)
    save_scatter_plots(pred_df, paths.scatter_valence_plot, paths.scatter_arousal_plot, args.experiment_name)
    save_error_hist(pred_df, paths.error_hist_plot, args.experiment_name)
    save_model_comparison(args, test_metrics)

    print("=" * 60)
    print("Deney tamamlandı")
    print(f"Best model: {paths.best_model}")
    print(f"Last model: {paths.last_model}")
    print(f"Predictions: {paths.predictions_csv}")
    print(f"History: {paths.history_csv}")
    print(f"Comparison row updated: {COMPARISON_OUT}")
    print("=" * 60)


def select_best_crnn_model():
    if not os.path.exists(COMPARISON_OUT):
        raise FileNotFoundError("results/model_comparison.csv bulunamadı.")

    df = pd.read_csv(COMPARISON_OUT)
    candidates = df[df["model_name"].str.startswith("crnn_")].copy()
    if candidates.empty:
        raise ValueError("crnn_ ile başlayan model bulunamadı.")

    candidates = candidates.sort_values(by=["avg_rmse", "avg_pearson"], ascending=[True, False])
    best_name = candidates.iloc[0]["model_name"]

    src_model = os.path.join(MODELS_DIR, f"{best_name}_best.pt")
    dst_model = os.path.join(MODELS_DIR, "crnn_best.pt")

    if not os.path.exists(src_model):
        raise FileNotFoundError(f"Best checkpoint bulunamadı: {src_model}")

    shutil.copy2(src_model, dst_model)
    print(f"Seçilen en iyi CRNN: {best_name}")
    print(f"Kopyalandı: {dst_model}")


def parse_args():
    parser = argparse.ArgumentParser(description="CRNN deneylerini sabit split ile çalıştır")

    parser.add_argument("--experiment-name", type=str, default="crnn_v1")
    parser.add_argument("--preset", type=str, default="crnn_v1", choices=list(EXPERIMENT_PRESETS.keys()))

    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.4)

    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "smoothl1", "mse_ccc"])
    parser.add_argument("--augmentation", type=str, default="none", choices=["none", "specaugment", "specaugment_noise"])

    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--bidirectional", dest="bidirectional", action="store_true")
    parser.add_argument("--no-bidirectional", dest="bidirectional", action="store_false")
    parser.set_defaults(bidirectional=True)

    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--early-patience", type=int, default=10)
    parser.add_argument("--notes", type=str, default="")

    parser.add_argument("--select-best-crnn", action="store_true", help="model_comparison.csv içinden en iyi CRNN'i seçip models/crnn_best.pt üretir")

    args = parser.parse_args()

    preset = EXPERIMENT_PRESETS[args.preset]
    args.optimizer = preset["optimizer"]
    args.lr = preset["lr"]
    args.weight_decay = preset["weight_decay"]
    args.dropout = preset["dropout"]
    args.loss = preset["loss"]
    args.augmentation = preset["augmentation"]
    args.hidden_size = preset["hidden_size"]
    args.num_layers = preset["num_layers"]
    args.bidirectional = preset["bidirectional"]
    args.epochs = preset.get("epochs", args.epochs)
    args.early_patience = preset.get("early_patience", args.early_patience)
    args.scheduler_patience = preset.get("scheduler_patience", args.scheduler_patience)

    if not args.notes:
        args.notes = preset["notes"]

    return args


def main():
    args = parse_args()
    ensure_dirs()

    if args.select_best_crnn:
        select_best_crnn_model()
        return

    train_one_experiment(args)


if __name__ == "__main__":
    main()
