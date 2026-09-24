"""
AŞAMA 2 - CNN optimizasyon deney scripti

Bu script, AŞAMA 1'de sabitlenen split'i (configs/fixed_split_seed42.csv) kullanarak
farklı CNN deneylerini aynı metrik/çıktı formatında çalıştırır.
"""

import argparse
import os
import random
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
DEFAULT_EPOCHS = 35
DEFAULT_LR = 1e-4
DEFAULT_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE_TOTAL = 0.15


EXPERIMENT_PRESETS: Dict[str, Dict] = {
    "cnn_v2_opt_adam": {
        "optimizer": "adam",
        "lr": 3e-4,
        "dropout": 0.4,
        "weight_decay": 1e-5,
        "loss": "mse",
        "augmentation": "none",
        "architecture": "baseline",
        "notes": "optimizer adam",
    },
    "cnn_v2_opt_adamw": {
        "optimizer": "adamw",
        "lr": 3e-4,
        "dropout": 0.4,
        "weight_decay": 1e-5,
        "loss": "mse",
        "augmentation": "none",
        "architecture": "baseline",
        "notes": "optimizer adamw",
    },
    "cnn_v2_lr_1e3": {
        "optimizer": "adam",
        "lr": 1e-3,
        "dropout": 0.4,
        "weight_decay": 1e-5,
        "loss": "mse",
        "augmentation": "none",
        "architecture": "baseline",
        "notes": "lr 1e-3",
    },
    "cnn_v2_lr_3e4": {
        "optimizer": "adam",
        "lr": 3e-4,
        "dropout": 0.4,
        "weight_decay": 1e-5,
        "loss": "mse",
        "augmentation": "none",
        "architecture": "baseline",
        "notes": "lr 3e-4",
    },
    "cnn_v2_lr_1e4": {
        "optimizer": "adam",
        "lr": 1e-4,
        "dropout": 0.4,
        "weight_decay": 1e-5,
        "loss": "mse",
        "augmentation": "none",
        "architecture": "baseline",
        "notes": "lr 1e-4",
    },
    "cnn_v2_dropout_03": {
        "optimizer": "adam",
        "lr": 3e-4,
        "dropout": 0.3,
        "weight_decay": 1e-5,
        "loss": "mse",
        "augmentation": "none",
        "architecture": "baseline",
        "notes": "dropout 0.3",
    },
    "cnn_v2_dropout_04": {
        "optimizer": "adam",
        "lr": 3e-4,
        "dropout": 0.4,
        "weight_decay": 1e-5,
        "loss": "mse",
        "augmentation": "none",
        "architecture": "baseline",
        "notes": "dropout 0.4",
    },
    "cnn_v2_dropout_05": {
        "optimizer": "adam",
        "lr": 3e-4,
        "dropout": 0.5,
        "weight_decay": 1e-5,
        "loss": "mse",
        "augmentation": "none",
        "architecture": "baseline",
        "notes": "dropout 0.5",
    },
    "cnn_v2_wd_0": {
        "optimizer": "adam",
        "lr": 3e-4,
        "dropout": 0.4,
        "weight_decay": 0.0,
        "loss": "mse",
        "augmentation": "none",
        "architecture": "baseline",
        "notes": "weight decay 0",
    },
    "cnn_v2_wd_1e5": {
        "optimizer": "adam",
        "lr": 3e-4,
        "dropout": 0.4,
        "weight_decay": 1e-5,
        "loss": "mse",
        "augmentation": "none",
        "architecture": "baseline",
        "notes": "weight decay 1e-5",
    },
    "cnn_v2_wd_1e4": {
        "optimizer": "adam",
        "lr": 3e-4,
        "dropout": 0.4,
        "weight_decay": 1e-4,
        "loss": "mse",
        "augmentation": "none",
        "architecture": "baseline",
        "notes": "weight decay 1e-4",
    },
    "cnn_v2_noaug": {
        "optimizer": "adam",
        "lr": 3e-4,
        "dropout": 0.4,
        "weight_decay": 1e-5,
        "loss": "mse",
        "augmentation": "none",
        "architecture": "baseline",
        "notes": "v2 no augmentation",
    },
    "cnn_v2_specaugment": {
        "optimizer": "adam",
        "lr": 3e-4,
        "dropout": 0.4,
        "weight_decay": 1e-5,
        "loss": "mse",
        "augmentation": "specaugment",
        "architecture": "baseline",
        "notes": "specaugment",
    },
    "cnn_v2_specaugment_noise": {
        "optimizer": "adam",
        "lr": 3e-4,
        "dropout": 0.4,
        "weight_decay": 1e-5,
        "loss": "mse",
        "augmentation": "specaugment_noise",
        "architecture": "baseline",
        "notes": "specaugment + mild noise",
    },
    "cnn_v2_mse": {
        "optimizer": "adam",
        "lr": 3e-4,
        "dropout": 0.4,
        "weight_decay": 1e-5,
        "loss": "mse",
        "augmentation": "none",
        "architecture": "baseline",
        "notes": "loss mse",
    },
    "cnn_v2_smoothl1": {
        "optimizer": "adam",
        "lr": 3e-4,
        "dropout": 0.4,
        "weight_decay": 1e-5,
        "loss": "smoothl1",
        "augmentation": "none",
        "architecture": "baseline",
        "notes": "loss smoothl1",
    },
    "cnn_v2_mse_ccc": {
        "optimizer": "adam",
        "lr": 3e-4,
        "dropout": 0.4,
        "weight_decay": 1e-5,
        "loss": "mse_ccc",
        "augmentation": "none",
        "architecture": "baseline",
        "notes": "loss mse + ccc",
    },
    "cnn_optimized": {
        "optimizer": "adamw",
        "lr": 3e-4,
        "dropout": 0.4,
        "weight_decay": 1e-4,
        "loss": "mse_ccc",
        "augmentation": "specaugment_noise",
        "architecture": "optimized",
        "notes": "deeper cnn 32-64-128-256",
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
            noise = torch.randn_like(x) * 0.01
            x = x + noise

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


class CNNVA(nn.Module):
    def __init__(self, dropout: float = 0.4, architecture: str = "baseline"):
        super().__init__()

        if architecture == "optimized":
            channels = [32, 64, 128, 256]
        else:
            channels = [32, 64, 128]

        layers: List[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )
            in_ch = out_ch

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.net = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


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


def save_config_yaml(
    args,
    train_count: int,
    val_count: int,
    test_count: int,
    paths: ArtifactPaths,
):
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
        f"  architecture: {args.architecture}",
        "artifacts:",
        f"  best_model: {paths.best_model}",
        f"  last_model: {paths.last_model}",
        f"  predictions: {paths.predictions_csv}",
        f"  history: {paths.history_csv}",
    ]

    with open(paths.config_yaml, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def append_experiment_log(args, train_count: int, val_count: int, test_count: int, config_path: str):
    row = pd.DataFrame(
        [
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
                "architecture": args.architecture,
                "config_path": config_path,
                "split_path": SPLIT_OUT,
                "notes": args.notes,
            }
        ]
    )

    if os.path.exists(EXPERIMENT_LOG_OUT):
        old = pd.read_csv(EXPERIMENT_LOG_OUT)
        old = old[old["experiment_name"] != args.experiment_name]
        row = pd.concat([old, row], ignore_index=True)

    row.to_csv(EXPERIMENT_LOG_OUT, index=False)


def save_history(history_rows: List[Dict], history_out: str):
    pd.DataFrame(history_rows).to_csv(history_out, index=False)


def save_loss_curve(history_rows: List[Dict], loss_plot_out: str, title_prefix: str):
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


def save_scatter_plots(pred_df: pd.DataFrame, out_v: str, out_a: str, title_prefix: str):
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


def save_error_hist(pred_df: pd.DataFrame, out_hist: str, title_prefix: str):
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


def save_model_comparison(args, metrics: Dict[str, float]):
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


def evaluate_on_test(model, test_df: pd.DataFrame, batch_size: int, predictions_out: str):
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
    print(f"CNN Experiment: {args.experiment_name}")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Preset: {args.preset}")
    print(
        f"optimizer={args.optimizer}, lr={args.lr}, dropout={args.dropout}, "
        f"wd={args.weight_decay}, loss={args.loss}, aug={args.augmentation}, arch={args.architecture}"
    )

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

    model = CNNVA(dropout=args.dropout, architecture=args.architecture).to(DEVICE)
    optimizer = build_optimizer(args, model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.scheduler_patience, factor=0.5)

    if args.loss == "mse_ccc":
        criterion = None
    else:
        criterion = get_loss_function(args.loss)

    best_val_loss = float("inf")
    early_counter = 0
    history_rows: List[Dict] = []

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


def select_best_cnn_model():
    if not os.path.exists(COMPARISON_OUT):
        raise FileNotFoundError("results/model_comparison.csv bulunamadı.")

    df = pd.read_csv(COMPARISON_OUT)
    if df.empty:
        raise ValueError("model_comparison.csv boş.")

    candidates = df[df["model_name"].str.startswith("cnn_")].copy()
    if candidates.empty:
        raise ValueError("cnn_ ile başlayan model bulunamadı.")

    candidates = candidates.sort_values(by=["avg_rmse", "avg_pearson"], ascending=[True, False])
    best_name = candidates.iloc[0]["model_name"]

    src_model = os.path.join(MODELS_DIR, f"{best_name}_best.pt")
    dst_model = os.path.join(MODELS_DIR, "cnn_best.pt")

    if not os.path.exists(src_model):
        raise FileNotFoundError(f"Best checkpoint bulunamadı: {src_model}")

    shutil.copy2(src_model, dst_model)
    print(f"Seçilen en iyi CNN: {best_name}")
    print(f"Kopyalandı: {dst_model}")


def parse_args():
    parser = argparse.ArgumentParser(description="CNN deneylerini sabit split ile çalıştır")

    parser.add_argument("--experiment-name", type=str, default="cnn_v2_noaug")
    parser.add_argument("--preset", type=str, default="cnn_v2_noaug", choices=list(EXPERIMENT_PRESETS.keys()))

    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--augmentation", type=str, default="none", choices=["none", "specaugment", "specaugment_noise"])
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "smoothl1", "mse_ccc"])
    parser.add_argument("--architecture", type=str, default="baseline", choices=["baseline", "optimized"])

    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--early-patience", type=int, default=6)

    parser.add_argument("--notes", type=str, default="")

    parser.add_argument("--select-best-cnn", action="store_true", help="model_comparison.csv içinden en iyi CNN'i seçip models/cnn_best.pt üretir")

    args = parser.parse_args()

    preset = EXPERIMENT_PRESETS[args.preset]
    args.optimizer = preset["optimizer"]
    args.lr = preset["lr"]
    args.dropout = preset["dropout"]
    args.weight_decay = preset["weight_decay"]
    args.loss = preset["loss"]
    args.augmentation = preset["augmentation"]
    args.architecture = preset["architecture"]

    if not args.notes:
        args.notes = preset["notes"]

    return args


def main():
    args = parse_args()
    ensure_dirs()

    if args.select_best_cnn:
        select_best_cnn_model()
        return

    train_one_experiment(args)


if __name__ == "__main__":
    main()
