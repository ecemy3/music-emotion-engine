"""
Önceden çıkarılmış CLAP embedding'leri (cache/embeddings/clap_deam.npy,
bkz. src/embeddings.py) üzerinde bir Valence-Arousal + quadrant tahmin
başlığı (MLP) eğitir.

configs/fixed_split_seed42.csv ile AYNI split kullanılır (train/val/test
zaten dosyada mevcut). Test seti eğitim sırasında HİÇ kullanılmaz - sadece
scripts/evaluate_clap_head.py gibi ayrı bir değerlendirme adımında.

Model: embedding(512) -> paylaşılan MLP gövde (dropout'lu) -> üç ayrı baş:
  - valence_head: Linear(-, 1)
  - arousal_head: Linear(-, 1)
  - quadrant_head: Linear(-, 4)  (etiketin TRAIN medyanına göre bölgesi:
    0=yüksekV/yüksekA, 1=yüksekV/düşükA, 2=düşükV/yüksekA, 3=düşükV/düşükA)

Loss = CCC_loss(V) + CCC_loss(A) + lambda * CrossEntropy(quadrant)
Etiketler (V, A) eğitimde TRAIN seti mean/std'siyle standardize edilir;
değerlendirme metrikleri (RMSE/Pearson/CCC) ham (1-9) ölçeğe geri
dönüştürülerek hesaplanır.

Kullanım:
    python src/train_clap_head.py --lambda-quadrant 0.3
    python src/train_clap_head.py --sweep   # 0, 0.3, 1.0 hepsini dener,
                                             # en iyisini models/clap_head_best.pt yapar
"""

import argparse
import os
import shutil
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO_ROOT, "cache", "embeddings")
EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "clap_deam.npy")
SONG_IDS_PATH = os.path.join(CACHE_DIR, "clap_deam_song_ids.npy")
ANN_PATH = os.path.join(REPO_ROOT, "data", "deam", "annotations.csv")
SPLIT_PATH = os.path.join(REPO_ROOT, "configs", "fixed_split_seed42.csv")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
LOGS_DIR = os.path.join(REPO_ROOT, "logs")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# src/embeddings.py ile birebir aynı olmalı (checkpoint'e de yazılır).
CLAP_MODEL_NAME = "laion/clap-htsat-unfused"
CLAP_SR = 48000
WINDOW_SEC = 10.0
HOP_SEC = 5.0
EMBED_DIM = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_HIDDEN_DIMS = (256, 128)
DEFAULT_DROPOUT = 0.3
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEED = 42
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 15
LAMBDA_SWEEP = (0.0, 0.3, 1.0)


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ClapVAHead(nn.Module):
    """Paylaşılan MLP gövde + üç ayrı baş (valence, arousal, quadrant)."""

    def __init__(self, embed_dim: int = EMBED_DIM, hidden_dims=DEFAULT_HIDDEN_DIMS, dropout: float = DEFAULT_DROPOUT, n_quadrants: int = 4):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = embed_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.valence_head = nn.Linear(in_dim, 1)
        self.arousal_head = nn.Linear(in_dim, 1)
        self.quadrant_head = nn.Linear(in_dim, n_quadrants)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        v = self.valence_head(h).squeeze(-1)
        a = self.arousal_head(h).squeeze(-1)
        q = self.quadrant_head(h)
        return v, a, q


class ClapEmbeddingDataset(Dataset):
    def __init__(self, song_ids, embeddings, valence, arousal, quadrant):
        self.song_ids = song_ids
        self.embeddings = embeddings
        self.valence = valence
        self.arousal = arousal
        self.quadrant = quadrant

    def __len__(self):
        return len(self.song_ids)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.embeddings[idx]).float(),
            torch.tensor(self.valence[idx], dtype=torch.float32),
            torch.tensor(self.arousal[idx], dtype=torch.float32),
            torch.tensor(self.quadrant[idx], dtype=torch.long),
        )


def compute_quadrant(valence: np.ndarray, arousal: np.ndarray, median_v: float, median_a: float) -> np.ndarray:
    """0=yüksekV/yüksekA, 1=yüksekV/düşükA, 2=düşükV/yüksekA, 3=düşükV/düşükA (TRAIN medyanına göre)."""
    high_v = valence >= median_v
    high_a = arousal >= median_a
    quadrant = np.full(len(valence), -1, dtype=np.int64)
    quadrant[high_v & high_a] = 0
    quadrant[high_v & ~high_a] = 1
    quadrant[~high_v & high_a] = 2
    quadrant[~high_v & ~high_a] = 3
    return quadrant


def load_datasets(seed: int) -> Tuple[Dataset, Dataset, Dataset, Dict]:
    embeddings = np.load(EMBEDDINGS_PATH)
    song_ids = np.load(SONG_IDS_PATH)
    id_to_idx = {int(sid): i for i, sid in enumerate(song_ids)}

    ann = pd.read_csv(ANN_PATH).set_index("song_id")
    split_df = pd.read_csv(SPLIT_PATH)

    def gather(split_name: str):
        ids = [int(sid) for sid in split_df.loc[split_df["split"] == split_name, "song_id"] if int(sid) in id_to_idx]
        idxs = [id_to_idx[sid] for sid in ids]
        v = ann.loc[ids, "valence"].to_numpy(dtype=np.float64)
        a = ann.loc[ids, "arousal"].to_numpy(dtype=np.float64)
        return ids, embeddings[idxs], v, a

    train_ids, train_emb, train_v_raw, train_a_raw = gather("train")
    val_ids, val_emb, val_v_raw, val_a_raw = gather("val")
    test_ids, test_emb, test_v_raw, test_a_raw = gather("test")

    v_mean, v_std = float(train_v_raw.mean()), float(train_v_raw.std())
    a_mean, a_std = float(train_a_raw.mean()), float(train_a_raw.std())
    median_v, median_a = float(np.median(train_v_raw)), float(np.median(train_a_raw))

    def standardize(arr, mean, std):
        return (arr - mean) / std

    train_quad = compute_quadrant(train_v_raw, train_a_raw, median_v, median_a)
    val_quad = compute_quadrant(val_v_raw, val_a_raw, median_v, median_a)
    test_quad = compute_quadrant(test_v_raw, test_a_raw, median_v, median_a)

    train_ds = ClapEmbeddingDataset(train_ids, train_emb, standardize(train_v_raw, v_mean, v_std), standardize(train_a_raw, a_mean, a_std), train_quad)
    val_ds = ClapEmbeddingDataset(val_ids, val_emb, standardize(val_v_raw, v_mean, v_std), standardize(val_a_raw, a_mean, a_std), val_quad)
    test_ds = ClapEmbeddingDataset(test_ids, test_emb, standardize(test_v_raw, v_mean, v_std), standardize(test_a_raw, a_mean, a_std), test_quad)

    meta = {
        "v_mean": v_mean, "v_std": v_std, "a_mean": a_mean, "a_std": a_std,
        "median_v": median_v, "median_a": median_a,
        "n_train": len(train_ids), "n_val": len(val_ids), "n_test": len(test_ids),
    }
    return train_ds, val_ds, test_ds, meta


def ccc(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred_mean, target_mean = pred.mean(), target.mean()
    pred_var, target_var = pred.var(unbiased=False), target.var(unbiased=False)
    covariance = ((pred - pred_mean) * (target - target_mean)).mean()
    return (2 * covariance) / (pred_var + target_var + (pred_mean - target_mean) ** 2 + eps)


def ccc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1 - ccc(pred, target)


def ccc_numpy(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred_mean, target_mean = pred.mean(), target.mean()
    pred_var, target_var = pred.var(), target.var()
    covariance = ((pred - pred_mean) * (target - target_mean)).mean()
    return float((2 * covariance) / (pred_var + target_var + (pred_mean - target_mean) ** 2 + eps))


@dataclass
class EpochMetrics:
    train_loss: float
    train_ccc_v_loss: float
    train_ccc_a_loss: float
    train_ce: float
    val_rmse_v: float
    val_rmse_a: float
    val_ccc_v: float
    val_ccc_a: float
    val_pearson_v: float
    val_pearson_a: float
    val_quad_acc: float
    val_pred_va_corr: float

    @property
    def val_mean_ccc(self) -> float:
        return (self.val_ccc_v + self.val_ccc_a) / 2.0


def run_validation(model: nn.Module, loader: DataLoader, v_mean, v_std, a_mean, a_std) -> Tuple[EpochMetrics, None]:
    model.eval()
    pred_v_all, pred_a_all, true_v_all, true_a_all = [], [], [], []
    pred_quad_all, true_quad_all = [], []

    with torch.no_grad():
        for emb, v, a, q in loader:
            emb, v, a, q = emb.to(DEVICE), v.to(DEVICE), a.to(DEVICE), q.to(DEVICE)
            pv, pa, pq = model(emb)
            pred_v_all.append(pv.cpu().numpy())
            pred_a_all.append(pa.cpu().numpy())
            true_v_all.append(v.cpu().numpy())
            true_a_all.append(a.cpu().numpy())
            pred_quad_all.append(pq.argmax(dim=1).cpu().numpy())
            true_quad_all.append(q.cpu().numpy())

    pred_v = np.concatenate(pred_v_all) * v_std + v_mean
    pred_a = np.concatenate(pred_a_all) * a_std + a_mean
    true_v = np.concatenate(true_v_all) * v_std + v_mean
    true_a = np.concatenate(true_a_all) * a_std + a_mean
    pred_quad = np.concatenate(pred_quad_all)
    true_quad = np.concatenate(true_quad_all)

    rmse_v = float(np.sqrt(mean_squared_error(true_v, pred_v)))
    rmse_a = float(np.sqrt(mean_squared_error(true_a, pred_a)))
    ccc_v = ccc_numpy(pred_v, true_v)
    ccc_a = ccc_numpy(pred_a, true_a)
    pearson_v = float(pearsonr(pred_v, true_v)[0])
    pearson_a = float(pearsonr(pred_a, true_a)[0])
    quad_acc = float((pred_quad == true_quad).mean())
    pred_va_corr = float(pearsonr(pred_v, pred_a)[0]) if np.std(pred_v) > 1e-8 and np.std(pred_a) > 1e-8 else 0.0

    return rmse_v, rmse_a, ccc_v, ccc_a, pearson_v, pearson_a, quad_acc, pred_va_corr


def train_one_lambda(lam: float, seed: int = DEFAULT_SEED, verbose: bool = True) -> Dict:
    set_seed(seed)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_ds, val_ds, test_ds, meta = load_datasets(seed)
    train_loader = DataLoader(train_ds, batch_size=DEFAULT_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=DEFAULT_BATCH_SIZE, shuffle=False)

    model = ClapVAHead().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
    ce_loss_fn = nn.CrossEntropyLoss()

    best_val_mean_ccc = -1e9
    best_epoch = 0
    early_counter = 0
    history_rows = []

    print(f"\n{'=' * 70}\nCLAP head eğitimi - lambda_quadrant={lam}\n{'=' * 70}")
    print(f"train={meta['n_train']} val={meta['n_val']} test={meta['n_test']} (test HİÇ kullanılmıyor)")
    print(f"device={DEVICE}\n")

    best_model_path = os.path.join(MODELS_DIR, f"clap_head_lambda{lam}_best.pt")
    last_model_path = os.path.join(MODELS_DIR, f"clap_head_lambda{lam}_last.pt")
    history_path = os.path.join(LOGS_DIR, f"clap_head_lambda{lam}_history.csv")
    curves_path = os.path.join(RESULTS_DIR, f"clap_head_lambda{lam}_curves.png")

    checkpoint_base = {
        "lambda_quadrant": lam,
        "hyperparameters": {
            "hidden_dims": list(DEFAULT_HIDDEN_DIMS),
            "dropout": DEFAULT_DROPOUT,
            "lr": DEFAULT_LR,
            "weight_decay": DEFAULT_WEIGHT_DECAY,
            "batch_size": DEFAULT_BATCH_SIZE,
            "seed": seed,
            "median_valence": meta["median_v"],
            "median_arousal": meta["median_a"],
        },
        "label_standardization": {
            "valence_mean": meta["v_mean"], "valence_std": meta["v_std"],
            "arousal_mean": meta["a_mean"], "arousal_std": meta["a_std"],
        },
        "backbone_model_name": CLAP_MODEL_NAME,
        "embedding_settings": {
            "sample_rate": CLAP_SR, "window_sec": WINDOW_SEC, "hop_sec": HOP_SEC, "embed_dim": EMBED_DIM,
        },
    }

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        loss_sum, ccc_v_loss_sum, ccc_a_loss_sum, ce_sum, n_batches = 0.0, 0.0, 0.0, 0.0, 0

        for emb, v, a, q in train_loader:
            emb, v, a, q = emb.to(DEVICE), v.to(DEVICE), a.to(DEVICE), q.to(DEVICE)
            optimizer.zero_grad()
            pv, pa, pq = model(emb)

            l_ccc_v = ccc_loss(pv, v)
            l_ccc_a = ccc_loss(pa, a)
            l_ce = ce_loss_fn(pq, q)
            loss = l_ccc_v + l_ccc_a + lam * l_ce

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            ccc_v_loss_sum += l_ccc_v.item()
            ccc_a_loss_sum += l_ccc_a.item()
            ce_sum += l_ce.item()
            n_batches += 1

        train_loss = loss_sum / n_batches
        train_ccc_v_loss = ccc_v_loss_sum / n_batches
        train_ccc_a_loss = ccc_a_loss_sum / n_batches
        train_ce = ce_sum / n_batches

        rmse_v, rmse_a, ccc_v, ccc_a, pearson_v, pearson_a, quad_acc, pred_va_corr = run_validation(
            model, val_loader, meta["v_mean"], meta["v_std"], meta["a_mean"], meta["a_std"]
        )
        val_mean_ccc = (ccc_v + ccc_a) / 2.0
        scheduler.step(val_mean_ccc)

        is_best = val_mean_ccc > best_val_mean_ccc
        if is_best:
            best_val_mean_ccc = val_mean_ccc
            best_epoch = epoch
            early_counter = 0
            torch.save(
                {
                    **checkpoint_base,
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": {
                        "rmse_v": rmse_v, "rmse_a": rmse_a, "ccc_v": ccc_v, "ccc_a": ccc_a,
                        "pearson_v": pearson_v, "pearson_a": pearson_a,
                        "quadrant_accuracy": quad_acc, "pred_va_correlation": pred_va_corr,
                        "mean_ccc": val_mean_ccc,
                    },
                },
                best_model_path,
            )
        else:
            early_counter += 1

        history_rows.append(
            {
                "epoch": epoch, "train_loss": train_loss, "train_ccc_v_loss": train_ccc_v_loss,
                "train_ccc_a_loss": train_ccc_a_loss, "train_ce": train_ce,
                "val_rmse_v": rmse_v, "val_rmse_a": rmse_a, "val_ccc_v": ccc_v, "val_ccc_a": ccc_a,
                "val_pearson_v": pearson_v, "val_pearson_a": pearson_a, "val_quad_acc": quad_acc,
                "val_pred_va_corr": pred_va_corr, "val_mean_ccc": val_mean_ccc, "is_best": is_best,
            }
        )

        if verbose:
            line = (
                f"Epoch {epoch:3d} | train_loss={train_loss:.4f} "
                f"(ccc_v={train_ccc_v_loss:.4f} ccc_a={train_ccc_a_loss:.4f} ce={train_ce:.4f}) | "
                f"val RMSE_V={rmse_v:.3f} RMSE_A={rmse_a:.3f} CCC_V={ccc_v:.3f} CCC_A={ccc_a:.3f} "
                f"Pearson_V={pearson_v:.3f} Pearson_A={pearson_a:.3f} QuadAcc={quad_acc:.3f} "
                f"PredVACorr={pred_va_corr:.3f}"
            )
            if is_best:
                line += " * best"
            print(line)

        if early_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping (patience={EARLY_STOP_PATIENCE}) - epoch {epoch}")
            break

    torch.save(
        {
            **checkpoint_base,
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_metrics": {
                "rmse_v": rmse_v, "rmse_a": rmse_a, "ccc_v": ccc_v, "ccc_a": ccc_a,
                "pearson_v": pearson_v, "pearson_a": pearson_a,
                "quadrant_accuracy": quad_acc, "pred_va_correlation": pred_va_corr,
                "mean_ccc": val_mean_ccc,
            },
        },
        last_model_path,
    )

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(history_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Toplam loss")
    axes[0].plot(history_df["epoch"], history_df["train_ccc_v_loss"], label="CCC_loss(V)", alpha=0.7)
    axes[0].plot(history_df["epoch"], history_df["train_ccc_a_loss"], label="CCC_loss(A)", alpha=0.7)
    axes[0].plot(history_df["epoch"], history_df["train_ce"], label="CE(quadrant)", alpha=0.7)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Train Loss (lambda={lam})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history_df["epoch"], history_df["val_ccc_v"], label="Val CCC (Valence)")
    axes[1].plot(history_df["epoch"], history_df["val_ccc_a"], label="Val CCC (Arousal)")
    axes[1].axvline(best_epoch, color="gray", linestyle="--", alpha=0.5, label=f"En iyi epoch ({best_epoch})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("CCC")
    axes[1].set_title(f"Validation CCC (lambda={lam})")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(curves_path, dpi=150)
    plt.close(fig)

    print(f"\nEn iyi epoch: {best_epoch} (val_mean_ccc={best_val_mean_ccc:.4f})")
    print(f"Best model: {best_model_path}")
    print(f"Last model: {last_model_path}")
    print(f"History: {history_path}")
    print(f"Curves: {curves_path}")

    return {
        "lambda": lam,
        "best_epoch": best_epoch,
        "best_val_mean_ccc": best_val_mean_ccc,
        "best_model_path": best_model_path,
        "history_path": history_path,
    }


def sweep(seed: int = DEFAULT_SEED):
    results = []
    for lam in LAMBDA_SWEEP:
        results.append(train_one_lambda(lam, seed=seed))

    print(f"\n{'=' * 70}\nLAMBDA TARAMASI ÖZETİ\n{'=' * 70}")
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))

    best = max(results, key=lambda r: r["best_val_mean_ccc"])
    best_out_path = os.path.join(MODELS_DIR, "clap_head_best.pt")
    shutil.copy2(best["best_model_path"], best_out_path)
    print(f"\nSeçilen lambda: {best['lambda']} (val_mean_ccc={best['best_val_mean_ccc']:.4f})")
    print(f"Kopyalandı: {best_out_path}")
    return results, best


def parse_args():
    parser = argparse.ArgumentParser(description="CLAP embedding'leri üzerinde V/A + quadrant başlığı eğit")
    parser.add_argument("--lambda-quadrant", type=float, default=None, help="Tek bir lambda değeriyle eğit")
    parser.add_argument("--sweep", action="store_true", help="LAMBDA_SWEEP içindeki tüm değerleri dene")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sweep:
        sweep(seed=args.seed)
    elif args.lambda_quadrant is not None:
        train_one_lambda(args.lambda_quadrant, seed=args.seed)
    else:
        print("--lambda-quadrant <değer> ya da --sweep belirtin.")


if __name__ == "__main__":
    main()
