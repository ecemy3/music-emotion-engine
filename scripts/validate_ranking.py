"""
src/ranking.py doğrulaması.

data/deam/audio'dan duygusal olarak birbirinden çok farklı 5 klip seçilir
(annotations.csv'ye göre elle: düşük V/düşük A, yüksek V/yüksek A, düşük
V/yüksek A, yüksek V/düşük A, orta V/orta A). Hedef, bunlardan birinin
(TARGET_SONG_ID) gerçek VA değeri olarak verilir; o klibin sıralamada ilk
sırada ya da "ayırt edilemez" grupta çıkması beklenir.

Kullanım:
    python scripts/validate_ranking.py                  # varsayılan: ensemble
    python scripts/validate_ranking.py --model clap_head # CLAP-tabanlı model
"""

import argparse
import os
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

from inference import load_clap_head_runtime, predict_multi_window  # noqa: E402
from ranking import INDISTINGUISHABLE_THRESHOLD, rank_candidates, score_candidate  # noqa: E402
from train_cnn_experiments import CNNVA  # noqa: E402
from train_crnn_experiments import CRNNVA  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AUDIO_DIR = os.path.join(REPO_ROOT, "data", "deam", "audio")
ANN_PATH = os.path.join(REPO_ROOT, "data", "deam", "annotations.csv")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
CLAP_HEAD_CKPT = os.path.join(MODELS_DIR, "clap_head_best.pt")

ENSEMBLE_COMPONENTS = ("cnn_optimized", "cnn_baseline", "crnn_v2")
ENSEMBLE_WEIGHTS = {"cnn_optimized": 0.30, "cnn_baseline": 0.30, "crnn_v2": 0.40}

# Duygusal olarak birbirinden çok farklı 5 klip - annotations.csv'ye
# bakılarak elle seçildi (bkz. PR açıklaması / dörtgen dağılımı).
CANDIDATE_SONGS = {
    375: "düşük Valence / düşük Arousal",
    115: "yüksek Valence / yüksek Arousal",
    1903: "düşük Valence / yüksek Arousal",
    631: "yüksek Valence / düşük Arousal",
    40: "orta Valence / orta Arousal",
}

# Hedef = bu şarkının annotations.csv'deki gerçek (V, A) değeri.
TARGET_SONG_ID = 115


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


def parse_args():
    parser = argparse.ArgumentParser(description="src/ranking.py doğrulaması")
    parser.add_argument("--model", choices=["ensemble", "clap_head"], default="ensemble")
    return parser.parse_args()


def main():
    args = parse_args()
    ann = pd.read_csv(ANN_PATH).set_index("song_id")

    if args.model == "clap_head":
        if not os.path.exists(CLAP_HEAD_CKPT):
            print(f"HATA: {CLAP_HEAD_CKPT} bulunamadı. Önce src/train_clap_head.py --sweep çalıştırılmalı.")
            return
        runtime = load_clap_head_runtime(CLAP_HEAD_CKPT, device=DEVICE)
        print(f"Model: clap_head ({CLAP_HEAD_CKPT})\n")
    else:
        models = load_ensemble_models()
        runtime = {"type": "ensemble", "models": models, "weights": ENSEMBLE_WEIGHTS}
        print("Model: ensemble (cnn_optimized 0.30, cnn_baseline 0.30, crnn_v2 0.40)\n")

    target_row = ann.loc[TARGET_SONG_ID]
    target_v, target_a = float(target_row["valence"]), float(target_row["arousal"])
    print(
        f"Hedef: song_id={TARGET_SONG_ID} ({CANDIDATE_SONGS[TARGET_SONG_ID]}), "
        f"gerçek VA=({target_v}, {target_a})\n"
    )

    candidates = []
    display_names = {}
    t0 = time.time()
    for sid, desc in CANDIDATE_SONGS.items():
        wav_path = os.path.join(AUDIO_DIR, f"{sid}.wav")
        if not os.path.exists(wav_path):
            print(f"UYARI: {wav_path} bulunamadı, atlanıyor.")
            continue
        true_v = float(ann.loc[sid, "valence"])
        true_a = float(ann.loc[sid, "arousal"])
        mw = predict_multi_window(wav_path, runtime, device=DEVICE)
        internal_name = f"song_{sid}"
        display_names[internal_name] = f"song_{sid} ({desc}, gerçek V={true_v:.1f}/A={true_a:.1f})"
        candidates.append(score_candidate(internal_name, mw, target_v, target_a))
    elapsed = time.time() - t0

    ranked = rank_candidates(candidates)

    header = (
        f"{'Sıra':<5}{'Aday':<62}{'Tahmin V':>10}{'Tahmin A':>10}"
        f"{'Uzaklık':>10}{'Tutarlılık':>12}{'%Yakın':>8}{'AyırtEdilemez':>15}"
    )
    print(header)
    print("-" * len(header))
    for c in ranked:
        print(
            f"{c.rank:<5}{display_names[c.name]:<62}{c.valence:>10.3f}{c.arousal:>10.3f}"
            f"{c.distance:>10.3f}{c.consistency:>12.3f}{c.pct_windows_close:>8.1f}"
            f"{str(c.indistinguishable_from_best):>15}"
        )

    print(f"\nÇalışma süresi (5 klip): {elapsed:.2f} sn")
    print(f"Ayırt edilemezlik eşiği: {INDISTINGUISHABLE_THRESHOLD}")

    target_candidate = next(c for c in ranked if c.name == f"song_{TARGET_SONG_ID}")
    passed = target_candidate.rank == 1 or target_candidate.indistinguishable_from_best

    print(
        f"\nHedef klip (song_{TARGET_SONG_ID}) -> sıra: {target_candidate.rank}, "
        f"ayırt edilemez mi: {target_candidate.indistinguishable_from_best}"
    )
    print(
        "SONUÇ: BAŞARILI - hedef klip ilk sırada ya da ayırt edilemez grupta"
        if passed
        else "SONUÇ: BAŞARISIZ - hedef klip beklenen konumda değil"
    )


if __name__ == "__main__":
    main()
