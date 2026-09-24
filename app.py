import os
import io
import glob
import re
import pandas as pd
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import json
import traceback
from collections import OrderedDict

def load_dotenv(path: str = ".env"):
    try:
        if not os.path.exists(path):
            return False

        with open(path, "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"'")
                if key and key not in os.environ:
                    os.environ[key] = value

        return True
    except Exception:
        return False

# Survey utils fonksiyonlarını import et
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.survey_utils import survey_row_to_va
from src.advanced_analytics import (
    demographic_analysis,
    emotion_change_analysis,
    human_vs_ai_perception,
    create_demographic_plots,
    create_emotion_change_plots,
    create_human_ai_plots,
)
from src.db_utils import (
    is_local_db_available,
    save_song_analysis,
    save_gemini_analysis,
    get_all_song_analyses,
    get_all_gemini_analyses,
    delete_song_analysis,
    delete_gemini_analysis,
)
from src.flux_image_generator import (
    DEFAULT_MODEL_ID as DEFAULT_FLUX_MODEL_ID,
    FluxConnectionError,
    FluxImageGenerationError,
    FluxRateLimitError,
    FluxServerError,
    FluxTimeoutError,
    compute_prompt_hash as compute_flux_prompt_hash,
    generate_flux_image,
    load_flux_cached_image,
)
from src.art_dna_engine import analyze_art_dna


load_dotenv()


SR = 16000
DURATION = 30
SAMPLES = SR * DURATION
N_FFT = 1024
HOP = 512
N_MELS = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COMPARISON_CSV = os.path.join("results", "model_comparison.csv")
MODELS_DIR = "models"

ENSEMBLE_COMPONENTS = ("cnn_optimized", "cnn_baseline", "crnn_v2")
ENSEMBLE_WEIGHTS = {
    "cnn_optimized": 0.30,
    "cnn_baseline": 0.30,
    "crnn_v2": 0.40,
}
ENSEMBLE_COMPARISON_NAME = "ensemble_best"

mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR,
    n_fft=N_FFT,
    hop_length=HOP,
    n_mels=N_MELS,
)
to_db = torchaudio.transforms.AmplitudeToDB()


class CNNVA(nn.Module):
    def __init__(self, dropout: float = 0.4, architecture: str = "baseline"):
        super().__init__()
        channels = [32, 64, 128, 256] if architecture == "optimized" else [32, 64, 128]

        layers = []
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

    def forward(self, x):
        return self.head(self.net(x))


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

        self.lstm = nn.LSTM(
            input_size=2048,
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

    def forward(self, x):
        feat = self.cnn(x)
        b, c, f, t = feat.shape
        feat = feat.permute(0, 3, 1, 2).reshape(b, t, c * f)
        out, _ = self.lstm(feat)
        pooled = out[:, -1, :]
        return self.head(pooled)


class CRNNVALegacy(nn.Module):
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
        )

        self.lstm = nn.LSTM(
            input_size=128,
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

    def forward(self, x):
        feat = self.cnn(x)
        feat = torch.mean(feat, dim=2)
        feat = feat.permute(0, 2, 1)
        out, _ = self.lstm(feat)
        pooled = out[:, -1, :]
        return self.head(pooled)


class AudioTransformerVA(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=4, dropout=0.3):
        super().__init__()
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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x)


class CNNTransformerVA(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=4, dropout=0.3):
        super().__init__()
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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = torch.mean(x, dim=2)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = torch.mean(x, dim=1)
        return self.head(x)


def _sanitize_sensitive_error_text(message: str) -> str:
    text = str(message or "")
    text = re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "[REDACTED_GOOGLE_KEY]", text)
    text = re.sub(r"AQ\.[0-9A-Za-z_\-\.]{10,}", "[REDACTED_GEMINI_KEY]", text)
    text = re.sub(r"hf_[0-9A-Za-z]{10,}", "[REDACTED_HF_KEY]", text)
    return text


def _show_gemini_error(context_label: str, exc: Exception) -> None:
    safe_message = _sanitize_sensitive_error_text(str(exc))
    st.error(f"{context_label} başarısız: {safe_message}")


def _extract_gemini_response_text(response) -> str:
    if response is None:
        return ""

    direct_text = getattr(response, "text", None)
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                return part_text

    return ""


def _gemini_file_state_name(uploaded_file) -> str:
    state = getattr(uploaded_file, "state", None)
    if state is None:
        return ""

    state_name = getattr(state, "name", None)
    if state_name:
        return str(state_name).upper()

    return str(state).upper()


def _resolve_hf_api_key(raw_value: str = "") -> str:
    value = str(raw_value or "").strip()
    if value:
        return value
    return str(os.getenv("HF_API_KEY", "")).strip()


def _validate_hf_api_key_input(api_key: str):
    key = str(api_key or "").strip()
    if not key:
        return False, ""

    if not key.startswith("hf_") or len(key) < 16:
        return False, "Geçersiz Hugging Face API key formatı. 'hf_' ile başlamalıdır."

    return True, ""


def _validate_gemini_api_key_input(api_key: str):
    key = str(api_key or "").strip()
    if not key:
        return False, ""

    if key.startswith("AIza"):
        return False, "Eski API key formatı tespit edildi. Lütfen yeni 'AQ.' formatındaki Gemini key kullanın."

    if not key.startswith("AQ."):
        return False, "Geçersiz Gemini API key formatı. Yeni anahtarlar 'AQ.' ile başlamalıdır."

    if len(key) < 12:
        return False, "Gemini API key çok kısa görünüyor."

    return True, ""


def detect_model_family(state_dict, model_path: str = ""):
    keys = list(state_dict.keys())
    if any(k.startswith("patch_embed") for k in keys):
        return "transformer"
    if any(k.startswith("lstm.") for k in keys):
        return "crnn"
    if any(k.startswith("transformer.layers") for k in keys):
        return "cnn_transformer"
    return "cnn"


def _count_transformer_layers(state_dict) -> int:
    layer_ids = set()
    for key in state_dict.keys():
        match = re.search(r"transformer\.layers\.(\d+)\.", key)
        if match:
            layer_ids.add(int(match.group(1)))
    return max(layer_ids) + 1 if layer_ids else 4


def build_model_from_state_dict(state_dict, family: str):
    if family == "cnn":
        linear_weight = state_dict.get("head.1.weight")
        architecture = "optimized" if linear_weight is not None and linear_weight.shape[1] >= 256 else "baseline"
        return CNNVA(architecture=architecture)

    if family == "crnn":
        lstm_ih = state_dict.get("lstm.weight_ih_l0")
        input_size = int(lstm_ih.shape[1]) if lstm_ih is not None else 2048
        bidirectional = any("_reverse" in k for k in state_dict.keys())

        layer_ids = set()
        for key in state_dict.keys():
            match = re.search(r"lstm\.weight_ih_l(\d+)$", key)
            if match:
                layer_ids.add(int(match.group(1)))
        num_layers = max(layer_ids) + 1 if layer_ids else 1

        lstm_hh = state_dict.get("lstm.weight_hh_l0")
        hidden_size = int(lstm_hh.shape[1]) if lstm_hh is not None else 128

        if input_size <= 256:
            return CRNNVALegacy(
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
            )

        return CRNNVA(
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )

    if family == "transformer":
        patch_weight = state_dict.get("patch_embed.weight")
        embed_dim = int(patch_weight.shape[0]) if patch_weight is not None else 256
        num_layers = _count_transformer_layers(state_dict)
        return AudioTransformerVA(embed_dim=embed_dim, num_layers=num_layers)

    if family == "cnn_transformer":
        in_proj = state_dict.get("transformer.layers.0.self_attn.in_proj_weight")
        embed_dim = int(in_proj.shape[1]) if in_proj is not None else 256
        num_layers = _count_transformer_layers(state_dict)
        return CNNTransformerVA(embed_dim=embed_dim, num_layers=num_layers)

    return CNNVA()


def resolve_ensemble_component_paths():
    component_paths = {}
    missing = []

    for name in ENSEMBLE_COMPONENTS:
        best_path = os.path.join(MODELS_DIR, f"{name}_best.pt")
        last_path = os.path.join(MODELS_DIR, f"{name}_last.pt")

        if os.path.exists(best_path):
            component_paths[name] = {"path": best_path, "tag": "best"}
        elif os.path.exists(last_path):
            component_paths[name] = {"path": last_path, "tag": "last"}
        else:
            missing.append(name)

    if missing:
        raise FileNotFoundError(
            "Ensemble için checkpoint bulunamayan modeller: " + ", ".join(missing)
        )

    return component_paths


def discover_model_options():
    options = OrderedDict()
    options["⭐ ensemble_best"] = {
        "type": "ensemble",
        "comparison_name": ENSEMBLE_COMPARISON_NAME,
    }

    model_files = []
    model_files.extend(glob.glob(os.path.join(MODELS_DIR, "*_best.pt")))
    model_files.extend(glob.glob(os.path.join(MODELS_DIR, "*_last.pt")))

    for model_path in sorted(model_files):
        file_name = os.path.basename(model_path)
        if file_name == "ensemble_best.pt":
            continue

        if file_name.endswith("_best.pt"):
            base_name = file_name[: -len("_best.pt")]
            tag = "best"
        elif file_name.endswith("_last.pt"):
            base_name = file_name[: -len("_last.pt")]
            tag = "last"
        else:
            continue

        display_name = f"{base_name} [{tag}]"
        options[display_name] = {
            "type": "single",
            "path": model_path,
            "comparison_name": base_name,
        }

    return options


MODEL_OPTIONS = discover_model_options()


def normalize_song_name(song_name):
    """Şarkı adını karşılaştırma için normalize et."""
    if not isinstance(song_name, str):
        return ""

    normalized = re.sub(r"\s+", " ", song_name.strip()).casefold()
    alias_map = {
        "maried life": "married life",
        "marryed life": "married life",
        "marid life": "married life",
    }
    return alias_map.get(normalized, normalized)


def canonical_song_display(song_name):
    """Arayüzde gösterilecek standart şarkı adını döndür."""
    if not isinstance(song_name, str):
        return song_name

    normalized = normalize_song_name(song_name)
    display_map = {
        "married life": "Married Life",
    }
    if normalized in display_map:
        return display_map[normalized]

    return re.sub(r"\s+", " ", song_name.strip())


def find_song_column(df):
    """CSV içinden şarkı kolonunu dinamik bul."""
    def _score(col_name):
        low = str(col_name).casefold()
        score = 0
        if "dinlediniz" in low:
            score += 4
        if "hangi" in low:
            score += 2
        if "müzi" in low or "muz" in low:
            score += 2
        if "amaç" in low or "amac" in low:
            score -= 5
        return score

    candidates = []
    for col in df.columns:
        score = _score(col)
        if score <= 0:
            continue

        non_empty_count = int(
            df[col].astype(str).str.strip().replace("", np.nan).notna().sum()
        )
        candidates.append((score, non_empty_count, col))

    if not candidates:
        return None

    # Önce yüksek skor, eşitlikte en dolu kolon seçilir.
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def normalize_song_column(df, song_col):
    """Şarkı kolonundaki adları standartlaştır."""
    if song_col not in df.columns:
        return df

    df = df.copy()
    df[song_col] = df[song_col].apply(
        lambda v: canonical_song_display(v) if isinstance(v, str) else v
    )
    return df


def get_song_options(df, song_col):
    """Seçim kutuları için benzersiz şarkı adları."""
    if song_col not in df.columns:
        return []

    songs = df[song_col].dropna().tolist()
    songs = [s for s in songs if isinstance(s, str) and s.strip()]
    songs = [canonical_song_display(s) for s in songs]
    return sorted(set(songs))


def filter_df_by_song(df, song_col, selected_song):
    """Seçilen şarkıya göre DataFrame filtrele."""
    if selected_song == "Tüm şarkılar":
        return df

    selected_key = normalize_song_name(selected_song)
    mask = df[song_col].apply(normalize_song_name) == selected_key
    return df[mask].copy()


def load_audio(audio_bytes, file_extension="wav"):
    """
    Audio dosyasını yükle ve işle (WAV veya MP3)
    
    Args:
        audio_bytes: WAV veya MP3 dosyası bytes
        file_extension: Dosya uzantısı ("wav" veya "mp3")
    
    Returns:
        torch.Tensor: İşlenmiş audio waveform
    """
    # Librosa hem WAV hem MP3 destekler ve ek bağımlılık gerektirmez
    wav_np, sr = librosa.load(io.BytesIO(audio_bytes), sr=SR, mono=True)
    
    # Numpy'den torch tensor'e çevir ve (1, samples) şekline getir
    wav = torch.from_numpy(wav_np).unsqueeze(0)
    
    # Pad or trim to fixed duration
    if wav.shape[1] < SAMPLES:
        wav = torch.nn.functional.pad(wav, (0, SAMPLES - wav.shape[1]))
    else:
        wav = wav[:, :SAMPLES]
    
    return wav


def wav_to_logmel(wav):
    """
    Waveform'u log-mel spectrogram'a çevir
    
    Args:
        wav: torch.Tensor waveform
    
    Returns:
        torch.Tensor: Normalize edilmiş log-mel spectrogram
    """
    m = mel_spec(wav)
    m = to_db(m)
    m = (m - m.mean()) / (m.std() + 1e-6)
    return m


def predict_va(model, wav):
    """
    Model ile Valence-Arousal tahmini yap
    
    Args:
        model: Eğitilmiş CNN model
        wav: Audio waveform
    
    Returns:
        tuple: (valence, arousal) tahminleri
    """
    mel = wav_to_logmel(wav).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        v, a = model(mel).cpu().numpy()[0]
    return float(v), float(a)


def predict_va_from_mel(model, mel):
    """Hazır mel-spectrogram ile model tahmini yap."""
    model.eval()
    with torch.no_grad():
        v, a = model(mel).cpu().numpy()[0]
    return float(v), float(a)


def predict_with_selected_model(selected_model_runtime, wav):
    """Tek model veya ensemble için tahmini ortak arayüzle üret."""
    if selected_model_runtime["type"] == "single":
        return predict_va(selected_model_runtime["model"], wav)

    mel = wav_to_logmel(wav).unsqueeze(0).to(DEVICE)
    weighted_v = 0.0
    weighted_a = 0.0
    for model_name, model in selected_model_runtime["models"].items():
        weight = selected_model_runtime["weights"][model_name]
        pv, pa = predict_va_from_mel(model, mel)
        weighted_v += weight * pv
        weighted_a += weight * pa

    return float(weighted_v), float(weighted_a)


def compare(song_name, audio_bytes, df, selected_model_runtime, file_extension="wav", song_col="Hangi müziği dinlediniz"):
    """
    İnsan ve model tahminlerini karşılaştır
    
    Args:
        song_name: Şarkı adı
        audio_bytes: WAV veya MP3 dosyası bytes
        df: Anket verileri DataFrame
        selected_model_runtime: Seçili model çalışma bilgisi
        file_extension: Dosya uzantısı ("wav" veya "mp3")
    
    Returns:
        dict: Karşılaştırma sonuçları
    """
    # Model tahmini
    wav = load_audio(audio_bytes, file_extension)
    mv, ma = predict_with_selected_model(selected_model_runtime, wav)
    
    # İnsan tahminleri (anket)
    selected_key = normalize_song_name(song_name)
    song_df = df[df[song_col].apply(normalize_song_name) == selected_key]
    
    if len(song_df) == 0:
        return None
    
    hv_list, ha_list = [], []
    for _, row in song_df.iterrows():
        hv, ha = survey_row_to_va(row)
        hv_list.append(hv)
        ha_list.append(ha)
    
    hv_mean = np.mean(hv_list)
    ha_mean = np.mean(ha_list)
    hv_std = np.std(hv_list)
    ha_std = np.std(ha_list)
    
    return {
        "Model_Valence": mv,
        "Model_Arousal": ma,
        "Human_Valence_Mean": hv_mean,
        "Human_Arousal_Mean": ha_mean,
        "Human_Valence_Std": hv_std,
        "Human_Arousal_Std": ha_std,
        "Delta_V": abs(hv_mean - mv),
        "Delta_A": abs(ha_mean - ma),
        "N_Responses": len(song_df)
    }


@st.cache_resource
def load_model(model_path):
    """Model'i yükle (cache ile)"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=True)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    family = detect_model_family(state_dict, model_path)
    model = build_model_from_state_dict(state_dict, family).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model, family


@st.cache_resource
def load_ensemble_models(component_items):
    """Ensemble bileşen modellerini cache ile yükle."""
    loaded_models = {}
    loaded_families = {}

    for model_name, model_path in component_items:
        model, family = load_model(model_path=model_path)
        loaded_models[model_name] = model
        loaded_families[model_name] = family

    return loaded_models, loaded_families


@st.cache_data
def load_comparison_table():
    if not os.path.exists(COMPARISON_CSV):
        return None
    try:
        return pd.read_csv(COMPARISON_CSV)
    except Exception:
        return None


def _pick_column(columns, candidates):
    for col in candidates:
        if col in columns:
            return col
    return None


def _load_prediction_file_for_metrics(model_name):
    path = os.path.join("results", f"{model_name}_predictions.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tahmin dosyası bulunamadı: {path}")

    raw = pd.read_csv(path)
    id_col = _pick_column(raw.columns, ["song_id", "sample_id"])
    pred_v_col = _pick_column(raw.columns, ["valence", "pred_valence"])
    pred_a_col = _pick_column(raw.columns, ["arousal", "pred_arousal"])
    true_v_col = _pick_column(raw.columns, ["true_valence", "valence_true", "gt_valence"])
    true_a_col = _pick_column(raw.columns, ["true_arousal", "arousal_true", "gt_arousal"])

    if id_col is None or pred_v_col is None or pred_a_col is None:
        raise ValueError(f"{model_name} dosyasında gerekli kolonlar eksik")
    if true_v_col is None or true_a_col is None:
        raise ValueError(f"{model_name} dosyasında true_valence/true_arousal kolonları eksik")

    df = pd.DataFrame(
        {
            "song_id": pd.to_numeric(raw[id_col], errors="coerce").round().astype("Int64"),
            "pred_valence": pd.to_numeric(raw[pred_v_col], errors="coerce"),
            "pred_arousal": pd.to_numeric(raw[pred_a_col], errors="coerce"),
            "true_valence": pd.to_numeric(raw[true_v_col], errors="coerce"),
            "true_arousal": pd.to_numeric(raw[true_a_col], errors="coerce"),
        }
    )

    if df.isna().any().any():
        raise ValueError(f"{model_name} tahmin dosyasında NaN değer var")

    df["song_id"] = df["song_id"].astype(np.int64)
    return df


def _validate_alignment(reference_ids, candidate_ids, model_name):
    if len(reference_ids) != len(candidate_ids):
        raise ValueError(f"{model_name} satır sayısı farklı")
    if not np.array_equal(reference_ids, candidate_ids):
        raise ValueError(f"{model_name} song_id sırası/samples farklı")


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _safe_pearson(y_true, y_pred):
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


@st.cache_data
def compute_ensemble_metrics_for_comparison():
    """Weighted ensemble metriklerini results/*_predictions.csv üstünden hesapla."""
    pred_frames = {
        model_name: _load_prediction_file_for_metrics(model_name)
        for model_name in ENSEMBLE_COMPONENTS
    }

    ref_ids = pred_frames[ENSEMBLE_COMPONENTS[0]]["song_id"].to_numpy(dtype=np.int64)
    for model_name in ENSEMBLE_COMPONENTS[1:]:
        _validate_alignment(ref_ids, pred_frames[model_name]["song_id"].to_numpy(dtype=np.int64), model_name)

    y_true_v = pred_frames[ENSEMBLE_COMPONENTS[0]]["true_valence"].to_numpy(dtype=np.float64)
    y_true_a = pred_frames[ENSEMBLE_COMPONENTS[0]]["true_arousal"].to_numpy(dtype=np.float64)

    for model_name in ENSEMBLE_COMPONENTS[1:]:
        frame = pred_frames[model_name]
        if not np.allclose(y_true_v, frame["true_valence"].to_numpy(dtype=np.float64), atol=1e-7):
            raise ValueError(f"true_valence uyumsuz: {model_name}")
        if not np.allclose(y_true_a, frame["true_arousal"].to_numpy(dtype=np.float64), atol=1e-7):
            raise ValueError(f"true_arousal uyumsuz: {model_name}")

    ensemble_v = np.zeros_like(y_true_v)
    ensemble_a = np.zeros_like(y_true_a)
    for model_name, weight in ENSEMBLE_WEIGHTS.items():
        ensemble_v += weight * pred_frames[model_name]["pred_valence"].to_numpy(dtype=np.float64)
        ensemble_a += weight * pred_frames[model_name]["pred_arousal"].to_numpy(dtype=np.float64)

    return {
        "model_name": ENSEMBLE_COMPARISON_NAME,
        "rmse_v": _rmse(y_true_v, ensemble_v),
        "rmse_a": _rmse(y_true_a, ensemble_a),
        "pearson_v": _safe_pearson(y_true_v, ensemble_v),
        "pearson_a": _safe_pearson(y_true_a, ensemble_a),
    }


@st.cache_data
def load_comparison_table_with_ensemble():
    """results/model_comparison.csv tablosunu ensemble satırı ekleyerek döndür."""
    comp_df = load_comparison_table()
    if comp_df is None or comp_df.empty:
        return None

    if "model_name" not in comp_df.columns:
        if "model" in comp_df.columns:
            comp_df = comp_df.rename(columns={"model": "model_name"})
        else:
            return None

    needed_cols = ["model_name", "rmse_v", "rmse_a", "pearson_v", "pearson_a"]
    if any(col not in comp_df.columns for col in needed_cols):
        return None

    table = comp_df[needed_cols].copy()

    try:
        ensemble_row = compute_ensemble_metrics_for_comparison()
        table = table[table["model_name"] != ENSEMBLE_COMPARISON_NAME]
        table = pd.concat([table, pd.DataFrame([ensemble_row])], ignore_index=True)
    except Exception:
        pass

    return table


def render_selected_model_metrics(selected_model_name, selected_model):
    comp_df = load_comparison_table_with_ensemble()
    if comp_df is None or comp_df.empty or "model_name" not in comp_df.columns:
        st.sidebar.info("Model metrik tablosu bulunamadı")
        return

    row = comp_df[comp_df["model_name"] == selected_model["comparison_name"]]
    if row.empty:
        st.sidebar.info(f"{selected_model_name} için metrik kaydı yok")
        return

    r = row.iloc[0]
    st.sidebar.markdown("### Seçili Model Metrikleri")
    st.sidebar.metric("RMSE (Valence)", f"{r['rmse_v']:.4f}")
    st.sidebar.metric("RMSE (Arousal)", f"{r['rmse_a']:.4f}")
    st.sidebar.metric("Pearson (Valence)", f"{r['pearson_v']:.4f}")
    st.sidebar.metric("Pearson (Arousal)", f"{r['pearson_a']:.4f}")


def render_model_performance_tab(selected_model):
    """results/model_comparison.csv + ensemble satırı ile model kıyas tabı."""
    st.subheader("Model Karşılaştırma")
    st.info(
        "Ensemble model, birden fazla modelin tahminlerini birleştirerek daha yüksek doğruluk sağlar."
    )

    comp_df = load_comparison_table_with_ensemble()
    if comp_df is None or comp_df.empty:
        st.warning("Model karşılaştırma tablosu yüklenemedi")
        return

    # Modelleri ortalama RMSE'ye göre sırala (eşitlikte ortalama Pearson daha yüksek olan önde)
    comp_df = comp_df.copy()
    comp_df["avg_rmse"] = (comp_df["rmse_v"] + comp_df["rmse_a"]) / 2.0
    comp_df["avg_pearson"] = (comp_df["pearson_v"] + comp_df["pearson_a"]) / 2.0
    comp_df = comp_df.sort_values(
        by=["avg_rmse", "avg_pearson"],
        ascending=[True, False],
    ).reset_index(drop=True)

    top3 = comp_df.head(3).copy()
    st.markdown("### Önerilen Modeller (Top 3)")
    top_lines = []
    for idx, row in top3.iterrows():
        top_lines.append(
            f"{idx + 1}. {row['model_name']} | avg_rmse={row['avg_rmse']:.4f} | avg_pearson={row['avg_pearson']:.4f}"
        )
    st.success("\n".join(top_lines))

    display_df = comp_df.copy()
    display_df = display_df.rename(
        columns={
            "model_name": "Model",
            "rmse_v": "RMSE_V",
            "rmse_a": "RMSE_A",
            "pearson_v": "Pearson_V",
            "pearson_a": "Pearson_A",
            "avg_rmse": "AVG_RMSE",
            "avg_pearson": "AVG_Pearson",
        }
    )

    st.markdown("### Metrik Tablosu")
    st.dataframe(
        display_df[["Model", "RMSE_V", "RMSE_A", "Pearson_V", "Pearson_A", "AVG_RMSE", "AVG_Pearson"]],
        use_container_width=True,
        hide_index=True,
    )

    model_names = comp_df["model_name"].tolist()
    x = np.arange(len(model_names))

    rmse_v = comp_df["rmse_v"].to_numpy(dtype=np.float64)
    rmse_a = comp_df["rmse_a"].to_numpy(dtype=np.float64)
    p_v = comp_df["pearson_v"].to_numpy(dtype=np.float64)
    p_a = comp_df["pearson_a"].to_numpy(dtype=np.float64)

    ensemble_color_main = "#d62728"
    ensemble_color_soft = "#ff9896"
    normal_color_main = "#4c78a8"
    normal_color_soft = "#9ecae9"

    rmse_colors_v = [
        ensemble_color_main if name == ENSEMBLE_COMPARISON_NAME else normal_color_main
        for name in model_names
    ]
    rmse_colors_a = [
        ensemble_color_soft if name == ENSEMBLE_COMPARISON_NAME else normal_color_soft
        for name in model_names
    ]

    pearson_colors_v = [
        ensemble_color_main if name == ENSEMBLE_COMPARISON_NAME else "#59a14f"
        for name in model_names
    ]
    pearson_colors_a = [
        ensemble_color_soft if name == ENSEMBLE_COMPARISON_NAME else "#8cd17d"
        for name in model_names
    ]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### RMSE Karşılaştırması")
        fig_rmse, ax_rmse = plt.subplots(figsize=(10, 5))
        width = 0.35
        ax_rmse.bar(x - width / 2, rmse_v, width, label="RMSE_V", color=rmse_colors_v)
        ax_rmse.bar(x + width / 2, rmse_a, width, label="RMSE_A", color=rmse_colors_a)
        ax_rmse.set_ylabel("RMSE")
        ax_rmse.set_xticks(x)
        ax_rmse.set_xticklabels(model_names, rotation=35, ha="right")
        ax_rmse.grid(axis="y", alpha=0.3)
        ax_rmse.legend()
        st.pyplot(fig_rmse)

    with col2:
        st.markdown("### Pearson Karşılaştırması")
        fig_p, ax_p = plt.subplots(figsize=(10, 5))
        width = 0.35
        ax_p.bar(x - width / 2, p_v, width, label="Pearson_V", color=pearson_colors_v)
        ax_p.bar(x + width / 2, p_a, width, label="Pearson_A", color=pearson_colors_a)
        ax_p.set_ylabel("Pearson")
        ax_p.set_xticks(x)
        ax_p.set_xticklabels(model_names, rotation=35, ha="right")
        ax_p.grid(axis="y", alpha=0.3)
        ax_p.legend()
        st.pyplot(fig_p)

    if selected_model["comparison_name"] == ENSEMBLE_COMPARISON_NAME:
        st.success("Seçili model: ensemble_best (kırmızı ile vurgulandı)")


def main():
    st.set_page_config(page_title="Human vs CNN Emotion", page_icon="🎵", layout="wide")
    
    st.title("🎵 Human vs CNN Emotion Analysis")
    st.markdown("---")
    
    # CSV yükleme
    st.sidebar.header("Ayarlar")
    if not MODEL_OPTIONS:
        st.error("models klasorunde test edilebilir model bulunamadi")
        return

    selected_model_name = st.sidebar.selectbox(
        "Model Seç",
        list(MODEL_OPTIONS.keys()),
        index=0,
    )
    selected_model = MODEL_OPTIONS[selected_model_name]
    st.sidebar.info(
        "Ensemble model, birden fazla modelin tahminlerini birleştirerek daha yüksek doğruluk sağlar."
    )

    if "hf_api_key" not in st.session_state:
        st.session_state["hf_api_key"] = _resolve_hf_api_key()

    st.sidebar.text_input(
        "🔑 Hugging Face API Key",
        type="password",
        key="hf_api_key",
        help="FLUX.1-schnell gorsel uretimi icin kullanilir. Bos birakirsaniz HF_API_KEY env/.env degeri kullanilir.",
    )

    csv_file = st.sidebar.file_uploader("Anket CSV Dosyası", type=["csv"])
    
    if csv_file is None:
        st.warning("Lütfen anket CSV dosyasını yükleyin")
        return
    
    # DataFrame yükle
    df = pd.read_csv(csv_file)
    
    # Şarkı kolonunu dinamik bul
    song_col = find_song_column(df)
    if song_col is None:
        st.error("CSV'de şarkı kolonu bulunamadı")
        return

    # Şarkı adlarını normalize et (örn. Maried Life -> Married Life)
    df = normalize_song_column(df, song_col)

    # Gerekli kolonları kontrol et (CSV'deki gerçek kolon adları - sonda boşluk var!)
    required_cols = [
        "Bu müzik beni huzurlu ve sakin hissettirdi. ",
        "Bu müzik beni neşeli ve enerjik hissettirdi. ",
        "Bu müzik bana nostaljik bir his verdi. ",
        "Bu müzik beni üzgün veya melankolik hissettirdi. ",
        "Bu müzik bana hayranlık ve şaşkınlık hisleri uyandırdı. ",
        "Bu müzik beni gergin veya huzursuz hissettirdi. ",
        "Bu müzik beni güçlü, motive olmuş hissettirdi. "
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"CSV'de eksik kolonlar: {missing_cols}")
        return

    song_options = get_song_options(df, song_col)
    if not song_options:
        st.error("CSV içinde analiz edilebilir şarkı verisi bulunamadı")
        return

    selected_song_filter = st.sidebar.selectbox(
        "Analiz Şarkı Filtresi",
        ["Tüm şarkılar"] + song_options,
        index=0,
    )
    analysis_df = filter_df_by_song(df, song_col, selected_song_filter)

    if selected_song_filter != "Tüm şarkılar":
        st.sidebar.info(
            f"Filtre: {selected_song_filter} ({len(analysis_df)} yanıt)"
        )

    if analysis_df.empty:
        st.warning("Seçilen filtre için veri bulunamadı")
        return
    
    st.sidebar.success(f"{len(analysis_df)} anket cevabı hazır")
    
    # Model yükle
    try:
        if selected_model.get("type") == "ensemble":
            component_paths = resolve_ensemble_component_paths()
            component_items = tuple(
                (name, comp["path"])
                for name, comp in sorted(component_paths.items(), key=lambda x: x[0])
            )
            ensemble_models, ensemble_families = load_ensemble_models(component_items)
            family_text = ", ".join(
                f"{name}:{ensemble_families[name]}"
                for name in sorted(ensemble_families.keys())
            )
            selected_model_runtime = {
                "type": "ensemble",
                "models": ensemble_models,
                "weights": ENSEMBLE_WEIGHTS,
            }
            st.sidebar.success(
                f"Ensemble aktif: {selected_model_name} ({family_text}, {DEVICE})"
            )
            st.sidebar.caption(
                "Bileşenler: "
                + ", ".join(
                    f"{name} [{component_paths[name]['tag']}]"
                    for name in ENSEMBLE_COMPONENTS
                )
            )
        else:
            model, family = load_model(model_path=selected_model["path"])
            selected_model_runtime = {
                "type": "single",
                "model": model,
            }
            st.sidebar.success(f"Model yüklendi: {selected_model_name} ({family}, {DEVICE})")

        render_selected_model_metrics(selected_model_name, selected_model)
    except Exception as e:
        st.error(f"Model yüklenemedi: {e}")
        return

    # DynamoDB Local durum göstergesi (sidebar)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### DynamoDB Local")
    if is_local_db_available():
        st.sidebar.success("Bağlı (localhost:8000)")
    else:
        st.sidebar.warning("Bağlı değil – analizler kaydedilmeyecek")
        st.sidebar.caption("Başlatmak için: `docker-compose up -d`")

    # TAB YAPISI
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Model Karşılaştırma",
        "Şarkı Bazlı Karşılaştırma",
        "Demografik Analiz",
        "Duygu Değişimi",
        "İnsan vs Gemini AI",
        "İnsan vs Seçili Model vs Gemini",
        "🎨 Şarkının Art DNA Görselleştirmesi",
        "Geçmiş Analizler",
    ])
    
    # TAB 1: Genel model performans karşılaştırması
    with tab1:
        render_model_performance_tab(selected_model)

    # TAB 2: Şarkı bazlı model karşılaştırması
    with tab2:
        render_model_comparison_tab(analysis_df, selected_model_runtime, song_col)
    
    # TAB 3: Demografik Analiz
    with tab3:
        render_demographic_tab(analysis_df)
    
    # TAB 4: Duygu Değişimi
    with tab4:
        render_emotion_change_tab(df, song_col)
    
    # TAB 5: İnsan vs Gemini
    with tab5:
        render_gemini_tab(analysis_df, song_col)
    
    # TAB 6: Üçlü Karşılaştırma
    with tab6:
        render_triple_comparison_tab(
            analysis_df,
            selected_model_runtime,
            selected_model_name,
            song_col,
        )

    # TAB 7: Şarkının Art DNA Görselleştirmesi
    with tab7:
        render_abstract_emotion_visualization_tab(df, song_col)

    # TAB 8: Geçmiş Analizler (DynamoDB Local)
    with tab8:
        render_history_tab()


def render_model_comparison_tab(df, selected_model_runtime, song_col):
    """Orijinal model karşılaştırma sekmesi"""
    st.subheader("Şarkı Bazlı Model Karşılaştırması")
    
    # Açıklama kutusu
    st.info("""
    **Bu bölümde ne yapıyoruz?**
    
    Aynı şarkıyı hem **insanlara** hem de **yapay zeka modeline** dinletiyoruz ve duygusal tepkilerini karşılaştırıyoruz.
    
    **Valence (Değerlik)**: Müziğin ne kadar pozitif/negatif hissettirdiği
    - 1-3: Negatif (üzgün, melankolik, kızgın)
    - 4-6: Nötr (kararsız, karmaşık)
    - 7-9: Pozitif (mutlu, neşeli, iyimser)
    
    **Arousal (Uyarılmışlık)**: Müziğin enerji seviyesi
    - 1-3: Düşük enerji (sakin, huzurlu, durgun)
    - 4-6: Orta enerji (dengeli)
    - 7-9: Yüksek enerji (heyecanlı, enerjik, gergin)
    
    **Δ (Delta)**: İnsan ve model tahminleri arasındaki fark - ne kadar küçükse o kadar benzer!
    """)
    
    # Ana arayüz
    col1, col2 = st.columns(2)
    
    with col1:
        # Şarkı seçimi (NaN değerleri temizle)
        songs = get_song_options(df, song_col)
        if not songs:
            st.warning("Şarkı seçimi için veri bulunamadı")
            return

        song = st.selectbox("Şarkı Seç", songs, key="demographic_song")
        
        # Kaç kişi cevap vermiş
        song_mask = df[song_col].apply(normalize_song_name) == normalize_song_name(song)
        n_responses = int(song_mask.sum())
        st.info(f"Bu şarkı için {n_responses} anket cevabı mevcut")
    
    with col2:
        # Audio upload
        audio = st.file_uploader(" Şarkı Dosyasını Yükle (.wav, .mp3)", type=["wav", "mp3"])
    
    # Karşılaştırma butonu
    if st.button("Karşılaştır", type="primary", use_container_width=True):
        if audio is None:
            st.warning("Lütfen şarkı dosyasını yükleyin (.wav veya .mp3)")
            return
        
        with st.spinner("Analiz ediliyor..."):
            # Dosya uzantısını tespit et
            file_ext = audio.name.split('.')[-1].lower()
            
            # Karşılaştırmayı yap
            res = compare(
                song,
                audio.getvalue(),
                df,
                selected_model_runtime,
                file_ext,
                song_col=song_col,
            )
            
            if res is None:
                st.error("Seçilen şarkı için anket verisi bulunamadı")
                return
            
            st.markdown("---")
            st.subheader("Sonuçlar")
            
            # Özet yorum
            delta_v = res['Delta_V']
            delta_a = res['Delta_A']
            avg_delta = (delta_v + delta_a) / 2
            
            if avg_delta < 1.0:
                st.success(f"**Mükemmel uyum!** Model insan tepkilerine çok yakın tahmin yaptı (Ortalama fark: {avg_delta:.2f})")
            elif avg_delta < 2.0:
                st.success(f"**İyi uyum!** Model insan tepkilerine yakın tahmin yaptı (Ortalama fark: {avg_delta:.2f})")
            elif avg_delta < 3.0:
                st.warning(f"**Orta uyum.** Model ile insanlar arasında farklar var (Ortalama fark: {avg_delta:.2f})")
            else:
                st.error(f"**Zayıf uyum.** Model insan tepkilerinden oldukça farklı (Ortalama fark: {avg_delta:.2f})")
            
            # Metrikler
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Human Valence",
                    f"{res['Human_Valence_Mean']:.2f}",
                    delta=f"±{res['Human_Valence_Std']:.2f}"
                )
            
            with col2:
                st.metric(
                    "Model Valence",
                    f"{res['Model_Valence']:.2f}",
                    delta=f"Δ {res['Delta_V']:.2f}"
                )
            
            with col3:
                st.metric(
                    "Human Arousal",
                    f"{res['Human_Arousal_Mean']:.2f}",
                    delta=f"±{res['Human_Arousal_Std']:.2f}"
                )
            
            with col4:
                st.metric(
                    " Model Arousal",
                    f"{res['Model_Arousal']:.2f}",
                    delta=f"Δ {res['Delta_A']:.2f}"
                )
            
            # Görselleştirme
            st.markdown("---")
            st.subheader(" Görselleştirme")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Valence-Arousal Karşılaştırması**")
                # Bar chart
                chart_data = pd.DataFrame({
                    "İnsan": [res['Human_Valence_Mean'], res['Human_Arousal_Mean']],
                    "Model": [res['Model_Valence'], res['Model_Arousal']]
                }, index=["Valence (Pozitiflik)", "Arousal (Enerji)"])
                
                st.bar_chart(chart_data)
                
                st.markdown("""
                **Bu grafikte ne görüyoruz?**
                
                **İnsan** (koyu mavi): İnsanların ortalama skorları  
                **Model** (açık mavi): Yapay zekanın tahminleri
                
                Çubuklar ne kadar yakınsa model o kadar başarılı!
                """)
            
            with col2:
                st.markdown("**Duygu Haritası**")
                # VA Space scatter plot
                fig, ax = plt.subplots(figsize=(6, 6))
                
                # Human point
                ax.scatter(
                    res['Human_Valence_Mean'],
                    res['Human_Arousal_Mean'],
                    s=200,
                    c='blue',
                    marker='o',
                    label='İnsan',
                    alpha=0.7
                )
                
                # Model point
                ax.scatter(
                    res['Model_Valence'],
                    res['Model_Arousal'],
                    s=200,
                    c='red',
                    marker='s',
                    label='Model',
                    alpha=0.7
                )
                
                # Error bars for human
                ax.errorbar(
                    res['Human_Valence_Mean'],
                    res['Human_Arousal_Mean'],
                    xerr=res['Human_Valence_Std'],
                    yerr=res['Human_Arousal_Std'],
                    fmt='none',
                    c='blue',
                    alpha=0.3
                )
                
                # Connection line
                ax.plot(
                    [res['Human_Valence_Mean'], res['Model_Valence']],
                    [res['Human_Arousal_Mean'], res['Model_Arousal']],
                    'k--',
                    alpha=0.3
                )
                
                ax.set_xlim(1, 9)
                ax.set_ylim(1, 9)
                ax.set_xlabel('Valence (Negatif ← → Pozitif)', fontsize=11)
                ax.set_ylabel('Arousal (Sakin ← → Enerjik)', fontsize=11)
                ax.set_title('Duygu Uzayı', fontsize=13)
                ax.grid(True, alpha=0.3)
                ax.legend()
                ax.axhline(5, color='gray', linewidth=0.5, alpha=0.5)
                ax.axvline(5, color='gray', linewidth=0.5, alpha=0.5)
                
                # Bölge etiketleri
                ax.text(7.5, 7.5, 'Mutlu\nHeyecanlı', ha='center', fontsize=9, alpha=0.5)
                ax.text(2.5, 7.5, 'Gergin\nKızgın', ha='center', fontsize=9, alpha=0.5)
                ax.text(7.5, 2.5, 'Huzurlu\nRahat', ha='center', fontsize=9, alpha=0.5)
                ax.text(2.5, 2.5, 'Üzgün\nMelankolik', ha='center', fontsize=9, alpha=0.5)
                
                st.pyplot(fig)
                
                st.markdown(f"""
                **Bu grafikte ne görüyoruz?**
                
                Mavi nokta: İnsanların ortalama tepkisi  
                Kırmızı kare: Model'in tahmini  
                
                Nokta ve kare ne kadar yakınsa başarı o kadar yüksek!
                Kesikli çizgi iki nokta arasındaki mesafeyi gösteriyor.
                
                **Sonuç**: Model bu şarkıyı {'çok benzer' if avg_delta < 2 else 'benzer' if avg_delta < 3 else 'farklı'} yorumladı.
                """, 9)
                ax.set_ylim(1, 9)
                ax.set_xlabel('Valence', fontsize=12)
                ax.set_ylabel('Arousal', fontsize=12)
                ax.set_title('Valence-Arousal Space', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.legend()
                ax.axhline(5, color='gray', linewidth=0.5, alpha=0.5)
                ax.axvline(5, color='gray', linewidth=0.5, alpha=0.5)
                
                st.pyplot(fig)
            
            # DynamoDB Local'e kaydet
            model_label = selected_model_runtime.get("type", "unknown")
            if model_label == "ensemble":
                model_label = "ensemble_best"
            saved = save_song_analysis(
                song_name=song,
                model_name=model_label,
                model_valence=res['Model_Valence'],
                model_arousal=res['Model_Arousal'],
                human_valence_mean=res['Human_Valence_Mean'],
                human_arousal_mean=res['Human_Arousal_Mean'],
                human_valence_std=res['Human_Valence_Std'],
                human_arousal_std=res['Human_Arousal_Std'],
                delta_v=res['Delta_V'],
                delta_a=res['Delta_A'],
                n_responses=res['N_Responses'],
            )
            if saved:
                st.caption("Analiz sonucu DynamoDB Local'e kaydedildi.")

            # Detaylı bilgi
            with st.expander("Detaylı Bilgi"):
                st.json({
                    "Şarkı": song,
                    "Anket Katılımcı Sayısı": res['N_Responses'],
                    "Human Valence": f"{res['Human_Valence_Mean']:.2f} ± {res['Human_Valence_Std']:.2f}",
                    "Human Arousal": f"{res['Human_Arousal_Mean']:.2f} ± {res['Human_Arousal_Std']:.2f}",
                    "Model Valence": f"{res['Model_Valence']:.2f}",
                    "Model Arousal": f"{res['Model_Arousal']:.2f}",
                    "Valence Farkı": f"{res['Delta_V']:.2f}",
                    "Arousal Farkı": f"{res['Delta_A']:.2f}"
                })


def render_demographic_tab(df):
    """Demografik analiz sekmesi"""
    st.subheader("Demografik Analiz")
    
    # Açıklama kutusu
    st.info("""
    **Bu analizde neler var?**
    
    Bu bölümde anket katılımcılarının yaş, cinsiyet ve müzik dinleme alışkanlıklarına göre 
    duygusal tepkilerini inceliyoruz. 
    
    **Valence (Değerlik)**: Duygunun pozitif mi (mutlu, neşeli) yoksa negatif mi (üzgün, kızgın) olduğunu gösterir. 
    1-9 skalasında: 1=Çok Negatif, 5=Nötr, 9=Çok Pozitif
    
    **Arousal (Uyarılmışlık)**: Duygunun enerji seviyesini gösterir.
    1-9 skalasında: 1=Çok Sakin/Durgun, 5=Orta, 9=Çok Heyecanlı/Enerjik
    """)
    
    with st.spinner("Analiz yapılıyor..."):
        results, analyzed_df = demographic_analysis(df)
    
    st.success("Analiz tamamlandı!")
    
    # Görselleştirmeler
    figures = create_demographic_plots(results, analyzed_df)
    
    st.markdown("###Yaş Gruplarına Göre Duygusal Tepkiler")
    st.pyplot(figures[0])
    st.markdown("""
    ** Bu grafikte ne görüyoruz?**
    
    Farklı yaş gruplarının müziğe verdiği duygusal tepkilerin ortalamaları. 
    Genç dinleyiciler mi yoksa yaşlı dinleyiciler mi müziğe daha pozitif/enerjik tepki veriyor?
    """)
    
    st.markdown("---")
    st.markdown("### Cinsiyete Göre Duygusal Farklılıklar")
    st.pyplot(figures[1])
    st.markdown("""
    ** Bu grafikte ne görüyoruz?**
    
    Kadın ve erkek dinleyicilerin aynı müziklere verdiği duygusal tepkilerin karşılaştırması.
    Yan yana duran çubuklar, her cinsiyetin Valence (mavi) ve Arousal (turuncu) skorlarını gösteriyor.
    """)
    
    st.markdown("---")
    st.markdown("###  Müzik Dinleme Sıklığı ve Duygu Yoğunluğu")
    st.pyplot(figures[2])
    st.markdown("""
    **Bu grafikte ne görüyoruz?**
    
    Müziği ne kadar sık dinlediğiniz, müziğe verdiğiniz duygusal tepkinin yoğunluğunu etkiliyor mu?
    Düzenli müzik dinleyenler daha yoğun duygular mı yaşıyor?
    """)
    
    st.markdown("---")
    st.markdown("### Duygusal Tepki Uzayı (Cinsiyet Bazlı)")
    st.pyplot(figures[3])
    st.markdown("""
    **Bu grafikte ne görüyoruz?**
    
    Bu grafik "duygu haritası" gibi düşünülebilir. Her nokta bir kişinin tepkisini temsil ediyor.
    - **Sağ yukarı**: Pozitif ve enerjik (örn: mutlu, heyecanlı)
    - **Sol yukarı**: Negatif ama enerjik (örn: gergin, kızgın)
    - **Sağ aşağı**: Pozitif ama sakin (örn: huzurlu, rahat)
    - **Sol aşağı**: Negatif ve sakin (örn: üzgün, melankolik)
    """)
    
    # İstatistikler
    with st.expander("Detaylı İstatistikler"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Yaş Grupları**")
            st.dataframe(results['age_groups'])
        
        with col2:
            st.markdown("**Cinsiyet**")
            st.dataframe(results['gender'])
        
        st.markdown("**Müzik Dinleme Sıklığı**")
        st.dataframe(results['frequency'])
        
        st.markdown("**Sıklık vs Duygu Yoğunluğu**")
        st.dataframe(results['frequency_intensity'])


def render_emotion_change_tab(df, song_col):
    """Duygu değişimi analiz sekmesi"""
    st.subheader("Duygu Durumu Değişimi")
    
    st.info("""
    ** Müzik duygularımızı nasıl etkiliyor?**
    
    Bu bölümde müzik dinlemeden önceki ve sonraki duygu durumlarınızı karşılaştırıyoruz.
    Hangi şarkılar insanları daha çok etkiliyor? Müzik gerçekten ruh halimizi değiştiriyor mu?
    """)

    songs = get_song_options(df, song_col)
    selected_song = st.selectbox(
        "Şarkı Bazlı Değerlendirme",
        ["Tüm şarkılar"] + songs,
        key="emotion_change_song",
    )
    selected_df = filter_df_by_song(df, song_col, selected_song)

    if selected_df.empty:
        st.warning("Seçilen şarkı için veri bulunamadı")
        return

    if selected_song == "Tüm şarkılar":
        st.caption(f"Analiz kapsamı: tüm şarkılar ({len(selected_df)} yanıt)")
    else:
        st.caption(f"Analiz kapsamı: {selected_song} ({len(selected_df)} yanıt)")
    
    with st.spinner("Analiz yapılıyor..."):
        selected_results = emotion_change_analysis(selected_df)
        all_results = emotion_change_analysis(df)
    
    st.success(" Analiz tamamlandı!")
    
    # Görselleştirmeler
    figures_selected = create_emotion_change_plots(selected_results)
    total_song_count = len(all_results['song_deep_impact'])
    top_song_count = min(10, total_song_count)
    if total_song_count >= 2:
        top_song_count = max(2, top_song_count)
    figures_all = create_emotion_change_plots(all_results, top_n=top_song_count)

    if selected_song != "Tüm şarkılar":
        selected_impact = all_results['song_deep_impact'].get(selected_song, np.nan)
        global_impact = float(all_results['song_deep_impact'].mean())
        col1, col2 = st.columns(2)

        with col1:
            impact_text = f"{float(selected_impact):.2f}" if not pd.isna(selected_impact) else "N/A"
            st.metric("Seçili Şarkı Etki Skoru", impact_text)

        with col2:
            if pd.isna(selected_impact):
                st.metric("Genel Ortalama Etki", f"{global_impact:.2f}")
            else:
                st.metric(
                    "Genel Ortalama Etki",
                    f"{global_impact:.2f}",
                    delta=f"{(float(selected_impact) - global_impact):+.2f}",
                )
    
    st.markdown("### Müzik Dinlemeden Önce vs Sonra")
    st.pyplot(figures_selected[0])
    st.markdown("""
    **Bu grafikte ne görüyoruz?**
    
    Sol taraf: İnsanlar müzik dinlemeden önce nasıl hissediyordu?
    Sağ taraf: Müzik dinledikten sonra nasıl hissetti?
    
    Örneğin, stresli olan insanlar müzik dinledikten sonra daha rahat mı hissediyor?
    """)
    
    st.markdown("---")
    st.markdown(f"### En Etkili Şarkılar (Tüm Veri, İlk {top_song_count})")
    st.pyplot(figures_all[1])
    st.markdown("""
    **Bu grafikte ne görüyoruz?**
    
    "Bu müzik beni derinden etkiledi" sorusuna en yüksek puanları alan şarkılar.
    Skorlar 1-5 arasında, 5 = Çok derinden etkiledi.
    
    Bu şarkılar duygusal olarak en güçlü tepkileri yaratan parçalar!
    """)
    
    # Detaylı istatistikler
    with st.expander("Detaylı İstatistikler"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Dinlemeden Önce Dağılımı**")
            st.dataframe(selected_results['before_distribution'])
        
        with col2:
            st.markdown("**Dinledikten Sonra Dağılımı**")
            st.dataframe(selected_results['after_distribution'])
        
        st.markdown(f"**Şarkılara Göre Etki Skorları (Tüm Veri, İlk {top_song_count})**")
        st.dataframe(all_results['song_deep_impact'].head(top_song_count))


def render_human_ai_tab(df):
    """İnsan vs AI algısı analiz sekmesi"""
    st.subheader(" İnsan vs Yapay Zeka Algısı")
    
    st.info("""
    **İnsanlar ve yapay zeka müziği farklı mı yorumluyor?**
    
    Bu bölümde iki soruyu karşılaştırıyoruz:
    1. **Sizce bu müzik hangi duyguyu yansıtıyor?** (İnsan algısı)
    2. **Sizce yapay zeka bu müziği hangi duyguyla etiketlerdi?** (AI algısı tahmini)
    3. **Duygunun yoğunluğu nasıldı?** (Duygu yoğunluğu)
    
    İnsanlar, yapay zekanın müziği farklı yorumlayacağını düşünüyor mu? Ne kadar uyuşuyorlar?
    """)
    
    with st.spinner("Analiz yapılıyor..."):
        results = human_vs_ai_perception(df)
    
    st.success("Analiz tamamlandı!")
    
    # Ana metrik
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Uyuşma Oranı",
            f"%{results['agreement_rate']:.1f}",
            help="İnsan ve AI algısının ne kadar benzer olduğu"
        )
    
    with col2:
        st.metric(
            " Uyuşan Cevap",
            f"{int(results['agreement_rate'] * len(df) / 100)} kişi",
            help="Aynı duyguyu işaretleyen kişi sayısı"
        )
    
    with col3:
        st.metric(
            " Farklı Cevap",
            f"{results['disagreement_count']} kişi",
            help="Farklı duygu işaretleyen kişi sayısı"
        )
    
    # Görselleştirmeler
    figures = create_human_ai_plots(results)
    
    st.markdown("---")
    st.markdown("###  İnsan vs AI Duygu Etiketleri")
    st.pyplot(figures[0])
    st.markdown("""
    **Bu grafikte ne görüyoruz?**
    
    Sol: İnsanların müzikte gördüğü duygular
    Sağ: İnsanların AI'nın göreceğini düşündüğü duygular
    
    Farklılıklar varsa, insanlar AI'yı kendilerinden farklı algılıyor demektir!
    """)
    
    st.markdown("---")
    
    # Yoğunluk analizi varsa göster
    if len(figures) > 1 and 'intensity_distribution' in results:
        st.markdown("###  Duygu Yoğunluğu Analizi")
        st.pyplot(figures[1])
        st.markdown("""
        **Bu grafikte ne görüyoruz?**
        
        Sol: İnsanların hissettiği duygu yoğunluğu dağılımı (Çok hafif → Çok yoğun)
        Sağ: Her yoğunluk seviyesinde insan-AI uyuşma oranı
        
         Yeşil: Yüksek uyuşma (%50+) | Turuncu: Orta uyuşma | Kırmızı: Düşük uyuşma
        
        **Yorum**: Yoğun duygularda insanlar ve AI daha mı uyuşuyor, yoksa daha mı farklı mı düşünüyor?
        """)
        st.markdown("---")
    
    st.markdown("###  Genel Uyuşma Durumu")
    # Yoğunluk grafiği varsa index 2, yoksa 1
    pie_idx = 2 if len(figures) > 2 else 1
    st.pyplot(figures[pie_idx])
    st.markdown(f"""
    **Bu grafikte ne görüyoruz?**
    
    Ankete katılan {len(df)} kişiden **%{results['agreement_rate']:.1f}**'i 
    insan ve AI algısının aynı olduğunu düşünüyor.
    
    {'Çoğunluk uyuşuyor!' if results['agreement_rate'] > 50 else ' Çoğunluk farklı düşünüyor!'}
    """)
    
    st.markdown("---")
    st.markdown("### Şarkılara Göre Uyuşma Durumu")
    song_idx = pie_idx + 1
    st.pyplot(figures[song_idx])
    st.markdown("""
    ** Bu grafikte ne görüyoruz?**
    
    Her şarkı için insan-AI algısı ne kadar benzer?
    
     **Yeşil çubuklar**: Yüksek uyuşma (%50+) - İnsanlar AI'nın da benzer yorumlayacağını düşünüyor
     **Turuncu çubuklar**: Orta uyuşma (%30-50) - Kararsızlık var
     **Kırmızı çubuklar**: Düşük uyuşma (%30-) - İnsanlar AI'nın farklı yorumlayacağını düşünüyor
    
    Bazı şarkılar "evrensel" duygular mı uyandırıyor? Yoksa yoruma mı açık?
    """)
    
    # Farklılık nedenleri
    with st.expander("İnsanlar Neden Farklı Olduğunu Düşünüyor?"):
        reasons = results['disagreement_reasons']
        if len(reasons) > 0:
            st.markdown(f"**Toplam {len(reasons)} kişi farklılık nedeni belirtti:**")
            st.markdown("**İlk 15 yanıt:**")
            for i, reason in enumerate(reasons[:15], 1):
                st.markdown(f"**{i}.** {reason}")
            
            if len(reasons) > 15:
                st.info(f"... ve {len(reasons) - 15} yanıt daha")
        else:
            st.info("Farklılık nedeni belirtilmemiş")
    
    # Detaylı istatistikler
    with st.expander(" Detaylı İstatistikler"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**İnsan Etiketleri**")
            st.dataframe(results['human_labels'])
        
        with col2:
            st.markdown("**AI Etiketleri (Tahmin)**")
            st.dataframe(results['ai_labels'])
        
        if 'intensity_distribution' in results:
            st.markdown("**Duygu Yoğunluğu Dağılımı**")
            st.dataframe(results['intensity_distribution'])
        
        st.markdown("**Şarkılara Göre Uyuşma (Top 10)**")
        st.dataframe(results['song_agreement'].head(10))


def render_gemini_tab(df, song_col):
    """İnsan vs Gemini AI müzik analizi karşılaştırma sekmesi"""
    st.subheader("İnsan vs Gemini AI Müzik Analizi")
    
    st.info("""
    ** Gemini AI müziği nasıl yorumluyor?**
    
    Bu bölümde:
    1. **Bir şarkı seçin** ve dosyasını yükleyin
    2. **Gemini AI'a gönderin** - Gemini müziği dinleyip duygusal analiz yapar
    3. **Karşılaştırın** - Gemini'nin analizi ile anket katılımcılarının cevapları karşılaştırılır
    
    """)
    
    # API Key girişi
    st.markdown("###  Google Gemini API Anahtarı")
      
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Google AI Studio'dan aldığın ücretsiz API key'i buraya gir"
    )

    api_key = str(api_key or "").strip()
    is_valid_key, key_warning = _validate_gemini_api_key_input(api_key)
    if not is_valid_key:
        if not api_key:
            st.warning("Devam etmek için Gemini API key gereklidir")
        else:
            st.warning(key_warning)
        return
    
    # Şarkı seçimi
    st.markdown("### Şarkı Seçimi")
    
    songs = get_song_options(df, song_col)
    if not songs:
        st.error("Şarkı verisi bulunamadı")
        return
    
    selected_song = st.selectbox("Şarkı Seç", songs, key="gemini_song")
    
    # Şarkı dosyası yükleme
    st.markdown("### Şarkı Dosyası")
    audio_file = st.file_uploader(
        f"'{selected_song}' şarkısını yükleyin (.wav, .mp3)",
        type=['wav', 'mp3'],
        help="Seçtiğiniz şarkının audio dosyasını buraya yükleyin"
    )
    
    if not audio_file:
        st.info(f"Lütfen '{selected_song}' şarkısının dosyasını yükleyin")
        return
    
    # Gemini Analiz butonu
    if st.button("Gemini AI ile Analiz Et", type="primary"):
        with st.spinner("Gemini AI müziği analiz ediyor..."):
            try:
                # Audio dosyasını oku
                audio_bytes = audio_file.read()
                
                # Gemini ile analiz yap
                gemini_result = analyze_music_with_gemini(audio_bytes, audio_file.name, api_key)
                
                if gemini_result is None:
                    st.error(" Gemini analizi başarısız oldu!")
                    return
                
                # Sonuçları session state'e kaydet
                st.session_state['gemini_analysis'] = gemini_result
                st.session_state['selected_song'] = selected_song
                st.success("Gemini AI analizi tamamlandı!")

                # İnsan verisini bu şarkı için al (DynamoDB kaydı için)
                _song_key = normalize_song_name(selected_song)
                _song_resp = df[df[song_col].apply(normalize_song_name) == _song_key]
                _hv_scores = [survey_row_to_va(r) for _, r in _song_resp.iterrows()]
                _hv_valid = [(v, a) for v, a in _hv_scores if v is not None and a is not None]
                if _hv_valid:
                    _hv_mean = float(np.mean([v for v, _ in _hv_valid]))
                    _ha_mean = float(np.mean([a for _, a in _hv_valid]))
                    _dv = abs(gemini_result['valence'] - _hv_mean)
                    _da = abs(gemini_result['arousal'] - _ha_mean)
                    _saved = save_gemini_analysis(
                        song_name=selected_song,
                        gemini_valence=gemini_result['valence'],
                        gemini_arousal=gemini_result['arousal'],
                        gemini_emotion=gemini_result.get('emotion', ''),
                        human_valence_mean=_hv_mean,
                        human_arousal_mean=_ha_mean,
                        delta_v=_dv,
                        delta_a=_da,
                    )
                    if _saved:
                        st.caption("Gemini analizi DynamoDB Local'e kaydedildi.")
                
            except Exception as e:
                _show_gemini_error("Gemini analizi", e)
                return
    
    # Eğer analiz yapıldıysa sonuçları göster
    if 'gemini_analysis' in st.session_state and st.session_state['selected_song'] == selected_song:
        st.markdown("---")
        st.markdown("### Gemini AI Analiz Sonuçları")
        
        result = st.session_state['gemini_analysis']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Gemini AI'ın Değerlendirmesi")
            st.metric("Valence (Pozitiflik)", f"{result['valence']:.2f}")
            st.metric("Arousal (Enerji)", f"{result['arousal']:.2f}")
            st.markdown(f"**Tespit Edilen Duygu**: {result['emotion']}")
            
            if 'explanation' in result:
                with st.expander("Gemini 2.5 Flash'ın Açıklaması"):
                    st.write(result['explanation'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Gemini 2.5 Flash'ın Değerlendirmesi")
            st.metric("Valence (Pozitiflik)", f"{result['valence']:.2f}")
            st.metric("Arousal (Enerji)", f"{result['arousal']:.2f}")
            st.markdown(f"**Tespit Edilen Duygu**: {result['emotion']}")
            
            if 'explanation' in result:
                with st.expander("Gemini'nin Açıklaması"):
                    st.write(result['explanation'])
        
        with col2:
            st.markdown("#### İnsan Cevapları (Anket)")
            
            # Bu şarkı için insan cevaplarını al
            song_key = normalize_song_name(selected_song)
            song_responses = df[df[song_col].apply(normalize_song_name) == song_key]
            
            if len(song_responses) == 0:
                st.warning("Bu şarkı için anket cevabı bulunamadı!")
            else:
                # Survey utils ile VA hesapla
                human_va_scores = []
                for _, row in song_responses.iterrows():
                    v, a = survey_row_to_va(row)
                    if v is not None and a is not None:
                        human_va_scores.append((v, a))
                
                if len(human_va_scores) > 0:
                    human_v = np.mean([v for v, a in human_va_scores])
                    human_a = np.mean([a for v, a in human_va_scores])
                    human_v_std = np.std([v for v, a in human_va_scores])
                    human_a_std = np.std([a for v, a in human_va_scores])
                    
                    st.metric(
                        "Valence (Pozitiflik)",
                        f"{human_v:.2f}",
                        delta=f"±{human_v_std:.2f}"
                    )
                    st.metric(
                        "Arousal (Enerji)",
                        f"{human_a:.2f}",
                        delta=f"±{human_a_std:.2f}"
                    )
                    st.info(f"📊 {len(song_responses)} kişi bu şarkıyı dinledi")
                    
                    # Karşılaştırma
                    st.markdown("---")
                    st.markdown("### Karşılaştırma")
                    
                    delta_v = abs(result['valence'] - human_v)
                    delta_a = abs(result['arousal'] - human_a)
                    avg_delta = (delta_v + delta_a) / 2
                    
                    if avg_delta < 1.0:
                        st.success(f"**Mükemmel uyum!** Gemini insanlarla çok benzer yorumladı (Fark: {avg_delta:.2f})")
                    elif avg_delta < 2.0:
                        st.success(f" **İyi uyum!** Gemini insanlara yakın yorumladı (Fark: {avg_delta:.2f})")
                    elif avg_delta < 3.0:
                        st.warning(f" **Orta uyum.** Gemini ile insanlar arasında farklar var (Fark: {avg_delta:.2f})")
                    else:
                        st.error(f" **Zayıf uyum.** Gemini insanlardan oldukça farklı yorumladı (Fark: {avg_delta:.2f})")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Δ Valence", f"{delta_v:.2f}")
                    with col2:
                        st.metric("Δ Arousal", f"{delta_a:.2f}")
                    
                    # Görselleştirme
                    st.markdown("---")
                    st.markdown("### Görselleştirme")
                    
                    fig, ax = plt.subplots(figsize=(8, 8))
                    
                    # Gemini point
                    ax.scatter(
                        result['valence'],
                        result['arousal'],
                        s=300,
                        c='green',
                        marker='*',
                        label='Gemini AI',
                        alpha=0.8,
                        edgecolors='black',
                        linewidths=2
                    )
                    
                    # Human point
                    ax.scatter(
                        human_v,
                        human_a,
                        s=200,
                        c='blue',
                        marker='o',
                        label='İnsan',
                        alpha=0.7
                    )
                    
                    # Error bars for human
                    ax.errorbar(
                        human_v,
                        human_a,
                        xerr=human_v_std,
                        yerr=human_a_std,
                        fmt='none',
                        c='blue',
                        alpha=0.3
                    )
                    
                    # Connection line
                    ax.plot(
                        [human_v, result['valence']],
                        [human_a, result['arousal']],
                        'k--',
                        alpha=0.3,
                        linewidth=2
                    )
                    
                    ax.set_xlim(1, 9)
                    ax.set_ylim(1, 9)
                    ax.set_xlabel('Valence (Negatif ← → Pozitif)', fontsize=12)
                    ax.set_ylabel('Arousal (Sakin ← → Enerjik)', fontsize=12)
                    ax.set_title(f'İnsan vs Gemini AI: {selected_song}', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=11)
                    ax.axhline(5, color='gray', linewidth=0.5, alpha=0.5)
                    ax.axvline(5, color='gray', linewidth=0.5, alpha=0.5)
                    
                    # Bölge etiketleri
                    ax.text(7.5, 7.5, 'Mutlu\nHeyecanlı', ha='center', fontsize=9, alpha=0.5)
                    ax.text(2.5, 7.5, 'Gergin\nKızgın', ha='center', fontsize=9, alpha=0.5)
                    ax.text(7.5, 2.5, 'Huzurlu\nRahat', ha='center', fontsize=9, alpha=0.5)
                    ax.text(2.5, 2.5, 'Üzgün\nMelankolik', ha='center', fontsize=9, alpha=0.5)
                    
                    st.pyplot(fig)
                    
                    st.markdown(f"""
                    ** Bu grafikte ne görüyoruz?**
                    
                     Yeşil yıldız: Gemini AI'ın analizi  
                     Mavi nokta: İnsanların ortalama cevabı  
                    
                    Yıldız ve nokta ne kadar yakınsa, Gemini insanlarla o kadar benzer düşünüyor!
                    
                    **Sonuç**: Gemini bu şarkıyı {'insanlara çok benzer' if avg_delta < 2 else 'insanlara yakın' if avg_delta < 3 else 'insanlardan farklı'} yorumladı.
                    """)
                    
                    # Gemini ile detaylı analiz
                    st.markdown("---")
                    st.markdown("###  Gemini'nin Detaylı Duygusal Yorumu")
                    
                    if st.button(" Gemini'den Detaylı Yorum Al", type="secondary"):
                        with st.spinner("Gemini detaylı analiz yapıyor..."):
                            detailed_analysis = generate_detailed_emotion_analysis(
                                api_key=api_key,
                                song_name=selected_song,
                                gemini_analysis=result,
                                human_valence=human_v,
                                human_arousal=human_a
                            )
                            
                            if detailed_analysis:
                                st.markdown("### 🎭 Gemini'nin Duygusal Analiz Raporu")
                                st.markdown(detailed_analysis)
                            else:
                                st.error("Detaylı analiz oluşturulamadı!")
                else:
                    st.warning("Bu şarkı için geçerli anket cevabı bulunamadı!")


def analyze_music_with_gemini(audio_bytes, filename, api_key):
    """
    Gemini AI ile müzik analizi yapar
    
    Args:
        audio_bytes: Audio dosyasının byte verisi
        filename: Dosya adı
        api_key: Google Gemini API key
        
    Returns:
        dict: {'valence': float, 'arousal': float, 'emotion': str, 'explanation': str}
    """
    try:
        import google.genai as modern_genai
        import tempfile
        import time

        client = modern_genai.Client(api_key=api_key)

        suffix = os.path.splitext(filename or "")[1] or ".wav"
        tmp_path = ""
        uploaded_file = None

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            st.info("Dosya Gemini sunucularına yükleniyor...")
            uploaded_file = client.files.upload(file=tmp_path)

            for _ in range(120):
                state_name = _gemini_file_state_name(uploaded_file)
                if state_name in {"READY", "ACTIVE", "PROCESSED", "SUCCEEDED", "SUCCESS"}:
                    break

                if state_name == "FAILED":
                    raise ValueError("Audio dosyası işlenemedi")

                st.info("Dosya işleniyor, lütfen bekleyin...")
                time.sleep(2)
                uploaded_file = client.files.get(name=uploaded_file.name)
            else:
                raise TimeoutError("Audio dosyası işleme süresi aşıldı")

            st.info("Dosya hazır, analiz başlıyor...")

            # Prompt
            prompt = """
            Bu müzik parçasını dinle ve duygusal bir analiz yap. Aşağıdaki kriterlere göre değerlendir:
            
            1. **Valence (Pozitiflik)**: 1-9 arası bir değer ver
               - 1-3: Negatif (üzgün, melankolik, kızgın, kasvetli)
               - 4-6: Nötr (karışık, belirsiz, nötr)
               - 7-9: Pozitif (mutlu, neşeli, iyimser, coşkulu)
            
            2. **Arousal (Enerji Seviyesi)**: 1-9 arası bir değer ver
               - 1-3: Düşük enerji (sakin, huzurlu, durgun, yavaş)
               - 4-6: Orta enerji (dengeli, ılımlı)
               - 7-9: Yüksek enerji (heyecanlı, enerjik, gergin, dinamik)
            
            3. **Tespit Edilen Ana Duygu**: Örn: Mutlu, Üzgün, Sakin, Gergin, Nostaljik, Heyecanlı, Romantik, vb.
            
            4. **Açıklama**: Müziğin duygusal karakterini detaylı şekilde analiz et:
               - Melodinin yapısı ve hissettirdikleri
               - Enstrümanların kullanımı ve etkileri
               - Ritim ve temponun duygusal katkısı
               - Varsa vokal/şarkı sözlerinin duygusal tonu
               - Genel atmosfer ve senin kişisel yorumun
               - Bu müziği dinlerken ne hissedebileceğini anlat (3-5 cümle)
            
            Cevabını SADECE JSON formatında ver, başka hiçbir metin ekleme:
            {
                "valence": 5.5,
                "arousal": 6.2,
                "emotion": "Nostaljik",
                "explanation": "Bu müzik parçası yavaş bir tempo ve yumuşak piyano melodisiyle başlıyor. Kemanların devreye girmesiyle melankolik bir hava yaratılıyor. Ritim oldukça sakin ve dinleyiciyi düşünceye sevk eden bir yapıda. Genel olarak geçmişe özlem duyulan anları hatırlatan, içsel bir yolculuğa çıkaran bir atmosfere sahip. Dinlerken huzur bulabilir ama aynı zamanda hafif bir üzüntü de hissedilebilir."
            }
            """

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, uploaded_file],
            )

            # JSON yanıtını parse et
            result_text = _extract_gemini_response_text(response).strip()
            if not result_text:
                raise RuntimeError("Gemini boş yanıt döndürdü")

            # JSON dışındaki markdown formatlarını temizle
            if result_text.startswith('```json'):
                result_text = result_text[7:]
            if result_text.startswith('```'):
                result_text = result_text[3:]
            if result_text.endswith('```'):
                result_text = result_text[:-3]
            result_text = result_text.strip()
            
            result = json.loads(result_text)
            
            # Validate result
            if 'valence' not in result or 'arousal' not in result:
                st.error("Gemini yanıtı beklenen formatda değil!")
                return None
            
            # Ensure values are in range 1-9
            result['valence'] = max(1, min(9, float(result['valence'])))
            result['arousal'] = max(1, min(9, float(result['arousal'])))

            return result

        finally:
            if uploaded_file is not None and getattr(uploaded_file, "name", None):
                try:
                    client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass

            # Geçici dosyayı sil
            try:
                if tmp_path:
                    os.unlink(tmp_path)
            except Exception:
                pass

    except Exception as e:
        _show_gemini_error("Gemini API çağrısı", e)
        return None


def generate_detailed_emotion_analysis(api_key, song_name, gemini_analysis, human_valence, human_arousal):
    """
    Gemini AI ile detaylı duygusal karşılaştırma analizi
    
    Args:
        api_key: Google Gemini API key
        song_name: Şarkı adı
        gemini_analysis: Gemini'nin müzik analizi
        human_valence: İnsan valence ortalaması
        human_arousal: İnsan arousal ortalaması
        
    Returns:
        str: Detaylı analiz metni (markdown formatında)
    """
    try:
        import google.genai as modern_genai

        client = modern_genai.Client(api_key=api_key)
        
        # Fark hesapla
        delta_v = abs(gemini_analysis['valence'] - human_valence)
        delta_a = abs(gemini_analysis['arousal'] - human_arousal)
        
        prompt = f"""
        Aşağıdaki verilere dayanarak, "{song_name}" şarkısının duygusal analizini karşılaştıran detaylı bir rapor oluştur.
        
        **Yapay Zeka (Gemini) Analizi:**
        - Tespit Edilen Duygu: {gemini_analysis['emotion']}
        - Valence (Pozitiflik): {gemini_analysis['valence']:.2f}/9
        - Arousal (Enerji Seviyesi): {gemini_analysis['arousal']:.2f}/9
        - Açıklama: {gemini_analysis['explanation']}
        
        **İnsan Dinleyiciler Analizi (Anket Verileri):**
        - Ortalama Valence: {human_valence:.2f}/9
        - Ortalama Arousal: {human_arousal:.2f}/9
        
        **Farklar:**
        - Valence Farkı: {delta_v:.2f}
        - Arousal Farkı: {delta_a:.2f}
        
        Lütfen şu başlıklar altında detaylı bir analiz yaz:
        
        1. **Genel Değerlendirme**: AI ve insan algısı ne kadar uyumlu? Farklar anlamlı mı?
        
        2. **Valence Karşılaştırması**: Pozitiflik/Negatiflik açısından farklılıklar neler? 
           İnsanlar mı daha pozitif/negatif algılıyor, yoksa AI mı?
        
        3. **Arousal Karşılaştırması**: Enerji seviyesi algısındaki farklar neler?
           AI ile insanlar arasında tempo/dinamik algısında tutarsızlık var mı?
        
        4. **Duygu Tanımlaması**: AI'ın tespit ettiği duygu insanların genel hissiyle uyumlu mu?
           Farklılıkların nedeni ne olabilir?
        
        5. **Sonuç ve Yorum**: Bu müzik parçası için hangi algı daha doğru olabilir? 
           Subjektif mi objektif mi? AI'ın gözden kaçırdığı nüanslar var mı?
        
        Cevabını Markdown formatında, başlıklar ve alt başlıklarla düzenli şekilde yaz.
        Akademik ama anlaşılır bir dil kullan.
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        response_text = _extract_gemini_response_text(response)
        if not response_text:
            raise RuntimeError("Gemini boş yanıt döndürdü")

        return response_text
        
    except Exception as e:
        _show_gemini_error("Detaylı Gemini analizi", e)
        return None


def render_triple_comparison_tab(df, selected_model_runtime, selected_model_name, song_col):
    """İnsan vs CNN vs Gemini üçlü karşılaştırma sekmesi"""
    st.subheader("İnsan vs Seçili Model vs Gemini 2.5 Flash")
    
    st.info("""
    **Üç farklı yaklaşımı karşılaştırıyoruz:**
    
    1. **İnsan Algısı**: Anket katılımcılarının subjektif duygusal tepkileri
    2. **Seçili Model**: Kenar çubuğundan seçtiğiniz modelin tahmini (tek model veya ensemble)
    3. **Gemini 2.5 Flash**: Google'ın multimodal AI'sının müzik analizi
    
    **Hangi yaklaşım daha doğru?** Üçünü de karşılaştırarak görelim!
    """)
    
    # Gemini API Key
    st.markdown("###Gemini API Key")
    api_key = st.text_input(
        "API Key",
        type="password",
        help="Google AI Studio'dan aldığınız API key",
        key="triple_api_key"
    )

    api_key = str(api_key or "").strip()
    is_valid_key, key_warning = _validate_gemini_api_key_input(api_key)
    if not is_valid_key:
        if not api_key:
            st.warning("Lütfen Gemini API key girin")
        else:
            st.warning(key_warning)
        return
    
    # Şarkı seçimi
    st.markdown("###Şarkı Seçimi")
    
    songs = get_song_options(df, song_col)
    if not songs:
        st.error("Şarkı verisi bulunamadı")
        return
    
    selected_song = st.selectbox("Şarkı Seç", songs, key="triple_song")
    
    # Audio dosyası
    st.markdown("### Şarkı Dosyası")
    audio_file = st.file_uploader(
        f"'{selected_song}' şarkısını yükleyin (.wav, .mp3)",
        type=['wav', 'mp3'],
        key="triple_audio"
    )
    
    if not audio_file:
        st.info(f"Lütfen '{selected_song}' şarkısının dosyasını yükleyin")
        return
    
    # Karşılaştır butonu
    if st.button("Üçlü Karşılaştırma Yap", type="primary"):
        with st.spinner("Analiz ediliyor..."):
            try:
                # 1. İnsan verilerini al
                song_key = normalize_song_name(selected_song)
                song_responses = df[df[song_col].apply(normalize_song_name) == song_key]
                
                if len(song_responses) == 0:
                    st.error("Bu şarkı için anket verisi bulunamadı!")
                    return
                
                human_va_scores = []
                for _, row in song_responses.iterrows():
                    v, a = survey_row_to_va(row)
                    if v is not None and a is not None:
                        human_va_scores.append((v, a))
                
                if len(human_va_scores) == 0:
                    st.error("Geçerli insan verisi bulunamadı!")
                    return
                
                human_v = np.mean([v for v, a in human_va_scores])
                human_a = np.mean([a for v, a in human_va_scores])
                human_v_std = np.std([v for v, a in human_va_scores])
                human_a_std = np.std([a for v, a in human_va_scores])
                
                # 2. Seçili modelden tahmin al
                audio_bytes = audio_file.getvalue()
                file_ext = audio_file.name.split('.')[-1].lower()
                wav = load_audio(audio_bytes, file_ext)
                selected_v, selected_a = predict_with_selected_model(selected_model_runtime, wav)
                
                # 3. Gemini'den analiz al
                gemini_result = analyze_music_with_gemini(audio_bytes, audio_file.name, api_key)
                
                if gemini_result is None:
                    st.error("Gemini analizi başarısız oldu!")
                    return
                
                gemini_v = gemini_result['valence']
                gemini_a = gemini_result['arousal']
                
                # Sonuçları göster
                st.success("Analiz tamamlandı!")
                
                # Metrikler
                st.markdown("---")
                st.markdown("### Karşılaştırmalı Sonuçlar")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### İnsan Algısı")
                    st.metric("Valence", f"{human_v:.2f}", delta=f"±{human_v_std:.2f}")
                    st.metric("Arousal", f"{human_a:.2f}", delta=f"±{human_a_std:.2f}")
                    st.info(f"{len(song_responses)} katılımcı")
                
                with col2:
                    st.markdown(f"#### {selected_model_name}")
                    st.metric("Valence", f"{selected_v:.2f}")
                    st.metric("Arousal", f"{selected_a:.2f}")
                    delta_model_v = abs(selected_v - human_v)
                    delta_model_a = abs(selected_a - human_a)
                    st.info(f"Δ İnsan: {((delta_model_v + delta_model_a)/2):.2f}")
                
                with col3:
                    st.markdown("#### Gemini 2.5")
                    st.metric("Valence", f"{gemini_v:.2f}")
                    st.metric("Arousal", f"{gemini_a:.2f}")
                    delta_gem_v = abs(gemini_v - human_v)
                    delta_gem_a = abs(gemini_a - human_a)
                    st.info(f"Δ İnsan: {((delta_gem_v + delta_gem_a)/2):.2f}")
                
                # Fark analizi
                st.markdown("---")
                st.markdown("### Fark Analizi")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart
                    import matplotlib.pyplot as plt
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    
                    categories = ['Valence\n(Pozitiflik)', 'Arousal\n(Enerji)']
                    x = np.arange(len(categories))
                    width = 0.25
                    
                    ax.bar(x - width, [human_v, human_a], width, label='İnsan', color='blue', alpha=0.7)
                    ax.bar(x, [selected_v, selected_a], width, label='Seçili Model', color='red', alpha=0.7)
                    ax.bar(x + width, [gemini_v, gemini_a], width, label='Gemini', color='green', alpha=0.7)
                    
                    ax.set_ylabel('Skor (1-9)', fontsize=11)
                    ax.set_title('Üç Yaklaşımın Karşılaştırması', fontsize=13, fontweight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels(categories)
                    ax.legend()
                    ax.set_ylim(1, 9)
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    st.pyplot(fig)
                
                with col2:
                    # VA Space
                    fig, ax = plt.subplots(figsize=(8, 8))
                    
                    # Human
                    ax.scatter(human_v, human_a, s=250, c='blue', marker='o', 
                              label='İnsan', alpha=0.7, edgecolors='black', linewidths=2)
                    ax.errorbar(human_v, human_a, xerr=human_v_std, yerr=human_a_std,
                               fmt='none', c='blue', alpha=0.3)
                    
                    # Seçili model
                    ax.scatter(selected_v, selected_a, s=250, c='red', marker='s',
                              label='Seçili Model', alpha=0.7, edgecolors='black', linewidths=2)
                    
                    # Gemini
                    ax.scatter(gemini_v, gemini_a, s=300, c='green', marker='*',
                              label='Gemini', alpha=0.8, edgecolors='black', linewidths=2)
                    
                    ax.set_xlim(1, 9)
                    ax.set_ylim(1, 9)
                    ax.set_xlabel('Valence (Negatif ← → Pozitif)', fontsize=12)
                    ax.set_ylabel('Arousal (Sakin ← → Enerjik)', fontsize=12)
                    ax.set_title(f'Duygu Uzayı: {selected_song}', fontsize=13, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=11)
                    ax.axhline(5, color='gray', linewidth=0.5, alpha=0.5)
                    ax.axvline(5, color='gray', linewidth=0.5, alpha=0.5)
                    
                    # Bölge etiketleri
                    ax.text(7.5, 7.5, 'Mutlu\nHeyecanlı', ha='center', fontsize=9, alpha=0.5)
                    ax.text(2.5, 7.5, 'Gergin\nKızgın', ha='center', fontsize=9, alpha=0.5)
                    ax.text(7.5, 2.5, 'Huzurlu\nRahat', ha='center', fontsize=9, alpha=0.5)
                    ax.text(2.5, 2.5, 'Üzgün\nMelankolik', ha='center', fontsize=9, alpha=0.5)
                    
                    st.pyplot(fig)
                
                # Yorumlama
                st.markdown("---")
                st.markdown("### Analiz ve Yorum")
                
                # Hangi yaklaşım insana daha yakın?
                model_total_delta = (delta_model_v + delta_model_a) / 2
                gemini_total_delta = (delta_gem_v + delta_gem_a) / 2
                
                if model_total_delta < gemini_total_delta:
                    winner = selected_model_name
                    winner_delta = model_total_delta
                    winner_icon = ""
                else:
                    winner = "Gemini"
                    winner_delta = gemini_total_delta
                    winner_icon = ""
                
                st.success(f"{winner_icon} **{winner}** insanlara daha yakın! (Ortalama fark: {winner_delta:.2f})")

                # DynamoDB Local'e her iki sonucu da kaydet
                _triple_model_label = selected_model_name if selected_model_runtime.get("type") != "ensemble" else "ensemble_best"
                save_song_analysis(
                    song_name=selected_song,
                    model_name=_triple_model_label,
                    model_valence=selected_v,
                    model_arousal=selected_a,
                    human_valence_mean=human_v,
                    human_arousal_mean=human_a,
                    human_valence_std=human_v_std,
                    human_arousal_std=human_a_std,
                    delta_v=delta_model_v,
                    delta_a=delta_model_a,
                    n_responses=len(song_responses),
                )
                save_gemini_analysis(
                    song_name=selected_song,
                    gemini_valence=gemini_v,
                    gemini_arousal=gemini_a,
                    gemini_emotion=gemini_result.get('emotion', ''),
                    human_valence_mean=human_v,
                    human_arousal_mean=human_a,
                    delta_v=delta_gem_v,
                    delta_a=delta_gem_a,
                )
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Seçili Model → İnsan Farkı", f"{model_total_delta:.2f}")
                
                with col2:
                    st.metric("Gemini → İnsan Farkı", f"{gemini_total_delta:.2f}")
                
                with col3:
                    st.metric(
                        "Seçili Model ↔ Gemini Farkı",
                        f"{(abs(selected_v - gemini_v) + abs(selected_a - gemini_a)) / 2:.2f}",
                    )
                
                # Gemini'nin açıklaması
                st.markdown("---")
                st.markdown("### Gemini'nin Müzik Yorumu")
                
                st.markdown(f"**Tespit Edilen Duygu:** {gemini_result['emotion']}")
                
                with st.expander("Detaylı Açıklama"):
                    st.write(gemini_result['explanation'])
                
                # İstatistiksel özet
                with st.expander("Detaylı İstatistikler"):
                    summary_df = pd.DataFrame({
                        'Yaklaşım': ['İnsan (Ort.)', selected_model_name, 'Gemini'],
                        'Valence': [f"{human_v:.2f}", f"{selected_v:.2f}", f"{gemini_v:.2f}"],
                        'Arousal': [f"{human_a:.2f}", f"{selected_a:.2f}", f"{gemini_a:.2f}"],
                        'V Std': [f"{human_v_std:.2f}", '-', '-'],
                        'A Std': [f"{human_a_std:.2f}", '-', '-']
                    })
                    st.dataframe(summary_df, use_container_width=True)
                    
            except Exception as e:
                _show_gemini_error("Üçlü karşılaştırma", e)


def render_abstract_emotion_visualization_tab(df, song_col):
    """Deterministic Art DNA Engine ile soyut görsel üretimi."""
    st.markdown(
        """
        <style>
        .abstract-hero {
            background: linear-gradient(120deg, #1f3b4d 0%, #235789 55%, #f1a208 120%);
            border-radius: 18px;
            padding: 1.1rem 1.3rem;
            color: #ffffff;
            margin-bottom: 1rem;
            box-shadow: 0 10px 26px rgba(18, 55, 80, 0.24);
        }
        .abstract-card {
            border: 1px solid rgba(35, 87, 137, 0.26);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            background: linear-gradient(180deg, #f8fbff 0%, #eef5fb 100%);
            margin-bottom: 0.75rem;
        }
        .palette-chip {
            display: inline-block;
            margin: 0.2rem 0.3rem 0.2rem 0;
            padding: 0.28rem 0.58rem;
            border-radius: 999px;
            background: #e9f2ff;
            border: 1px solid #c5daf8;
            font-size: 0.84rem;
            color: #1b3a5b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="abstract-hero">
            <h2 style="margin:0; font-size:1.35rem;">🎨 Şarkının Art DNA Görselleştirmesi</h2>
            <p style="margin:0.45rem 0 0 0; font-size:0.96rem; line-height:1.35;">
                Bu modül tamamen Python tabanlı, açıklanabilir ve deterministik Art DNA Engine kullanır.
                Gemini veya herhangi bir Google servisi bu sekmede kullanılmaz.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    songs = get_song_options(df, song_col)
    if not songs:
        st.error("CSV içinde analiz edilebilir şarkı verisi bulunamadı")
        return

    if "hf_api_key" not in st.session_state:
        st.session_state["hf_api_key"] = _resolve_hf_api_key()

    image_key = "abstract_viz_image_bytes"
    prompt_key = "abstract_viz_prompt"
    image_model_key = "abstract_viz_image_model"
    song_key = "abstract_viz_song"
    signature_key = "abstract_viz_signature"
    analysis_key = "abstract_viz_analysis"
    in_flight_key = "abstract_viz_request_in_flight"

    if in_flight_key not in st.session_state:
        st.session_state[in_flight_key] = False

    left_col, right_col = st.columns([1, 1.4], gap="large")

    with left_col:
        st.markdown('<div class="abstract-card">', unsafe_allow_html=True)
        selected_song = st.selectbox(
            "CSV içinden analiz edilecek şarkı",
            songs,
            key="abstract_viz_song_select",
        )
        aspect_ratio = st.selectbox(
            "Görsel Oranı",
            ["1:1", "16:9", "9:16"],
            index=0,
            key="abstract_viz_aspect_ratio",
        )
        st.caption(
            "Analiz otomatik olarak seçilen şarkının anket verileri ile yapılır. "
            "Görsel üretimi Hugging Face FLUX ile gerçekleştirilir."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    selected_song_df = filter_df_by_song(df, song_col, selected_song)
    if selected_song_df.empty:
        st.warning("Seçilen şarkı için anket verisi bulunamadı")
        return

    try:
        analysis_payload = analyze_art_dna(selected_song, selected_song_df)
    except Exception as analysis_exc:
        st.error(f"Art DNA analizi oluşturulamadı: {analysis_exc}")
        return

    profile_signature = str(analysis_payload.get("profile_signature", ""))
    generated_prompt = str(analysis_payload.get("generated_prompt", "")).strip()
    prompt_text = f"{generated_prompt}\n\nPreferred aspect ratio: {aspect_ratio}.".strip()

    if (
        st.session_state.get(song_key) != selected_song
        or st.session_state.get(signature_key) != profile_signature
        or st.session_state.get(prompt_key) != prompt_text
    ):
        st.session_state[song_key] = selected_song
        st.session_state[signature_key] = profile_signature
        st.session_state[analysis_key] = analysis_payload
        st.session_state[prompt_key] = prompt_text
        st.session_state.pop(image_key, None)
        st.session_state.pop(image_model_key, None)

    dominant_emotions = analysis_payload.get("dominant_emotions", [])
    dominant_labels = [item.get("emotion", "") for item in dominant_emotions]

    with right_col:
        st.markdown('<div class="abstract-card">', unsafe_allow_html=True)
        st.markdown(f"**Seçilen Şarkı:** {selected_song}")
        st.markdown(f"**Anket Katılımcı Sayısı:** {analysis_payload.get('response_count', 0)}")
        st.markdown(
            "**Baskın Duygular:** "
            + (", ".join(label for label in dominant_labels if label) if dominant_labels else "Yok")
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Art DNA Analizi")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Valence", f"{analysis_payload.get('valence_mean', 0.0):.3f}")
        st.caption(str(analysis_payload.get("valence_category", "")))
    with metric_col2:
        st.metric("Arousal", f"{analysis_payload.get('arousal_mean', 0.0):.3f}")
        st.caption(str(analysis_payload.get("arousal_category", "")))
    with metric_col3:
        st.metric("Atmosfer", str(analysis_payload.get("atmosphere", "-")))
        st.caption(str(analysis_payload.get("emotion_intensity", "")))

    emotion_profile = analysis_payload.get("emotion_profile", {})
    full_distribution = analysis_payload.get("emotion_distribution", [])
    art_style = analysis_payload.get("art_style", [])
    color_dna = analysis_payload.get("color_dna", [])
    symbolism = analysis_payload.get("symbolism", [])

    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.markdown("#### Emotion Profile")
        profile_rows = [
            {"Metrik": "Valence", "Değer": f"{float(emotion_profile.get('valence', 0.0)):.3f}"},
            {"Metrik": "Valence Level", "Değer": str(emotion_profile.get("valence_level", "-"))},
            {"Metrik": "Arousal", "Değer": f"{float(emotion_profile.get('arousal', 0.0)):.3f}"},
            {"Metrik": "Arousal Level", "Değer": str(emotion_profile.get("arousal_level", "-"))},
            {"Metrik": "Entropy", "Değer": f"{float(emotion_profile.get('entropy', 0.0)):.6f}"},
            {"Metrik": "Diversity", "Değer": f"{float(emotion_profile.get('diversity', 0.0)):.6f}"},
            {
                "Metrik": "Emotional Intensity",
                "Değer": f"{float(emotion_profile.get('emotional_intensity', 0.0)):.6f}",
            },
            {"Metrik": "Intensity Label", "Değer": str(emotion_profile.get("intensity_label", "-"))},
            {"Metrik": "Positive Ratio", "Değer": f"{float(emotion_profile.get('positive_ratio', 0.0)):.6f}"},
            {"Metrik": "Negative Ratio", "Değer": f"{float(emotion_profile.get('negative_ratio', 0.0)):.6f}"},
            {
                "Metrik": "Emotional Balance",
                "Değer": f"{float(emotion_profile.get('emotional_balance', 0.0)):.6f}",
            },
        ]
        st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)

    with detail_col2:
        st.markdown("#### Duygu Dağılımı")
        distribution_rows = []
        for item in full_distribution:
            distribution_rows.append(
                {
                    "Duygu": item.get("emotion", ""),
                    "Skor": f"{float(item.get('score', 0.0)):.4f}",
                    "Adet": item.get("count", 0),
                    "Oran": f"{float(item.get('ratio', 0.0)):.4f}",
                    "Yüzde": f"{float(item.get('percentage', 0.0)):.2f}%",
                }
            )
        if distribution_rows:
            st.dataframe(pd.DataFrame(distribution_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("Duygu dağılımı bulunamadı")

    style_col1, style_col2 = st.columns(2)
    with style_col1:
        st.markdown("#### Art Style")
        if art_style:
            st.dataframe(pd.DataFrame(art_style), use_container_width=True, hide_index=True)
        else:
            st.caption("Sanat stili üretilemedi")

        st.markdown("#### Color DNA")
        if color_dna:
            st.dataframe(pd.DataFrame(color_dna), use_container_width=True, hide_index=True)
        else:
            st.caption("Color DNA üretilemedi")

        st.markdown(f"**Art DNA Seed:** `{analysis_payload.get('seed', '-')}`")
        st.markdown(f"**Profile Signature:** `{analysis_payload.get('profile_signature', '-')}`")

    with style_col2:
        st.markdown("#### Structural DNA")
        structural_rows = [
            {"Katman": "Composition DNA", "Değer": ", ".join(analysis_payload.get("composition_dna", []))},
            {"Katman": "Motion DNA", "Değer": ", ".join(analysis_payload.get("motion_dna", []))},
            {"Katman": "Texture DNA", "Değer": ", ".join(analysis_payload.get("texture_dna", []))},
            {"Katman": "Lighting DNA", "Değer": ", ".join(analysis_payload.get("lighting_dna", []))},
            {"Katman": "Spatial Layout", "Değer": ", ".join(analysis_payload.get("spatial_layout", []))},
        ]
        st.dataframe(pd.DataFrame(structural_rows), use_container_width=True, hide_index=True)

        st.markdown("#### Symbolism")
        if symbolism:
            st.dataframe(pd.DataFrame(symbolism), use_container_width=True, hide_index=True)
        else:
            st.caption("Sembolizm üretilemedi")

    st.info(str(analysis_payload.get("academic_rationale", "")))

    with st.expander("Oluşturulan Prompt"):
        st.code(prompt_text, language="text")

    generate_clicked = st.button(
        "Soyut Görsel Oluştur",
        type="primary",
        use_container_width=True,
        key="abstract_viz_generate_btn",
        disabled=bool(st.session_state.get(in_flight_key, False)),
    )

    if generate_clicked:
        hf_api_key = _resolve_hf_api_key(st.session_state.get("hf_api_key", ""))
        is_valid_key, key_warning = _validate_hf_api_key_input(hf_api_key)

        if not is_valid_key:
            if not hf_api_key:
                st.warning(
                    "Gorsel uretimi icin Hugging Face API key gereklidir. "
                    "Lutfen sidebar alanini doldurun veya .env icine HF_API_KEY ekleyin."
                )
            else:
                st.warning(key_warning)
        else:
            st.session_state[in_flight_key] = True
            try:
                with st.spinner("🎨 Soyut sanat oluşturuluyor..."):
                    prompt_hash = compute_flux_prompt_hash(prompt_text)
                    cached_bytes = load_flux_cached_image(prompt_text)

                    if cached_bytes is not None:
                        image_bytes = cached_bytes
                        used_model = f"{DEFAULT_FLUX_MODEL_ID} (cache)"
                    else:
                        image_bytes = generate_flux_image(api_key=hf_api_key, prompt=prompt_text)
                        used_model = DEFAULT_FLUX_MODEL_ID

                    st.session_state[image_key] = image_bytes
                    st.session_state[image_model_key] = used_model
                    st.success("✅ Görsel başarıyla oluşturuldu")
                    st.caption(f"Prompt SHA256: {prompt_hash}")
                    st.caption(f"Model: {used_model}")
            except Exception as generation_exc:
                safe_message = _sanitize_sensitive_error_text(str(generation_exc))

                if isinstance(generation_exc, FluxTimeoutError):
                    st.error("Hugging Face istegi zaman asimina ugradi. Lutfen tekrar deneyin.")
                elif isinstance(generation_exc, FluxConnectionError):
                    st.error(
                        "Hugging Face API baglanti/DNS hatasi olustu. "
                        "Internet, DNS veya ag erisimini kontrol edin."
                    )
                elif isinstance(generation_exc, FluxServerError):
                    st.error(f"Hugging Face sunucu hatasi: {safe_message}")
                elif isinstance(generation_exc, FluxRateLimitError):
                    st.error(f"Hugging Face kota limiti hatasi: {safe_message}")
                elif isinstance(generation_exc, FluxImageGenerationError):
                    st.error(f"FLUX gorsel uretimi basarisiz oldu: {safe_message}")
                else:
                    st.error(f"Gorsel uretimi basarisiz oldu: {safe_message}")
            finally:
                st.session_state[in_flight_key] = False

    if st.session_state.get(image_key):
        st.markdown("### Üretilen Soyut Görsel")
        st.image(st.session_state[image_key], use_container_width=True)

        if st.session_state.get(image_model_key):
            st.caption(f"Kullanılan model: {st.session_state[image_model_key]}")


def render_history_tab():
    """Geçmiş analizler sekmesi – DynamoDB Local'den okur."""
    st.subheader("Geçmiş Analizler (DynamoDB Local)")

    if not is_local_db_available():
        st.warning(
            "DynamoDB Local çalışmıyor. Geçmiş analizler görüntülenemiyor.\n\n"
            "Başlatmak için terminalde çalıştırın:\n```\ndocker-compose up -d\n```"
        )
        return

    # ---- Model analizleri ------------------------------------------------
    st.markdown("### Model Karşılaştırma Geçmişi")

    song_items = get_all_song_analyses()
    if not song_items:
        st.info("Henüz model analizi kaydedilmemiş.")
    else:
        song_df = pd.DataFrame(song_items)
        # Kolon sırası
        display_cols = [
            c for c in [
                "timestamp", "song_name", "model_name",
                "model_valence", "model_arousal",
                "human_valence_mean", "human_arousal_mean",
                "delta_v", "delta_a", "n_responses",
            ] if c in song_df.columns
        ]
        song_df = song_df[display_cols].sort_values("timestamp", ascending=False).reset_index(drop=True)

        # Sayısal kolonları 2 ondalıkla göster
        for col in ["model_valence", "model_arousal", "human_valence_mean", "human_arousal_mean", "delta_v", "delta_a"]:
            if col in song_df.columns:
                song_df[col] = song_df[col].apply(lambda x: round(float(x), 2) if pd.notna(x) else x)

        st.dataframe(song_df, use_container_width=True, hide_index=True)
        st.caption(f"Toplam {len(song_items)} kayıt")

        # Silme seçeneği
        with st.expander("Kayıt Sil"):
            if "analysis_id" in pd.DataFrame(song_items).columns:
                id_options = [item["analysis_id"] for item in song_items]
                del_id = st.selectbox("Silinecek kayıt ID", id_options, key="del_song_id")
                if st.button("Model Analizini Sil", key="btn_del_song"):
                    if delete_song_analysis(del_id):
                        st.success("Kayıt silindi. Sayfayı yenileyin.")
                    else:
                        st.error("Silme başarısız.")

    st.markdown("---")

    # ---- Gemini analizleri -----------------------------------------------
    st.markdown("### Gemini AI Analiz Geçmişi")

    gemini_items = get_all_gemini_analyses()
    if not gemini_items:
        st.info("Henüz Gemini analizi kaydedilmemiş.")
    else:
        gemini_df = pd.DataFrame(gemini_items)
        display_cols_g = [
            c for c in [
                "timestamp", "song_name", "gemini_emotion",
                "gemini_valence", "gemini_arousal",
                "human_valence_mean", "human_arousal_mean",
                "delta_v", "delta_a",
            ] if c in gemini_df.columns
        ]
        gemini_df = gemini_df[display_cols_g].sort_values("timestamp", ascending=False).reset_index(drop=True)

        for col in ["gemini_valence", "gemini_arousal", "human_valence_mean", "human_arousal_mean", "delta_v", "delta_a"]:
            if col in gemini_df.columns:
                gemini_df[col] = gemini_df[col].apply(lambda x: round(float(x), 2) if pd.notna(x) else x)

        st.dataframe(gemini_df, use_container_width=True, hide_index=True)
        st.caption(f"Toplam {len(gemini_items)} kayıt")

        with st.expander("Kayıt Sil"):
            if "analysis_id" in pd.DataFrame(gemini_items).columns:
                id_options_g = [item["analysis_id"] for item in gemini_items]
                del_id_g = st.selectbox("Silinecek kayıt ID", id_options_g, key="del_gemini_id")
                if st.button("Gemini Analizini Sil", key="btn_del_gemini"):
                    if delete_gemini_analysis(del_id_g):
                        st.success("Kayıt silindi. Sayfayı yenileyin.")
                    else:
                        st.error("Silme başarısız.")


if __name__ == "__main__":
    main()
