from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"

BASE_MODELS = ["cnn_optimized", "cnn_baseline", "crnn_v2"]

PREDICTION_FILES: Dict[str, Path] = {
    "cnn_optimized": RESULTS_DIR / "cnn_optimized_predictions.csv",
    "cnn_baseline": RESULTS_DIR / "cnn_baseline_predictions.csv",
    "crnn_v2": RESULTS_DIR / "crnn_v2_predictions.csv",
}

GROUND_TRUTH_CANDIDATES = [
    ROOT_DIR / "ground_truth.csv",
    RESULTS_DIR / "ground_truth.csv",
]

SIMPLE_ENSEMBLE_OUT = RESULTS_DIR / "ensemble_predictions.csv"
WEIGHTED_ENSEMBLE_OUT = RESULTS_DIR / "ensemble_weighted_predictions.csv"
STACKING_PREDICTIONS_OUT = RESULTS_DIR / "stacking_predictions.csv"
COMPARISON_WITH_STACKING_OUT = RESULTS_DIR / "model_comparison_with_stacking.csv"
LEARNED_WEIGHTS_OUT = RESULTS_DIR / "learned_weights.txt"

MANUAL_WEIGHT_VALENCE = np.array([0.30, 0.30, 0.40], dtype=np.float64)
MANUAL_WEIGHT_AROUSAL = np.array([0.55, 0.05, 0.40], dtype=np.float64)


def _pick_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def _as_song_id(series: pd.Series, source_name: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(values).any():
        raise ValueError(f"{source_name}: song_id has NaN values")
    if not np.allclose(values, np.round(values), atol=1e-8):
        raise ValueError(f"{source_name}: song_id has non-integer values")
    return np.round(values).astype(np.int64)


def _infer_scale(min_value: float, max_value: float) -> str:
    if min_value >= 0.0 and max_value <= 1.05:
        return "0-1"
    if min_value >= -1.05 and max_value <= 1.05:
        return "-1-1"
    if min_value >= 0.8 and max_value <= 9.2:
        return "1-9"
    return "unknown"


def _safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan")
    return float(pearsonr(y_true, y_pred)[0])


def _ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mean_true = float(np.mean(y_true))
    mean_pred = float(np.mean(y_pred))
    var_true = float(np.var(y_true))
    var_pred = float(np.var(y_pred))
    cov = float(np.mean((y_true - mean_true) * (y_pred - mean_pred)))
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denom < 1e-12:
        return float("nan")
    return float((2.0 * cov) / denom)


def _compute_metrics(
    y_true_v: np.ndarray,
    y_true_a: np.ndarray,
    y_pred_v: np.ndarray,
    y_pred_a: np.ndarray,
) -> Dict[str, float]:
    rmse_v = float(np.sqrt(np.mean((y_true_v - y_pred_v) ** 2)))
    rmse_a = float(np.sqrt(np.mean((y_true_a - y_pred_a) ** 2)))
    pearson_v = _safe_pearson(y_true_v, y_pred_v)
    pearson_a = _safe_pearson(y_true_a, y_pred_a)
    ccc_v = _ccc(y_true_v, y_pred_v)
    ccc_a = _ccc(y_true_a, y_pred_a)
    return {
        "rmse_v": rmse_v,
        "rmse_a": rmse_a,
        "pearson_v": pearson_v,
        "pearson_a": pearson_a,
        "ccc_v": ccc_v,
        "ccc_a": ccc_a,
        "avg_rmse": (rmse_v + rmse_a) / 2.0,
        "avg_pearson": (pearson_v + pearson_a) / 2.0,
        "avg_ccc": (ccc_v + ccc_a) / 2.0,
    }


def _extract_weights(model) -> np.ndarray:
    if hasattr(model, "coef_"):
        return np.asarray(model.coef_, dtype=np.float64).reshape(-1)
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=np.float64).reshape(-1)
    return np.full(3, np.nan, dtype=np.float64)


def _load_prediction_file(path: Path, model_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{model_name}: prediction file not found -> {path}")

    raw = pd.read_csv(path)
    id_col = _pick_column(raw.columns, ["song_id", "sample_id"])
    v_col = _pick_column(raw.columns, ["valence", "pred_valence"])
    a_col = _pick_column(raw.columns, ["arousal", "pred_arousal"])
    true_v_col = _pick_column(raw.columns, ["true_valence", "valence_true", "gt_valence"])
    true_a_col = _pick_column(raw.columns, ["true_arousal", "arousal_true", "gt_arousal"])

    missing = [
        key
        for key, value in {
            "song_id/sample_id": id_col,
            "valence/pred_valence": v_col,
            "arousal/pred_arousal": a_col,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(
            f"{model_name}: missing required columns -> {missing}. Found: {list(raw.columns)}"
        )

    song_ids = _as_song_id(raw[id_col], model_name)
    valence = pd.to_numeric(raw[v_col], errors="coerce").to_numpy(dtype=np.float64)
    arousal = pd.to_numeric(raw[a_col], errors="coerce").to_numpy(dtype=np.float64)

    if np.isnan(valence).any() or np.isnan(arousal).any():
        raise ValueError(f"{model_name}: valence/arousal has NaN values")
    if len(np.unique(song_ids)) != len(song_ids):
        raise ValueError(f"{model_name}: duplicate song_id detected")

    df = pd.DataFrame(
        {
            "song_id": song_ids,
            "valence": valence,
            "arousal": arousal,
        }
    )

    if true_v_col is not None and true_a_col is not None:
        df["true_valence"] = pd.to_numeric(raw[true_v_col], errors="coerce").to_numpy(dtype=np.float64)
        df["true_arousal"] = pd.to_numeric(raw[true_a_col], errors="coerce").to_numpy(dtype=np.float64)
    else:
        df["true_valence"] = np.nan
        df["true_arousal"] = np.nan

    return df


def _load_ground_truth_from_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    id_col = _pick_column(raw.columns, ["song_id", "sample_id"])
    v_col = _pick_column(raw.columns, ["valence", "true_valence", "valence_true", "gt_valence"])
    a_col = _pick_column(raw.columns, ["arousal", "true_arousal", "arousal_true", "gt_arousal"])

    missing = [
        key
        for key, value in {
            "song_id": id_col,
            "valence": v_col,
            "arousal": a_col,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(f"ground_truth.csv missing required columns -> {missing}")

    gt = pd.DataFrame(
        {
            "song_id": _as_song_id(raw[id_col], f"ground_truth:{path}"),
            "valence": pd.to_numeric(raw[v_col], errors="coerce").to_numpy(dtype=np.float64),
            "arousal": pd.to_numeric(raw[a_col], errors="coerce").to_numpy(dtype=np.float64),
        }
    )
    if gt[["valence", "arousal"]].isna().any().any():
        raise ValueError(f"ground_truth.csv has NaN values -> {path}")
    if len(np.unique(gt["song_id"].to_numpy())) != len(gt):
        raise ValueError(f"ground_truth.csv has duplicate song_id -> {path}")
    return gt


def _validate_same_ids(reference_ids: np.ndarray, candidate_ids: np.ndarray, candidate_name: str) -> None:
    if len(reference_ids) != len(candidate_ids):
        raise ValueError(
            f"song_id row count mismatch with {candidate_name}: {len(reference_ids)} vs {len(candidate_ids)}"
        )

    if np.array_equal(reference_ids, candidate_ids):
        return

    reference_set = set(reference_ids.tolist())
    candidate_set = set(candidate_ids.tolist())
    if reference_set != candidate_set:
        missing = sorted(reference_set - candidate_set)[:10]
        extra = sorted(candidate_set - reference_set)[:10]
        raise ValueError(
            f"song_id mismatch with {candidate_name}; stopping. Missing: {missing}, Extra: {extra}"
        )

    mismatch_positions = int(np.sum(reference_ids != candidate_ids))
    raise ValueError(
        f"song_id order mismatch with {candidate_name}; stopping. Mismatched positions: {mismatch_positions}"
    )


def _resolve_ground_truth(frames: Dict[str, pd.DataFrame], reference_ids: np.ndarray) -> pd.DataFrame:
    found_path = None
    for candidate in GROUND_TRUTH_CANDIDATES:
        if candidate.exists():
            found_path = candidate
            break

    if found_path is not None:
        gt = _load_ground_truth_from_csv(found_path)
        _validate_same_ids(reference_ids, gt["song_id"].to_numpy(dtype=np.int64), str(found_path))
        print(f"- ground_truth.csv loaded from: {found_path}")
        return gt

    # Auto-create ground_truth.csv from true_* columns if explicit file is missing.
    for model_name in BASE_MODELS:
        frame = frames[model_name]
        if not frame["true_valence"].isna().any() and not frame["true_arousal"].isna().any():
            gt = pd.DataFrame(
                {
                    "song_id": frame["song_id"].to_numpy(dtype=np.int64),
                    "valence": frame["true_valence"].to_numpy(dtype=np.float64),
                    "arousal": frame["true_arousal"].to_numpy(dtype=np.float64),
                }
            )
            out_path = ROOT_DIR / "ground_truth.csv"
            gt.to_csv(out_path, index=False)
            print(f"- ground_truth.csv not found; created automatically from {model_name}: {out_path}")
            return gt

    raise FileNotFoundError(
        "ground_truth.csv not found and true labels are not available in prediction files"
    )


def _validate_ground_truth_consistency(gt: pd.DataFrame, frames: Dict[str, pd.DataFrame]) -> None:
    yv = gt["valence"].to_numpy(dtype=np.float64)
    ya = gt["arousal"].to_numpy(dtype=np.float64)
    for model_name, frame in frames.items():
        tv = frame["true_valence"].to_numpy(dtype=np.float64)
        ta = frame["true_arousal"].to_numpy(dtype=np.float64)
        if np.isnan(tv).all() or np.isnan(ta).all():
            continue
        if not np.allclose(yv, tv, atol=1e-7) or not np.allclose(ya, ta, atol=1e-7):
            raise ValueError(f"ground truth mismatch between ground_truth.csv and {model_name}")


def _build_scale_table(frames: Dict[str, pd.DataFrame], gt: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model_name, frame in frames.items():
        for target in ["valence", "arousal"]:
            min_value = float(frame[target].min())
            max_value = float(frame[target].max())
            rows.append(
                {
                    "source": model_name,
                    "target": target,
                    "min": min_value,
                    "max": max_value,
                    "inferred_scale": _infer_scale(min_value, max_value),
                }
            )

    for target in ["valence", "arousal"]:
        min_value = float(gt[target].min())
        max_value = float(gt[target].max())
        rows.append(
            {
                "source": "ground_truth",
                "target": target,
                "min": min_value,
                "max": max_value,
                "inferred_scale": _infer_scale(min_value, max_value),
            }
        )
    return pd.DataFrame(rows)


def _run_stacking_cv(
    model_name: str,
    estimator_factory: Callable[[], object],
    X_valence: np.ndarray,
    X_arousal: np.ndarray,
    y_valence: np.ndarray,
    y_arousal: np.ndarray,
    n_splits: int = 5,
) -> Dict[str, object]:
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    n = len(y_valence)
    oof_valence = np.zeros(n, dtype=np.float64)
    oof_arousal = np.zeros(n, dtype=np.float64)
    fold_assignment = np.zeros(n, dtype=np.int64)
    fold_rows: List[Dict[str, float]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(kfold.split(X_valence), start=1):
        model_v = estimator_factory()
        model_a = estimator_factory()

        model_v.fit(X_valence[train_idx], y_valence[train_idx])
        model_a.fit(X_arousal[train_idx], y_arousal[train_idx])

        pred_v = np.asarray(model_v.predict(X_valence[valid_idx]), dtype=np.float64)
        pred_a = np.asarray(model_a.predict(X_arousal[valid_idx]), dtype=np.float64)

        oof_valence[valid_idx] = pred_v
        oof_arousal[valid_idx] = pred_a
        fold_assignment[valid_idx] = fold_idx

        fold_metrics = _compute_metrics(
            y_valence[valid_idx],
            y_arousal[valid_idx],
            pred_v,
            pred_a,
        )
        fold_metrics["fold"] = float(fold_idx)
        fold_rows.append(fold_metrics)

    cv_metrics = _compute_metrics(y_valence, y_arousal, oof_valence, oof_arousal)

    full_model_v = estimator_factory()
    full_model_a = estimator_factory()
    full_model_v.fit(X_valence, y_valence)
    full_model_a.fit(X_arousal, y_arousal)
    fit_valence = np.asarray(full_model_v.predict(X_valence), dtype=np.float64)
    fit_arousal = np.asarray(full_model_a.predict(X_arousal), dtype=np.float64)
    train_metrics = _compute_metrics(y_valence, y_arousal, fit_valence, fit_arousal)

    return {
        "model_name": model_name,
        "cv_metrics": cv_metrics,
        "train_metrics": train_metrics,
        "oof_valence": oof_valence,
        "oof_arousal": oof_arousal,
        "fit_valence": fit_valence,
        "fit_arousal": fit_arousal,
        "fold_assignment": fold_assignment,
        "weights_valence": _extract_weights(full_model_v),
        "weights_arousal": _extract_weights(full_model_a),
        "intercept_valence": float(getattr(full_model_v, "intercept_", np.nan)),
        "intercept_arousal": float(getattr(full_model_a, "intercept_", np.nan)),
        "fold_metrics": pd.DataFrame(fold_rows),
    }


def _build_diversity_tables(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    valence_df = pd.DataFrame({name: frames[name]["valence"].to_numpy(dtype=np.float64) for name in BASE_MODELS})
    arousal_df = pd.DataFrame({name: frames[name]["arousal"].to_numpy(dtype=np.float64) for name in BASE_MODELS})
    return {
        "valence": valence_df.corr(method="pearson"),
        "arousal": arousal_df.corr(method="pearson"),
    }


def _best_model_name(results: Dict[str, Dict[str, object]]) -> str:
    return sorted(
        results.keys(),
        key=lambda name: (
            float(results[name]["cv_metrics"]["avg_rmse"]),
            -float(results[name]["cv_metrics"]["avg_pearson"]),
        ),
    )[0]


def _weights_text(weights: np.ndarray) -> str:
    parts = [f"{model}={weight:+.6f}" for model, weight in zip(BASE_MODELS, weights.tolist())]
    return ", ".join(parts)


def main() -> None:
    print("STEP 1/9 - VERIYI YUKLE VE DOGRULA")

    frames = {name: _load_prediction_file(path, name) for name, path in PREDICTION_FILES.items()}
    reference_ids = frames[BASE_MODELS[0]]["song_id"].to_numpy(dtype=np.int64)

    for model_name in BASE_MODELS[1:]:
        _validate_same_ids(reference_ids, frames[model_name]["song_id"].to_numpy(dtype=np.int64), model_name)

    ground_truth = _resolve_ground_truth(frames, reference_ids)
    _validate_ground_truth_consistency(ground_truth, frames)

    print(f"- prediction files loaded: {', '.join(BASE_MODELS)}")
    print("- required columns normalized as: song_id, valence, arousal")
    print(f"- sample count: {len(reference_ids)}")
    print("- song_id alignment: OK")

    scale_table = _build_scale_table(frames, ground_truth)
    print("\nSTEP 4/9 - SCALE KONTROL")
    print(scale_table.to_string(index=False))

    valence_scales = set(scale_table[scale_table["target"] == "valence"]["inferred_scale"].tolist())
    arousal_scales = set(scale_table[scale_table["target"] == "arousal"]["inferred_scale"].tolist())
    if len(valence_scales) != 1 or len(arousal_scales) != 1:
        raise ValueError(
            "Scale mismatch detected; stopping. "
            f"Valence scales: {sorted(valence_scales)} | Arousal scales: {sorted(arousal_scales)}"
        )
    if "unknown" in valence_scales or "unknown" in arousal_scales:
        raise ValueError("Unknown scale detected; stopping")
    print("- scale consistency: OK")

    print("\nSTEP 2/9 - FEATURE MATRISI (STACKING INPUT)")
    X_valence = np.column_stack([frames[name]["valence"].to_numpy(dtype=np.float64) for name in BASE_MODELS])
    X_arousal = np.column_stack([frames[name]["arousal"].to_numpy(dtype=np.float64) for name in BASE_MODELS])
    y_valence = ground_truth["valence"].to_numpy(dtype=np.float64)
    y_arousal = ground_truth["arousal"].to_numpy(dtype=np.float64)
    print(f"- X_valence shape: {X_valence.shape}")
    print(f"- X_arousal shape: {X_arousal.shape}")

    print("\nSTEP 5/9 - KARSI LASTIRMA PIPELINE ICIN BASELINE ENSEMBLELER")
    simple_valence = np.mean(X_valence, axis=1)
    simple_arousal = np.mean(X_arousal, axis=1)
    simple_metrics = _compute_metrics(y_valence, y_arousal, simple_valence, simple_arousal)
    simple_df = pd.DataFrame(
        {
            "song_id": reference_ids,
            "valence": simple_valence,
            "arousal": simple_arousal,
            "true_valence": y_valence,
            "true_arousal": y_arousal,
        }
    )
    simple_df.to_csv(SIMPLE_ENSEMBLE_OUT, index=False)

    weighted_valence = X_valence @ MANUAL_WEIGHT_VALENCE
    weighted_arousal = X_arousal @ MANUAL_WEIGHT_AROUSAL
    weighted_metrics = _compute_metrics(y_valence, y_arousal, weighted_valence, weighted_arousal)
    weighted_df = pd.DataFrame(
        {
            "song_id": reference_ids,
            "valence": weighted_valence,
            "arousal": weighted_arousal,
            "true_valence": y_valence,
            "true_arousal": y_arousal,
            "w_valence_cnn_optimized": MANUAL_WEIGHT_VALENCE[0],
            "w_valence_cnn_baseline": MANUAL_WEIGHT_VALENCE[1],
            "w_valence_crnn_v2": MANUAL_WEIGHT_VALENCE[2],
            "w_arousal_cnn_optimized": MANUAL_WEIGHT_AROUSAL[0],
            "w_arousal_cnn_baseline": MANUAL_WEIGHT_AROUSAL[1],
            "w_arousal_crnn_v2": MANUAL_WEIGHT_AROUSAL[2],
        }
    )
    weighted_df.to_csv(WEIGHTED_ENSEMBLE_OUT, index=False)
    print(f"- saved simple ensemble: {SIMPLE_ENSEMBLE_OUT}")
    print(f"- saved weighted ensemble: {WEIGHTED_ENSEMBLE_OUT}")

    print("\nSTEP 3/9 ve STEP 4/9 - STACKING MODELLERI ve 5-FOLD CV")
    linear_factories: Dict[str, Callable[[], object]] = {
        "linear_regression": lambda: LinearRegression(),
        "ridge_alpha_1.0": lambda: Ridge(alpha=1.0),
        "lasso_alpha_0.001": lambda: Lasso(alpha=0.001, max_iter=10000),
    }

    linear_results: Dict[str, Dict[str, object]] = {}
    for model_name, factory in linear_factories.items():
        linear_results[model_name] = _run_stacking_cv(
            model_name=model_name,
            estimator_factory=factory,
            X_valence=X_valence,
            X_arousal=X_arousal,
            y_valence=y_valence,
            y_arousal=y_arousal,
            n_splits=5,
        )

    best_linear_name = _best_model_name(linear_results)
    best_linear = linear_results[best_linear_name]

    print("- linear stacking candidates (CV):")
    for model_name in linear_factories:
        metrics = linear_results[model_name]["cv_metrics"]
        print(
            f"  {model_name}: RMSE(V)={metrics['rmse_v']:.4f}, RMSE(A)={metrics['rmse_a']:.4f}, "
            f"Pearson(V)={metrics['pearson_v']:.4f}, Pearson(A)={metrics['pearson_a']:.4f}"
        )
    print(f"- best linear stacker: {best_linear_name}")

    stacking_df = pd.DataFrame(
        {
            "song_id": reference_ids,
            "valence": best_linear["oof_valence"],
            "arousal": best_linear["oof_arousal"],
            "true_valence": y_valence,
            "true_arousal": y_arousal,
            "fold": best_linear["fold_assignment"],
            "stacking_model": best_linear_name,
            "full_fit_valence": best_linear["fit_valence"],
            "full_fit_arousal": best_linear["fit_arousal"],
        }
    )
    stacking_df.to_csv(STACKING_PREDICTIONS_OUT, index=False)

    print("\nSTEP 8/9 - DIVERSITY ANALIZI")
    diversity_tables = _build_diversity_tables(frames)
    print("- Valence pairwise correlation:")
    print(diversity_tables["valence"].to_string())
    print("- Arousal pairwise correlation:")
    print(diversity_tables["arousal"].to_string())

    print("\nBONUS - NON-LINEAR STACKING (RANDOM FOREST) ve OVERFITTING KONTROL")
    rf_result = _run_stacking_cv(
        model_name="random_forest",
        estimator_factory=lambda: RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        X_valence=X_valence,
        X_arousal=X_arousal,
        y_valence=y_valence,
        y_arousal=y_arousal,
        n_splits=5,
    )

    single_metrics = {
        "cnn_optimized": _compute_metrics(
            y_valence,
            y_arousal,
            frames["cnn_optimized"]["valence"].to_numpy(dtype=np.float64),
            frames["cnn_optimized"]["arousal"].to_numpy(dtype=np.float64),
        ),
        "cnn_baseline": _compute_metrics(
            y_valence,
            y_arousal,
            frames["cnn_baseline"]["valence"].to_numpy(dtype=np.float64),
            frames["cnn_baseline"]["arousal"].to_numpy(dtype=np.float64),
        ),
        "crnn_v2": _compute_metrics(
            y_valence,
            y_arousal,
            frames["crnn_v2"]["valence"].to_numpy(dtype=np.float64),
            frames["crnn_v2"]["arousal"].to_numpy(dtype=np.float64),
        ),
    }

    print("\nSTEP 6/9 - METRIKLER ve KARSILASTIRMA TABLOSU")
    comparison_rows = [
        {"model": "cnn_optimized", **single_metrics["cnn_optimized"]},
        {"model": "cnn_baseline", **single_metrics["cnn_baseline"]},
        {"model": "crnn_v2", **single_metrics["crnn_v2"]},
        {"model": "simple_average_ensemble", **simple_metrics},
        {"model": "weighted_ensemble_manual", **weighted_metrics},
        {"model": f"stacking_ensemble_cv_{best_linear_name}", **best_linear["cv_metrics"]},
    ]

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(COMPARISON_WITH_STACKING_OUT, index=False)
    print(comparison_df.to_string(index=False))
    print(f"- saved comparison table: {COMPARISON_WITH_STACKING_OUT}")

    print("\nSTEP 7/9 - ANALIZ")
    best_required_row = comparison_df.sort_values(
        by=["avg_rmse", "avg_pearson"], ascending=[True, False]
    ).iloc[0]
    stacking_metrics = best_linear["cv_metrics"]

    delta_weighted_rmse_v = float(weighted_metrics["rmse_v"] - stacking_metrics["rmse_v"])
    delta_weighted_rmse_a = float(weighted_metrics["rmse_a"] - stacking_metrics["rmse_a"])
    delta_weighted_p_v = float(stacking_metrics["pearson_v"] - weighted_metrics["pearson_v"])
    delta_weighted_p_a = float(stacking_metrics["pearson_a"] - weighted_metrics["pearson_a"])

    weight_v = np.asarray(best_linear["weights_valence"], dtype=np.float64)
    weight_a = np.asarray(best_linear["weights_arousal"], dtype=np.float64)
    dominant_v = BASE_MODELS[int(np.argmax(np.abs(weight_v)))]
    dominant_a = BASE_MODELS[int(np.argmax(np.abs(weight_a)))]

    pairwise_val = [
        float(diversity_tables["valence"].loc[left, right])
        for left, right in combinations(BASE_MODELS, 2)
    ]
    pairwise_aro = [
        float(diversity_tables["arousal"].loc[left, right])
        for left, right in combinations(BASE_MODELS, 2)
    ]

    print(
        f"- Stacking en iyi model mi? {'EVET' if str(best_required_row['model']).startswith('stacking_ensemble') else 'HAYIR'}"
    )
    print(
        "- Weighted ensemble farki (stacking - weighted): "
        f"delta RMSE(V)={-delta_weighted_rmse_v:+.4f}, "
        f"delta RMSE(A)={-delta_weighted_rmse_a:+.4f}, "
        f"delta Pearson(V)={delta_weighted_p_v:+.4f}, "
        f"delta Pearson(A)={delta_weighted_p_a:+.4f}"
    )
    print(f"- Valence icin en yuksek agirlik: {dominant_v} ({weight_v[np.argmax(np.abs(weight_v))]:+.4f})")
    print(f"- Arousal icin en yuksek agirlik: {dominant_a} ({weight_a[np.argmax(np.abs(weight_a))]:+.4f})")
    print(f"- Valence ve Arousal agirliklari farkli mi? {'EVET' if not np.allclose(weight_v, weight_a, atol=1e-6) else 'HAYIR'}")
    print(
        "- Diversity ortalama korelasyonlari: "
        f"Valence={np.mean(pairwise_val):.4f}, Arousal={np.mean(pairwise_aro):.4f}"
    )

    # Save the learned weights and analysis as plain text report.
    lines: List[str] = []
    lines.append("STACKING ENSEMBLE REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Data and Validation")
    lines.append(f"- Samples: {len(reference_ids)}")
    lines.append(f"- Base models: {', '.join(BASE_MODELS)}")
    lines.append(f"- Ground truth source: {ROOT_DIR / 'ground_truth.csv' if (ROOT_DIR / 'ground_truth.csv').exists() else 'generated in memory'}")
    lines.append("")

    lines.append("Linear Stacking CV Results")
    for model_name in linear_factories:
        m = linear_results[model_name]["cv_metrics"]
        lines.append(
            f"- {model_name}: RMSE(V)={m['rmse_v']:.6f}, RMSE(A)={m['rmse_a']:.6f}, "
            f"Pearson(V)={m['pearson_v']:.6f}, Pearson(A)={m['pearson_a']:.6f}, "
            f"CCC(V)={m['ccc_v']:.6f}, CCC(A)={m['ccc_a']:.6f}"
        )
    lines.append(f"- Selected best stacking model: {best_linear_name}")
    lines.append("")

    lines.append("Learned Weights (fit on full data)")
    for model_name in linear_factories:
        result = linear_results[model_name]
        lines.append(f"- {model_name} / valence: {_weights_text(np.asarray(result['weights_valence'], dtype=np.float64))}")
        lines.append(f"  intercept={float(result['intercept_valence']):+.6f}")
        lines.append(f"- {model_name} / arousal: {_weights_text(np.asarray(result['weights_arousal'], dtype=np.float64))}")
        lines.append(f"  intercept={float(result['intercept_arousal']):+.6f}")
    lines.append("")

    lines.append("Diversity Correlation - Valence")
    lines.append(diversity_tables["valence"].to_string())
    lines.append("")
    lines.append("Diversity Correlation - Arousal")
    lines.append(diversity_tables["arousal"].to_string())
    lines.append("")

    lines.append("Required Comparison")
    lines.append(comparison_df.to_string(index=False))
    lines.append("")

    lines.append("Bonus: RandomForest Stacking")
    rf_cv = rf_result["cv_metrics"]
    rf_train = rf_result["train_metrics"]
    lines.append(
        f"- RF CV: RMSE(V)={rf_cv['rmse_v']:.6f}, RMSE(A)={rf_cv['rmse_a']:.6f}, "
        f"Pearson(V)={rf_cv['pearson_v']:.6f}, Pearson(A)={rf_cv['pearson_a']:.6f}"
    )
    lines.append(
        f"- RF Train: RMSE(V)={rf_train['rmse_v']:.6f}, RMSE(A)={rf_train['rmse_a']:.6f}, "
        f"Pearson(V)={rf_train['pearson_v']:.6f}, Pearson(A)={rf_train['pearson_a']:.6f}"
    )
    lines.append(
        f"- RF overfitting gap avg_rmse(train-cv)={rf_train['avg_rmse'] - rf_cv['avg_rmse']:+.6f}"
    )
    lines.append("")

    best_train = best_linear["train_metrics"]
    lines.append("Overfitting Check - Selected Stacking")
    lines.append(
        f"- CV avg_rmse={stacking_metrics['avg_rmse']:.6f}, Train avg_rmse={best_train['avg_rmse']:.6f}, "
        f"gap(train-cv)={best_train['avg_rmse'] - stacking_metrics['avg_rmse']:+.6f}"
    )

    LEARNED_WEIGHTS_OUT.write_text("\n".join(lines), encoding="utf-8")

    print("\nSTEP 9/9 - CIKTILAR")
    print(f"- saved stacking predictions: {STACKING_PREDICTIONS_OUT}")
    print(f"- saved comparison with stacking: {COMPARISON_WITH_STACKING_OUT}")
    print(f"- saved learned weights report: {LEARNED_WEIGHTS_OUT}")


if __name__ == "__main__":
    main()
