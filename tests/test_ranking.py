"""
src/ranking.py için pytest testleri.

Model yüklemez, ses dosyası okumaz - sadece sentetik (sahte) tahmin
verileriyle sıralama/uzaklık/"ayırt edilemez" mantığını test eder.
"""

import math

import pytest

from inference import MultiWindowResult, WindowPrediction
from ranking import (
    CLOSE_WINDOW_THRESHOLD,
    INDISTINGUISHABLE_THRESHOLD,
    CandidateRanking,
    euclidean_distance,
    load_emotion_targets,
    rank_candidates,
    resolve_target,
    score_candidate,
)


def make_window(valence, arousal, skipped=False, rms=0.05, padding_ratio=0.0):
    return WindowPrediction(
        start_sec=0.0,
        end_sec=30.0,
        rms=rms,
        skipped=skipped,
        padding_ratio=padding_ratio,
        valence=valence,
        arousal=arousal,
    )


def make_mw(window_va, valence=None, arousal=None, valence_std=None, arousal_std=None):
    """window_va: [(v, a), ...] sadece kullanılan (skip edilmemiş) pencereler için."""
    windows = [make_window(v, a) for v, a in window_va]
    vs = [w.valence for w in windows]
    as_ = [w.arousal for w in windows]

    return MultiWindowResult(
        valence=valence if valence is not None else sum(vs) / len(vs),
        arousal=arousal if arousal is not None else sum(as_) / len(as_),
        valence_std=valence_std if valence_std is not None else 0.0,
        arousal_std=arousal_std if arousal_std is not None else 0.0,
        windows=windows,
        n_windows_total=len(windows),
        n_windows_used=len(windows),
        n_windows_skipped=0,
    )


# --- euclidean_distance ---

def test_euclidean_distance_zero_when_identical():
    assert euclidean_distance(5.0, 5.0, 5.0, 5.0) == 0.0


def test_euclidean_distance_known_triangle():
    # 3-4-5 üçgeni
    assert euclidean_distance(0.0, 0.0, 3.0, 4.0) == pytest.approx(5.0)


# --- load_emotion_targets / resolve_target ---

def test_load_emotion_targets_has_expected_labels():
    targets = load_emotion_targets()
    expected = {
        "sakin-huzurlu",
        "sıcak-nostaljik",
        "enerjik-pozitif",
        "gergin-dramatik",
        "hüzünlü-duygusal",
    }
    assert expected.issubset(targets.keys())
    for t in targets.values():
        assert 1.0 <= t.valence <= 9.0
        assert 1.0 <= t.arousal <= 9.0


def test_resolve_target_by_label():
    targets = load_emotion_targets()
    v, a = resolve_target(target_label="enerjik-pozitif", targets=targets)
    assert (v, a) == (targets["enerjik-pozitif"].valence, targets["enerjik-pozitif"].arousal)


def test_resolve_target_manual():
    v, a = resolve_target(manual_valence=5, manual_arousal=7)
    assert v == 5.0
    assert a == 7.0


def test_resolve_target_unknown_label_raises():
    with pytest.raises(ValueError):
        resolve_target(target_label="olmayan-etiket", targets=load_emotion_targets())


def test_resolve_target_missing_args_raises():
    with pytest.raises(ValueError):
        resolve_target()


# --- score_candidate ---

def test_score_candidate_distance_matches_target():
    mw = make_mw([(5.0, 5.0), (5.0, 5.0)], valence=5.0, arousal=5.0, valence_std=0.0, arousal_std=0.0)
    cand = score_candidate("clip_a", mw, target_v=5.0, target_a=5.0)
    assert cand.distance == pytest.approx(0.0)
    assert cand.consistency == pytest.approx(0.0)
    assert cand.pct_windows_close == pytest.approx(100.0)


def test_score_candidate_pct_windows_close():
    # Hedef (5, 5). Pencerelerden biri tam hedefte (uzaklık 0 <= 1.0),
    # diğeri çok uzak (uzaklık 1.0'dan büyük) -> %50 yakın.
    mw = make_mw([(5.0, 5.0), (9.0, 9.0)], valence=7.0, arousal=7.0, valence_std=2.0, arousal_std=2.0)
    cand = score_candidate("clip_b", mw, target_v=5.0, target_a=5.0)
    assert cand.pct_windows_close == pytest.approx(50.0)
    assert cand.consistency == pytest.approx(math.sqrt(2.0 ** 2 + 2.0 ** 2))


def test_score_candidate_skipped_windows_excluded_from_pct():
    windows = [make_window(5.0, 5.0), make_window(None, None, skipped=True)]
    mw = MultiWindowResult(
        valence=5.0,
        arousal=5.0,
        valence_std=0.0,
        arousal_std=0.0,
        windows=windows,
        n_windows_total=2,
        n_windows_used=1,
        n_windows_skipped=1,
    )
    cand = score_candidate("clip_c", mw, target_v=5.0, target_a=5.0)
    assert cand.pct_windows_close == pytest.approx(100.0)


def test_score_candidate_no_used_windows_pct_is_zero():
    windows = [make_window(None, None, skipped=True)]
    mw = MultiWindowResult(
        valence=5.0,
        arousal=5.0,
        valence_std=0.0,
        arousal_std=0.0,
        windows=windows,
        n_windows_total=1,
        n_windows_used=0,
        n_windows_skipped=1,
        used_all_windows_fallback=False,
    )
    cand = score_candidate("clip_d", mw, target_v=5.0, target_a=5.0)
    assert cand.pct_windows_close == pytest.approx(0.0)


# --- rank_candidates ---

def _candidate(name, distance):
    return CandidateRanking(
        name=name,
        valence=5.0,
        arousal=5.0,
        valence_std=0.1,
        arousal_std=0.1,
        distance=distance,
        consistency=0.14,
        pct_windows_close=100.0,
        n_windows_used=5,
    )


def test_rank_candidates_orders_ascending_by_distance():
    candidates = [_candidate("C", 2.0), _candidate("A", 0.2), _candidate("B", 1.0)]
    ranked = rank_candidates(candidates)
    assert [c.name for c in ranked] == ["A", "B", "C"]
    assert [c.rank for c in ranked] == [1, 2, 3]


def test_rank_candidates_indistinguishable_from_best():
    # A ve B birbirinden 0.3 (< 0.5 eşiği) farklı -> B, A'dan (en iyi) ayırt edilemez.
    # C, A'dan 1.8 farklı (>= eşik) -> ayırt edilebilir.
    candidates = [_candidate("A", 0.2), _candidate("B", 0.5), _candidate("C", 2.0)]
    ranked = rank_candidates(candidates)

    by_name = {c.name: c for c in ranked}
    assert by_name["A"].indistinguishable_from_best is True  # kendisi
    assert by_name["B"].indistinguishable_from_best is True
    assert by_name["C"].indistinguishable_from_best is False


def test_rank_candidates_group_chaining():
    candidates = [_candidate("A", 0.2), _candidate("B", 0.5), _candidate("C", 2.0)]
    ranked = rank_candidates(candidates)
    by_name = {c.name: c for c in ranked}
    assert by_name["A"].group_id == by_name["B"].group_id
    assert by_name["C"].group_id != by_name["A"].group_id


def test_rank_candidates_custom_threshold():
    candidates = [_candidate("A", 0.0), _candidate("B", 0.9)]
    # Varsayılan eşikle (0.5) ayırt edilebilir olmalı.
    default_ranked = {c.name: c for c in rank_candidates(candidates)}
    assert default_ranked["B"].indistinguishable_from_best is False

    # Daha gevşek bir eşikle (1.0) ayırt edilemez olmalı.
    loose_ranked = {c.name: c for c in rank_candidates(candidates, threshold=1.0)}
    assert loose_ranked["B"].indistinguishable_from_best is True


def test_rank_candidates_empty_list():
    assert rank_candidates([]) == []


def test_rank_candidates_does_not_mutate_input():
    candidates = [_candidate("A", 1.0)]
    original_rank = candidates[0].rank
    rank_candidates(candidates)
    assert candidates[0].rank == original_rank


def test_indistinguishable_threshold_is_fixed_constant():
    # Eşiğin sabit ve dokümante edilen değeri koruduğunu doğrula (model
    # RMSE'sine ~0.8 dayanan tasarım kararı - bkz. modül dosyası başı).
    assert INDISTINGUISHABLE_THRESHOLD == 0.5


def test_close_window_threshold_is_fixed_constant():
    assert CLOSE_WINDOW_THRESHOLD == 1.0
