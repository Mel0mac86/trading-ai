"""
Test della correzione per multiple testing (Deflated Sharpe Ratio).

Verifichiamo le proprieta' matematiche di PSR, Expected Maximum Sharpe e DSR,
e l'integrazione nel filtro della Pattern Discovery (il DSR e' un requisito
aggiuntivo: non puo' aumentare il numero di pattern accettati).
"""

from __future__ import annotations

import numpy as np
import pytest

from trading_ai.data_engine import generate_ohlcv
from trading_ai.feature_engineering import FeatureEngine
from trading_ai.pattern_discovery import PatternDiscovery
from trading_ai.validation.multiple_testing import (
    deflated_sharpe_ratio, expected_max_sharpe, probabilistic_sharpe_ratio,
)


def test_psr_zero_sharpe_is_half():
    """Con Sharpe osservato = 0 e soglia 0, P(vero>0) = 0.5."""
    assert probabilistic_sharpe_ratio(0.0, n_obs=100) == pytest.approx(0.5, abs=1e-6)


def test_psr_monotonic_in_sharpe():
    """A parita' di campione, uno Sharpe piu' alto da' un PSR piu' alto."""
    low = probabilistic_sharpe_ratio(0.1, n_obs=200)
    high = probabilistic_sharpe_ratio(0.5, n_obs=200)
    assert high > low


def test_psr_increases_with_sample():
    """Lo stesso Sharpe positivo e' piu' credibile con piu' osservazioni."""
    few = probabilistic_sharpe_ratio(0.2, n_obs=30)
    many = probabilistic_sharpe_ratio(0.2, n_obs=1000)
    assert many > few


def test_psr_small_sample_returns_zero():
    """Con meno di 2 osservazioni il PSR non e' definito -> 0."""
    assert probabilistic_sharpe_ratio(1.0, n_obs=1) == 0.0


def test_expected_max_sharpe_grows_with_trials():
    """Piu' trial -> Sharpe massimo atteso (sotto nulla) piu' alto."""
    s10 = expected_max_sharpe(10, variance_of_sharpes=0.04)
    s100 = expected_max_sharpe(100, variance_of_sharpes=0.04)
    assert s100 > s10 > 0


def test_expected_max_sharpe_single_trial_zero():
    """Con un solo trial non c'e' selezione: soglia 0."""
    assert expected_max_sharpe(1, 0.04) == 0.0


def test_dsr_below_psr_due_to_deflation():
    """Il DSR (con molti trial) deve essere <= del PSR grezzo (deflazione)."""
    sharpe, n = 0.3, 300
    raw = probabilistic_sharpe_ratio(sharpe, n)
    dsr = deflated_sharpe_ratio(sharpe, n, n_trials=50, variance_of_sharpes=0.04)
    assert dsr <= raw
    assert 0.0 <= dsr <= 1.0


def test_dsr_is_additional_filter():
    """
    Con use_dsr attivo il numero di pattern stabili non puo' superare quello
    senza DSR: il DSR e' un requisito AGGIUNTIVO, mai un rilassamento.
    """
    df = generate_ohlcv(n=9000, seed=7)
    feats = FeatureEngine().transform(df, dropna=True)
    common = dict(n_clusters=12, horizon=8, min_count_test=10, min_profit_factor=1.0)

    no_dsr = PatternDiscovery(use_dsr=False, **common)
    no_dsr.discover(feats)
    with_dsr = PatternDiscovery(use_dsr=True, min_dsr=0.90, **common)
    with_dsr.discover(feats)

    assert len(with_dsr.stable_patterns()) <= len(no_dsr.stable_patterns())
    # La colonna dsr deve essere presente nella tabella.
    table = PatternDiscovery(use_dsr=True, **common).discover(feats)
    assert "dsr" in table.columns
