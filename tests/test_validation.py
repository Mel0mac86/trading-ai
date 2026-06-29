"""
Test automatici del Modulo 5 - Validation.

Verifichiamo: il Monte Carlo distingue una serie vincente da una perdente, il
walk-forward calcola la consistenza, la robustezza esplora la griglia e il
verdetto del Validator scarta correttamente le strategie deboli.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_ai.data_engine import generate_ohlcv
from trading_ai.feature_engineering import FeatureEngine
from trading_ai.pattern_discovery import PatternDiscovery
from trading_ai.strategy_generator import StrategyGenerator, RiskParams
from trading_ai.validation import (
    Validator, monte_carlo_trades, parameter_robustness, walk_forward,
)
from trading_ai.validation.walk_forward import walk_forward as wf_fn


def test_monte_carlo_winning_vs_losing():
    """Una distribuzione di R positivi deve dare prob_profit alta; negativi bassa."""
    winning = np.array([1.5, -1, 2, -1, 1.5, -1, 2, 1, -1, 1.5] * 5)  # edge positivo
    losing = -winning                                                  # edge negativo
    mc_win = monte_carlo_trades(winning, n_sims=500)
    mc_lose = monte_carlo_trades(losing, n_sims=500)
    assert mc_win["prob_profit"] > 0.8
    assert mc_lose["prob_profit"] < 0.2
    assert mc_win["maxdd_p95"] <= 0.0      # il drawdown e' <= 0 per definizione


def test_monte_carlo_empty():
    """Nessun trade -> esito neutro che non passa i filtri."""
    mc = monte_carlo_trades(np.array([]))
    assert mc["n_trades"] == 0
    assert mc["prob_profit"] == 0.0


def test_monte_carlo_is_reproducible():
    """Stesso seme -> stessi risultati (riproducibilita')."""
    r = np.array([1.0, -1.0, 2.0, -1.0, 1.0])
    a = monte_carlo_trades(r, seed=1, n_sims=300)
    b = monte_carlo_trades(r, seed=1, n_sims=300)
    assert a["final_p50"] == pytest.approx(b["final_p50"])


@pytest.fixture
def strategy_and_feats():
    """Costruisce una strategia reale dalla pipeline per i test d'integrazione."""
    df = generate_ohlcv(n=12000, seed=42)
    feats = FeatureEngine().transform(df, dropna=True)
    disc = PatternDiscovery(n_clusters=12, horizon=8, min_count_test=10,
                            min_profit_factor=1.0)
    disc.discover(feats)
    gen = StrategyGenerator(disc, risk=RiskParams(max_bars=20))
    strategies = gen.build()
    return strategies, feats


def test_walk_forward_structure(strategy_and_feats):
    """Il walk-forward deve produrre una finestra per split e una consistenza in [0,1]."""
    strategies, feats = strategy_and_feats
    if not strategies:
        pytest.skip("Nessuna strategia stabile su dati sintetici (atteso).")
    wf = wf_fn(strategies[0], feats, n_splits=4)
    assert len(wf["per_window"]) == 4
    assert 0.0 <= wf["consistency"] <= 1.0


def test_robustness_grid(strategy_and_feats):
    """La robustezza deve valutare l'intera griglia 3x3 di SL/TP."""
    strategies, feats = strategy_and_feats
    if not strategies:
        pytest.skip("Nessuna strategia stabile su dati sintetici (atteso).")
    rob = parameter_robustness(strategies[0], feats)
    assert len(rob["grid"]) == 9           # 3 fattori SL x 3 fattori TP
    assert 0.0 <= rob["profitable_fraction"] <= 1.0


def test_validator_rejects_insufficient_trades(strategy_and_feats):
    """Con soglia trade altissima, qualsiasi strategia deve essere scartata."""
    strategies, feats = strategy_and_feats
    if not strategies:
        pytest.skip("Nessuna strategia stabile su dati sintetici (atteso).")
    v = Validator(min_trades=10_000)        # soglia irraggiungibile
    report = v.validate(strategies[0], feats)
    assert report.robust is False
    assert any("trade insufficienti" in r for r in report.reasons)


def test_validator_verdict_fields(strategy_and_feats):
    """Il report di validazione deve essere completo e serializzabile."""
    strategies, feats = strategy_and_feats
    if not strategies:
        pytest.skip("Nessuna strategia stabile su dati sintetici (atteso).")
    v = Validator(min_trades=5, n_mc_sims=300, n_splits=4)
    report = v.validate(strategies[0], feats)
    s = report.summary()
    for key in ["name", "robust", "mc_prob_profit", "wf_consistency",
                "rob_profitable_fraction"]:
        assert key in s
    assert isinstance(report.robust, bool)
