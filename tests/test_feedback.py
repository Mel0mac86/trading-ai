"""
Test automatici del Modulo 7 - AI Feedback.

Verifichiamo: la diagnosi rileva i problemi attesi, l'obiettivo robusto esclude
i campioni piccoli, l'ottimizzazione trova parametri validi, e il versioning
mantiene lo storico con persistenza JSON.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_ai.data_engine import generate_ohlcv
from trading_ai.feature_engineering import FeatureEngine
from trading_ai.feedback import FeedbackEngine
from trading_ai.feedback.analysis import analyze
from trading_ai.feedback.optimizer import optimize_risk, robust_objective
from trading_ai.pattern_discovery.clustering import FeatureClusterer
from trading_ai.strategy_generator import RiskParams, Strategy


def test_analyze_flags_high_drawdown():
    """Drawdown elevato deve generare issue e una spinta a stringere lo SL."""
    metrics = {"n_trades": 100, "win_rate": 0.5, "profit_factor": 1.5,
               "max_drawdown": -0.40, "avg_win": 0.02, "avg_loss": -0.02,
               "expectancy": 0.001}
    diag = analyze(metrics)
    assert any("Drawdown" in i for i in diag["issues"])
    assert diag["nudges"].get("sl_atr", 0) < 0


def test_analyze_flags_no_edge():
    """Expectancy non positiva deve essere segnalata come assenza di edge."""
    metrics = {"n_trades": 50, "win_rate": 0.45, "profit_factor": 0.9,
               "max_drawdown": -0.1, "avg_win": 0.01, "avg_loss": -0.012,
               "expectancy": -0.001}
    diag = analyze(metrics)
    assert any("edge" in i.lower() for i in diag["issues"])


def test_robust_objective_excludes_small_sample():
    """Con pochi trade l'obiettivo robusto e' -inf (configurazione esclusa)."""
    summ = {"n_trades": 5, "total_return": 1.0, "max_drawdown": -0.05}
    assert robust_objective(summ, min_trades=30) == float("-inf")
    summ2 = {"n_trades": 100, "total_return": 0.2, "max_drawdown": -0.05}
    assert np.isfinite(robust_objective(summ2, min_trades=30))


@pytest.fixture
def strategy_and_feats():
    df = generate_ohlcv(n=10000, seed=33)
    feats = FeatureEngine().transform(df, groups=["indicator", "volatility"], dropna=True)
    cols = [c for c in feats.columns if c not in ("open", "high", "low", "close", "volume")]
    clu = FeatureClusterer(n_clusters=6, feature_columns=cols).fit(feats)
    strat = Strategy(name="PAT01_LONG", cluster_id=1, direction=1, clusterer=clu,
                     risk=RiskParams(sl_atr=2, tp_atr=3, max_bars=20))
    return strat, feats


def test_optimize_returns_valid_params(strategy_and_feats):
    """L'ottimizzazione deve ritornare RiskParams validi e una tabella popolata."""
    strat, feats = strategy_and_feats
    opt = optimize_risk(strat, feats, min_trades=5)
    assert opt["best_risk"].sl_atr > 0
    assert opt["best_risk"].tp_atr > 0
    assert len(opt["table"]) > 0
    # La tabella e' ordinata per score decrescente.
    scores = opt["table"]["score"].to_numpy()
    assert (scores[:-1] >= scores[1:]).all()


def test_feedback_creates_versions(strategy_and_feats):
    """Il feedback deve produrre baseline (v0) + versione migliorata (v1)."""
    strat, feats = strategy_and_feats
    fb = FeedbackEngine(min_trades=5)
    res = fb.improve(strat, feats)
    assert len(res.versions) == 2
    assert res.versions[0].version == 0 and res.versions[0].parent is None
    assert res.versions[1].version == 1 and res.versions[1].parent == 0
    assert "issues" in res.diagnosis

    # La strategia migliorata e' un oggetto Strategy con (eventualmente) nuovi parametri.
    assert hasattr(res.improved_strategy, "risk")


def test_history_persistence(strategy_and_feats, tmp_path):
    """Lo storico delle versioni deve salvarsi e ricaricarsi senza perdite."""
    strat, feats = strategy_and_feats
    fb = FeedbackEngine(min_trades=5)
    fb.improve(strat, feats)
    path = fb.save_history(tmp_path / "history.json")
    assert path.exists()
    loaded = FeedbackEngine.load_history(path)
    assert len(loaded) == len(fb.history)
    assert loaded[0].name == strat.name


def test_iterate_multiple_rounds(strategy_and_feats):
    """L'iterazione su piu' cicli deve accumulare versioni nello storico."""
    strat, feats = strategy_and_feats
    fb = FeedbackEngine(min_trades=5)
    fb.iterate(strat, feats, rounds=2)
    assert len(fb.history) == 4   # 2 cicli x (v0 + v1)
