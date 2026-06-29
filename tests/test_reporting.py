"""
Test automatici del Modulo 9 - Report.

Verifichiamo le metriche su curve di equity note, la robustezza ai casi limite
(nessun trade) e la generazione effettiva di grafici PNG e report Markdown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_ai.reporting import ReportGenerator, compute_report_metrics
from trading_ai.reporting.metrics import (
    calmar_ratio, max_drawdown, sharpe_ratio,
)
from trading_ai.strategy_generator.backtest import BacktestResult


def _equity(values, freq="D"):
    idx = pd.date_range("2021-01-01", periods=len(values), freq=freq)
    return pd.Series(values, index=idx, dtype=float)


def test_max_drawdown_known():
    """Drawdown su una curva nota: 100 -> 120 -> 90 -> 110 -> dd = -25%."""
    eq = _equity([100, 120, 90, 110])
    assert max_drawdown(eq) == pytest.approx((90 - 120) / 120)


def test_sharpe_zero_when_flat():
    """Equity piatta -> volatilita' nulla -> Sharpe 0 (niente divisione per zero)."""
    eq = _equity([100, 100, 100, 100, 100])
    assert sharpe_ratio(eq) == 0.0


def test_sharpe_positive_for_uptrend():
    """Una crescita costante con poca volatilita' deve dare Sharpe positivo."""
    eq = _equity(list(np.linspace(100, 200, 100)))
    assert sharpe_ratio(eq) > 0


def test_calmar_uses_cagr_and_dd():
    """Calmar e' definito (finito) per una curva crescente con drawdown."""
    eq = _equity(list(np.linspace(100, 150, 200)) + [120])
    c = calmar_ratio(eq)
    assert np.isfinite(c)


def test_report_metrics_winrate_profitfactor():
    """Win rate e profit factor su rendimenti noti."""
    tr = np.array([0.02, -0.01, 0.03, -0.02])
    eq = _equity([100, 102, 101, 104, 102])
    m = compute_report_metrics(eq, tr)
    assert m["win_rate"] == pytest.approx(0.5)
    # PF = (0.02+0.03)/(0.01+0.02) = 0.05/0.03
    assert m["profit_factor"] == pytest.approx(0.05 / 0.03)
    assert m["n_trades"] == 4


def test_report_metrics_empty_trades():
    """Nessun trade: metriche neutre senza crash."""
    eq = _equity([100, 100])
    m = compute_report_metrics(eq, np.array([]))
    assert m["n_trades"] == 0
    assert m["win_rate"] == 0.0


def test_generate_full_report(tmp_path):
    """La generazione deve produrre PNG, Markdown e CSV su disco."""
    eq = _equity(list(np.linspace(10000, 11000, 60)))
    trades = pd.DataFrame({"return": np.random.default_rng(0).normal(0.001, 0.01, 30)})
    result = BacktestResult(trades=trades, equity=eq,
                            returns=trades["return"].to_numpy())

    class _Strat:        # strategia fittizia minimale per il test
        name = "TEST_LONG"
        cluster_id = 3
        direction = 1

    out = ReportGenerator(reports_dir=tmp_path).generate(_Strat(), result, validation=None)
    assert out["markdown"].exists()
    assert out["equity"].exists() and out["equity"].suffix == ".png"
    assert out["drawdown"].exists()
    assert out["distribution"].exists()
    assert out["metrics_csv"].exists()
    # Il report deve contenere le sezioni chiave.
    text = out["markdown"].read_text()
    assert "Metriche di performance" in text
    assert "Limiti e avvertenze" in text
    assert "Sharpe Ratio" in text
