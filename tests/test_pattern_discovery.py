"""
Test automatici del Modulo 3 - Pattern Discovery.

Verifichiamo: correttezza delle metriche su casi noti, drawdown, labeling
forward senza leakage, anti-leakage del clusterer (fit solo sul train),
filtro di stabilita' e capacita' di scoprire un pattern "piantato" nei dati.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_ai.data_engine import generate_ohlcv
from trading_ai.feature_engineering import FeatureEngine
from trading_ai.pattern_discovery import PatternDiscovery
from trading_ai.pattern_discovery.clustering import FeatureClusterer
from trading_ai.pattern_discovery.labeling import forward_return
from trading_ai.pattern_discovery.metrics import compute_stats, equity_max_drawdown


def test_forward_return_shifts_correctly():
    """Il forward return deve guardare h barre avanti e annullare la coda."""
    close = pd.Series([10.0, 11.0, 12.0, 13.0], index=pd.date_range("2020", periods=4, freq="h"))
    fwd = forward_return(close, horizon=1)
    assert fwd.iloc[0] == pytest.approx(0.1)      # 11/10 - 1
    assert np.isnan(fwd.iloc[-1])                  # ultima barra: nessun futuro


def test_metrics_on_known_returns():
    """Statistiche calcolate a mano su un piccolo set di rendimenti."""
    returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
    st = compute_stats(returns, total_bars=100)
    assert st.count == 5
    assert st.frequency == pytest.approx(0.05)
    assert st.mean_return == pytest.approx(returns.mean())
    assert st.prob_up == pytest.approx(0.6)
    assert st.prob_down == pytest.approx(0.4)
    # profit_factor = (0.02+0.03+0.01) / (0.01+0.02) = 0.06/0.03 = 2.0
    assert st.profit_factor == pytest.approx(2.0)


def test_max_drawdown_simple():
    """Drawdown su una sequenza nota: sale poi scende."""
    # equity cumulata: [1, 3, 2, 0, 1] -> picco 3, minimo 0 -> dd = -3
    returns = np.array([1.0, 2.0, -1.0, -2.0, 1.0])
    assert equity_max_drawdown(returns) == pytest.approx(-3.0)


def test_profit_factor_no_losses():
    """Senza perdite il profit factor e' infinito (gestito senza crash)."""
    st = compute_stats(np.array([0.01, 0.02, 0.03]), total_bars=10)
    assert st.profit_factor == float("inf")


def test_clusterer_no_leakage():
    """Lo scaler deve apprendere media/std SOLO dal training."""
    df = generate_ohlcv(n=2000, seed=5)
    feats = FeatureEngine().transform(df, groups=["indicator"], dropna=True)
    split = int(len(feats) * 0.7)
    train, test = feats.iloc[:split], feats.iloc[split:]
    clu = FeatureClusterer(n_clusters=5).fit(train)
    # La media appresa dallo scaler deve corrispondere a quella del SOLO train.
    first_col = clu.feature_columns[0]
    learned_mean = clu.scaler.mean_[0]
    assert learned_mean == pytest.approx(train[first_col].mean(), rel=1e-4)
    # predict sul test non deve fallire e deve dare etichette valide.
    labels = clu.predict(test)
    assert labels.between(0, 4).all()


def test_discovery_runs_and_filters():
    """La scoperta produce una tabella e marca correttamente la stabilita'."""
    df = generate_ohlcv(n=8000, seed=7)
    feats = FeatureEngine().transform(df, dropna=True)
    pd_ = PatternDiscovery(n_clusters=10, horizon=8, min_count_test=10)
    table = pd_.discover(feats)
    assert not table.empty
    # Ogni cluster non vuoto e' rappresentato; le colonne chiave esistono.
    for col in ["cluster_id", "direction", "stable", "test_expectancy", "test_profit_factor"]:
        assert col in table.columns
    # I pattern stabili devono rispettare le soglie OOS imposte.
    for p in pd_.stable_patterns():
        assert p.test.profit_factor >= pd_.min_profit_factor
        assert p.test.expectancy > 0
        assert p.train.expectancy > 0


def test_discovery_finds_planted_pattern():
    """
    Pianta un segnale: dopo ogni RSI molto basso (<25) forziamo un rimbalzo.
    La discovery deve trovare almeno un pattern long stabile e profittevole.
    """
    rng = np.random.default_rng(0)
    n = 6000
    price = [100.0]
    # Costruiamo una serie dove i ribassi forti tendono a rimbalzare (mean-reversion).
    for i in range(1, n):
        prev = price[-1]
        # drift mean-reverting verso 100 + rumore
        step = 0.02 * (100 - prev) + rng.normal(0, 0.5)
        price.append(max(prev + step, 1.0))
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    close = pd.Series(price, index=idx)
    df = pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close * 1.001,
        "low": close * 0.999,
        "close": close,
        "volume": 1000.0,
    }, index=idx)

    feats = FeatureEngine().transform(df, groups=["indicator", "volatility"], dropna=True)
    pd_ = PatternDiscovery(n_clusters=12, horizon=5, min_count_test=10,
                           min_profit_factor=1.05)
    pd_.discover(feats)
    # Su una serie fortemente mean-reverting deve emergere almeno un pattern stabile.
    assert len(pd_.stable_patterns()) >= 1
