"""
Test automatici del Modulo 2 - Feature Engineering.

Verifichiamo correttezza degli indicatori (contro implementazioni di
riferimento semplici), forma dei pattern, causalita' anti-leakage della
market structure, estensibilita' del registry e l'integrazione end-to-end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_ai.data_engine import generate_ohlcv
from trading_ai.feature_engineering import FeatureEngine, list_features
from trading_ai.feature_engineering.indicators import atr, ema, rsi, sma, true_range
from trading_ai.feature_engineering.market_structure import (
    causal_levels, structure_events, fair_value_gap,
)
from trading_ai.feature_engineering.registry import FEATURE_REGISTRY, feature


@pytest.fixture
def df() -> pd.DataFrame:
    return generate_ohlcv(n=3_000, seed=99)


def test_sma_matches_reference(df):
    """La SMA deve coincidere con la media mobile di pandas."""
    out = sma(df["close"], 20)
    ref = df["close"].rolling(20).mean()
    pd.testing.assert_series_equal(out, ref, check_names=False)


def test_ema_first_valid_equals_seed(df):
    """Con adjust=False, il primo valore EMA coincide col primo prezzo."""
    out = ema(df["close"], 10)
    assert out.iloc[0] == pytest.approx(df["close"].iloc[0])


def test_rsi_bounded_0_100(df):
    """L'RSI deve restare nell'intervallo [0, 100]."""
    out = rsi(df["close"], 14).dropna()
    assert out.min() >= 0.0
    assert out.max() <= 100.0


def test_atr_positive_and_ge_componentwise(df):
    """ATR deve essere positivo e il True Range >= range della barra."""
    tr = true_range(df["high"], df["low"], df["close"])
    assert (tr.dropna() >= (df["high"] - df["low"]).loc[tr.dropna().index] - 1e-9).all()
    a = atr(df["high"], df["low"], df["close"], 14).dropna()
    assert (a > 0).all()


def test_fvg_detects_known_gap():
    """Costruiamo un gap rialzista esplicito e verifichiamo il rilevamento."""
    idx = pd.date_range("2020", periods=4, freq="h")
    data = pd.DataFrame({
        "open": [10, 10, 12, 13],
        "high": [10.5, 11.0, 12.5, 13.5],  # high[0]=10.5
        "low":  [9.5, 10.0, 11.0, 12.0],   # low[2]=11.0 > high[0]=10.5 -> bullish FVG
        "close": [10, 10.8, 12.2, 13.0],
        "volume": [1, 1, 1, 1],
    }, index=idx)
    out = fair_value_gap(data)
    assert out["fvg"].iloc[2] == 1            # FVG rialzista alla terza candela
    assert out["fvg_size"].iloc[2] > 0


def test_causal_levels_no_lookahead(df):
    """
    I livelli causali NON devono cambiare se azzeriamo i dati FUTURI:
    proprieta' chiave per escludere il data leakage.
    """
    full = causal_levels(df, k=2)
    cutoff = 1500
    # Ricalcoliamo usando solo i dati fino a cutoff: i valori <= cutoff-k
    # devono restare identici (non dipendono dal futuro).
    partial = causal_levels(df.iloc[:cutoff], k=2)
    # Confrontiamo una zona ben dentro la parte comune (oltre il warm-up).
    a = full["last_swing_high"].iloc[100:cutoff - 10]
    b = partial["last_swing_high"].iloc[100:cutoff - 10]
    pd.testing.assert_series_equal(a, b, check_freq=False)


def test_structure_events_values(df):
    """BOS/CHoCH/trend devono assumere solo valori ammessi."""
    ev = structure_events(df, k=2)
    assert set(ev["bos"].unique()).issubset({-1, 0, 1})
    assert set(ev["choch"].unique()).issubset({-1, 0, 1})
    assert set(ev["trend"].unique()).issubset({-1, 0, 1})


def test_registry_is_populated():
    """Tutte le famiglie di feature attese devono essere registrate."""
    names = list_features()
    for expected in ["rsi", "atr", "macd", "adx", "bollinger", "vwap",
                     "cdl_engulfing", "fvg", "structure", "trend_strength"]:
        assert expected in names


def test_can_add_custom_feature(df):
    """Estensibilita': una nuova feature via decoratore deve comparire nell'output."""
    name = "test_body_size"
    if name not in FEATURE_REGISTRY:
        @feature(name, group="custom")
        def _body(d):
            return (d["close"] - d["open"]).abs()
    out = FeatureEngine().transform(df, features=[name])
    assert name in out.columns


def test_engine_end_to_end(df):
    """Smoke test: il FeatureEngine produce molte colonne senza errori."""
    out = FeatureEngine().transform(df, dropna=True)
    # OHLCV (5) + numerose feature.
    assert out.shape[1] > 25
    assert len(out) > 0
    assert not out.isna().any().any()         # dopo dropna nessun NaN residuo
