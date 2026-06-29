"""
Test automatici del Modulo 4 - Strategy Generator.

Usiamo scenari di prezzo COSTRUITI A MANO per verificare in modo deterministico
la meccanica del backtester (Take Profit, Stop Loss, time-stop, trailing) e poi
un test d'integrazione completo Moduli 1-2-3-4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_ai.data_engine import generate_ohlcv
from trading_ai.feature_engineering import FeatureEngine
from trading_ai.pattern_discovery import PatternDiscovery
from trading_ai.strategy_generator import (
    CostModel, RiskParams, StrategyGenerator, backtest, summarize_backtest,
)


def _make_df(prices_hlc: list[tuple[float, float, float]]) -> pd.DataFrame:
    """Costruisce un DataFrame OHLC da una lista di (high, low, close)."""
    idx = pd.date_range("2020-01-01", periods=len(prices_hlc), freq="h")
    high = [p[0] for p in prices_hlc]
    low = [p[1] for p in prices_hlc]
    close = [p[2] for p in prices_hlc]
    open_ = [close[0]] + close[:-1]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close},
                        index=idx)


def test_long_take_profit_hit():
    """Long con TP a 2*ATR: un rialzo deve chiudere in TP, non in stop."""
    # ATR fisso = 1.0. Entry alla barra 0 (close=100). TP=102, SL=98.
    df = _make_df([
        (100, 100, 100),   # barra 0: ingresso
        (103, 100, 102),   # barra 1: high 103 >= TP 102 -> chiude in TP
        (102, 99, 101),
    ])
    atr = pd.Series(1.0, index=df.index)
    entries = pd.Series([True, False, False], index=df.index)
    risk = RiskParams(sl_atr=2, tp_atr=2, be_atr=0, trail_atr=0, max_bars=5)
    res = backtest(df, entries, direction=1, atr=atr, risk=risk)
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["exit"] == pytest.approx(102.0)   # uscita esatta in TP
    assert res.trades.iloc[0]["return"] > 0                     # trade vincente


def test_long_stop_loss_hit():
    """Long: un ribasso deve chiudere in Stop Loss con perdita."""
    df = _make_df([
        (100, 100, 100),   # ingresso a 100, SL=98
        (100, 97, 98),     # low 97 <= SL 98 -> stop
        (99, 98, 99),
    ])
    atr = pd.Series(1.0, index=df.index)
    entries = pd.Series([True, False, False], index=df.index)
    risk = RiskParams(sl_atr=2, tp_atr=5, be_atr=0, trail_atr=0, max_bars=5)
    res = backtest(df, entries, direction=1, atr=atr, risk=risk)
    assert res.trades.iloc[0]["exit"] == pytest.approx(98.0)
    assert res.trades.iloc[0]["return"] < 0


def test_time_stop_exit():
    """Senza toccare barriere, il trade chiude al time-stop (alla close)."""
    df = _make_df([
        (100, 100, 100),   # ingresso
        (100.5, 99.5, 100.2),
        (100.4, 99.8, 100.1),   # max_bars=2 -> chiude qui alla close 100.1
        (100.3, 99.9, 100.0),
    ])
    atr = pd.Series(1.0, index=df.index)
    entries = pd.Series([True, False, False, False], index=df.index)
    risk = RiskParams(sl_atr=5, tp_atr=5, be_atr=0, trail_atr=0, max_bars=2)
    res = backtest(df, entries, direction=1, atr=atr, risk=risk)
    assert res.trades.iloc[0]["bars"] == 2
    assert res.trades.iloc[0]["exit"] == pytest.approx(100.1)


def test_breakeven_protects_profit():
    """Dopo BE, uno scivolone fino all'ingresso chiude a pareggio (~0), non in perdita."""
    df = _make_df([
        (100, 100, 100),    # ingresso a 100; BE a +1 ATR (>=101)
        (101.5, 100.5, 101),  # sale a 101.5 -> BE attivato, SL spostato a 100
        (101, 99.5, 100),     # scende: low 99.5 <= SL(=100) -> esce a 100 (pareggio)
    ])
    atr = pd.Series(1.0, index=df.index)
    entries = pd.Series([True, False, False], index=df.index)
    risk = RiskParams(sl_atr=3, tp_atr=10, be_atr=1, trail_atr=0, max_bars=5)
    res = backtest(df, entries, direction=1, atr=atr, risk=risk)
    assert res.trades.iloc[0]["exit"] == pytest.approx(100.0)   # uscita a pareggio
    assert res.trades.iloc[0]["return"] == pytest.approx(0.0)


def test_no_overlapping_positions():
    """Con segnali consecutivi, non si aprono posizioni sovrapposte."""
    df = _make_df([(100, 100, 100)] + [(101, 99, 100)] * 10)
    atr = pd.Series(1.0, index=df.index)
    entries = pd.Series([True] * 11, index=df.index)  # segnale ovunque
    risk = RiskParams(sl_atr=5, tp_atr=5, be_atr=0, trail_atr=0, max_bars=3)
    res = backtest(df, entries, direction=1, atr=atr, risk=risk)
    # I trade non si sovrappongono: ogni exit_time <= entry_time successivo.
    t = res.trades
    for k in range(len(t) - 1):
        assert t.iloc[k]["exit_time"] <= t.iloc[k + 1]["entry_time"]


def test_summarize_empty():
    """Il riepilogo gestisce il caso senza trade senza crashare."""
    df = _make_df([(100, 100, 100), (100, 100, 100)])
    atr = pd.Series(1.0, index=df.index)
    entries = pd.Series([False, False], index=df.index)
    res = backtest(df, entries, direction=1, atr=atr, risk=RiskParams())
    summ = summarize_backtest(res)
    assert summ["n_trades"] == 0


def test_cost_model_return():
    """Il costo per trade combina spread, slippage e commissioni correttamente."""
    cm = CostModel(spread=0.0002, slippage=0.0001, commission=0.00005)
    # cost = (0.0002 + 2*0.0001)/100 + 0.00005 = 0.0004/100 + 0.00005
    assert cm.cost_return(100.0) == pytest.approx(0.0004 / 100.0 + 0.00005)


def test_costs_reduce_returns():
    """Con costi, lo stesso trade vincente rende meno (drag) rispetto a senza."""
    df = _make_df([
        (100, 100, 100),
        (103, 100, 102),   # TP a 102
        (102, 99, 101),
    ])
    atr = pd.Series(1.0, index=df.index)
    entries = pd.Series([True, False, False], index=df.index)
    risk = RiskParams(sl_atr=2, tp_atr=2, be_atr=0, trail_atr=0, max_bars=5)
    free = backtest(df, entries, 1, atr, risk)
    costed = backtest(df, entries, 1, atr, risk,
                      costs=CostModel(spread=0.5, slippage=0.1, commission=0.001))
    assert costed.trades.iloc[0]["return"] < free.trades.iloc[0]["return"]
    # Il rendimento lordo resta invariato; cambia solo il netto.
    assert costed.trades.iloc[0]["gross_return"] == pytest.approx(free.trades.iloc[0]["return"])


def test_costs_can_flip_marginal_trade_to_loss():
    """Costi alti possono trasformare un piccolo guadagno lordo in perdita netta."""
    df = _make_df([
        (100, 100, 100),
        (100.6, 100, 100.5),   # piccolo TP a +0.5
        (100.5, 100, 100.4),
    ])
    atr = pd.Series(0.25, index=df.index)
    entries = pd.Series([True, False, False], index=df.index)
    risk = RiskParams(sl_atr=2, tp_atr=2, be_atr=0, trail_atr=0, max_bars=5)
    costed = backtest(df, entries, 1, atr, risk,
                      costs=CostModel(spread=0.8))   # spread maggiore del guadagno
    assert costed.trades.iloc[0]["return"] < 0


def test_end_to_end_pipeline():
    """Integrazione Moduli 1->4: dai dati grezzi alle strategie backtestate."""
    df = generate_ohlcv(n=8000, seed=21)
    feats = FeatureEngine().transform(df, dropna=True)
    disc = PatternDiscovery(n_clusters=10, horizon=8, min_count_test=10,
                            min_profit_factor=1.0)
    disc.discover(feats)
    gen = StrategyGenerator(disc)
    gen.build()
    table = gen.backtest_all(feats)
    # Se ci sono pattern stabili, la tabella ha le metriche attese.
    if not table.empty:
        for col in ["name", "n_trades", "win_rate", "profit_factor",
                    "total_return", "max_drawdown"]:
            assert col in table.columns
