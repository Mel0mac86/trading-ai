"""
Modulo 4 - Strategy Generator
============================

Trasforma i PATTERN STABILI scoperti dal Modulo 3 in STRATEGIE complete ed
eseguibili, e le misura con un backtester realistico.

Una strategia = un pattern (cluster + direzione) + regole di rischio
(SL/TP/BE/trailing/sizing) + filtri di contesto (trend, volatilita', orari,
max operazioni). L'ingresso e' "la barra appartiene al cluster del pattern";
l'uscita e' gestita dal backtester (Stop/Target/Break-Even/Trailing/Time-stop).

Uso rapido
----------
    gen = StrategyGenerator(pattern_discovery)   # PD gia' .discover()-ato
    strategies = gen.build()                     # una strategia per pattern stabile
    summary = gen.backtest_all(features)         # tabella metriche per strategia
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trading_ai.pattern_discovery import PatternDiscovery
from trading_ai.pattern_discovery.metrics import compute_stats, equity_max_drawdown
from trading_ai.strategy_generator.backtest import BacktestResult, backtest
from trading_ai.strategy_generator.risk import CostModel, Filters, RiskParams
from trading_ai.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["Strategy", "StrategyGenerator", "RiskParams", "Filters", "CostModel",
           "backtest", "BacktestResult"]


@dataclass
class Strategy:
    """Una strategia completa, autoconsistente ed eseguibile."""

    name: str                              # identificativo leggibile
    cluster_id: int                        # pattern di riferimento (cluster KMeans)
    direction: int                         # +1 long / -1 short
    clusterer: object                      # FeatureClusterer gia' addestrato (per i segnali)
    risk: RiskParams = field(default_factory=RiskParams)
    filters: Filters = field(default_factory=Filters)
    costs: CostModel = field(default_factory=CostModel)  # costi di transazione (spread/slippage/commissioni)

    # --- Generazione segnali -------------------------------------------------
    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """
        Ritorna una Series booleana: True dove la strategia entra.
        Segnale base = appartenenza al cluster; poi si applicano i filtri.
        """
        clusters = self.clusterer.predict(features)        # cluster per ogni barra
        signal = (clusters == self.cluster_id)             # ingresso grezzo del pattern
        return self._apply_filters(features, signal)

    def _apply_filters(self, feats: pd.DataFrame, signal: pd.Series) -> pd.Series:
        """Applica i filtri di contesto, restringendo i segnali."""
        f = self.filters
        out = signal.copy()

        # Filtro di trend: opera solo se la forza/direzione del trend concorda.
        if f.use_trend and "trend_strength" in feats.columns:
            trend_ok = np.sign(feats["trend_strength"]) == self.direction
            out &= trend_ok

        # Filtro ADX: richiede un trend abbastanza forte.
        if f.min_adx > 0 and "adx_adx" in feats.columns:
            out &= feats["adx_adx"] >= f.min_adx

        # Filtro volatilita': evita le fasi troppo volatili (ATR% sopra soglia).
        if f.max_volatility > 0 and "volatility_atr_pct" in feats.columns:
            out &= feats["volatility_atr_pct"] <= f.max_volatility

        # Filtro orario: opera solo nella finestra [start_hour, end_hour].
        if f.start_hour is not None and f.end_hour is not None:
            hours = feats.index.hour
            out &= (hours >= f.start_hour) & (hours <= f.end_hour)

        return out.fillna(False)

    # --- Esecuzione ----------------------------------------------------------
    def run(self, features: pd.DataFrame, initial_equity: float = 10_000.0) -> BacktestResult:
        """Genera i segnali e li backtesta. Richiede la colonna 'atr' nelle feature."""
        if "atr" not in features.columns:
            raise ValueError("Le feature devono includere 'atr' (Modulo 2) per il backtest.")
        signals = self.generate_signals(features)
        return backtest(features, signals, self.direction,
                        features["atr"], self.risk, initial_equity, self.costs)

    def describe(self) -> dict:
        """Riassunto serializzabile della strategia (per i report)."""
        return {
            "name": self.name, "cluster_id": self.cluster_id,
            "direction": self.direction,
            "risk": self.risk.as_dict(), "filters": self.filters.as_dict(),
            "costs": self.costs.as_dict(),
        }


def summarize_backtest(result: BacktestResult) -> dict:
    """
    Metriche aggregate di un backtest (anticipo del Modulo 9).
    Combina le statistiche per-trade con il drawdown sull'equity compounded.
    """
    n = len(result.trades)
    if n == 0:
        return {"n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "expectancy": 0.0, "total_return": 0.0, "max_drawdown": 0.0,
                "sharpe_like": 0.0}
    stats = compute_stats(result.returns, total_bars=n)   # PF, expectancy, sharpe-like...
    eq = result.equity
    total_return = eq.iloc[-1] / eq.iloc[0] - 1.0          # rendimento complessivo
    # Drawdown sull'equity reale (relativo, in frazione del picco).
    running_max = eq.cummax()
    dd = ((eq - running_max) / running_max).min()
    return {
        "n_trades": n,
        "win_rate": float((result.returns > 0).mean()),
        "profit_factor": stats.profit_factor,
        "expectancy": stats.expectancy,
        "total_return": float(total_return),
        "max_drawdown": float(dd),
        "sharpe_like": stats.sharpe_like,
    }


class StrategyGenerator:
    """Costruisce strategie dai pattern stabili e le backtesta."""

    def __init__(self, discovery: PatternDiscovery,
                 risk: RiskParams | None = None, filters: Filters | None = None,
                 costs: CostModel | None = None):
        if discovery.clusterer is None:
            raise ValueError("Esegui prima discovery.discover() sul Modulo 3.")
        self.discovery = discovery
        self.risk = risk or RiskParams()
        self.filters = filters or Filters()
        self.costs = costs or CostModel()
        self.strategies: list[Strategy] = []

    def build(self) -> list[Strategy]:
        """Una strategia per ogni pattern STABILE trovato dal Modulo 3."""
        self.strategies = []
        for p in self.discovery.stable_patterns():
            strat = Strategy(
                name=f"PAT{p.cluster_id:02d}_{'LONG' if p.direction == 1 else 'SHORT'}",
                cluster_id=p.cluster_id, direction=p.direction,
                clusterer=self.discovery.clusterer,
                risk=self.risk, filters=self.filters, costs=self.costs,
            )
            self.strategies.append(strat)
        logger.info("Strategie costruite: %d", len(self.strategies))
        return self.strategies

    def backtest_all(self, features: pd.DataFrame,
                     initial_equity: float = 10_000.0) -> pd.DataFrame:
        """Backtesta tutte le strategie e ritorna una tabella di metriche."""
        if not self.strategies:
            self.build()
        rows = []
        for strat in self.strategies:
            res = strat.run(features, initial_equity)
            row = {"name": strat.name, "direction": strat.direction}
            row.update(summarize_backtest(res))
            rows.append(row)
        table = pd.DataFrame(rows)
        if not table.empty:
            table = table.sort_values("total_return", ascending=False).reset_index(drop=True)
        return table
