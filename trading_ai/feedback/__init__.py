"""
Modulo 7 - AI Feedback
=====================

Dopo ogni backtest: analizza gli errori, propone miglioramenti, ottimizza
automaticamente i parametri, crea una NUOVA VERSIONE della strategia e mantiene
lo STORICO delle versioni.

Il ciclo di feedback e' volutamente CONSERVATIVO: l'ottimizzazione usa una
griglia grossolana e un obiettivo che penalizza il drawdown, per evitare di
"inseguire il rumore" (overfitting). Ogni versione conserva metriche, parametri
e il riferimento alla versione genitrice, cosi' l'evoluzione e' tracciabile.

Uso rapido
----------
    fb = FeedbackEngine()
    result = fb.improve(strategy, features)
    print(result.versions[-1].metrics)     # metriche della versione migliorata
    fb.save_history("strategies/PAT00_LONG_history.json")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from trading_ai.feedback.analysis import analyze
from trading_ai.feedback.optimizer import optimize_risk
from trading_ai.strategy_generator import summarize_backtest
from trading_ai.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["FeedbackEngine", "StrategyVersion", "FeedbackResult",
           "analyze", "optimize_risk"]


@dataclass
class StrategyVersion:
    """Una versione di strategia con i suoi parametri e le sue metriche."""

    version: int                       # numero progressivo (0 = baseline)
    name: str
    risk: dict                         # parametri di rischio (serializzati)
    metrics: dict                      # metriche del backtest di questa versione
    parent: int | None = None          # versione da cui deriva (None per la 0)
    note: str = ""                     # motivazione del cambiamento
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeedbackResult:
    """Esito di un ciclo di feedback su una strategia."""

    strategy_name: str
    diagnosis: dict                    # output di analyze() sulla baseline
    versions: list[StrategyVersion]    # storico (baseline + migliorate)
    improved_strategy: object          # nuova strategia con i parametri ottimizzati
    optimization_table: pd.DataFrame   # griglia esplorata


class FeedbackEngine:
    """Motore di feedback: analizza, ottimizza, versiona."""

    def __init__(self, min_trades: int = 30):
        self.min_trades = min_trades
        # Storico globale di tutte le versioni viste (anche tra strategie diverse).
        self.history: list[StrategyVersion] = []

    def improve(self, strategy, features: pd.DataFrame) -> FeedbackResult:
        """
        Esegue un ciclo completo: baseline -> diagnosi -> ottimizzazione ->
        nuova versione, registrando tutto nello storico.
        """
        # --- Versione 0: baseline -------------------------------------------
        base_summary = summarize_backtest(strategy.run(features))
        diagnosis = analyze(base_summary)               # analizza gli errori
        v0 = StrategyVersion(0, strategy.name, strategy.risk.as_dict(),
                             base_summary, parent=None, note="baseline")

        # --- Ottimizzazione robusta dei parametri ---------------------------
        opt = optimize_risk(strategy, features, min_trades=self.min_trades)
        improved = replace(strategy, risk=opt["best_risk"])  # nuova strategia
        improved_summary = opt["best_summary"] or base_summary

        v1 = StrategyVersion(
            1, strategy.name, opt["best_risk"].as_dict(), improved_summary,
            parent=0,
            note="ottimizzazione risk-params; " + "; ".join(diagnosis["suggestions"][:2]),
        )

        versions = [v0, v1]
        self.history.extend(versions)
        logger.info(
            "Feedback %s: baseline ret %.4f -> migliorata ret %.4f",
            strategy.name, base_summary["total_return"],
            improved_summary["total_return"],
        )
        return FeedbackResult(strategy.name, diagnosis, versions, improved,
                              opt["table"])

    def iterate(self, strategy, features: pd.DataFrame, rounds: int = 2):
        """
        Applica piu' cicli di feedback in sequenza, ciascuno partendo dalla
        strategia migliorata al ciclo precedente. Ritorna l'ultima FeedbackResult.
        """
        result = None
        current = strategy
        for _ in range(max(1, rounds)):
            result = self.improve(current, features)
            current = result.improved_strategy
        return result

    # --- Persistenza dello storico ------------------------------------------
    def save_history(self, path: str | Path) -> Path:
        """Salva l'intero storico delle versioni in JSON (leggibile/versionabile)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [v.as_dict() for v in self.history]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @staticmethod
    def load_history(path: str | Path) -> list[StrategyVersion]:
        """Ricarica uno storico salvato."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return [StrategyVersion(**d) for d in data]
