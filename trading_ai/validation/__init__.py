"""
Modulo 5 - Validation
====================

Il "guardiano" anti-illusione: sottopone ogni strategia a una batteria di test
e ne scarta AUTOMATICAMENTE quelle non robuste. Combina:

  - Out-of-Sample / Walk-Forward : profittevole in molte finestre temporali?
  - Monte Carlo                  : profittevole in molti riordinamenti dei trade?
  - Robustness                   : profittevole anche perturbando i parametri?
  - Sensitivity                  : performance poco sensibile ai parametri?

Una strategia e' "robusta" solo se supera TUTTE le soglie. Cosi' riduciamo
drasticamente la probabilita' di portare in produzione un overfitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trading_ai.strategy_generator import summarize_backtest
from trading_ai.validation.monte_carlo import monte_carlo_trades
from trading_ai.validation.robustness import parameter_robustness
from trading_ai.validation.walk_forward import walk_forward
from trading_ai.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["Validator", "ValidationReport", "monte_carlo_trades",
           "walk_forward", "parameter_robustness"]


@dataclass
class ValidationReport:
    """Esito completo della validazione di una strategia."""

    name: str
    robust: bool                       # verdetto finale
    reasons: list[str] = field(default_factory=list)  # perche' (non) e' robusta
    backtest: dict = field(default_factory=dict)
    monte_carlo: dict = field(default_factory=dict)
    walk_forward: dict = field(default_factory=dict)
    robustness: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """Versione piatta e serializzabile (per i report del Modulo 9)."""
        return {
            "name": self.name, "robust": self.robust,
            "reasons": "; ".join(self.reasons),
            "bt_total_return": self.backtest.get("total_return"),
            "bt_profit_factor": self.backtest.get("profit_factor"),
            "bt_n_trades": self.backtest.get("n_trades"),
            "mc_prob_profit": self.monte_carlo.get("prob_profit"),
            "mc_maxdd_p95": self.monte_carlo.get("maxdd_p95"),
            "wf_consistency": self.walk_forward.get("consistency"),
            "rob_profitable_fraction": self.robustness.get("profitable_fraction"),
            "rob_sensitivity": self.robustness.get("sensitivity"),
        }


class Validator:
    """Applica la batteria di test e produce un verdetto robusto/non robusto."""

    def __init__(
        self,
        min_trades: int = 30,             # minimo di trade per significativita' statistica
        min_mc_prob_profit: float = 0.60,  # prob. di profitto Monte Carlo
        min_wf_consistency: float = 0.50,  # frazione minima di finestre profittevoli
        min_robust_fraction: float = 0.60,  # frazione minima di varianti profittevoli
        max_sensitivity: float = 1.50,    # dispersione massima ammessa ai parametri
        n_splits: int = 5,
        n_mc_sims: int = 2000,
    ):
        self.min_trades = min_trades
        self.min_mc_prob_profit = min_mc_prob_profit
        self.min_wf_consistency = min_wf_consistency
        self.min_robust_fraction = min_robust_fraction
        self.max_sensitivity = max_sensitivity
        self.n_splits = n_splits
        self.n_mc_sims = n_mc_sims

    def validate(self, strategy, features: pd.DataFrame) -> ValidationReport:
        """Esegue tutti i test su una strategia e ne decide la robustezza."""
        reasons: list[str] = []

        # --- Backtest di riferimento (intero periodo) -----------------------
        result = strategy.run(features)
        bt = summarize_backtest(result)

        # Gate iniziale: senza abbastanza trade non si puo' validare nulla.
        if bt["n_trades"] < self.min_trades:
            reasons.append(f"trade insufficienti ({bt['n_trades']} < {self.min_trades})")
            return ValidationReport(strategy.name, False, reasons, bt)

        # --- Monte Carlo -----------------------------------------------------
        mc = monte_carlo_trades(
            result.trades["r_multiple"].to_numpy(),
            risk_per_trade=strategy.risk.risk_per_trade,
            n_sims=self.n_mc_sims,
        )
        if mc["prob_profit"] < self.min_mc_prob_profit:
            reasons.append(
                f"Monte Carlo prob_profit {mc['prob_profit']:.2f} < {self.min_mc_prob_profit}")

        # --- Walk-Forward ----------------------------------------------------
        wf = walk_forward(strategy, features, n_splits=self.n_splits)
        if wf["consistency"] < self.min_wf_consistency:
            reasons.append(
                f"walk-forward consistency {wf['consistency']:.2f} < {self.min_wf_consistency}")

        # --- Robustness + Sensitivity ---------------------------------------
        rob = parameter_robustness(strategy, features)
        if rob["profitable_fraction"] < self.min_robust_fraction:
            reasons.append(
                f"robustezza {rob['profitable_fraction']:.2f} < {self.min_robust_fraction}")
        if rob["sensitivity"] > self.max_sensitivity:
            reasons.append(
                f"sensibilita' {rob['sensitivity']:.2f} > {self.max_sensitivity}")

        # Verdetto: robusta solo se NESSUN test ha aggiunto un motivo di scarto.
        robust = len(reasons) == 0
        if robust:
            reasons.append("supera tutti i test di robustezza")
        logger.info("Validazione %s -> %s", strategy.name,
                    "ROBUSTA" if robust else "SCARTATA")
        return ValidationReport(strategy.name, robust, reasons, bt, mc, wf, rob)

    def validate_many(self, strategies: list, features: pd.DataFrame) -> pd.DataFrame:
        """Valida una lista di strategie e ritorna una tabella riassuntiva."""
        reports = [self.validate(s, features) for s in strategies]
        return pd.DataFrame([r.summary() for r in reports])
