"""
Modulo 9 - Report
================

Per ogni strategia genera automaticamente un report completo: grafici (equity,
drawdown, distribuzione profitti), tabella di metriche (Sharpe, Sortino, Calmar,
Win Rate, Profit Factor, Expectancy, Recovery Factor, numero trade) e una
spiegazione in linguaggio naturale del PERCHE' la strategia e' stata selezionata
e dei suoi LIMITI (regola di progetto).

Uso rapido
----------
    from trading_ai.reporting import ReportGenerator

    rep = ReportGenerator()
    out = rep.generate(strategy, backtest_result, validation_report)
    print(out["markdown"])     # percorso del report .md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_ai.config import PATHS
from trading_ai.reporting.metrics import compute_report_metrics
from trading_ai.reporting.plots import (
    plot_drawdown, plot_equity, plot_profit_distribution,
)
from trading_ai.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["ReportGenerator", "compute_report_metrics"]


class ReportGenerator:
    """Genera report per-strategia (grafici + metriche + spiegazione)."""

    def __init__(self, reports_dir: Path | None = None):
        self.reports_dir = Path(reports_dir) if reports_dir else PATHS.reports

    def generate(self, strategy, result, validation=None) -> dict:
        """
        Crea la cartella del report, salva i grafici e scrive il file Markdown.

        Parametri
        ---------
        strategy : Strategy
            La strategia documentata.
        result : BacktestResult
            Output del backtest (equity + trade).
        validation : ValidationReport | None
            Esito del Modulo 5 (per spiegare perche' e' stata accettata/scartata).

        Ritorna
        -------
        dict con i percorsi dei file generati e le metriche calcolate.
        """
        name = getattr(strategy, "name", "strategy")
        out_dir = self.reports_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        metrics = compute_report_metrics(result.equity, result.returns)

        # --- Grafici ---------------------------------------------------------
        eq_png = plot_equity(result.equity, out_dir / "equity.png", title=f"Equity - {name}")
        dd_png = plot_drawdown(result.equity, out_dir / "drawdown.png")
        dist_png = plot_profit_distribution(result.returns, out_dir / "profit_dist.png")

        # --- Report Markdown -------------------------------------------------
        md_path = out_dir / "report.md"
        md_path.write_text(self._render_markdown(strategy, metrics, validation),
                           encoding="utf-8")

        # --- Metriche in CSV (comode per confronti) --------------------------
        csv_path = out_dir / "metrics.csv"
        pd.DataFrame([metrics]).to_csv(csv_path, index=False)

        logger.info("Report generato in %s", out_dir)
        return {"dir": out_dir, "markdown": md_path, "equity": eq_png,
                "drawdown": dd_png, "distribution": dist_png,
                "metrics_csv": csv_path, "metrics": metrics}

    # --- Composizione del testo ---------------------------------------------
    def _render_markdown(self, strategy, m: dict, validation) -> str:
        """Costruisce il contenuto Markdown del report."""
        direction = "LONG" if getattr(strategy, "direction", 1) == 1 else "SHORT"

        def pct(x: float) -> str:
            return f"{x * 100:.2f}%"

        lines = [
            f"# Report strategia: {getattr(strategy, 'name', 'strategy')}",
            "",
            f"- **Direzione:** {direction}",
            f"- **Pattern (cluster):** {getattr(strategy, 'cluster_id', '?')}",
            "",
            "## Metriche di performance",
            "",
            "| Metrica | Valore |",
            "|---|---|",
            f"| Numero di trade | {m['n_trades']} |",
            f"| Rendimento totale | {pct(m['total_return'])} |",
            f"| CAGR | {pct(m['cagr'])} |",
            f"| Sharpe Ratio | {m['sharpe']:.2f} |",
            f"| Sortino Ratio | {m['sortino']:.2f} |",
            f"| Calmar Ratio | {m['calmar']:.2f} |",
            f"| Max Drawdown | {pct(m['max_drawdown'])} |",
            f"| Recovery Factor | {m['recovery_factor']:.2f} |",
            f"| Win Rate | {pct(m['win_rate'])} |",
            f"| Profit Factor | {m['profit_factor']:.2f} |",
            f"| Expectancy (per trade) | {pct(m['expectancy'])} |",
            f"| Trade migliore / peggiore | {pct(m['best_trade'])} / {pct(m['worst_trade'])} |",
            "",
            "## Grafici",
            "",
            "![Equity](equity.png)",
            "",
            "![Drawdown](drawdown.png)",
            "",
            "![Distribuzione profitti](profit_dist.png)",
            "",
            "## Perche' e' stata selezionata",
            "",
        ]

        # Razionale di selezione dalla validazione (Modulo 5), se disponibile.
        if validation is not None:
            verdict = "ROBUSTA ✅" if validation.robust else "SCARTATA ❌"
            lines.append(f"**Verdetto di validazione:** {verdict}")
            lines.append("")
            for r in validation.reasons:
                lines.append(f"- {r}")
            if validation.monte_carlo:
                mc = validation.monte_carlo
                lines += [
                    "",
                    f"- Monte Carlo: probabilita' di profitto "
                    f"{pct(mc.get('prob_profit', 0))}, drawdown di coda "
                    f"{pct(mc.get('maxdd_p95', 0))}.",
                ]
        else:
            lines.append("_Nessun report di validazione associato._")

        # Limiti: trasparenza obbligatoria.
        lines += [
            "",
            "## Limiti e avvertenze",
            "",
            "- I risultati derivano da un BACKTEST: la performance passata non "
            "garantisce quella futura.",
            "- Costi non inclusi di default: spread/commissioni/slippage reali "
            "riducono il rendimento; testare sul broker target.",
            f"- Significativita' statistica legata al numero di trade "
            f"({m['n_trades']}): valori bassi rendono le metriche meno affidabili.",
            "- Il pattern e' stato scoperto su dati storici: monitorare il "
            "decadimento (regime change) e rivalidare periodicamente.",
            "",
        ]
        return "\n".join(lines)
