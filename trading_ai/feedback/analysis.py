"""
Analisi diagnostica di un backtest (Modulo 7).

Trasforma le metriche di una strategia in DIAGNOSI leggibili e in suggerimenti
di direzione per i parametri. E' la parte "analizza gli errori e propone
miglioramenti" del modulo di feedback.
"""

from __future__ import annotations


def analyze(metrics: dict) -> dict:
    """
    Esamina le metriche e produce diagnosi + suggerimenti di modifica.

    Ritorna
    -------
    dict con:
      'issues'      : lista di problemi rilevati (testo)
      'suggestions' : lista di azioni proposte (testo)
      'nudges'      : dizionario di "spinte" sui parametri (es. {'tp_atr': +1})
    """
    issues: list[str] = []
    suggestions: list[str] = []
    nudges: dict[str, float] = {}

    n = metrics.get("n_trades", 0)
    win = metrics.get("win_rate", 0.0)
    pf = metrics.get("profit_factor", 0.0)
    mdd = metrics.get("max_drawdown", 0.0)
    avg_win = metrics.get("avg_win", 0.0)
    avg_loss = metrics.get("avg_loss", 0.0)
    expectancy = metrics.get("expectancy", 0.0)

    # Pochi trade -> metriche poco affidabili.
    if n < 30:
        issues.append(f"Campione ridotto ({n} trade): bassa significativita'.")
        suggestions.append("Allargare il periodo dati o allentare i filtri.")

    # Win rate basso ma payoff potenzialmente alto -> alza il TP / abbassa lo SL.
    if win < 0.4:
        issues.append(f"Win rate basso ({win:.0%}).")
        suggestions.append("Valutare TP piu' vicino o filtri d'ingresso piu' selettivi.")
        nudges["tp_atr"] = -1.0

    # Profit factor debole -> il rapporto vincite/perdite non paga.
    if pf < 1.2:
        issues.append(f"Profit factor debole ({pf:.2f}).")
        suggestions.append("Aumentare il rapporto rischio/rendimento (TP/SL).")
        nudges["tp_atr"] = nudges.get("tp_atr", 0.0) + 1.0

    # Drawdown elevato -> riduci il rischio per trade e stringi lo stop.
    if mdd < -0.25:
        issues.append(f"Drawdown elevato ({mdd:.0%}).")
        suggestions.append("Ridurre il rischio per trade e/o stringere lo stop.")
        nudges["sl_atr"] = -0.5

    # Perdite medie molto piu' grandi delle vincite medie -> taglia prima le perdite.
    if avg_loss != 0 and abs(avg_loss) > 1.8 * max(avg_win, 1e-9):
        issues.append("Le perdite medie superano nettamente le vincite medie.")
        suggestions.append("Introdurre/anticipare break-even e trailing stop.")
        nudges["be_atr"] = -0.5
        nudges["trail_atr"] = 1.0

    if expectancy <= 0:
        issues.append("Expectancy non positiva: la strategia non ha edge attuale.")
        suggestions.append("Rivedere il pattern d'ingresso o scartare la strategia.")

    if not issues:
        issues.append("Nessuna criticita' grave rilevata.")
        suggestions.append("Tentare un fine-tuning dei parametri di rischio.")

    return {"issues": issues, "suggestions": suggestions, "nudges": nudges}
