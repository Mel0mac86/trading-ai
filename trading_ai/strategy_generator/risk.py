"""
Parametri di rischio e filtri di una strategia (Modulo 4).

Separiamo la CONFIGURAZIONE (questi dataclass) dalla LOGICA (backtest.py), cosi'
gli stessi parametri sono facilmente serializzabili (per i report del Modulo 9)
e ottimizzabili (dal Modulo 7 - AI Feedback).

Tutte le distanze di SL/TP sono espresse in MULTIPLI DI ATR: cosi' la strategia
si adatta automaticamente alla volatilita' dello strumento e del periodo,
invece di usare pip fissi che non hanno senso tra strumenti diversi.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class RiskParams:
    """Gestione di stop, target, break-even, trailing e dimensionamento."""

    sl_atr: float = 2.0          # Stop Loss = sl_atr * ATR dall'ingresso
    tp_atr: float = 3.0          # Take Profit = tp_atr * ATR dall'ingresso
    be_atr: float = 1.0          # dopo +be_atr*ATR di profitto -> SL a pareggio (0 = off)
    trail_atr: float = 0.0       # trailing stop a trail_atr*ATR dal massimo (0 = off)
    risk_per_trade: float = 0.01  # frazione di equity rischiata per trade (1%)
    max_bars: int = 50           # time stop: chiusura forzata dopo N barre
    max_trades_per_day: int = 0  # 0 = nessun limite giornaliero

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class CostModel:
    """
    Costi di transazione applicati a ogni trade (spread, slippage, commissioni).

    Modelliamo i costi come un "drag" sul rendimento del trade: e' un'ottima
    approssimazione, conservativa e semplice. Tutte le grandezze di prezzo sono
    nell'unita' dello strumento (es. 0.0001 = 1 pip su EURUSD a 5 cifre).

    cost_return = (spread + 2*slippage) / prezzo_ingresso + commission
      - spread     : differenza denaro/lettera, pagata una volta a giro (round-trip)
      - slippage   : scostamento medio per lato (entrata e uscita -> *2)
      - commission : commissione round-trip come FRAZIONE del nozionale (es. 0.00007)
    """

    spread: float = 0.0          # spread in unita' di prezzo (round-trip)
    slippage: float = 0.0        # slippage in unita' di prezzo, per lato
    commission: float = 0.0      # commissione round-trip come frazione del nozionale

    def cost_return(self, entry_price: float) -> float:
        """Costo totale del trade espresso come riduzione di rendimento."""
        if entry_price <= 0:
            return self.commission
        price_cost = (self.spread + 2.0 * self.slippage) / entry_price
        return price_cost + self.commission

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Filters:
    """Filtri di contesto che possono BLOCCARE un ingresso valido del pattern."""

    use_trend: bool = False      # se True, entra solo se il trend concorda con la direzione
    min_adx: float = 0.0         # ADX minimo richiesto (0 = filtro disattivo)
    max_volatility: float = 0.0  # ATR% massimo consentito (0 = disattivo)
    start_hour: int | None = None  # ora minima (inclusa) per operare, 0-23
    end_hour: int | None = None    # ora massima (inclusa) per operare, 0-23

    def as_dict(self) -> dict:
        return asdict(self)
