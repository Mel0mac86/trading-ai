"""
Backtester event-driven (Modulo 4).

Simula i trade barra per barra applicando realmente Stop Loss, Take Profit,
Break-Even e Trailing Stop, con dimensionamento a rischio fisso frazionario.
Niente look-ahead: ogni decisione usa solo l'informazione disponibile fino alla
barra corrente; l'ingresso avviene alla CHIUSURA della barra-segnale e la
gestione parte dalla barra successiva.

Assunzioni prudenti:
  - una sola posizione aperta alla volta (niente pyramiding di default);
  - se in una stessa barra vengono toccati sia SL sia TP, assumiamo il caso
    PEGGIORE (prima lo SL): backtest conservativo, non ottimistico.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_ai.strategy_generator.risk import CostModel, RiskParams


@dataclass
class BacktestResult:
    """Output di un backtest: lista trade, curva equity e metriche aggregate."""

    trades: pd.DataFrame        # un record per trade (entry, exit, return, R, durata...)
    equity: pd.Series           # curva di equity compounded nel tempo
    returns: np.ndarray         # rendimenti per-trade (per le metriche del Modulo 9)


def backtest(
    df: pd.DataFrame,
    entries: pd.Series,
    direction: int,
    atr: pd.Series,
    risk: RiskParams,
    initial_equity: float = 10_000.0,
    costs: CostModel | None = None,
) -> BacktestResult:
    """
    Esegue il backtest di una strategia direzionale.

    Parametri
    ---------
    df : pd.DataFrame
        OHLC con DatetimeIndex (serve per l'esecuzione intrabar).
    entries : pd.Series[bool]
        True sulle barre in cui il pattern genera un segnale d'ingresso.
    direction : int
        +1 per long, -1 per short.
    atr : pd.Series
        ATR allineato a df, usato per dimensionare SL/TP in volatilita'.
    risk : RiskParams
        Parametri di gestione del rischio.
    initial_equity : float
        Capitale iniziale.
    """
    # Convertiamo in array numpy per un loop veloce e leggibile.
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    atr_arr = atr.to_numpy(dtype=float)
    sig = entries.to_numpy(dtype=bool)
    times = df.index
    n = len(df)
    costs = costs or CostModel()       # zero costi se non specificato (retro-compatibile)

    equity = initial_equity            # capitale corrente (compounded)
    equity_curve = np.full(n, np.nan)  # equity registrata nel tempo
    records: list[dict] = []           # accumulatore dei trade chiusi

    i = 0                              # indice di barra corrente
    trades_today = 0                   # contatore trade del giorno (per max_trades_per_day)
    current_day = None                 # giorno corrente per il reset del contatore

    while i < n - 1:
        # Reset del contatore giornaliero al cambio di data.
        day = times[i].date()
        if day != current_day:
            current_day, trades_today = day, 0

        # Condizioni per NON aprire: nessun segnale, ATR non valido, limite giornaliero.
        atr_i = atr_arr[i]
        can_trade = (
            sig[i] and np.isfinite(atr_i) and atr_i > 0 and
            (risk.max_trades_per_day == 0 or trades_today < risk.max_trades_per_day)
        )
        if not can_trade:
            equity_curve[i] = equity
            i += 1
            continue

        # --- Apertura trade alla chiusura della barra-segnale ---------------
        entry_price = close[i]
        sl_dist = risk.sl_atr * atr_i               # distanza stop in prezzo
        tp_dist = risk.tp_atr * atr_i               # distanza target in prezzo
        # Long: SL sotto, TP sopra. Short: speculare (direction = -1).
        sl = entry_price - direction * sl_dist
        tp = entry_price + direction * tp_dist
        be_moved = False                            # break-even gia' applicato?
        extreme = entry_price                       # estremo favorevole (per trailing)

        exit_price = None
        exit_idx = None
        # --- Gestione barra per barra dalla successiva ----------------------
        last = min(i + risk.max_bars, n - 1)
        for j in range(i + 1, last + 1):
            hi, lo = high[j], low[j]

            if direction == 1:                      # ---- gestione LONG ----
                # Caso peggiore prima: controlliamo lo STOP.
                if lo <= sl:
                    exit_price, exit_idx = sl, j
                    break
                if hi >= tp:                        # poi il TARGET
                    exit_price, exit_idx = tp, j
                    break
                # Aggiorniamo l'estremo favorevole (massimo) per BE/trailing.
                extreme = max(extreme, hi)
                profit = extreme - entry_price
                if risk.be_atr > 0 and not be_moved and profit >= risk.be_atr * atr_i:
                    sl = max(sl, entry_price)       # SL a pareggio
                    be_moved = True
                if risk.trail_atr > 0:
                    sl = max(sl, extreme - risk.trail_atr * atr_i)  # trailing sotto il max
            else:                                   # ---- gestione SHORT ----
                if hi >= sl:                        # stop sopra
                    exit_price, exit_idx = sl, j
                    break
                if lo <= tp:                        # target sotto
                    exit_price, exit_idx = tp, j
                    break
                extreme = min(extreme, lo)          # estremo favorevole (minimo)
                profit = entry_price - extreme
                if risk.be_atr > 0 and not be_moved and profit >= risk.be_atr * atr_i:
                    sl = min(sl, entry_price)
                    be_moved = True
                if risk.trail_atr > 0:
                    sl = min(sl, extreme + risk.trail_atr * atr_i)

        # Time stop: se nessuna barriera e' stata toccata, usciamo alla close.
        if exit_price is None:
            exit_idx = last
            exit_price = close[last]

        # --- Calcolo PnL ----------------------------------------------------
        # Rendimento lordo del trade orientato per direzione.
        gross_ret = direction * (exit_price - entry_price) / entry_price
        # Sottraiamo i costi di transazione (spread + slippage + commissioni).
        trade_ret = gross_ret - costs.cost_return(entry_price)
        # R-multiple: profitto NETTO in unita' di rischio iniziale (return / rischio%).
        risk_pct = sl_dist / entry_price
        r_multiple = trade_ret / risk_pct if risk_pct > 0 else 0.0
        # Equity a rischio fisso frazionario: guadagno = equity * risk% * R.
        equity *= (1.0 + risk.risk_per_trade * r_multiple)

        records.append({
            "entry_time": times[i], "exit_time": times[exit_idx],
            "direction": direction, "entry": entry_price, "exit": exit_price,
            "bars": exit_idx - i, "gross_return": gross_ret, "return": trade_ret,
            "r_multiple": r_multiple, "equity": equity,
        })

        # Segniamo l'equity su tutte le barre attraversate dal trade.
        equity_curve[i:exit_idx + 1] = equity
        trades_today += 1
        i = exit_idx + 1            # nessun pyramiding: si riparte dopo l'uscita

    # Riempimento finale dell'equity per le barre rimaste.
    equity_curve[i:] = equity
    # Forward-fill iniziale: prima del primo trade l'equity e' quella iniziale.
    eq_series = pd.Series(equity_curve, index=times).ffill().fillna(initial_equity)

    trades_df = pd.DataFrame(records)
    returns = trades_df["return"].to_numpy() if not trades_df.empty else np.array([])
    return BacktestResult(trades=trades_df, equity=eq_series, returns=returns)
