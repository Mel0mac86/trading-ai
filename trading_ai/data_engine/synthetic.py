"""
Generatore di dati OHLCV sintetici ma realistici.

Serve a due scopi:
  1) Testare l'intera pipeline senza dipendere da download esterni (utile su
     Kaggle/CI dove la rete puo' mancare).
  2) Avere un "ground truth" controllato per i test automatici del Data Engine.

Il prezzo segue un Geometric Brownian Motion (moto browniano geometrico), il
modello classico per i prezzi finanziari: i rendimenti logaritmici sono
gaussiani, quindi il prezzo resta positivo e ha volatilita' realistica.
"""

from __future__ import annotations

import numpy as np   # generazione numeri casuali e calcolo vettoriale
import pandas as pd  # costruzione del DataFrame finale


def generate_ohlcv(
    n: int = 10_000,                 # numero di candele da generare
    start: str = "2020-01-01",       # timestamp della prima candela
    freq: str = "1min",              # frequenza (1min = candele M1)
    start_price: float = 1.1000,     # prezzo iniziale (es. EURUSD)
    annual_vol: float = 0.10,        # volatilita' annualizzata (10%)
    seed: int | None = 42,           # seme RNG per risultati riproducibili
) -> pd.DataFrame:
    """
    Genera un DataFrame OHLCV sintetico indicizzato per data.

    Ritorna
    -------
    pd.DataFrame
        Colonne: open, high, low, close, volume. Indice: DatetimeIndex.
    """
    rng = np.random.default_rng(seed)  # RNG moderno e riproducibile di NumPy

    # --- Timestamp -----------------------------------------------------------
    # Creiamo n istanti equispaziati a partire da `start` con passo `freq`.
    index = pd.date_range(start=start, periods=n, freq=freq)

    # --- Rendimenti log per il prezzo di CHIUSURA ---------------------------
    # Convertiamo la volatilita' annua in volatilita' per-barra. Assumiamo
    # ~525_600 minuti in un anno (365*24*60); sqrt perche' la varianza scala
    # linearmente nel tempo, la deviazione standard con la radice.
    minutes_per_year = 365 * 24 * 60
    bar_vol = annual_vol / np.sqrt(minutes_per_year)

    # Rendimenti log gaussiani: media 0 (nessun drift), dev.std = bar_vol.
    log_returns = rng.normal(loc=0.0, scale=bar_vol, size=n)

    # Prezzo di chiusura = prezzo iniziale * exp(somma cumulata dei rendimenti).
    close = start_price * np.exp(np.cumsum(log_returns))

    # --- Costruzione di open/high/low coerenti ------------------------------
    # L'open di ogni barra e' la close di quella precedente (mercato continuo).
    open_ = np.empty(n)                 # array vuoto per le aperture
    open_[0] = start_price              # la primissima apertura e' il prezzo iniziale
    open_[1:] = close[:-1]              # le successive = chiusura della barra prima

    # Ampiezza casuale del "wick" (ombra) proporzionale alla volatilita' di barra.
    wick = np.abs(rng.normal(0.0, bar_vol, size=n)) * close

    # High = max(open, close) + ombra superiore; Low = min(open, close) - ombra.
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick

    # --- Volume sintetico ----------------------------------------------------
    # Volume positivo con distribuzione di Poisson (numeri interi, tipico tick).
    volume = rng.poisson(lam=1000, size=n).astype(float)

    # --- Assemblaggio DataFrame ---------------------------------------------
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
    df.index.name = "time"  # nome dell'indice coerente con il resto della pipeline
    return df
