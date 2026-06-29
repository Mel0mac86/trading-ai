"""
Correzione per test multipli (Modulo 5 - estensione anti-illusione).

Quando si cercano pattern provando MOLTE configurazioni (qui: N cluster) e si
tengono le migliori, lo Sharpe osservato e' gonfiato dalla selezione: anche con
rendimenti puramente casuali, il "migliore tra N" sembrera' buono. E' il
problema del multiple testing / p-hacking.

Implementiamo gli strumenti di Bailey & Lopez de Prado:

  - Probabilistic Sharpe Ratio (PSR): probabilita' che lo Sharpe VERO superi una
    soglia di riferimento, dato lo Sharpe osservato, la lunghezza del campione e
    i momenti superiori (skew, curtosi) dei rendimenti.

  - Expected Maximum Sharpe: lo Sharpe atteso del MIGLIORE tra N trial sotto
    l'ipotesi nulla (nessun edge). E' la soglia con cui deflazionare.

  - Deflated Sharpe Ratio (DSR): PSR calcolato usando come soglia l'Expected
    Maximum Sharpe -> probabilita' che l'edge sia REALE e non frutto della
    ricerca su N candidati. Vicino a 1 = robusto; vicino a 0 = probabile fortuna.

Tutti gli Sharpe qui sono PER-OSSERVAZIONE (mean/std dei rendimenti per trade),
non annualizzati: e' la convenzione richiesta dalle formule PSR/DSR.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

# Costante di Eulero-Mascheroni, usata nella stima dell'Expected Maximum Sharpe.
_EULER_GAMMA = 0.5772156649015329


def probabilistic_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sr_benchmark: float = 0.0,
) -> float:
    """
    Probabilistic Sharpe Ratio: P(Sharpe_vero > sr_benchmark).

    Parametri
    ---------
    sharpe : float
        Sharpe osservato per-osservazione (mean/std dei rendimenti per trade).
    n_obs : int
        Numero di osservazioni (trade).
    skew : float
        Asimmetria dei rendimenti.
    kurtosis : float
        Curtosi NON in eccesso (per la normale vale 3.0).
    sr_benchmark : float
        Soglia di Sharpe da superare (0 = "meglio del caso").

    Ritorna
    -------
    float in [0, 1]: probabilita' che lo Sharpe vero superi la soglia.
    """
    if n_obs < 2:
        return 0.0
    # Varianza dello stimatore dello Sharpe (corretta per skew e curtosi).
    denom = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe ** 2
    if denom <= 0:
        # Momenti estremi rendono la stima instabile: restiamo prudenti.
        return 0.0
    z = (sharpe - sr_benchmark) * np.sqrt(n_obs - 1.0) / np.sqrt(denom)
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, variance_of_sharpes: float) -> float:
    """
    Sharpe atteso del MIGLIORE tra n_trials configurazioni sotto ipotesi nulla.

    Approssimazione di Lopez de Prado basata sulle statistiche degli estremi:
        SR* = sqrt(V) * [ (1-gamma)*Z^-1(1 - 1/N) + gamma*Z^-1(1 - 1/(N*e)) ]
    dove V e' la varianza delle stime di Sharpe tra i trial, gamma la costante di
    Eulero-Mascheroni, Z^-1 la quantile della normale standard.
    """
    if n_trials <= 1 or variance_of_sharpes <= 0:
        return 0.0                                       # con 1 solo trial non c'e' selezione
    sqrt_v = np.sqrt(variance_of_sharpes)
    # Quantili della normale ai due punti previsti dalla formula.
    q1 = norm.ppf(1.0 - 1.0 / n_trials)
    q2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sqrt_v * ((1.0 - _EULER_GAMMA) * q1 + _EULER_GAMMA * q2))


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    n_trials: int,
    variance_of_sharpes: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Deflated Sharpe Ratio: PSR con soglia = Expected Maximum Sharpe.

    Ritorna la probabilita' che lo Sharpe osservato rifletta un edge REALE,
    una volta scontato il fatto di aver scelto il migliore tra n_trials candidati.
    """
    sr_star = expected_max_sharpe(n_trials, variance_of_sharpes)
    return probabilistic_sharpe_ratio(sharpe, n_obs, skew, kurtosis, sr_benchmark=sr_star)
