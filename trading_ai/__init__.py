"""
trading_ai
==========

Piattaforma di ricerca automatica di pattern e strategie di trading
statisticamente robuste.

Il package e' organizzato in moduli INDIPENDENTI, uno per ogni fase della
pipeline descritta nel progetto:

    data_engine        -> Modulo 1: import, pulizia, normalizzazione dati
    feature_engineering-> Modulo 2: estrazione feature (in arrivo)
    pattern_discovery  -> Modulo 3: scoperta pattern via ML (in arrivo)
    strategy_generator -> Modulo 4: generazione strategie (in arrivo)
    validation         -> Modulo 5: walk-forward, Monte Carlo... (in arrivo)
    ea_generator       -> Modulo 6: export MQL4/MQL5 (in arrivo)
    feedback           -> Modulo 7: ottimizzazione iterativa (in arrivo)
    reporting          -> Modulo 9: report e metriche (in arrivo)

Ogni modulo puo' essere importato e usato da solo, sia da script Python
sia da notebook Kaggle.
"""

# Versione semantica del package. La incrementiamo a ogni modulo completato.
__version__ = "0.1.0"

# Esponiamo il sotto-package del Modulo 1 cosi' che
# `import trading_ai; trading_ai.data_engine` funzioni subito.
from trading_ai import data_engine  # noqa: F401

__all__ = ["data_engine", "__version__"]
