"""
Configurazione centralizzata e condivisa da TUTTI i moduli.

Tenere costanti e percorsi in un solo posto evita "stringhe magiche" sparse
nel codice e rende la piattaforma facilmente riconfigurabile (anche su Kaggle,
dove i percorsi cambiano: /kaggle/working invece della root del repo).
"""

from __future__ import annotations  # consente type hint moderni anche su Python vecchi

import os                            # per leggere variabili d'ambiente (override Kaggle)
from dataclasses import dataclass, field  # per creare config strutturate e immutabili
from pathlib import Path             # gestione percorsi cross-platform (Linux/Windows/Kaggle)


# --- Radice del progetto -----------------------------------------------------
# __file__ e' .../trading_ai/config.py; saliamo di 2 livelli per arrivare alla
# root del repository (cartella che contiene /datasets, /models, ecc.).
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_root() -> Path:
    """
    Determina la cartella di lavoro radice.

    Su Kaggle il repo viene clonato in sola lettura sotto /kaggle/input, quindi
    gli OUTPUT vanno scritti in /kaggle/working. Permettiamo un override tramite
    la variabile d'ambiente TRADING_AI_ROOT senza toccare il codice.
    """
    env = os.environ.get("TRADING_AI_ROOT")  # eventuale override esterno
    if env:                                  # se l'utente/Kaggle l'ha impostata...
        return Path(env)                     # ...la usiamo cosi' com'e'
    return PROJECT_ROOT                       # altrimenti la root del repo


# I timeframe supportati, mappati al numero di minuti che rappresentano.
# Serve al resampler (Modulo 1) per aggregare correttamente le candele.
TIMEFRAME_MINUTES: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}

# Nomi di colonna canonici usati in TUTTA la piattaforma. Qualsiasi sorgente
# dati (MT4, MT5, CSV, API) viene normalizzata a questo schema.
OHLCV_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)  # frozen=True -> config immutabile (niente modifiche accidentali)
class Paths:
    """Raccolta centralizzata di tutti i percorsi delle cartelle del progetto."""

    root: Path = field(default_factory=_resolve_root)  # cartella radice (vedi sopra)

    # property: percorsi derivati calcolati a partire dalla root.
    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def strategies(self) -> Path:
        return self.root / "strategies"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def mql4(self) -> Path:
        return self.root / "mql4"

    @property
    def mql5(self) -> Path:
        return self.root / "mql5"

    def ensure(self) -> "Paths":
        """Crea sul disco tutte le cartelle se non esistono (idempotente)."""
        for p in [self.datasets, self.models, self.strategies,
                  self.reports, self.logs, self.mql4, self.mql5]:
            p.mkdir(parents=True, exist_ok=True)  # parents=True crea anche i genitori
        return self  # ritorniamo self per permettere chiamate concatenate


# Istanza singleton pronta all'uso: `from trading_ai.config import PATHS`.
PATHS = Paths()
