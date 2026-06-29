"""
Persistenza di modelli e strategie.

Permette di SALVARE e RICARICARE gli artefatti della pipeline (clusterer
addestrato, strategie, oggetti vari) cosi' l'autopilota non deve rifare la
scoperta da zero a ogni esecuzione e i risultati sono riproducibili.

Usiamo joblib, lo standard de-facto per serializzare oggetti scikit-learn
(piu' efficiente di pickle sui grandi array NumPy).
"""

from __future__ import annotations

from pathlib import Path

import joblib

from trading_ai.utils.logging import get_logger

logger = get_logger(__name__)


def save_object(obj: object, path: str | Path) -> Path:
    """Serializza un qualsiasi oggetto Python su file (.joblib)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)                              # compressione automatica di joblib
    return path


def load_object(path: str | Path) -> object:
    """Ricarica un oggetto serializzato con save_object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artefatto non trovato: {path}")
    return joblib.load(path)


def save_strategy(strategy, dir_path: str | Path) -> Path:
    """
    Salva una strategia completa (incluso il clusterer addestrato e i parametri).

    Una strategia e' interamente serializzabile: clusterer scikit-learn +
    dataclass di rischio/filtri/costi. La salviamo come singolo .joblib.
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / f"{strategy.name}.joblib"
    save_object(strategy, path)
    logger.info("Strategia salvata: %s", path.name)
    return path


def load_strategy(path: str | Path):
    """Ricarica una strategia salvata con save_strategy."""
    return load_object(path)
