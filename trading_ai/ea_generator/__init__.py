"""
Modulo 6 - EA Generator
======================

Genera Expert Advisor MQL4 e MQL5 a partire da una strategia validata. Il
modello (scaler + KMeans) viene embeddato nel sorgente, cosi' l'EA e'
autosufficiente. Le feature usate devono essere ESPORTABILI (calcolabili con
indicatori nativi MetaTrader): la funzione di export lo verifica e fallisce con
un messaggio chiaro in caso contrario.

Uso rapido
----------
    from trading_ai.ea_generator import EAGenerator

    eag = EAGenerator()
    paths = eag.export(strategy)        # scrive mql4/<name>.mq4 e mql5/<name>.mq5
"""

from __future__ import annotations

from pathlib import Path

from trading_ai.config import PATHS
from trading_ai.ea_generator.features_map import EXPORTABLE_FEATURES, check_exportable
from trading_ai.ea_generator.mql4 import generate_mql4
from trading_ai.ea_generator.mql5 import generate_mql5
from trading_ai.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["EAGenerator", "generate_mql4", "generate_mql5", "EXPORTABLE_FEATURES"]


class EAGenerator:
    """Esporta strategie in codice MQL4/MQL5 su file."""

    def __init__(self, mql4_dir: Path | None = None, mql5_dir: Path | None = None):
        # Default: le cartelle /mql4 e /mql5 del progetto (Modulo 8).
        self.mql4_dir = Path(mql4_dir) if mql4_dir else PATHS.mql4
        self.mql5_dir = Path(mql5_dir) if mql5_dir else PATHS.mql5

    def _validate(self, strategy) -> None:
        """Controlla i prerequisiti per un export fedele e compilabile."""
        if getattr(strategy, "clusterer", None) is None:
            raise ValueError("La strategia non ha un clusterer addestrato.")
        # Tutte le feature devono essere ricalcolabili nativamente in MQL.
        check_exportable(list(strategy.clusterer.feature_columns))

    def to_mql4(self, strategy) -> str:
        """Ritorna il sorgente MQL4 (senza scrivere su file)."""
        self._validate(strategy)
        return generate_mql4(strategy)

    def to_mql5(self, strategy) -> str:
        """Ritorna il sorgente MQL5 (senza scrivere su file)."""
        self._validate(strategy)
        return generate_mql5(strategy)

    def export(self, strategy) -> dict[str, Path]:
        """
        Genera e SCRIVE i file .mq4 e .mq5. Ritorna i percorsi creati.
        """
        self._validate(strategy)
        self.mql4_dir.mkdir(parents=True, exist_ok=True)
        self.mql5_dir.mkdir(parents=True, exist_ok=True)

        p4 = self.mql4_dir / f"{strategy.name}.mq4"
        p5 = self.mql5_dir / f"{strategy.name}.mq5"
        p4.write_text(generate_mql4(strategy), encoding="utf-8")
        p5.write_text(generate_mql5(strategy), encoding="utf-8")

        logger.info("EA esportato: %s | %s", p4.name, p5.name)
        return {"mql4": p4, "mql5": p5}

    def export_many(self, strategies: list) -> list[dict[str, Path]]:
        """Esporta una lista di strategie, ignorando con log quelle non esportabili."""
        out = []
        for s in strategies:
            try:
                out.append(self.export(s))
            except ValueError as e:
                logger.warning("Salto %s: %s", getattr(s, "name", "?"), e)
        return out
