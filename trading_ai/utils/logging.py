"""
Logging unificato per tutta la piattaforma.

Usiamo il modulo standard `logging` (niente dipendenze esterne). Un logger
configurato bene ci permette di tracciare quante candele vengono scartate,
quali pattern superano i filtri, ecc. — fondamentale per la riproducibilita'.
"""

from __future__ import annotations

import logging  # libreria standard per il logging
import sys      # per scrivere i log su stdout (visibile nei notebook Kaggle)

# Teniamo traccia dei logger gia' configurati per non aggiungere handler
# duplicati (causa tipica di righe di log ripetute nei notebook).
_CONFIGURED: set[str] = set()


def get_logger(name: str = "trading_ai", level: int = logging.INFO) -> logging.Logger:
    """
    Restituisce un logger configurato in modo coerente.

    Parametri
    ---------
    name : str
        Nome del logger (di solito il nome del modulo chiamante).
    level : int
        Soglia minima dei messaggi da mostrare (default INFO).

    Ritorna
    -------
    logging.Logger
        Logger pronto, con formattazione standard e output su stdout.
    """
    logger = logging.getLogger(name)  # logging.getLogger restituisce SEMPRE la stessa istanza per lo stesso nome
    logger.setLevel(level)            # imposta la soglia dei messaggi

    if name not in _CONFIGURED:       # configuriamo gli handler una sola volta
        handler = logging.StreamHandler(sys.stdout)  # output su stdout (Kaggle-friendly)
        # Formato: orario - nome modulo - livello - messaggio
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)  # applichiamo il formato all'handler
        logger.addHandler(handler)       # colleghiamo l'handler al logger
        logger.propagate = False         # evitiamo che il messaggio risalga al root logger (doppi log)
        _CONFIGURED.add(name)            # segniamo questo logger come gia' configurato

    return logger
