"""
Logging unificato per tutta la piattaforma.

Usiamo il modulo standard `logging` (niente dipendenze esterne). Un logger
configurato bene ci permette di tracciare quante candele vengono scartate,
quali pattern superano i filtri, ecc. — fondamentale per la riproducibilita'.
"""

from __future__ import annotations

import logging  # libreria standard per il logging
import sys      # per scrivere i log su stdout (visibile nei notebook Kaggle)
from pathlib import Path  # gestione percorsi del file di log

# Teniamo traccia dei logger gia' configurati per non aggiungere handler
# duplicati (causa tipica di righe di log ripetute nei notebook).
_CONFIGURED: dict[str, logging.Logger] = {}

# File handler attivi: applicati a TUTTI i logger (i figli hanno propagate=False,
# quindi non basta agganciarli al solo logger radice).
_FILE_HANDLERS: list[logging.FileHandler] = []

# Formato unico riutilizzato da tutti gli handler (stdout e file).
_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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
        handler.setFormatter(_FORMATTER)  # applichiamo il formato condiviso
        logger.addHandler(handler)       # colleghiamo l'handler al logger
        logger.propagate = False         # evitiamo che il messaggio risalga al root logger (doppi log)
        for fh in _FILE_HANDLERS:        # agganciamo anche eventuali file handler attivi
            logger.addHandler(fh)
        _CONFIGURED[name] = logger       # registriamo questo logger configurato

    return logger


def add_file_handler(path: str | Path, level: int = logging.INFO) -> Path:
    """
    Aggiunge un file di log al logger radice 'trading_ai': tutti i moduli vi
    scriveranno (oltre che su stdout). Utile per l'autopilota, che lascia una
    traccia completa e riproducibile di ogni esecuzione in /logs.

    Idempotente sullo stesso percorso: non duplica l'handler.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(path.resolve())
    # Evitiamo handler doppi sullo stesso file (es. ri-esecuzioni nel notebook).
    for h in _FILE_HANDLERS:
        if getattr(h, "baseFilename", None) == resolved:
            return path
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(_FORMATTER)
    _FILE_HANDLERS.append(fh)
    # Lo agganciamo a tutti i logger gia' creati; get_logger lo aggiungera' ai futuri.
    get_logger("trading_ai", level)                 # garantisce almeno il logger radice
    for lg in _CONFIGURED.values():
        lg.addHandler(fh)
    return path
