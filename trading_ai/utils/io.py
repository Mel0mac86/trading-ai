"""
Funzioni di I/O efficienti e riutilizzabili.

Per gestire "milioni di candele" preferiamo il formato Parquet (colonnare,
compresso, tipizzato) ai CSV: e' 5-10x piu' piccolo e molto piu' veloce da
leggere. Queste helper centralizzano la logica cosi' ogni modulo salva/legge
allo stesso modo.
"""

from __future__ import annotations

from pathlib import Path  # gestione percorsi

import pandas as pd  # I/O DataFrame


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """
    Salva un DataFrame in formato Parquet creando le cartelle mancanti.

    Parametri
    ---------
    df : pd.DataFrame
        Dati da salvare.
    path : str | Path
        Percorso del file di destinazione (.parquet).

    Ritorna
    -------
    Path
        Il percorso effettivo del file salvato.
    """
    path = Path(path)                                  # normalizziamo a Path
    path.parent.mkdir(parents=True, exist_ok=True)     # creiamo la cartella se manca
    # engine pyarrow + compressione snappy = buon compromesso velocita'/dimensione
    df.to_parquet(path, engine="pyarrow", compression="snappy")
    return path


def load_parquet(path: str | Path) -> pd.DataFrame:
    """
    Carica un DataFrame da un file Parquet.

    Solleva FileNotFoundError con messaggio chiaro se il file non esiste,
    cosi' l'errore e' immediatamente comprensibile nei notebook.
    """
    path = Path(path)              # normalizziamo a Path
    if not path.exists():          # controllo esplicito per un errore leggibile
        raise FileNotFoundError(f"File Parquet non trovato: {path}")
    return pd.read_parquet(path, engine="pyarrow")  # lettura veloce con pyarrow
