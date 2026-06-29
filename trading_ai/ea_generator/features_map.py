"""
Mappatura feature -> indicatori nativi MetaTrader (Modulo 6).

Un EA deve ricalcolare in tempo reale le STESSE feature su cui e' stato
addestrato il modello. Solo gli indicatori NATIVI di MetaTrader sono ricalcolabili
in modo affidabile e compilabile; per questo definiamo qui il sottoinsieme di
feature "esportabili" e, per ciascuna, l'espressione MQL4 e MQL5 che la calcola
alla barra di indice `{s}` (shift).

Per generare EA fedeli, la Pattern Discovery destinata all'export va eseguita
restringendo le feature a EXPORTABLE_FEATURES (vedi notebook 06).
"""

from __future__ import annotations

# Codice MQL5 per creare l'handle di ciascun indicatore (in OnInit).
MQL5_HANDLE_CREATE: dict[str, str] = {
    "rsi": "iRSI(_Symbol,_Period,14,PRICE_CLOSE)",
    "atr": "iATR(_Symbol,_Period,14)",
    "adx": "iADX(_Symbol,_Period,14)",
    "macd": "iMACD(_Symbol,_Period,12,26,9,PRICE_CLOSE)",
    "ema10": "iMA(_Symbol,_Period,10,0,MODE_EMA,PRICE_CLOSE)",
    "ema20": "iMA(_Symbol,_Period,20,0,MODE_EMA,PRICE_CLOSE)",
    "ema50": "iMA(_Symbol,_Period,50,0,MODE_EMA,PRICE_CLOSE)",
    "sma10": "iMA(_Symbol,_Period,10,0,MODE_SMA,PRICE_CLOSE)",
    "sma20": "iMA(_Symbol,_Period,20,0,MODE_SMA,PRICE_CLOSE)",
    "sma50": "iMA(_Symbol,_Period,50,0,MODE_SMA,PRICE_CLOSE)",
    "bands": "iBands(_Symbol,_Period,20,0,2.0,PRICE_CLOSE)",
}

# Per ogni colonna-feature esportabile:
#   handles : indicatori (handle) richiesti -> per creare solo quelli necessari in MQL5
#   mql4    : espressione che calcola il valore in MQL4 (shift {s})
#   mql5    : espressione che calcola il valore in MQL5 usando IndVal(handle,buffer,{s})
EXPORT_SPEC: dict[str, dict] = {
    "rsi": {"handles": ["rsi"],
            "mql4": "iRSI(_Symbol,_Period,14,PRICE_CLOSE,{s})",
            "mql5": "IndVal(h_rsi,0,{s})"},
    "atr": {"handles": ["atr"],
            "mql4": "iATR(_Symbol,_Period,14,{s})",
            "mql5": "IndVal(h_atr,0,{s})"},
    "adx_adx": {"handles": ["adx"],
                "mql4": "iADX(_Symbol,_Period,14,PRICE_CLOSE,MODE_MAIN,{s})",
                "mql5": "IndVal(h_adx,0,{s})"},
    "adx_plus_di": {"handles": ["adx"],
                    "mql4": "iADX(_Symbol,_Period,14,PRICE_CLOSE,MODE_PLUSDI,{s})",
                    "mql5": "IndVal(h_adx,1,{s})"},
    "adx_minus_di": {"handles": ["adx"],
                     "mql4": "iADX(_Symbol,_Period,14,PRICE_CLOSE,MODE_MINUSDI,{s})",
                     "mql5": "IndVal(h_adx,2,{s})"},
    "macd_line": {"handles": ["macd"],
                  "mql4": "iMACD(_Symbol,_Period,12,26,9,PRICE_CLOSE,MODE_MAIN,{s})",
                  "mql5": "IndVal(h_macd,0,{s})"},
    "macd_signal": {"handles": ["macd"],
                    "mql4": "iMACD(_Symbol,_Period,12,26,9,PRICE_CLOSE,MODE_SIGNAL,{s})",
                    "mql5": "IndVal(h_macd,1,{s})"},
    "macd_hist": {"handles": ["macd"],
                  "mql4": "(iMACD(_Symbol,_Period,12,26,9,PRICE_CLOSE,MODE_MAIN,{s})"
                          "-iMACD(_Symbol,_Period,12,26,9,PRICE_CLOSE,MODE_SIGNAL,{s}))",
                  "mql5": "(IndVal(h_macd,0,{s})-IndVal(h_macd,1,{s}))"},
    "ema_10": {"handles": ["ema10"],
               "mql4": "iMA(_Symbol,_Period,10,0,MODE_EMA,PRICE_CLOSE,{s})",
               "mql5": "IndVal(h_ema10,0,{s})"},
    "ema_20": {"handles": ["ema20"],
               "mql4": "iMA(_Symbol,_Period,20,0,MODE_EMA,PRICE_CLOSE,{s})",
               "mql5": "IndVal(h_ema20,0,{s})"},
    "ema_50": {"handles": ["ema50"],
               "mql4": "iMA(_Symbol,_Period,50,0,MODE_EMA,PRICE_CLOSE,{s})",
               "mql5": "IndVal(h_ema50,0,{s})"},
    "sma_10": {"handles": ["sma10"],
               "mql4": "iMA(_Symbol,_Period,10,0,MODE_SMA,PRICE_CLOSE,{s})",
               "mql5": "IndVal(h_sma10,0,{s})"},
    "sma_20": {"handles": ["sma20"],
               "mql4": "iMA(_Symbol,_Period,20,0,MODE_SMA,PRICE_CLOSE,{s})",
               "mql5": "IndVal(h_sma20,0,{s})"},
    "sma_50": {"handles": ["sma50"],
               "mql4": "iMA(_Symbol,_Period,50,0,MODE_SMA,PRICE_CLOSE,{s})",
               "mql5": "IndVal(h_sma50,0,{s})"},
    "bollinger_mid": {"handles": ["bands"],
                      "mql4": "iBands(_Symbol,_Period,20,2,0,PRICE_CLOSE,MODE_MAIN,{s})",
                      "mql5": "IndVal(h_bands,0,{s})"},
    "bollinger_upper": {"handles": ["bands"],
                        "mql4": "iBands(_Symbol,_Period,20,2,0,PRICE_CLOSE,MODE_UPPER,{s})",
                        "mql5": "IndVal(h_bands,1,{s})"},
    "bollinger_lower": {"handles": ["bands"],
                        "mql4": "iBands(_Symbol,_Period,20,2,0,PRICE_CLOSE,MODE_LOWER,{s})",
                        "mql5": "IndVal(h_bands,2,{s})"},
    "bollinger_width": {"handles": ["bands"],
                        "mql4": "((iBands(_Symbol,_Period,20,2,0,PRICE_CLOSE,MODE_UPPER,{s})"
                                "-iBands(_Symbol,_Period,20,2,0,PRICE_CLOSE,MODE_LOWER,{s}))"
                                "/iBands(_Symbol,_Period,20,2,0,PRICE_CLOSE,MODE_MAIN,{s}))",
                        "mql5": "((IndVal(h_bands,1,{s})-IndVal(h_bands,2,{s}))/IndVal(h_bands,0,{s}))"},
    "bollinger_pctb": {"handles": ["bands"],
                       "mql4": "((iClose(_Symbol,_Period,{s})"
                               "-iBands(_Symbol,_Period,20,2,0,PRICE_CLOSE,MODE_LOWER,{s}))"
                               "/(iBands(_Symbol,_Period,20,2,0,PRICE_CLOSE,MODE_UPPER,{s})"
                               "-iBands(_Symbol,_Period,20,2,0,PRICE_CLOSE,MODE_LOWER,{s})))",
                       "mql5": "((iClose(_Symbol,_Period,{s})-IndVal(h_bands,2,{s}))"
                               "/(IndVal(h_bands,1,{s})-IndVal(h_bands,2,{s})))"},
    "volatility_atr_pct": {"handles": ["atr"],
                           "mql4": "(iATR(_Symbol,_Period,14,{s})/iClose(_Symbol,_Period,{s}))",
                           "mql5": "(IndVal(h_atr,0,{s})/iClose(_Symbol,_Period,{s}))"},
}

# Set di feature esportabili (in un EA compilabile e fedele).
EXPORTABLE_FEATURES: list[str] = list(EXPORT_SPEC)


def check_exportable(feature_columns: list[str]) -> None:
    """
    Verifica che TUTTE le feature del modello siano esportabili in MQL.
    Solleva un errore chiaro (con la lista di quelle non supportate) altrimenti.
    """
    bad = [c for c in feature_columns if c not in EXPORT_SPEC]
    if bad:
        raise ValueError(
            "Feature non esportabili in MQL: " + ", ".join(bad) + ".\n"
            "Esegui la Pattern Discovery con "
            "feature_columns=EXPORTABLE_FEATURES per generare EA fedeli."
        )


def required_handles(feature_columns: list[str]) -> list[str]:
    """Insieme ordinato degli handle MQL5 necessari per queste feature."""
    handles: list[str] = []
    for col in feature_columns:
        for h in EXPORT_SPEC[col]["handles"]:
            if h not in handles:
                handles.append(h)
    return handles
