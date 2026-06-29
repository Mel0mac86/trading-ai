"""
Test automatici del Modulo 1 - Data Engine.

Verifichiamo (regola di progetto "verifica che il modulo funzioni"):
  - generazione dati sintetici coerenti
  - normalizzazione schema da CSV stile MetaTrader
  - rimozione effettiva di righe corrotte
  - correttezza del resampling multi-timeframe
  - assenza di data leakage nella normalizzazione
  - round-trip di salvataggio/caricamento Parquet

Esecuzione:  pytest -q   (oppure  python -m pytest -q)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_ai.data_engine import (
    DataEngine, Normalizer, clean, generate_ohlcv, load_csv, resample,
)


# --- Fixture: dataset sintetico riutilizzabile in piu' test ------------------
@pytest.fixture
def synth() -> pd.DataFrame:
    return generate_ohlcv(n=5_000, seed=123)


def test_synthetic_is_consistent(synth):
    """I dati sintetici devono rispettare le relazioni OHLC di base."""
    assert list(synth.columns) == ["open", "high", "low", "close", "volume"]
    assert (synth["high"] >= synth["low"]).all()                 # high >= low
    assert (synth["high"] >= synth[["open", "close"]].max(axis=1)).all()
    assert (synth["low"] <= synth[["open", "close"]].min(axis=1)).all()
    assert (synth[["open", "high", "low", "close"]] > 0).all().all()  # prezzi positivi
    assert isinstance(synth.index, pd.DatetimeIndex)             # indice temporale


def test_loader_normalizes_metatrader_csv(tmp_path):
    """Un CSV stile MT (date+time separati, header maiuscolo) va normalizzato."""
    csv = tmp_path / "EURUSD.csv"
    csv.write_text(
        "Date,Time,Open,High,Low,Close,Volume\n"
        "2020.01.01,00:00,1.1000,1.1010,1.0995,1.1005,100\n"
        "2020.01.01,00:01,1.1005,1.1012,1.1001,1.1009,120\n"
    )
    df = load_csv(csv, has_header=True)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) == 2


def test_loader_real_metatrader_export(tmp_path):
    """
    Formato export MetaTrader reale: tab-separated, datetime puntato
    'YYYY.MM.DD HH:MM', colonna spread. Spread conservato e point_value inferito.
    """
    csv = tmp_path / "XAUUSD_M5.csv"
    csv.write_text(
        "date\topen\thigh\tlow\tclose\tvolume\tspread\n"
        "2024.11.18 00:00\t2564.258\t2568.575\t2563.835\t2567.455\t595\t81\n"
        "2024.11.18 00:05\t2567.455\t2569.100\t2566.200\t2568.300\t610\t79\n"
    )
    df = load_csv(csv)
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "spread"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index[0] == pd.Timestamp("2024-11-18 00:00")
    # 3 decimali -> point_value 0.001 (oro).
    assert df.attrs["point_value"] == pytest.approx(0.001)


def test_resample_averages_spread(tmp_path):
    """Lo spread (in punti) deve essere mediato nel resampling, non sommato."""
    csv = tmp_path / "XAU.csv"
    lines = ["date\topen\thigh\tlow\tclose\tvolume\tspread"]
    t = pd.Timestamp("2024-01-01 00:00")
    for i in range(20):
        lines.append(f"{t.strftime('%Y.%m.%d %H:%M')}\t100.0\t100.5\t99.5\t100.1\t10\t{80 + (i % 4)}")
        t += pd.Timedelta(minutes=1)
    csv.write_text("\n".join(lines))
    df = load_csv(csv)
    m5 = resample(df, "M5")
    assert "spread" in m5.columns
    # La media di 5 valori attorno a ~80-83 resta in quel range (non somma a ~400).
    assert 79 <= m5["spread"].iloc[0] <= 84


def test_cleaner_removes_corrupted_rows(synth):
    """Iniettiamo righe corrotte e verifichiamo che vengano rimosse."""
    df = synth.copy()
    # high < low (incoerente)
    df.iloc[10, df.columns.get_loc("high")] = df.iloc[10]["low"] - 0.5
    # prezzo negativo
    df.iloc[20, df.columns.get_loc("close")] = -1.0
    # NaN su open
    df.iloc[30, df.columns.get_loc("open")] = np.nan

    cleaned, report = clean(df)
    assert report.inconsistent >= 1
    assert report.non_positive >= 1
    assert report.missing_ohlc >= 1
    assert len(cleaned) < len(df)
    # Dopo la pulizia, le relazioni OHLC devono valere ovunque.
    assert (cleaned["high"] >= cleaned["low"]).all()


def test_cleaner_removes_duplicate_timestamps(synth):
    """Timestamp duplicati devono essere collassati a uno solo."""
    dup = pd.concat([synth.iloc[:100], synth.iloc[:100]])  # raddoppiamo 100 righe
    cleaned, report = clean(dup)
    assert report.duplicates == 100
    assert not cleaned.index.duplicated().any()


def test_resample_aggregates_correctly(synth):
    """Il resampling M1->M5 deve seguire le regole OHLCV standard."""
    m5 = resample(synth, "M5")
    # Un blocco di 5 candele M1 -> 1 candela M5: il numero di barre cala ~5x.
    assert len(m5) <= len(synth) // 5 + 1
    # Confrontiamo la prima barra M5 con le prime 5 M1 corrispondenti.
    first_block = synth.iloc[:5]
    assert m5.iloc[0]["open"] == pytest.approx(first_block.iloc[0]["open"], rel=1e-5)
    assert m5.iloc[0]["high"] == pytest.approx(first_block["high"].max(), rel=1e-5)
    assert m5.iloc[0]["low"] == pytest.approx(first_block["low"].min(), rel=1e-5)
    assert m5.iloc[0]["close"] == pytest.approx(first_block.iloc[-1]["close"], rel=1e-5)
    assert m5.iloc[0]["volume"] == pytest.approx(first_block["volume"].sum(), rel=1e-5)


def test_resample_rejects_unknown_timeframe(synth):
    """Un timeframe non valido deve sollevare ValueError (fail-fast)."""
    with pytest.raises(ValueError):
        resample(synth, "M7")


def test_normalizer_zscore_has_no_leakage(synth):
    """I parametri appresi sul training NON devono dipendere dal test."""
    train, test = synth.iloc[:4000], synth.iloc[4000:]
    norm = Normalizer(method="zscore", columns=["close"]).fit(train)
    learned_mean = norm.params_["close"]["mean"]
    # La media appresa deve coincidere con quella del SOLO training.
    assert learned_mean == pytest.approx(train["close"].mean(), rel=1e-5)
    # transform sul test usa i parametri del training (non li ricalcola).
    out = norm.transform(test)
    assert "close" in out.columns


def test_normalizer_returns_are_stationary(synth):
    """Il metodo 'returns' deve produrre rendimenti centrati attorno a 0."""
    out = Normalizer(method="returns", columns=["close"]).fit_transform(synth)
    assert abs(out["close"].mean()) < 0.01  # media dei rendimenti ~ 0


def test_engine_end_to_end(synth, tmp_path):
    """Smoke test della facciata DataEngine: import->clean->resample->save->load."""
    eng = DataEngine()
    df = eng.load_dataframe(synth)            # normalizza + pulisce
    assert eng.last_report is not None        # il report deve essere popolato
    h1 = eng.to_timeframe(df, "H1")           # resample
    assert len(h1) > 0
    path = eng.save(h1, tmp_path / "h1.parquet")  # salvataggio
    reloaded = eng.load(path)                  # ricaricamento
    # check_freq=False: Parquet non serializza l'attributo 'freq' dell'indice
    # (e' un metadato, non un dato). I valori devono comunque essere identici.
    pd.testing.assert_frame_equal(h1, reloaded, check_freq=False)
