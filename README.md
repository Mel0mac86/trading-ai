# trading-ai

Piattaforma AI per la **ricerca automatica di pattern e strategie di trading
statisticamente robuste**, sviluppata su Kaggle e sincronizzata con GitHub.

> L'obiettivo **non** è prevedere il mercato con certezza, ma costruire un
> sistema di ricerca automatica di configurazioni ricorrenti e strategie
> validate, con difese rigorose contro **overfitting** e **data leakage**.

## Filosofia architetturale

La logica vive in un **package Python riutilizzabile e testato** (`trading_ai/`);
i notebook in `notebooks/` sono **sottili** e si limitano a orchestrare il
package. Così il codice è coperto da test automatici, riusabile fuori da Kaggle
e privo di duplicazioni copia-incolla tra notebook.

Ogni modulo è **indipendente**: può essere importato e usato da solo.

## Struttura del repository

```
trading-ai/
├── trading_ai/            # package core riutilizzabile
│   ├── config.py          # percorsi, timeframe, schema colonne canonico
│   ├── utils/             # logging e I/O (Parquet) condivisi
│   ├── data_engine/       # MODULO 1 (loader, cleaner, resampler, normalizer)
│   ├── feature_engineering/ # MODULO 2 (indicatori, candlestick, market structure)
│   ├── pattern_discovery/ # MODULO 3 (clustering, labeling, metriche, validazione OOS)
│   ├── strategy_generator/ # MODULO 4 (risk, filtri, backtester event-driven)
│   ├── validation/        # MODULO 5 (walk-forward, monte carlo, robustness)
│   ├── ea_generator/      # MODULO 6 (export MQL4/MQL5 con modello embeddato)
│   ├── feedback/          # MODULO 7 (analisi, ottimizzazione, versioning)
│   ├── reporting/         # MODULO 9 (metriche, grafici, report)
│   └── pipeline.py        # orchestratore end-to-end (tutti i moduli)
├── notebooks/             # notebook Kaggle (uno per modulo)
├── datasets/              # dati (non versionati, vedi .gitignore)
├── models/                # modelli ML serializzati
├── strategies/            # strategie generate + report di selezione
├── mql4/  /  mql5/        # Expert Advisor esportati
├── reports/               # report di performance (Modulo 9)
├── logs/                  # log di esecuzione
└── tests/                 # test automatici (pytest)
```

## Roadmap dei moduli

| # | Modulo | Stato |
|---|--------|-------|
| 1 | **Data Engine** — import, pulizia, resampling, normalizzazione | ✅ Completato |
| 2 | **Feature Engineering** — ATR, RSI, MACD, ADX, EMA/SMA, VWAP, Bollinger, pattern, FVG, BOS/CHoCH, swing, S/R, liquidità, volatilità, trend strength | ✅ Completato |
| 3 | **Pattern Discovery** — scoperta non supervisionata via ML + statistica, validazione OOS, **Deflated Sharpe Ratio** (correzione multiple testing) | ✅ Completato |
| 4 | **Strategy Generator** — entry/exit/SL/TP/BE/trailing/filtri + backtester con costi (spread/slippage/commissioni) | ✅ Completato |
| 5 | **Validation** — walk-forward, OOS, Monte Carlo, robustness, sensitivity, PSR/Deflated Sharpe | ✅ Completato |
| 6 | **EA Generator** — export MQL4/MQL5 con modello KMeans embeddato | ✅ Completato |
| 7 | **AI Feedback** — analisi errori, ottimizzazione, versioning strategie | ✅ Completato |
| 8 | GitHub — organizzazione automatica del repo | ✅ (struttura attiva) |
| 9 | **Report** — equity, drawdown, Sharpe/Sortino/Calmar, ecc. | ✅ Completato |

**Tutti i 9 moduli sono implementati e testati (63 test verdi).** La pipeline
end-to-end è orchestrata da `trading_ai/pipeline.py` e dal notebook
[`notebooks/00_full_pipeline.ipynb`](notebooks/00_full_pipeline.ipynb).

## Installazione

```bash
pip install -r requirements.txt
# oppure, per il comando CLI installato:
pip install -e .
```

## Autopilota (zero input)

La piattaforma gira **da sola, senza inserire nulla**: acquisisce i dati (CSV
locali in `datasets/` → download opzionale → fallback sintetico garantito),
esegue l'intera pipeline su più strumenti, valida, genera report ed EA,
**persiste i modelli** e scrive un report consolidato per ogni run.

```bash
python -m trading_ai                       # run completa con i default
python -m trading_ai --instruments EURUSD XAUUSD US500
python -m trading_ai --config config/default.yaml
python -m trading_ai --no-download         # solo dati locali/sintetici
# Dati direttamente da un dataset Kaggle (richiede KAGGLE_API_TOKEN):
python -m trading_ai --instruments XAUUSD --kaggle-dataset owner/dataset
```

**Acquisizione dati** (in cascata, sempre con fallback): file locali in
`datasets/` → **dataset Kaggle** (se `--kaggle-dataset` + credenziali) → download
yfinance → dati sintetici. Più file annuali dello stesso strumento vengono
fusi automaticamente in una serie continua.

Output di ogni run (timestamped):

```
reports/runs/<run>/manifest.json   # esiti macchina-leggibili
reports/runs/<run>/summary.md      # sintesi + strategie robuste
reports/runs/<run>/<Strategia>/    # report di dettaglio (equity, drawdown...)
models/<run>/<strumento>/          # clusterer e strategie serializzati (joblib)
mql4/  mql5/                       # Expert Advisor esportati
logs/autopilot_<run>.log           # log completo dell'esecuzione
```

### Formato dati supportato

Il loader riconosce gli **export MetaTrader** out-of-the-box (oltre a CSV generici):

```
date	open	high	low	close	volume	spread
2024.11.18 00:00	2564.258	2568.575	2563.835	2567.455	595	81
```

- datetime puntato `YYYY.MM.DD HH:MM` (anche `date`+`time` separati), con parsing a
  **formato esplicito** → veloce su milioni di candele;
- separatore (tab/virgola/`;`) rilevato automaticamente;
- la colonna **`spread`** (in punti) alimenta i **costi di transazione reali**
  (conversione punti→prezzo con `point_value` inferito dai decimali), poi viene
  rimossa dalle feature per non inquinare il modello.

Metti il file in `datasets/` con il simbolo nel nome (es. `XAUUSD_M5.csv`):
l'autopilota lo abbina automaticamente allo strumento `XAUUSD`.

## Esempio rapido (Modulo 1)

```python
from trading_ai.data_engine import DataEngine

eng = DataEngine()
df  = eng.load_csv("datasets/EURUSD_M1.csv")   # import + pulizia automatica
h1  = eng.to_timeframe(df, "H1")               # resample multi-timeframe
norm = eng.normalize(h1, method="returns")     # normalizzazione anti-leakage
eng.save(h1, "datasets/EURUSD_H1.parquet")     # persistenza Parquet
```

Vedi il notebook dimostrativo: [`notebooks/01_data_engine.ipynb`](notebooks/01_data_engine.ipynb).

## Test

```bash
python -m pytest -q
```

Ogni modulo deve avere i test **verdi** prima di passare al successivo.
