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
│   └── validation/        # MODULO 5 (walk-forward, monte carlo, robustness)
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
| 3 | **Pattern Discovery** — scoperta non supervisionata via ML + statistica, validazione OOS | ✅ Completato |
| 4 | **Strategy Generator** — entry/exit/SL/TP/BE/trailing/filtri + backtester | ✅ Completato |
| 5 | **Validation** — walk-forward, OOS, Monte Carlo, robustness, sensitivity | ✅ Completato |
| 6 | EA Generator — export MQL4/MQL5 compilabile | ⏳ |
| 7 | AI Feedback — analisi errori e ottimizzazione iterativa | ⏳ |
| 8 | GitHub — organizzazione automatica del repo | ✅ (struttura attiva) |
| 9 | Report — equity, drawdown, Sharpe/Sortino/Calmar, ecc. | ⏳ |

## Installazione

```bash
pip install -r requirements.txt
```

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
