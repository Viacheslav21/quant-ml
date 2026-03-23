# CLAUDE.md

## What This Is

Quant ML — ML service for Polymarket prediction model. Collects historical market data, trains XGBoost model to predict market outcomes, and serves predictions via API. Intended to become an additional evidence source in quant-engine's Bayesian fusion.

## Commands

```bash
pip install -r requirements.txt
cp .env.example .env  # add DATABASE_URL

python main.py collect   # Fetch historical data from Polymarket APIs
python main.py train     # Train model (Sprint 2 — TODO)
python main.py serve     # Start inference API (Sprint 3 — TODO)
```

## Architecture

### Data Collection (Sprint 1 — done)
- **Gamma API**: closed markets metadata (volume, theme, dates, token IDs). Filters: volume ≥ $1k, clean outcome (YES ≥ 0.95 or NO ≤ 0.05).
- **CLOB API**: price history per market (`/prices-history?market={token_id}`). Rate-limited via semaphore (30 concurrent).
- Per market: snapshot price at 14, 7, 3, 1, 0 days before expiry (~5 samples per market).
- Price lookup: bisect + 12h tolerance for closest timestamp match.
- **Features computed**: yes_price, theme (keyword matching, 15 themes), volume, market_age_days, price_momentum_7d, price_momentum_1d, price_volatility_7d, volume_per_day, price_distance_50, neg_risk, days_before_expiry.
- **Target**: outcome (YES=1, NO=0).
- Stored in shared PostgreSQL table `ml_training_data` (UNIQUE on market_id + days_before_expiry).
- Expected dataset: 5k–10k markets × 4-5 snapshots = 20k–50k training samples.

### Model Training (Sprint 2 — TODO)
- XGBoost classifier on tabular features.
- Time-series train/test split by market end_date (group by market_id to prevent leakage).
- Platt scaling for calibrated probabilities (CalibratedClassifierCV).
- Planned config: n_estimators=200, max_depth=5, learning_rate=0.05.
- Metrics: Brier score (primary), log loss, profit simulation.
- Output: `model.json`.

### Inference API (Sprint 3 — TODO)
- FastAPI endpoint `GET /predict`.
- quant-engine calls it during signal generation.
- Returns calibrated P(YES) for given features.

## Known Issues
- **Look-ahead bias**: price_momentum_7d and price_volatility_7d include prices up to observation point. At inference time, these features must be recomputed using only past data. Needs fix before training.
- **Nullable features**: momentum and volatility can be NULL if price history is sparse. Training code must handle NaN imputation.

## Modules
- **main.py** — Entry point with collect/train/serve mode dispatch.
- **ml/data_collector.py** — Gamma + CLOB API data fetching, feature engineering, async batch processing.
- **utils/db.py** — PostgreSQL: ml_training_data table schema, CRUD, batch upsert.

## Database
Uses shared PostgreSQL (same as quant-engine). Owns `ml_training_data` table (UNIQUE on market_id + days_before_expiry). Read-only access to `markets`, `positions`, `signals` for cross-referencing.
