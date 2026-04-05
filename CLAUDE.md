# CLAUDE.md

## What This Is

Quant ML — ML service for Polymarket prediction model. Collects historical market data from 3 sources (Polymarket, Kalshi, Manifold), trains two XGBoost models (P(YES) + P(mispriced)), and serves predictions via FastAPI. Used as an additional evidence source in quant-engine's Bayesian fusion (90% math + 10% ML blend).

## Commands

```bash
pip install -r requirements.txt
cp .env.example .env  # add DATABASE_URL

python main.py        # Starts FastAPI server on port 8080
```

Training and collection are triggered via HTTP endpoints, not CLI arguments:
- `POST /api/train` — Collect from all 3 sources + train model (background task)
- `POST /api/train-only` — Re-train on existing data only (no collection)
- `GET /predict?yes_price=X&theme=Y&...` — Inference endpoint
- `GET /health` — Model status and sample count
- `GET /api/training-status` — Training progress and phase

Deployed via Railway (`Procfile: web: python main.py`).

## Architecture

### Data Collection
- **Polymarket Gamma API**: Closed markets metadata (volume, theme, dates, token IDs). Filters: volume ≥ $100, clean outcome (YES ≥ 0.95 or NO ≤ 0.05).
- **Polymarket CLOB API**: Price history per market (`/prices-history?market={token_id}`). Rate-limited via semaphore (30 concurrent).
- **Kalshi API**: Settled markets with last price and open interest. Cursor-based pagination.
- **Manifold API**: Resolved binary markets with probability and volume. Offset-based pagination.
- Per Polymarket market: snapshot price at 14, 7, 3, 1, 0 days before expiry (~5 samples per market).
- Price lookup: bisect + 12h tolerance for closest timestamp match.
- **Features computed** (19 total): yes_price, theme, volume, market_age_days, price_momentum_7d, price_momentum_1d, price_volatility_7d, volume_per_day, price_distance_50, neg_risk, days_before_expiry, question_length, has_numbers, spread, hurst, book_imbalance, contrarian_conf, n_evidence, volume_ratio.
- **Target**: outcome (YES=1, NO=0). **Mispricing label**: yes_price < 0.35 & outcome=1, or yes_price > 0.65 & outcome=0.
- Stored in shared PostgreSQL tables `ml_training_data` and `ml_models`.

### Model Training
- Two XGBoost classifiers: **Model 1** P(YES outcome), **Model 2** P(market is mispriced).
- Time-series train/test split by `collected_at` chronological order (75/25).
- Config: n_estimators=300, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=5.
- Metrics: Brier score (primary), accuracy, feature importance, filtered accuracy (trades where mispricing > 0.5).
- Models serialized to JSON and stored in `ml_models` DB table (hot-reloaded after training).

### Inference API
- `GET /predict` returns `{p_yes, p_mispriced}` for given features.
- quant-engine calls it for top 5 signals during signal generation.
- Handles old models with fewer features via `_safe_predict` (adds missing cols as NaN).

## Known Issues
- **Look-ahead bias**: price_momentum_7d and price_volatility_7d may include prices up to observation point at training time. At inference time, quant-engine provides real-time features computed from past data only.
- **Nullable features**: Kalshi and Manifold samples have NULL momentum/volatility (only Polymarket has price history). XGBoost handles NaN natively.
- **No model versioning**: `ml_models` table has single row (id='main'), training overwrites previous model.

## Modules
- **main.py** (~260 lines) — FastAPI server, training orchestration via background tasks, inference endpoint.
- **ml/signal_model.py** (~275 lines) — XGBoost training (2 models) and inference, feature preparation, model serialization.
- **ml/data_collector.py** (~255 lines) — Polymarket Gamma + CLOB API data collection, feature engineering, async batch processing.
- **ml/kalshi_collector.py** (~145 lines) — Kalshi settled markets collection.
- **ml/manifold_collector.py** (~150 lines) — Manifold Markets resolved collection.
- **utils/db.py** (~150 lines) — PostgreSQL: `ml_training_data` + `ml_models` tables, CRUD, batch upsert, model persistence.
- **utils/telegram.py** (~27 lines) — Async Telegram notifications for training results.

## Database
Uses shared PostgreSQL (same as quant-engine). Owns two tables:
- `ml_training_data` (UNIQUE on market_id + days_before_expiry) — training samples with 19 features.
- `ml_models` (PK: id='main') — serialized model bytes (BYTEA) + metrics (JSONB) + trained_at timestamp.

Read-only access to `markets`, `positions`, `signals` for cross-referencing.
