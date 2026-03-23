# ML Signal Model — Plan

## Цель

Обучить локальную ML модель на исторических данных Polymarket.
Модель предсказывает реальную P(YES) для маркета → сравниваем с рыночной ценой → находим mispricing.

Работает как дополнительный evidence source в Bayesian fusion (рядом с prospect, history, volume и т.д.),
а в перспективе может заменить Claude confirmation (дешевле, быстрее, учится).

## Архитектура

```
[1. Data Collection]
    Gamma API: закрытые маркеты (metadata, outcome)
    CLOB API: price history по token_id (цены в прошлом)
        ↓
[2. Feature Engineering]
    Для каждого маркета в момент T (N дней до expiry):
    - yes_price_at_T, volume, theme, days_to_expiry, spread, price_momentum
        ↓
[3. Training]
    XGBoost classifier: features → P(YES outcome)
    Train/test split по времени (не random!)
        ↓
[4. Integration]
    math_engine.analyze() → p_ml = model.predict(features)
    → добавляется в Bayesian fusion как ещё один evidence source
        ↓
[5. Monitoring]
    Brier score на новых трейдах, retrain по расписанию
```

## Phase 1: Data Collection (`ml/data_collector.py`)

### Источники
1. **Gamma API** — metadata закрытых маркетов
   - `GET /markets?closed=true&order=volume&ascending=false&limit=100&offset=N`
   - Поля: id, question, volume, clobTokenIds, endDate, createdAt, outcomePrices, outcomes
   - Пагинация: до 5000-10000 маркетов

2. **CLOB API** — историческая цена по token_id
   - `GET /prices-history?market={token_id}&interval=all&fidelity=60`
   - Возвращает: [{t: timestamp, p: price}, ...] — вся история цен

### Логика сбора
```python
for market in closed_markets:
    token_id = market["clobTokenIds"][0]  # YES token
    price_history = clob_api.get_prices(token_id)

    # Определяем outcome
    outcome = 1 if outcomePrices[0] >= 0.95 else 0

    # Берём snapshot цены за 7 дней до expiry (имитируем момент входа)
    # + snapshot за 3 дня, 1 день — для разных горизонтов
    end_date = parse(market["endDate"])
    for days_before in [14, 7, 3, 1]:
        target_ts = (end_date - timedelta(days=days_before)).timestamp()
        price_at_T = find_closest_price(price_history, target_ts)
        if price_at_T:
            save_training_sample(market, price_at_T, days_before, outcome)
```

### Хранение
- PostgreSQL таблица `ml_training_data` или CSV файл
- ~5000 маркетов × 4 горизонта = ~20000 training samples

### Rate limits
- Gamma API: без лимита (public)
- CLOB API: 100 req/sec
- Сбор ~5000 маркетов ≈ 1-2 минуты

## Phase 2: Feature Engineering

### Фичи из маркета (на момент T)
| Feature | Описание | Источник |
|---|---|---|
| `yes_price` | Цена YES на момент T | CLOB price history |
| `theme` | Категория (encoded) | keyword detection |
| `volume` | Общий объём торгов | Gamma API |
| `days_to_expiry` | Дней до закрытия | endDate - T |
| `market_age` | Дней с создания | T - createdAt |
| `price_momentum_7d` | Изменение цены за 7 дней до T | price history |
| `price_momentum_1d` | Изменение цены за 1 день до T | price history |
| `price_volatility` | Std dev цены за последние 7 дней | price history |
| `volume_per_day` | volume / market_age | computed |
| `neg_risk` | Multi-outcome event | Gamma API |
| `is_sports` | Спортивный маркет (по keywords) | question text |
| `price_distance_50` | abs(yes_price - 0.50) | computed |

### Кодирование
- `theme` → one-hot или target encoding
- `is_sports` → binary
- Все numeric → без нормализации (XGBoost не нужна)

### Target
- `outcome`: 1 = YES resolved, 0 = NO resolved

## Phase 3: Training (`ml/signal_model.py`)

### Модель
```python
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

class SignalModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            objective='binary:logistic',  # выдаёт вероятность
            eval_metric='logloss',
        )

    def train(self, df):
        # Split по времени! Не random
        # Train: маркеты до 2026-01
        # Test: маркеты 2026-01 — 2026-03
        train = df[df['end_date'] < '2026-01-01']
        test = df[df['end_date'] >= '2026-01-01']

        self.model.fit(train[FEATURES], train['outcome'],
                       eval_set=[(test[FEATURES], test['outcome'])])

        # Калибровка: Platt scaling
        from sklearn.calibration import CalibratedClassifierCV
        self.calibrated = CalibratedClassifierCV(self.model, cv=5)
        self.calibrated.fit(train[FEATURES], train['outcome'])

    def predict(self, features: dict) -> float:
        """Returns calibrated P(YES)."""
        return self.calibrated.predict_proba([features])[0][1]

    def save(self, path='ml/model.json'):
        self.model.save_model(path)

    def load(self, path='ml/model.json'):
        self.model.load_model(path)
```

### Метрики
- **Brier score** — калибровка (главная метрика)
- **Log loss** — дискриминация
- **Profit simulation** — "если бы торговали по модели, какой P&L?"
- **Feature importance** — какие фичи самые полезные

### Валидация
- Time series split (не random!) — будущее не заглядываем
- Walk-forward: train на 2024-2025, test на 2026-01, retrain, test на 2026-02, ...
- Сравнение с baseline: "просто верить рыночной цене" (Brier = market)

## Phase 4: Integration в quant-engine

### Вариант A: дополнительный evidence source
```python
# В math_engine.analyze():
p_ml = self.signal_model.predict({
    "yes_price": p_market,
    "theme": theme,
    "volume": volume,
    "days_to_expiry": days_left,
    ...
})

# Добавляется в Bayesian fusion наравне с другими
evidence = [p_history, p_vol, p_time, p_momentum, p_ml, ...]
p_final = bayesian_update(p_prospect, *evidence)
```

### Вариант B: замена Claude confirmation
```python
# Вместо вызова Haiku:
p_ml = signal_model.predict(features)
ml_confirms = abs(p_ml - p_market) > 0.05  # модель тоже видит mispricing

if ml_confirms:
    sig["p_ml"] = p_ml
    sig["source"] = "ml"
    confirmed.append(sig)
```

### Рекомендация
Начать с **Вариант A** — ML как ещё один голос в ансамбле.
Когда накопится 100+ собственных трейдов — можно перейти к **Вариант B**.

## Phase 5: Мониторинг и Retrain

### Онлайн мониторинг
- Каждый закрытый трейд → сравнить p_ml vs outcome → rolling Brier score
- Если Brier растёт (модель деградирует) → Telegram alert
- Dashboard: график калибровки ML модели

### Retrain
- Каждую неделю (или по расписанию) перезапускать data collection + train
- Новые маркеты добавляются в training set
- Собственные трейды тоже добавляются (с бОльшим весом — это наш фидбек)
- Модель сохраняется как `ml/model.json`, подхватывается при рестарте

## Порядок реализации

### Sprint 1: Data Collection (первый)
1. `ml/data_collector.py` — скрипт сбора данных
2. Загрузить 5000+ закрытых маркетов + price history
3. Сохранить в PostgreSQL (`ml_training_data`) или CSV
4. Проверить качество данных

### Sprint 2: Model Training
1. `ml/signal_model.py` — XGBoost + калибровка
2. Feature engineering
3. Train/test split по времени
4. Оценить Brier score и profit simulation
5. Сравнить с baseline (market price = prediction)

### Sprint 3: Integration
1. Загрузка модели при старте (`math_engine.__init__`)
2. `predict()` в `analyze()` как evidence source
3. Логирование: `[ML] p_ml=0.35 vs market=0.25 for 'Will Gold...'`
4. A/B: CONFIG_TAG для сравнения с/без ML

### Sprint 4: Monitoring + Retrain
1. Rolling Brier в dashboard
2. Cron job для retrain (или в HISTORY_INTERVAL)
3. Собственные трейды как training data

## Зависимости
```
xgboost>=2.0.0
scikit-learn>=1.4.0
pandas>=2.0.0
```

## Вопросы для решения
- Сколько маркетов собирать? (5K vs 10K vs all)
- Какой горизонт входа? (7 дней до expiry? или разные?)
- Включать ли спортивные маркеты? (другие закономерности)
- Вес собственных трейдов в retrain? (2x? 5x?)
