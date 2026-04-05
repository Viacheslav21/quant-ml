"""
ML Signal Model — XGBoost classifier for predicting market outcomes.
Trained on historical Polymarket data, outputs calibrated P(YES).
"""

import json
import logging
import io
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

log = logging.getLogger("ml_model")

FEATURES = [
    # Original features
    "yes_price",
    "volume",
    "days_before_expiry",
    "market_age_days",
    "price_momentum_7d",
    "price_momentum_1d",
    "price_volatility_7d",
    "volume_per_day",
    "price_distance_50",
    "neg_risk",
    "theme_encoded",
    "question_length",
    "has_numbers",
    "spread",
    # New features from engine (v2)
    "hurst",              # Hurst exponent: trending vs mean-reverting
    "book_imbalance",     # Order book bid/ask imbalance [-1, 1]
    "contrarian_conf",    # Contrarian signal confidence [0, 1]
    "n_evidence",         # Number of active evidence sources
    "volume_ratio",       # Current vs average volume ratio
]

THEME_MAP = {
    "crypto": 0, "other": 1, "war": 2, "iran": 3, "israel": 4,
    "trump": 5, "election": 6, "oil": 7, "gold": 8, "fed": 9,
    "ukraine": 10, "russia": 11, "china": 12, "peace": 13,
    "social": 14, "tech": 15, "military": 16, "geopolitics": 17,
    "stocks": 18, "usgov": 19, "celeb": 20,
}


class SignalModel:
    def __init__(self):
        self.model = None           # P(YES) model
        self.mispricing_model = None # P(market is wrong) model
        self.metrics = {}

    def train(self, samples: list) -> dict:
        """Train both models. Returns metrics dict."""
        df = self._prepare_data(samples)
        if len(df) < 50:
            log.warning(f"[MODEL] Only {len(df)} samples, need at least 50")
            return {"error": "not enough data"}

        log.info(f"[MODEL] Training on {len(df)} samples ({df['outcome'].sum()} YES, {len(df) - df['outcome'].sum()} NO)")

        # Time-series split — sort chronologically to prevent data leakage
        if "collected_at" in df.columns:
            df = df.sort_values("collected_at", ascending=True, na_position="first").reset_index(drop=True)
        else:
            df = df.sort_values("id", ascending=True).reset_index(drop=True)
        split_idx = int(len(df) * 0.75)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        log.info(f"[MODEL] Split: train={len(train_df)}, test={len(test_df)}")

        # ── Model 1: P(YES) ──
        X_train = train_df[FEATURES]
        y_train = train_df["outcome"]
        X_test = test_df[FEATURES]
        y_test = test_df["outcome"]

        self.model = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", random_state=42, verbosity=0,
            subsample=0.8, colsample_bytree=0.8,  # regularization
            min_child_weight=5,
        )
        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        p_test = self.model.predict_proba(X_test)[:, 1]

        # ── Model 2: P(market is significantly wrong) ──
        # Target: was the market confident but wrong?
        # mispriced=1 if market said <35% YES but outcome=YES, or >65% YES but outcome=NO
        df["mispriced"] = (
            ((df["yes_price"] < 0.35) & (df["outcome"] == 1)) |
            ((df["yes_price"] > 0.65) & (df["outcome"] == 0))
        ).astype(int)
        train_mis = df.iloc[:split_idx]
        test_mis = df.iloc[split_idx:]
        y_train_mis = train_mis["mispriced"]
        y_test_mis = test_mis["mispriced"]

        base_mis = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", random_state=42, verbosity=0,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5,
        )
        base_mis.fit(
            train_mis[FEATURES], y_train_mis,
            eval_set=[(test_mis[FEATURES], y_test_mis)], verbose=False,
        )
        self.mispricing_model = base_mis
        p_test_mis = base_mis.predict_proba(test_mis[FEATURES])[:, 1]

        # ── Metrics ──
        self.metrics = {
            # Model 1: P(YES)
            "test_brier": round(brier_score_loss(y_test, p_test), 4),
            "test_accuracy": round(accuracy_score(y_test, (p_test > 0.5).astype(int)), 4),
            # Model 2: Mispricing
            "mis_accuracy": round(accuracy_score(y_test_mis, (p_test_mis > 0.5).astype(int)), 4),
            "mis_brier": round(brier_score_loss(y_test_mis, p_test_mis), 4),
            "mis_rate": round(df["mispriced"].mean(), 4),
            # General
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_total": len(df),
            "yes_rate": round(df["outcome"].mean(), 4),
        }

        # Market baseline
        if "yes_price" in test_df.columns:
            market_brier = round(brier_score_loss(y_test, test_df["yes_price"].clip(0.01, 0.99)), 4)
            self.metrics["market_brier"] = market_brier
            self.metrics["brier_improvement"] = round(market_brier - self.metrics["test_brier"], 4)

        # Profit simulation: if we only traded when mispricing model says >50%, what's the edge?
        mis_mask = p_test_mis > 0.5
        if mis_mask.sum() > 0:
            # For "mispriced" markets, our P(YES) model's accuracy
            filtered_acc = round(accuracy_score(
                y_test[mis_mask], (p_test[mis_mask] > 0.5).astype(int)), 4)
            self.metrics["filtered_accuracy"] = filtered_acc
            self.metrics["filtered_count"] = int(mis_mask.sum())
            self.metrics["filtered_pct"] = round(mis_mask.mean(), 4)

        # Feature importance (mispricing model — more actionable)
        importance = dict(zip(FEATURES, self.mispricing_model.feature_importances_))
        self.metrics["feature_importance"] = dict(sorted(importance.items(), key=lambda x: -x[1]))

        log.info(f"[MODEL 1] P(YES) — Brier: {self.metrics['test_brier']:.4f} (market: {self.metrics.get('market_brier', 'N/A')}) Acc: {self.metrics['test_accuracy']:.1%}")
        log.info(f"[MODEL 2] Mispricing — Brier: {self.metrics['mis_brier']:.4f} Acc: {self.metrics['mis_accuracy']:.1%} Rate: {self.metrics['mis_rate']:.1%}")
        if "filtered_accuracy" in self.metrics:
            log.info(f"[MODEL 2] Filtered trades: {self.metrics['filtered_count']}/{len(test_df)} ({self.metrics['filtered_pct']:.0%}) → Acc: {self.metrics['filtered_accuracy']:.1%}")
        log.info(f"[MODEL] Top mispricing features: {', '.join(f'{k}={v:.3f}' for k, v in list(self.metrics['feature_importance'].items())[:5])}")

        return self.metrics

    def predict(self, features: dict) -> dict:
        """Predict P(YES) and P(mispriced) for a single market."""
        row = self._features_to_row(features)
        df = pd.DataFrame([row])[FEATURES]
        # Ensure numeric types (None → NaN, object → float)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        result = {"p_yes": 0.5, "p_mispriced": 0.0}

        # Use only features the model was trained on (handles old 14-feature models)
        def _safe_predict(model, data):
            if hasattr(model, "feature_names_in_"):
                cols = list(model.feature_names_in_)
            elif hasattr(model, "get_booster"):
                cols = model.get_booster().feature_names
            else:
                cols = list(data.columns)
            # Only keep columns model knows; add missing as NaN
            for c in cols:
                if c not in data.columns:
                    data[c] = np.nan
            return model.predict_proba(data[cols])[:, 1][0]

        if self.model is not None:
            try:
                result["p_yes"] = float(_safe_predict(self.model, df.copy()))
            except Exception as e:
                log.warning(f"[MODEL] P(YES) predict failed: {e}")
        if self.mispricing_model is not None:
            try:
                result["p_mispriced"] = float(_safe_predict(self.mispricing_model, df.copy()))
            except Exception as e:
                log.warning(f"[MODEL] Mispricing predict failed: {e}")
        return result

    def save_bytes(self) -> bytes:
        """Serialize both models to bytes for DB storage."""
        import tempfile, os, json as _json
        models = {}
        for name, model in [("main", self.model), ("mispricing", self.mispricing_model)]:
            if model is None or not hasattr(model, "save_model"):
                continue
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                tmp_path = f.name
            try:
                model.save_model(tmp_path)
                with open(tmp_path, "rb") as f:
                    models[name] = f.read().decode("utf-8")
            finally:
                os.unlink(tmp_path)
        return _json.dumps(models).encode("utf-8")

    def load_bytes(self, data: bytes):
        """Load both models from bytes."""
        if not data:
            return
        import tempfile, os, json as _json
        models = _json.loads(data.decode("utf-8"))
        for name, content in models.items():
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
                f.write(content)
                tmp_path = f.name
            try:
                m = xgb.XGBClassifier()
                m.load_model(tmp_path)
                if name == "main":
                    self.model = m
                elif name == "mispricing":
                    self.mispricing_model = m
            finally:
                os.unlink(tmp_path)
        log.info(f"[MODEL] Loaded {len(models)} models from DB")

    def save_file(self, path: str = "model.json"):
        if self.model:
            self.model.save_model(path)
        if self.mispricing_model:
            self.mispricing_model.save_model(path.replace(".json", "_mis.json"))
        log.info(f"[MODEL] Saved to {path}")

    def load_file(self, path: str = "model.json"):
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)
        log.info(f"[MODEL] Loaded from {path}")

    def _prepare_data(self, samples: list) -> pd.DataFrame:
        """Convert DB samples to training DataFrame."""
        df = pd.DataFrame(samples)

        # Encode theme as integer
        df["theme_encoded"] = df["theme"].map(THEME_MAP).fillna(len(THEME_MAP))

        # Convert bools to int
        for col in ["neg_risk", "has_numbers"]:
            if col in df.columns:
                df[col] = df[col].fillna(False).astype(int)

        # Ensure all feature columns exist
        for col in FEATURES:
            if col not in df.columns:
                df[col] = np.nan

        # XGBoost handles NaN natively, no need to impute
        return df

    def _features_to_row(self, features: dict) -> dict:
        """Convert inference features dict to model input row."""
        row = {}
        for f in FEATURES:
            if f == "theme_encoded":
                row[f] = THEME_MAP.get(features.get("theme", "other"), len(THEME_MAP))
            elif f == "neg_risk":
                row[f] = int(features.get("neg_risk", False))
            else:
                row[f] = features.get(f)
        return row
