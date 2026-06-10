"""
train_synthetic_model.py

Trains a system-wide hourly Bluebikes demand model on SYNTHETIC data whose
48-feature schema matches exactly what the UI backend sends to /predict
(see bluebikes-ui/backend/server.js -> generateSystemWideFeaturesForCloudRun).

This is a stand-in artifact for the original team's trained model, which was
never published. It produces plausible, input-responsive predictions:
demand follows the real observed hourly ride patterns and is modulated by
temperature and precipitation, so the UI's weather sliders behave sensibly.

Run inside the SAME environment as the serving container (python:3.11-slim +
model_deployment/requirements.txt) so the pickle loads without version skew.

Usage:
    python train_synthetic_model.py /path/to/current_model.pkl
"""

import sys
import logging

import numpy as np
import joblib
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Real observed system-wide rides per hour, mirrored from server.js
WEEKDAY_PATTERNS = {
    0: 90, 1: 51, 2: 26, 3: 14, 4: 21, 5: 86,
    6: 254, 7: 616, 8: 1088, 9: 699, 10: 479, 11: 501,
    12: 582, 13: 593, 14: 636, 15: 784, 16: 1101,
    17: 1459, 18: 1141, 19: 824, 20: 575, 21: 434,
    22: 323, 23: 202,
}
WEEKEND_PATTERNS = {
    0: 234, 1: 209, 2: 123, 3: 36, 4: 23, 5: 35,
    6: 78, 7: 144, 8: 268, 9: 451, 10: 611, 11: 721,
    12: 821, 13: 860, 14: 883, 15: 891, 16: 877,
    17: 834, 18: 765, 19: 614, 20: 456, 21: 358,
    22: 302, 23: 229,
}


def build_features(hour, day_of_week, month, year, day, temperature_c, precipitation_mm):
    """Replicate server.js generateSystemWideFeaturesForCloudRun exactly (48 features)."""
    is_weekend = 1 if day_of_week in (0, 6) else 0
    base_rides = (WEEKEND_PATTERNS if is_weekend else WEEKDAY_PATTERNS)[hour]

    rides_last_hour = base_rides
    rides_same_hour_yesterday = round(base_rides * 0.95)
    rides_same_hour_last_week = round(base_rides * 0.98)
    rides_rolling_3h = round(base_rides * 1.05)
    rides_rolling_24h = round(base_rides * 8)

    temp_f = (temperature_c * 9 / 5) + 32
    temp_max = temp_f + 4
    temp_min = temp_f - 4
    temp_range = 8
    temp_avg = temp_f
    precip_in = precipitation_mm / 25.4

    morning_peak = 1 if 7 <= hour <= 9 else 0
    evening_peak = 1 if 17 <= hour <= 19 else 0

    features = [
        # 1-5 temporal
        hour, day_of_week, month, year, day,
        # 6-11 cyclical
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * day_of_week / 7), np.cos(2 * np.pi * day_of_week / 7),
        np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
        # 12-16 time-of-day flags
        morning_peak,
        evening_peak,
        1 if (hour >= 22 or hour <= 5) else 0,
        1 if 11 <= hour <= 14 else 0,
        is_weekend,
        # 17-19 interactions
        is_weekend * (1 if (hour >= 22 or hour <= 5) else 0),
        (0 if is_weekend else 1) * morning_peak,
        (0 if is_weekend else 1) * evening_peak,
        # 20-24 weather
        temp_max, temp_min, precip_in, temp_range, temp_avg,
        # 25-28 weather flags
        1 if precip_in > 0.01 else 0,
        1 if precip_in > 0.1 else 0,
        1 if temp_f < 50 else 0,
        1 if temp_f > 77 else 0,
        # 29-33 historical
        rides_last_hour, rides_same_hour_yesterday, rides_same_hour_last_week,
        rides_rolling_3h, rides_rolling_24h,
        # 34-40 trip stats (constants, as backend sends)
        15.5, 8.2, 12.0, 3.8, 1.9, 2.5, 0.65,
        # 41-48 bias-mitigation features
        1 if hour == 8 else 0,
        1 if hour in (17, 18) else 0,
        1.0 if hour in (8, 17, 18) else (0.5 if hour in (7, 9, 16, 19) else 0.0),
        1 if rides_last_hour > 800 else 0,
        1 if rides_last_hour < 200 else 0,
        abs(rides_last_hour - rides_rolling_3h),
        1 if ((1 if hour == 8 else 0)
              + (1 if hour in (17, 18) else 0)
              + (0 if is_weekend else 1) * morning_peak
              + (0 if is_weekend else 1) * evening_peak) > 0 else 0,
        (0 if 0 <= hour < 6 else 1 if 6 <= hour < 10 else 2 if 10 <= hour < 14
         else 3 if 14 <= hour < 18 else 4),
    ]
    return features, base_rides, temp_f, precip_in


def weather_multiplier(temp_f, precip_in, rng):
    """Plausible demand response to weather, so UI sliders matter."""
    mult = 1.0
    if precip_in > 0.3:
        mult *= 0.55
    elif precip_in > 0.1:
        mult *= 0.72
    elif precip_in > 0.01:
        mult *= 0.88
    if temp_f < 32:
        mult *= 0.6
    elif temp_f < 45:
        mult *= 0.8
    elif temp_f < 55:
        mult *= 0.92
    if temp_f > 92:
        mult *= 0.8
    elif 63 <= temp_f <= 80:
        mult *= 1.1
    return mult


def generate_dataset(n=40000, seed=42):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n):
        hour = int(rng.integers(0, 24))
        dow = int(rng.integers(0, 7))
        month = int(rng.integers(1, 13))
        year = 2025
        day = int(rng.integers(1, 29))
        temp_c = float(np.clip(rng.normal(15, 11), -12, 38))
        # precip: mostly dry, occasional rain
        precip_mm = float(max(0.0, rng.exponential(0.8) if rng.random() < 0.25 else 0.0)) * 6

        feats, base_rides, temp_f, precip_in = build_features(
            hour, dow, month, year, day, temp_c, precip_mm
        )
        mult = weather_multiplier(temp_f, precip_in, rng)
        noise = rng.normal(1.0, 0.08)
        target = max(0.0, base_rides * mult * noise)
        X.append(feats)
        y.append(target)
    return np.array(X, dtype=float), np.array(y, dtype=float)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "current_model.pkl"

    logger.info("Generating synthetic training data...")
    X, y = generate_dataset()
    logger.info("Dataset: X=%s, y range [%.1f, %.1f]", X.shape, y.min(), y.max())

    n_test = 4000
    X_train, y_train = X[:-n_test], y[:-n_test]
    X_test, y_test = X[-n_test:], y[-n_test:]

    logger.info("Training XGBRegressor...")
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=4,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    ss_res = float(np.sum((y_test - pred) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    mae = float(np.mean(np.abs(y_test - pred)))
    logger.info("Holdout R2=%.4f  MAE=%.2f", r2, mae)
    logger.info("n_features_in_=%s", getattr(model, "n_features_in_", "?"))

    joblib.dump(model, out_path)
    logger.info("Saved model to %s", out_path)


if __name__ == "__main__":
    main()
