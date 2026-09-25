"""Exploratory training-only comparison of absolute and relative band power."""

import json

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from resilience import ARTIFACTS


def main():
    saved = np.load(ARTIFACTS / "train_features.npz")
    x = saved["X"]
    y = saved["y"]
    log_power = x[:, [0, 4, 8, 12]]
    relative = log_power - log_power.mean(axis=1, keepdims=True)
    feature_sets = {
        "original_16": x,
        "relative_power_4": relative,
        "relative_plus_dynamics_12": np.column_stack(
            [relative, x[:, [1, 5, 9, 13, 3, 7, 11, 15]]]),
    }
    models = {
        "mean": lambda: DummyRegressor(),
        "ridge_100": lambda: make_pipeline(StandardScaler(), Ridge(alpha=100)),
        "forest": lambda: RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                                                 max_features=0.75, random_state=42,
                                                 n_jobs=-1),
    }
    report = {}
    for features_name, features in feature_sets.items():
        for model_name, factory in models.items():
            scores = []
            for seed in (42, 43, 44, 45, 46):
                predicted = np.empty(len(y))
                for fit_idx, held_out in KFold(n_splits=5, shuffle=True,
                                               random_state=seed).split(features):
                    model = factory().fit(features[fit_idx], y[fit_idx])
                    predicted[held_out] = np.clip(model.predict(features[held_out]), 0, 40)
                scores.append(float(np.sqrt(mean_squared_error(y, predicted))))
            report[f"{features_name}/{model_name}"] = {
                "mean_rmse": float(np.mean(scores)),
                "range_rmse": [float(min(scores)), float(max(scores))],
            }
    with (ARTIFACTS / "feature_set_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
