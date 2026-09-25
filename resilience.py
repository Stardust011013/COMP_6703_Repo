"""

Commands (run from assignment_2_):
  python resilience.py split
  python resilience.py inspect
  python resilience.py train
  python resilience.py predict
  python resilience.py score

"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from scipy import io, signal
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# in case of FR doesnt work well: use SVM
# edited Sept 25: neither works well
from sklearn.svm import SVR


ROOT = Path(__file__).resolve().parent

# directories. make sure to split validation/training later on
TRAIN_EEG = ROOT / "EEG"
val_eeg = ROOT / "validation" / "EEG"
val_ids = ROOT / "validation" / "ids.csv"
results = ROOT / "results"
sample_freq = 250.0
seconds_per_epoch = 30

# preprocessing. maybe try the github preproc pipeline later
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
FEATURE_NAMES = [f"{band}_{stat}" for band in BANDS for stat in
                 ("median_log_power", "temporal_iqr", "spatial_iqr", "temporal_slope")]

# These IDs come from the provided, fixed split. No labels are needed here.
VALIDATION_IDS = ("018", "129", "132", "136", "151", "161", "177", "182", "190", "210")


def read_rows(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        # if reader.fieldnames is None or not set(fields).issubset(reader.fieldnames):
        #     raise ValueError(f"{path} needs columns {fields}")
        rows = list(reader)
    # ids = [row["id"] for row in rows]
    # if len(ids) != len(set(ids)) or any(len(i) != 3 or not i.isdigit() for i in ids):
    #     raise ValueError(f"Duplicate or malformed IDs in {path}")
    return rows


def split() -> None:
    train_ids = {row["id"] for row in read_rows(ROOT / "label.csv", ("id", "label"))}
    val_ids = set(VALIDATION_IDS)
    # if len(train_ids) != 40 or len(val_ids) != 10 or train_ids & val_ids:
    #     raise ValueError("Expected disjoint 40/10 training and validation IDs")
    # all_ids = train_ids | val_ids
    # actual_ids = {p.stem for folder in (TRAIN_EEG, val_eeg) if folder.exists()
    #               for p in folder.glob("*.mat")}
    # if actual_ids != all_ids:
    #     raise ValueError(f"MAT ID mismatch: missing={sorted(all_ids - actual_ids)}, "
    #                      f"unexpected={sorted(actual_ids - all_ids)}")
    # if any((val_eeg / f"{i}.mat").exists() and (TRAIN_EEG / f"{i}.mat").exists()
    #        for i in val_ids):
    #     raise ValueError("Validation file exists in both directories")
    val_eeg.mkdir(parents=True, exist_ok=True)
    for participant_id in VALIDATION_IDS:
        source = TRAIN_EEG / f"{participant_id}.mat"
        destination = val_eeg / source.name
        if source.exists():
            source.rename(destination)
        elif not destination.exists():
            raise FileNotFoundError(source) # for debug purpose
    with val_ids.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("id",))
        writer.writerows((i,) for i in VALIDATION_IDS)
    print(f"Separated {len(train_ids)} training and {len(val_ids)} validation EEG files")


# vibe coded eeg loader from mat files
def load_eeg(path: Path) -> tuple[np.ndarray, float]:
    """Read only the large signal and sampling-rate variables from one MAT file."""
    try:
        entries = io.whosmat(path)
        candidates = [(name, shape) for name, shape, _ in entries
                      if len(shape) == 2 and 129 in shape and max(shape) >= 450_000]
        if len(candidates) != 1:
            raise ValueError(f"Expected one 129 x T signal in {path}, got {candidates}")
        name, _ = candidates[0]
        workspace = io.loadmat(path, variable_names=[name, "EEGSamplingRate"])
        x = np.asarray(workspace[name], dtype=np.float32)
        sample_freq = float(np.asarray(workspace["EEGSamplingRate"]).squeeze())
    except NotImplementedError as error:
        raise RuntimeError("MATLAB v7.3 files require an h5py loader; inspect the file format") from error
    if x.shape[0] != 129:
        x = x.T
    if x.shape[0] != 129 or not np.isclose(sample_freq, sample_freq):
        raise ValueError(f"Unexpected shape or sampling rate in {path}: {x.shape}, {sample_freq}")
    # Use the last 128 rows (E2-E129) 
    return x[1:], sample_freq


def inspect() -> None:
    for folder, name in ((TRAIN_EEG, "train"), (val_eeg, "validation")):
        paths = sorted(folder.glob("*.mat")) if folder.exists() else []
        print(f"{name}: {len(paths)} MAT files")
        if paths:
            x, sample_freq = load_eeg(paths[0])
            print(f"  example {paths[0].name}: {x.shape}, {sample_freq:g} Hz, "
                  f"{x.shape[1] / sample_freq / 60:.1f} min")


def eeg_features(path: Path) -> np.ndarray:
    x, sample_freq = load_eeg(path)
    epoch_samples = int(30 * sample_freq) # 30 seconds per epoch
    n_epochs = x.shape[1] // epoch_samples
    if n_epochs < 60:
        raise ValueError(f"Recording shorter than 30 minutes: {path}")
    sos = signal.butter(4, (0.3, 35.0), btype="bandpass", sample_freq=sample_freq, output="sos")
    by_epoch = []
    for k in range(n_epochs):
        epoch = x[:, k * epoch_samples:(k + 1) * epoch_samples].copy()
        finite = np.isfinite(epoch)
        valid = finite.mean(axis=1) >= 0.99
        if valid.sum() < 96:
            continue
        epoch[~finite] = 0.0
        epoch = epoch[valid]
        epoch -= np.median(epoch, axis=1, keepdims=True)
        clean = signal.sosfiltfilt(sos, epoch, axis=1)
        freqs, psd = signal.welch(clean, sample_freq=sample_freq, nperseg=1000, axis=1)
        powers = []
        for low, high in BANDS.values():
            bins = (freqs >= low) & (freqs < high)
            power = np.trapz(psd[:, bins], freqs[bins], axis=1)
            powers.append(np.log10(np.maximum(power, 1e-12)))
        by_epoch.append(np.stack(powers, axis=1))  # electrodes x bands
    if len(by_epoch) < 20:
        raise ValueError(f"Too few usable EEG epochs: {path}")
    epochs = np.stack(by_epoch)  # time | electrodes | bands structure 
    spatial_median = np.median(epochs, axis=1)
    spatial_iqr = np.percentile(epochs, 75, axis=1) - np.percentile(epochs, 25, axis=1)
    t = np.linspace(-0.5, 0.5, len(epochs))
    features = []
    for b in range(len(BANDS)):
        power_t = spatial_median[:, b]
        features.extend((
            np.median(power_t),
            np.percentile(power_t, 75) - np.percentile(power_t, 25),
            np.median(spatial_iqr[:, b]),
            np.dot(t, power_t - power_t.mean()) / np.dot(t, t),
        ))
    result = np.asarray(features, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"Non-finite EEG features: {path}")
    return result


def check_partition() -> list[dict[str, str]]:
    rows = read_rows(ROOT / "label.csv", ("id", "label"))
    if len(rows) != 40 or {p.stem for p in TRAIN_EEG.glob("*.mat")} != {r["id"] for r in rows}:
        raise ValueError("Training EEG folder must contain exactly the 40 labeled subjects. Run split.")
    if any((TRAIN_EEG / f"{i}.mat").exists() for i in VALIDATION_IDS):
        raise ValueError("Validation EEG found in training folder")
    return rows


def metrics(y: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {"rmse": float(np.sqrt(mean_squared_error(y, predicted))),
            "mae": float(mean_absolute_error(y, predicted)),
            "r2": float(r2_score(y, predicted))}


def train() -> None:
    rows = check_partition()
    ids = [row["id"] for row in rows]
    y = np.asarray([float(row["label"]) for row in rows])
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 40)):
        raise ValueError("Training labels must be finite and in [0, 40]")
    X = np.vstack([eeg_features(TRAIN_EEG / f"{i}.mat") for i in ids])
    results.mkdir(exist_ok=True)
    models = {
        "mean": DummyRegressor(strategy="mean"),
        "random_forest": RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                                                max_features=0.75, random_state=42,
                                                n_jobs=-1),
        "svr_rbf": make_pipeline(StandardScaler(), SVR(C=10, epsilon=1.0)),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_report = {}
    for name, model in models.items():
        prediction = np.clip(cross_val_predict(model, X, y, cv=cv), 0, 40)
        cv_report[name] = metrics(y, prediction)
        print(f"5-fold training CV {name}: {cv_report[name]}")
    # The forest is the predeclared primary model; validation never selects it.
    forest = models["random_forest"].fit(X, y)
    joblib.dump({"model": forest, "features": FEATURE_NAMES, "train_ids": ids},
                results / "forest.joblib")
    with (results / "train_cv.json").open("w", encoding="utf-8") as handle:
        json.dump({"cv": cv_report, "train_ids": ids, "feature_names": FEATURE_NAMES,
                   "primary_model": "random_forest"}, handle, indent=2)
    print(f"Fitted forest on {len(ids)} training participants")


def predict() -> None:
    rows = read_rows(val_ids, ("id",))
    ids = [row["id"] for row in rows]
    if len(ids) != 10 or set(ids) != set(VALIDATION_IDS):
        raise ValueError("Unexpected validation IDs")
    if {p.stem for p in val_eeg.glob("*.mat")} != set(ids):
        raise ValueError("Validation EEG folder does not match validation IDs")
    saved = joblib.load(results / "forest.joblib")
    if saved["features"] != FEATURE_NAMES or set(saved["train_ids"]) & set(ids):
        raise ValueError("Model schema or participant split mismatch")
    X = np.vstack([eeg_features(val_eeg / f"{i}.mat") for i in ids])
    scores = np.clip(saved["model"].predict(X), 0, 40)
    output = results / "validation_predictions.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("id", "prediction"))
        writer.writerows((i, f"{score:.6f}") for i, score in zip(ids, scores))
    print(f"Wrote {output}")


def score() -> None:
    predictions = read_rows(results / "validation_predictions.csv", ("id", "prediction"))
    truth = read_rows(ROOT / "val.csv", ("id", "label"))
    by_id = {row["id"]: float(row["prediction"]) for row in predictions}
    if set(by_id) != {row["id"] for row in truth}:
        raise ValueError("Predictions and validation labels refer to different IDs")
    y = np.asarray([float(row["label"]) for row in truth])
    p = np.asarray([by_id[row["id"]] for row in truth])
    report = metrics(y, p)
    with (results / "validation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    from write_evaluation_report import write_report
    write_report()
    print(f"Validation: {report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("split", "inspect", "train", "predict", "score"))
    args = parser.parse_args()
    globals()[args.command]()



