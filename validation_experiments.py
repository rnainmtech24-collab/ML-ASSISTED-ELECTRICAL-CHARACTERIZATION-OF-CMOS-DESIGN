"""Grouped validation and baseline benchmarks using only recorded project data."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from scipy.optimize import least_squares
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

from preprocess import VDD, _parse_suffix


ROOT = Path(__file__).resolve().parent
DEFAULT_SKY130 = ROOT / "data" / "sky130_reference"
DEFAULT_OUTPUT = ROOT / "reports" / "validation_experiments"
BASE_FEATURES = ["gate_v", "drain_v", "temperature_c", "width_m", "length_m"]
SEED = 42


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_180nm_device(device: str, csv_path: Path | None = None) -> pd.DataFrame:
    """Load cleaned 180 nm data into common physical feature columns."""
    device = device.upper()
    if device not in {"NMOS", "PMOS"}:
        raise ValueError(f"Unsupported device type: {device}")
    path = csv_path or ROOT / f"{device.lower()}.csv"
    raw = pd.read_csv(path)
    raw.columns = [str(column).strip() for column in raw.columns]

    temperature_column = "Temperature" if "Temperature" in raw else "Temp"
    if temperature_column not in raw:
        raw[temperature_column] = 27.0
    gate_column = "VGS"
    drain_column = "VDS"
    gate = pd.to_numeric(raw[gate_column], errors="coerce")
    drain = pd.to_numeric(raw[drain_column], errors="coerce")
    if device == "PMOS":
        gate = VDD - gate
        drain = VDD - drain

    data = pd.DataFrame({
        "gate_v": gate,
        "drain_v": drain,
        "temperature_c": pd.to_numeric(raw[temperature_column], errors="coerce"),
        "width_m": raw["Wval"].map(_parse_suffix),
        "length_m": raw["Lval"].map(_parse_suffix),
        "current_a": pd.to_numeric(raw["IDS"], errors="coerce").abs(),
    })
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data.loc[data["current_a"] > 0].copy()
    data["log_current"] = np.log10(data["current_a"])
    if data.empty:
        raise ValueError(f"No valid positive-current rows found in {path}")
    return data.reset_index(drop=True)


def load_sky130_device(data_dir: Path, device: str) -> pd.DataFrame:
    """Load one device from the versioned SKY130 ngspice DC reference data."""
    device = device.upper()
    path = data_dir / "device_dc.csv"
    if not path.is_file():
        raise FileNotFoundError(f"SKY130 reference file not found: {path}")
    raw = pd.read_csv(path)
    required = {
        "device_type", "corner", "temperature_c", "vgs_mag_v",
        "vds_mag_v", "w_um", "l_um", "ids_a",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    selected = raw.loc[raw["device_type"].str.upper() == device].copy()
    data = pd.DataFrame({
        "gate_v": pd.to_numeric(selected["vgs_mag_v"], errors="coerce"),
        "drain_v": pd.to_numeric(selected["vds_mag_v"], errors="coerce"),
        "temperature_c": pd.to_numeric(selected["temperature_c"], errors="coerce"),
        "width_m": pd.to_numeric(selected["w_um"], errors="coerce") * 1e-6,
        "length_m": pd.to_numeric(selected["l_um"], errors="coerce") * 1e-6,
        "current_a": pd.to_numeric(selected["ids_a"], errors="coerce"),
        "corner": selected["corner"].astype(str).str.upper(),
    })
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data.loc[data["current_a"] > 0].copy()
    data["log_current"] = np.log10(data["current_a"])
    if data.empty:
        raise ValueError(f"No valid {device} rows found in {path}")
    return data.reset_index(drop=True)


def group_holdout(
    data: pd.DataFrame,
    *,
    group_column: str,
    heldout_value: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by a whole category and assert the held-out group cannot leak."""
    if group_column not in data:
        raise ValueError(f"Missing group column: {group_column}")
    if group_column in {"width_m", "length_m"}:
        test_mask = np.isclose(
            data[group_column].to_numpy(dtype=float),
            float(heldout_value),
            rtol=1e-9,
            atol=1e-15,
        )
    else:
        test_mask = data[group_column].eq(heldout_value).to_numpy()
    train = data.loc[~test_mask].copy()
    test = data.loc[test_mask].copy()
    if train.empty or test.empty:
        raise ValueError(
            f"Holdout {group_column}={heldout_value!r} produced "
            f"{len(train)} train and {len(test)} test rows."
        )
    if group_column in {"width_m", "length_m"}:
        leaked = np.isclose(
            train[group_column].to_numpy(dtype=float),
            float(heldout_value),
            rtol=1e-9,
            atol=1e-15,
        ).any()
    else:
        leaked = train[group_column].eq(heldout_value).any()
    if leaked:
        raise AssertionError(f"Held-out {group_column} leaked into training rows.")
    return train, test


class SquareLawBaseline:
    """Fit a compact, temperature- and geometry-scaled MOS square-law baseline."""

    def __init__(self) -> None:
        self.parameters: np.ndarray | None = None

    @staticmethod
    def _predict_log(parameters: np.ndarray, features: np.ndarray) -> np.ndarray:
        log_beta, threshold, exponent, log_vsat, temp_slope = parameters
        gate = features[:, 0]
        drain = np.maximum(features[:, 1], 0.0)
        temperature = features[:, 2]
        width = np.maximum(features[:, 3], 1e-30)
        length = np.maximum(features[:, 4], 1e-30)
        overdrive = np.maximum(gate - threshold, 1e-6)
        vsat = np.exp(log_vsat)
        drain_factor = np.maximum(1.0 - np.exp(-drain / vsat), 1e-10)
        return (
            log_beta
            + np.log10(width / length)
            + exponent * np.log10(overdrive)
            + np.log10(drain_factor)
            + temp_slope * (temperature - 27.0)
        )

    def fit(self, features: np.ndarray, target: np.ndarray) -> "SquareLawBaseline":
        if len(features) > 5000:
            rng = np.random.default_rng(SEED)
            indices = rng.choice(len(features), 5000, replace=False)
            features = features[indices]
            target = target[indices]
        gate_min = float(np.min(features[:, 0]))
        gate_max = float(np.max(features[:, 0]))
        initial = np.array([-3.0, min(max(gate_max * 0.3, gate_min), gate_max), 2.0, np.log(0.5), 0.0])
        lower = np.array([-20.0, gate_min - 0.2, 0.5, np.log(0.005), -0.02])
        upper = np.array([5.0, gate_max, 4.0, np.log(10.0), 0.02])
        result = least_squares(
            lambda parameters: self._predict_log(parameters, features) - target,
            initial,
            bounds=(lower, upper),
            loss="soft_l1",
            max_nfev=250,
        )
        self.parameters = result.x
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.parameters is None:
            raise RuntimeError("SquareLawBaseline must be fitted before prediction.")
        return self._predict_log(self.parameters, features)


def build_estimators(seed: int) -> dict[str, Any]:
    """Return comparable log-current predictors, including a neural regressor."""
    mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 64, 32),
            activation="tanh",
            solver="adam",
            max_iter=140,
            early_stopping=True,
            n_iter_no_change=12,
            random_state=seed,
            batch_size=256,
        ),
    )
    mlp_with_scaled_target = TransformedTargetRegressor(
        regressor=mlp,
        transformer=StandardScaler(),
    )
    return {
        "square_law": SquareLawBaseline(),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=80,
            min_samples_leaf=2,
            max_features=1.0,
            n_jobs=-1,
            random_state=seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=seed,
        ),
        "neural_mlp": mlp_with_scaled_target,
    }


def _subsample(
    data: pd.DataFrame,
    maximum: int,
    seed: int,
) -> pd.DataFrame:
    if maximum <= 0 or len(data) <= maximum:
        return data
    return data.sample(n=maximum, random_state=seed).sort_index()


def _metrics(actual_log: np.ndarray, predicted_log: np.ndarray, current_floor: float) -> dict[str, float]:
    actual_current = np.power(10.0, actual_log)
    predicted_current = np.power(10.0, np.clip(predicted_log, -30.0, 5.0))
    relative_mask = actual_current >= current_floor
    relative_error = (
        np.abs(predicted_current[relative_mask] - actual_current[relative_mask])
        / actual_current[relative_mask]
        if np.any(relative_mask)
        else np.array([np.nan])
    )
    return {
        "r2_log_current": float(r2_score(actual_log, predicted_log)),
        "mae_log10_decades": float(mean_absolute_error(actual_log, predicted_log)),
        "rmse_log10_decades": float(np.sqrt(mean_squared_error(actual_log, predicted_log))),
        "mae_current_a": float(mean_absolute_error(actual_current, predicted_current)),
        "rmse_current_a": float(np.sqrt(mean_squared_error(actual_current, predicted_current))),
        "median_abs_relative_error_pct": float(np.nanmedian(relative_error) * 100.0),
        "relative_error_current_floor_a": float(current_floor),
    }


def _range_distance(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.min(train_x, axis=0)
    maximum = np.max(train_x, axis=0)
    scale = np.std(train_x, axis=0)
    scale[scale == 0] = 1.0
    outside_distance = np.maximum(minimum - test_x, 0) + np.maximum(test_x - maximum, 0)
    sigma_distance = np.max(outside_distance / scale, axis=1)
    outside = np.any((test_x < minimum) | (test_x > maximum), axis=1)
    return sigma_distance, outside


def _uncertainty_bins(
    fold_name: str,
    device: str,
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
    distance: np.ndarray,
    outside: np.ndarray,
    current_floor: float,
) -> list[dict[str, Any]]:
    ensemble_names = [
        name for name in ("extra_trees", "hist_gradient_boosting", "neural_mlp")
        if name in predictions
    ]
    ensemble = np.column_stack([predictions[name] for name in ensemble_names])
    disagreement = np.std(ensemble, axis=1)
    quantiles = np.quantile(disagreement, [0.25, 0.5, 0.75])
    bins = np.digitize(disagreement, quantiles, right=True)
    records = []
    for bin_index in range(4):
        mask = bins == bin_index
        if not np.any(mask):
            continue
        values = _metrics(actual[mask], np.mean(ensemble[mask], axis=1), current_floor)
        records.append({
            "fold": fold_name,
            "device": device,
            "uncertainty_proxy": "std of ExtraTrees, HistGradientBoosting, and MLP log-current predictions",
            "uncertainty_quartile": bin_index + 1,
            "samples": int(mask.sum()),
            "mean_disagreement_decades": float(np.mean(disagreement[mask])),
            "mean_feature_range_distance_sigma": float(np.mean(distance[mask])),
            "outside_training_feature_range_fraction": float(np.mean(outside[mask])),
            **values,
        })
    return records


def evaluate_fold(
    *,
    device: str,
    fold_name: str,
    holdout_group_type: str,
    holdout_group_value: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    maximum_train_samples: int,
    maximum_test_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train = _subsample(train, maximum_train_samples, seed)
    test = _subsample(test, maximum_test_samples, seed + 1)
    x_train = train[BASE_FEATURES].to_numpy(dtype=np.float64)
    y_train = train["log_current"].to_numpy(dtype=np.float64)
    x_test = test[BASE_FEATURES].to_numpy(dtype=np.float64)
    y_test = test["log_current"].to_numpy(dtype=np.float64)
    current_floor = float(np.quantile(np.power(10.0, y_train), 0.05))
    distance, outside = _range_distance(x_train, x_test)
    results: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}

    for model_name, estimator in build_estimators(seed).items():
        model = estimator
        fit_start = time.perf_counter()
        if model_name == "square_law":
            model.fit(x_train, y_train)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - fit_start
        prediction_start = time.perf_counter()
        predicted = np.asarray(model.predict(x_test), dtype=float).reshape(-1)
        prediction_seconds = time.perf_counter() - prediction_start
        predictions[model_name] = predicted
        results.append({
            "dataset": "180nm",
            "device": device,
            "fold": fold_name,
            "holdout_group_type": holdout_group_type,
            "holdout_group_value": holdout_group_value,
            "model": model_name,
            "train_samples": int(len(train)),
            "test_samples": int(len(test)),
            "fit_seconds": fit_seconds,
            "predict_seconds": prediction_seconds,
            "outside_training_feature_range_fraction": float(np.mean(outside)),
            "p95_feature_range_distance_sigma": float(np.quantile(distance, 0.95)),
            **_metrics(y_test, predicted, current_floor),
        })
    uncertainty = _uncertainty_bins(
        fold_name, device, y_test, predictions, distance, outside, current_floor
    )
    ensemble_names = [
        name for name in ("extra_trees", "hist_gradient_boosting", "neural_mlp")
        if name in predictions
    ]
    disagreement = np.std(
        np.column_stack([predictions[name] for name in ensemble_names]),
        axis=1,
    )
    prediction_rows: list[dict[str, Any]] = []
    for model_name, predicted in predictions.items():
        prediction_rows.extend({
            "dataset": "180nm",
            "device": device,
            "fold": fold_name,
            "holdout_group_type": holdout_group_type,
            "holdout_group_value": holdout_group_value,
            "model": model_name,
            "gate_v": float(features[0]),
            "drain_v": float(features[1]),
            "temperature_c": float(features[2]),
            "width_m": float(features[3]),
            "length_m": float(features[4]),
            "actual_log10_current": float(actual),
            "predicted_log10_current": float(predicted_value),
            "actual_current_a": float(10.0 ** actual),
            "predicted_current_a": float(10.0 ** np.clip(predicted_value, -30.0, 5.0)),
            "outside_training_feature_range": bool(is_outside),
            "feature_range_distance_sigma": float(range_distance),
            "ensemble_disagreement_decades": float(disagreement[index]),
            "high_disagreement_warning": bool(
                disagreement[index] >= np.quantile(disagreement, 0.90)
            ),
        } for index, (features, actual, predicted_value, is_outside, range_distance) in enumerate(
            zip(x_test, y_test, predicted, outside, distance)
        ))
    return results, uncertainty, prediction_rows


def run_180nm(
    *,
    holdout_temperature: float,
    holdout_width: float,
    holdout_length: float,
    maximum_train_samples: int,
    maximum_test_samples: int,
    seed: int,
    all_holdouts: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    uncertainty: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    folds_record: list[dict[str, Any]] = []
    for device in ("NMOS", "PMOS"):
        data = load_180nm_device(device)
        if all_holdouts:
            temperatures = sorted(float(value) for value in data["temperature_c"].unique())
            geometries = sorted(
                (float(width), float(length))
                for width, length in data[["width_m", "length_m"]].drop_duplicates().itertuples(index=False, name=None)
            )
        else:
            temperatures = [holdout_temperature]
            geometries = [(holdout_width, holdout_length)]

        temp_col = "temperature_c"
        for temperature in temperatures:
            train, test = group_holdout(
                data, group_column=temp_col, heldout_value=temperature
            )
            fold_name = f"temperature_{temperature:g}C"
            fold_metrics, fold_uncertainty, fold_predictions = evaluate_fold(
                device=device, fold_name=fold_name,
                holdout_group_type="temperature_c",
                holdout_group_value=f"{temperature:g}",
                train=train, test=test,
                maximum_train_samples=maximum_train_samples,
                maximum_test_samples=maximum_test_samples, seed=seed,
            )
            metrics.extend(fold_metrics)
            uncertainty.extend(fold_uncertainty)
            predictions.extend(fold_predictions)
            folds_record.append({
                "dataset": "180nm", "device": device, "fold": fold_name,
                "heldout_temperature_c": temperature,
                "train_rows_before_subsampling": len(train),
                "test_rows_before_subsampling": len(test),
            })

        for width, length in geometries:
            geometry_mask = np.isclose(data["width_m"], width) & np.isclose(data["length_m"], length)
            train = data.loc[~geometry_mask].copy()
            test = data.loc[geometry_mask].copy()
            if train.empty or test.empty:
                raise ValueError(f"Geometry holdout {width:g}m x {length:g}m is absent for {device}.")
            fold_name = f"geometry_W{width:g}m_L{length:g}m"
            fold_metrics, fold_uncertainty, fold_predictions = evaluate_fold(
                device=device, fold_name=fold_name,
                holdout_group_type="geometry_width_m_and_length_m",
                holdout_group_value=f"{width:g},{length:g}",
                train=train, test=test,
                maximum_train_samples=maximum_train_samples,
                maximum_test_samples=maximum_test_samples, seed=seed,
            )
            metrics.extend(fold_metrics)
            uncertainty.extend(fold_uncertainty)
            predictions.extend(fold_predictions)
            folds_record.append({
                "dataset": "180nm", "device": device, "fold": fold_name,
                "heldout_width_m": width, "heldout_length_m": length,
                "train_rows_before_subsampling": len(train),
                "test_rows_before_subsampling": len(test),
            })
    return metrics, uncertainty, predictions, folds_record


def run_sky130(
    *,
    data_dir: Path,
    maximum_train_samples: int,
    maximum_test_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    data_dir = data_dir.resolve()
    metrics: list[dict[str, Any]] = []
    uncertainty: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    folds_record: list[dict[str, Any]] = []
    raw = pd.read_csv(data_dir / "device_dc.csv")
    corners = sorted(raw["corner"].astype(str).str.upper().unique())
    for device in ("NMOS", "PMOS"):
        data = load_sky130_device(data_dir, device)
        for corner in corners:
            train, test = group_holdout(
                data, group_column="corner", heldout_value=corner
            )
            fold_name = f"heldout_SKY130_corner_{corner}"
            fold_metrics, fold_uncertainty, fold_predictions = evaluate_fold(
                device=device, fold_name=fold_name,
                holdout_group_type="corner",
                holdout_group_value=corner,
                train=train, test=test,
                maximum_train_samples=maximum_train_samples,
                maximum_test_samples=maximum_test_samples, seed=seed,
            )
            for record in fold_metrics:
                record["dataset"] = "SKY130"
                record["technology"] = "SKY130"
                record["pdk_variant"] = str(raw["pdk_variant"].iloc[0])
                record["pdk_revision"] = str(raw["pdk_revision"].iloc[0])
                record["temperature_c"] = float(data["temperature_c"].iloc[0])
                record["vdd_v"] = float(raw["vdd_v"].iloc[0])
                record["scope_note"] = (
                    "Exploratory held-out-corner prediction; corner is not a model input. "
                    "Dataset covers only recorded 27C, 1.8V, 0.42um/0.15um geometry."
                )
            for record in fold_uncertainty:
                record["dataset"] = "SKY130"
            for record in fold_predictions:
                record["dataset"] = "SKY130"
            metrics.extend(fold_metrics)
            uncertainty.extend(fold_uncertainty)
            predictions.extend(fold_predictions)
            folds_record.append({
                "dataset": "SKY130", "device": device, "fold": fold_name,
                "heldout_corner": corner,
                "train_rows_before_subsampling": len(train),
                "test_rows_before_subsampling": len(test),
                "note": "Corner omitted from model features; this tests transfer to unseen simulator variation, not corner-conditioned inference.",
            })
    return metrics, uncertainty, predictions, folds_record


def crosscheck_transient_estimates(data_dir: Path) -> list[dict[str, Any]]:
    """Compare existing SKY130 transient results to first-order project equations."""
    transient_path = data_dir / "inverter_transient.csv"
    metrics_path = data_dir / "device_dc_metrics.csv"
    if not transient_path.exists() or not metrics_path.exists():
        return []
    transient = pd.read_csv(transient_path)
    device_metrics = pd.read_csv(metrics_path)
    records: list[dict[str, Any]] = []
    for row in transient.to_dict(orient="records"):
        condition = (
            (device_metrics["corner"].astype(str).str.upper() == str(row["corner"]).upper())
            & np.isclose(device_metrics["temperature_c"], row["temperature_c"])
            & np.isclose(device_metrics["vdd_v"], row["vdd_v"])
        )
        selected = device_metrics.loc[condition]
        nmos = selected.loc[selected["device_type"].str.upper() == "NMOS"]
        pmos = selected.loc[selected["device_type"].str.upper() == "PMOS"]
        if nmos.empty or pmos.empty:
            continue
        nmos_row = nmos.iloc[0]
        pmos_row = pmos.iloc[0]
        n_width = float(row["wn_um"])
        p_width = float(row["wp_um"])
        n_width_factor = n_width / float(nmos_row["w_um"])
        p_width_factor = p_width / float(pmos_row["w_um"])
        n_ion = float(nmos_row["ion_a"]) * n_width_factor
        p_ion = float(pmos_row["ion_a"]) * p_width_factor
        capacitance = float(row["cload_f"])
        supply = float(row["vdd_v"])
        period = float(row["input_period_s"])
        tphl_estimate = capacitance * supply / n_ion if n_ion > 0 else np.nan
        tplh_estimate = capacitance * supply / p_ion if p_ion > 0 else np.nan
        power_estimate = capacitance * supply**2 / period if period > 0 else np.nan
        energy_estimate = capacitance * supply**2
        tphl_reference = float(row["tphl_s"])
        tplh_reference = float(row["tplh_s"])
        power_reference = float(row["average_power_w"])
        energy_reference = float(row["energy_per_period_j"])

        def percent_error(estimate: float, reference: float) -> float:
            return (
                abs(estimate - reference) / abs(reference) * 100
                if np.isfinite(estimate) and reference != 0
                else float("nan")
            )

        records.append({
            "technology": "SKY130",
            "corner": row["corner"],
            "temperature_c": row["temperature_c"],
            "vdd_v": supply,
            "cload_f": capacitance,
            "input_slew_s": row["input_slew_s"],
            "tphl_reference_s": tphl_reference,
            "tphl_estimate_s": tphl_estimate,
            "tphl_abs_percent_error": percent_error(tphl_estimate, tphl_reference),
            "tplh_reference_s": tplh_reference,
            "tplh_estimate_s": tplh_estimate,
            "tplh_abs_percent_error": percent_error(tplh_estimate, tplh_reference),
            "average_power_reference_w": power_reference,
            "dynamic_power_estimate_w": power_estimate,
            "dynamic_power_abs_percent_error": percent_error(power_estimate, power_reference),
            "energy_reference_j": energy_reference,
            "dynamic_energy_estimate_j": energy_estimate,
            "dynamic_energy_abs_percent_error": percent_error(energy_estimate, energy_reference),
            "note": (
                "First-order C*V/I and C*V^2 estimates; PMOS ION scaled linearly from "
                "device-metric width to inverter width. Existing references contain one "
                "load/slew condition, so this is not a load/slew sweep or sign-off validation."
            ),
        })
    return records


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        pd.DataFrame().to_csv(path, index=False)


def run(args: argparse.Namespace) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir).resolve() / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    all_metrics, all_uncertainty, all_predictions, all_folds = run_180nm(
        holdout_temperature=args.holdout_temperature,
        holdout_width=args.holdout_width_um * 1e-6,
        holdout_length=args.holdout_length_um * 1e-6,
        maximum_train_samples=args.max_train_samples,
        maximum_test_samples=args.max_test_samples,
        seed=args.seed,
        all_holdouts=args.all_holdouts,
    )
    if not args.skip_sky130:
        sky_metrics, sky_uncertainty, sky_predictions, sky_folds = run_sky130(
            data_dir=Path(args.sky130_dir),
            maximum_train_samples=args.max_train_samples,
            maximum_test_samples=args.max_test_samples,
            seed=args.seed,
        )
        all_metrics.extend(sky_metrics)
        all_uncertainty.extend(sky_uncertainty)
        all_predictions.extend(sky_predictions)
        all_folds.extend(sky_folds)

    _write_csv(output_dir / "validation_metrics.csv", all_metrics)
    _write_csv(output_dir / "uncertainty_summary.csv", all_uncertainty)
    _write_csv(output_dir / "heldout_predictions.csv", all_predictions)
    if all_metrics:
        summary_metrics = [
            "r2_log_current", "mae_log10_decades", "rmse_log10_decades",
            "mae_current_a", "rmse_current_a", "median_abs_relative_error_pct",
            "fit_seconds", "predict_seconds",
        ]
        summary = (
            pd.DataFrame(all_metrics)
            .groupby(["dataset", "device", "model"], dropna=False)[summary_metrics]
            .mean()
            .reset_index()
        )
        summary.to_csv(output_dir / "model_comparison_summary.csv", index=False)
    transient_records = (
        [] if args.skip_sky130
        else crosscheck_transient_estimates(Path(args.sky130_dir))
    )
    _write_csv(output_dir / "transient_power_crosscheck.csv", transient_records)

    inputs = [
        ROOT / "nmos.csv", ROOT / "pmos.csv",
        Path(args.sky130_dir) / "device_dc.csv",
        Path(args.sky130_dir) / "device_dc_metrics.csv",
        Path(args.sky130_dir) / "inverter_transient.csv",
        Path(args.sky130_dir) / "manifest.json",
    ]
    input_hashes = {}
    for path in inputs:
        if not path.is_file():
            continue
        try:
            label = str(path.resolve().relative_to(ROOT))
        except ValueError:
            label = str(path.resolve())
        input_hashes[label] = _sha256(path)
    manifest = {
        "experiment": "grouped_generalization_model_comparison_uncertainty_and_transient_crosscheck",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
        "seed": args.seed,
        "training_sample_cap_per_fold": args.max_train_samples,
        "test_sample_cap_per_fold": args.max_test_samples,
        "holdout_policy": {
            "180nm_default": {
                "temperature_c": args.holdout_temperature,
                "geometry_width_um": args.holdout_width_um,
                "geometry_length_um": args.holdout_length_um,
            },
            "all_180nm_groups": args.all_holdouts,
            "sky130": (
                "Leave one process corner out; corner omitted from features. "
                "Existing data is 27C, 1.8V, W=0.42um, L=0.15um only."
            ),
        },
        "models": [
            "square_law", "ridge", "extra_trees",
            "hist_gradient_boosting", "neural_mlp",
        ],
        "uncertainty_method": (
            "Heuristic disagreement standard deviation across ExtraTrees, "
            "HistGradientBoosting, and MLP predictions, grouped into quartiles. "
            "Not a calibrated confidence interval."
        ),
        "transient_power_scope": (
            "Cross-checks existing SKY130 transient records at the single recorded "
            "load/slew; no new ngspice runs are performed."
        ),
        "input_sha256": input_hashes,
        "folds": all_folds,
        "output_files": [
            "validation_metrics.csv", "model_comparison_summary.csv",
            "uncertainty_summary.csv",
            "heldout_predictions.csv", "transient_power_crosscheck.csv",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run grouped model validation using existing 180nm and SKY130 project data."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sky130-dir", default=str(DEFAULT_SKY130))
    parser.add_argument("--holdout-temperature", type=float, default=85.0)
    parser.add_argument("--holdout-width-um", type=float, default=1.0)
    parser.add_argument("--holdout-length-um", type=float, default=0.25)
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--max-test-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--all-holdouts", action="store_true")
    parser.add_argument("--skip-sky130", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.max_train_samples < 0 or args.max_test_samples < 0:
        parser.error("Sample caps must be >= 0 (0 uses every row in that split).")
    try:
        output_dir = run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.exit(2, f"ERROR: {error}\n")
    print(f"Validation experiment outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
