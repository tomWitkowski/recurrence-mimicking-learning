#!/usr/bin/env python3
"""Utility for reproducing the transaction-cost supervised learning experiment.

The script mirrors the two key notebook cells annotated with "### THIS CELL":
1. Train randomly initialised Trader models (with and without recurrency) N times.
2. Save diagnostics for downstream visualisation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Limit BLAS/numexpr threading to keep behaviour aligned with the notebook.
THREAD_ENV_VARS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
for key, value in THREAD_ENV_VARS.items():
    os.environ.setdefault(key, value)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from waves import WaveGrasper  # noqa: E402
from model_classifier_scratch import Trader as TClassifier  # noqa: E402
from utils import LightStrategy  # noqa: E402


def load_config(path: Path) -> Dict:
    """Load the experiment configuration."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def prepare_dataset(cfg: Dict) -> Dict[str, np.ndarray]:
    """Replicate the feature engineering block from the notebook."""
    data_cfg = cfg["data"]
    data_path = (REPO_ROOT / data_cfg["csv_path"]).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {data_path}")

    data = pd.read_csv(data_path)
    data["close"] = (data.bid + data.ask) / 2
    data["index"] = data.index

    subdf = data[["close"]].copy()
    spread = data_cfg.get("bid_ask_spread", 0.00008)
    subdf["bid"], subdf["ask"] = subdf.close - spread, subdf.close + spread

    DF = subdf.copy()
    DF.columns = [str(col) for col in DF.columns]
    pred_df = DF.copy()
    _ = pred_df[["close"]]

    wave = WaveGrasper(
        data.close,
        data_cfg["change"],
        gather_point_data=True,
        length=data_cfg["wave_length"],
    )
    wave.df.loc[0, "price"] = data.close.values[0]
    wave.df.sort_index(inplace=True)
    wave.df["index"] = wave.df.index

    DF = pd.DataFrame(
        [point[2] + [point[4], point[5]] for point in wave.point_data],
        columns=[f"tp{i}" for i in range(wave._length)][::-1] + ["assum", "now"],
        index=[point[0] for point in wave.point_data],
    )
    time_assum = pd.DataFrame(
        [point[0] - point[1] for point in wave.point_data],
        columns=["time_since"],
        index=[point[0] for point in wave.point_data],
    )

    DF = pd.concat([DF, subdf, time_assum], axis=1).dropna()

    df_wave = wave.df.copy()
    concated_other = pd.concat(
        [
            pd.Series(
                [point[0] for point in wave.point_data][:-1],
                index=[int(point[0]) for point in wave.point_data][:-1],
            )
        ],
        axis=1,
    )
    del concated_other[0]
    concated_other = concated_other.ffill().dropna()
    concated_other = pd.concat([df_wave.price.rename("y"), concated_other], axis=1).sort_index()
    concated_other["y"] = concated_other.y.bfill()
    concated_other.dropna(inplace=True)

    DF = pd.concat([DF, concated_other["y"]], axis=1).dropna()

    X = DF[[col for col in DF.columns if "tp" in col or col in ["assum", "now"]]]
    y = (DF.y > DF.now).astype(int)*2-1

    step = data_cfg.get("step", 1)
    titer = range(max(data_cfg["timeperiods"], data_cfg["returnperiod"]), len(X), step)

    X_vals = X.values
    bid_ask_vals = DF[["bid", "ask"]].values

    X_vals = X_vals[titer.start : titer.stop : step, :]
    norm_method = str(data_cfg.get("normalization", "difference")).lower()
    X_vals = _normalize_features(X_vals, norm_method)

    bid_ask_vals = bid_ask_vals[titer.start : titer.stop : step, :]
    y_vals = y.values[titer.start : titer.stop : step]

    return {"X_vals": X_vals, "y_vals": y_vals, "bid_ask_vals": bid_ask_vals}


def _compute_stats(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    preds = np.round(pred).reshape(-1)
    labels = target.reshape(-1)
    accuracy = float(np.mean(preds == labels))
    n_changes = int(np.sum(np.diff(preds) != 0)) if len(preds) > 1 else 0
    return {"accuracy": accuracy, "n_changes": n_changes}


def _evaluate_model(
    trader: TClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    features_test: np.ndarray,
    labels_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, float]]:
    train_pred = trader.train_iteration(features, labels)[1].numpy().reshape(-1)
    test_pred = trader.train_iteration(features_test, labels_test)[1].numpy().reshape(-1)
    train_stats = _compute_stats(train_pred, labels)
    test_stats = _compute_stats(test_pred, labels_test)
    return train_pred, test_pred, train_stats, test_stats


def _prediction_rows(pred: np.ndarray, run_id: int, split: str) -> List[Dict[str, float]]:
    split_flag = 1 if split == "test" else 0
    rounded = np.round(pred.reshape(-1), 5)
    return [
        {
            "id": run_id,
            "split": split_flag,
            "sample": idx,
            "prediction": float(f"{value:.5f}"),
        }
        for idx, value in enumerate(rounded)
    ]


def _decisions_from_pred(pred: np.ndarray, allow_zero: bool = False, threshold: float = 0.5) -> np.ndarray:
    values = pred.reshape(-1)
    if allow_zero:
        decisions = np.zeros_like(values, dtype=int)
        decisions[values >= threshold] = 1
        decisions[values <= -threshold] = -1
    else:
        rounded = np.round(values)
        decisions = np.where(rounded > 0, 1, -1).astype(int)
    return decisions


def _light_strategy_metrics(pred: np.ndarray, bid_ask: np.ndarray, allow_zero: bool = False) -> Tuple[float, float]:
    decisions = _decisions_from_pred(pred, allow_zero=allow_zero)
    df = pd.DataFrame(bid_ask, columns=["bid", "ask"])
    df["d"] = decisions
    strategy = LightStrategy()
    final_wallet = float(strategy.evaluate(df, collect_result=True))
    equity_curve = np.array(strategy.result, dtype=float)
    if equity_curve.size > 1:
        returns = np.diff(equity_curve) / equity_curve[:-1]
        returns = returns[np.isfinite(returns)]
    else:
        returns = np.array([], dtype=float)
    if returns.size == 0 or np.std(returns) == 0:
        sharpe = 0.0
    else:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
    return final_wallet, sharpe


def run_variant(
    features: np.ndarray,
    labels: np.ndarray,
    features_test: np.ndarray,
    labels_test: np.ndarray,
    bid_ask_train: np.ndarray,
    bid_ask_test: np.ndarray,
    cfg: Dict,
    rn: bool,
    start_id: int,
    checkpoint_interval: Optional[int],
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], int]:
    """Run N experiments and collect diagnostics and predictions."""
    runs = cfg["experiment"]["runs"]
    epochs = cfg["experiment"]["epochs"]
    lr = cfg["experiment"]["learning_rate"]

    diagnostics: List[Dict[str, float]] = []
    predictions: List[Dict[str, float]] = []
    next_id = start_id

    for run_idx in range(runs):
        trader = TClassifier(input_len=features.shape[1], rn=rn)
        epochs_trained = 0
        remaining_epochs = epochs
        checkpoint_stats: Dict[int, Tuple[Dict[str, float], Dict[str, float]]] = {}

        while remaining_epochs > 0:
            step_epochs = (
                min(checkpoint_interval, remaining_epochs)
                if checkpoint_interval
                else remaining_epochs
            )
            trader.set_lr(lr)
            trader.fit(features, labels, step_epochs, verbose=0)
            epochs_trained += step_epochs
            remaining_epochs -= step_epochs

            if checkpoint_interval and epochs_trained % checkpoint_interval == 0:
                trader.set_lr(0.0)
                _, _, train_cp, test_cp = _evaluate_model(
                    trader, features, labels, features_test, labels_test
                )
                checkpoint_stats[epochs_trained] = (train_cp, test_cp)

        trader.set_lr(0.0)
        train_pred, test_pred, train_stats, test_stats = _evaluate_model(
            trader, features, labels, features_test, labels_test
        )
        train_return, train_sharpe = _light_strategy_metrics(train_pred, bid_ask_train, allow_zero=False)
        test_return, test_sharpe = _light_strategy_metrics(test_pred, bid_ask_test, allow_zero=False)
        train_return_zero, train_sharpe_zero = _light_strategy_metrics(
            train_pred, bid_ask_train, allow_zero=True
        )
        test_return_zero, test_sharpe_zero = _light_strategy_metrics(
            test_pred, bid_ask_test, allow_zero=True
        )

        row = {
            "id": next_id,
            "run": run_idx,
            "rn": rn,
            "acc": train_stats["accuracy"],
            "acc_test": test_stats["accuracy"],
            "n": train_stats["n_changes"],
            "n_test": test_stats["n_changes"],
            "return_train": train_return,
            "return_test": test_return,
            "sharpe_train": train_sharpe,
            "sharpe_test": test_sharpe,
            "return_train_zero": train_return_zero,
            "return_test_zero": test_return_zero,
            "sharpe_train_zero": train_sharpe_zero,
            "sharpe_test_zero": test_sharpe_zero,
            "train_pred_mean": float(np.mean(train_pred)),
            "train_pred_std": float(np.std(train_pred)),
            "test_pred_mean": float(np.mean(test_pred)),
            "test_pred_std": float(np.std(test_pred)),
        }

        for epoch_idx in sorted(checkpoint_stats):
            train_cp, test_cp = checkpoint_stats[epoch_idx]
            row[f"acc_epoch_{epoch_idx}"] = train_cp["accuracy"]
            row[f"acc_test_epoch_{epoch_idx}"] = test_cp["accuracy"]
            row[f"n_epoch_{epoch_idx}"] = train_cp["n_changes"]
            row[f"n_test_epoch_{epoch_idx}"] = test_cp["n_changes"]

        diagnostics.append(row)
        predictions.extend(_prediction_rows(train_pred, next_id, "train"))
        predictions.extend(_prediction_rows(test_pred, next_id, "test"))

        next_id += 1

    return diagnostics, predictions, next_id


def _normalize_features(X_vals: np.ndarray, method: str) -> np.ndarray:
    """Normalize input features along the time dimension while preserving order."""
    method = method.lower()
    if method in ("difference", "diff", "none"):
        return X_vals[:, 1:] - X_vals[:, :-1]

    if method == "min_max":
        diff = X_vals[:, 1:] - X_vals[:, :-1]
        min_vals = diff.min(axis=0)
        max_vals = diff.max(axis=0)
        denom = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)
        return (diff - min_vals) / denom

    if method in ("normalized_difference", "norm_diff", "zscore_diff"):
        diff = X_vals[:, 1:] - X_vals[:, :-1]
        mean_vals = diff.mean(axis=0)
        std_vals = diff.std(axis=0)
        std_safe = np.where(std_vals == 0, 1.0, std_vals)
        return (diff - mean_vals) / std_safe

    prev_vals = X_vals[:, :-1]
    next_vals = X_vals[:, 1:]
    safe_prev = np.where(prev_vals == 0, 1e-12, prev_vals)

    if method == "log_returns":
        ratios = np.clip(next_vals / safe_prev, 1e-12, None)
        return np.log(ratios)

    if method in ("percentage_returns", "pct_returns", "percent_returns"):
        return (next_vals - prev_vals) / safe_prev

    raise ValueError(f"Unknown normalization method '{method}'.")


def _run_single_experiment(cfg: Dict, output_dir: Path) -> None:
    dataset = prepare_dataset(cfg)
    train_size = cfg["experiment"]["train_size"]
    checkpoint_interval = cfg["experiment"].get("checkpoint_interval")
    X_vals, y_vals = dataset["X_vals"], dataset["y_vals"]
    bid_ask_vals = dataset["bid_ask_vals"]

    features = X_vals[:train_size]
    labels = y_vals[:train_size]
    features_test = X_vals[train_size:]
    labels_test = y_vals[train_size:]
    bid_ask_train = bid_ask_vals[:train_size]
    bid_ask_test = bid_ask_vals[train_size:]
    if len(features_test) == 0:
        raise ValueError("Test split is empty. Increase the dataset or reduce train_size.")

    diagnostics_rows: List[Dict[str, float]] = []
    prediction_rows: List[Dict[str, float]] = []
    next_id = 0
    for _, use_rn in cfg["experiment"]["variants"].items():
        diag_rows, preds, next_id = run_variant(
            features,
            labels,
            features_test,
            labels_test,
            bid_ask_train,
            bid_ask_test,
            cfg,
            rn=use_rn,
            start_id=next_id,
            checkpoint_interval=checkpoint_interval,
        )
        diagnostics_rows.extend(diag_rows)
        prediction_rows.extend(preds)

    results = pd.DataFrame(diagnostics_rows)
    acc_cols = [col for col in results.columns if col.startswith("acc")]
    if acc_cols:
        results[acc_cols] = results[acc_cols].round(5)
    predictions_df = pd.DataFrame(prediction_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_to_store = deepcopy(cfg)
    cfg_to_store["experiment"]["output_dir"] = str(output_dir)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg_to_store, handle)

    results.to_csv(output_dir / "combined.csv", index=False)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    market_context = []
    train_context = pd.DataFrame(bid_ask_train, columns=["bid", "ask"])
    train_context["split"] = 0
    train_context["sample"] = np.arange(len(train_context))
    train_context["target"] = labels
    market_context.append(train_context)
    test_context = pd.DataFrame(bid_ask_test, columns=["bid", "ask"])
    test_context["split"] = 1
    test_context["sample"] = np.arange(len(test_context))
    test_context["target"] = labels_test
    market_context.append(test_context)
    market_df = pd.concat(market_context, ignore_index=True)
    market_df.to_csv(output_dir / "market_context.csv", index=False)
    print(f"Saved diagnostics to {output_dir}")


def _parse_list(raw: object) -> List[str]:
    """Accept a single value or comma-separated string and return clean entries."""
    if isinstance(raw, (list, tuple)):
        candidates = [str(item) for item in raw]
    else:
        candidates = [part.strip() for part in str(raw).split(",")]
    return [item for item in candidates if item]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the transaction-cost experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "config.yaml"),
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    csv_paths = _parse_list(cfg["data"]["csv_path"])
    if not csv_paths:
        raise ValueError("No csv_path provided. Set data.csv_path to at least one value.")

    normalization_options = _parse_list(cfg["data"].get("normalization", "difference"))
    if not normalization_options:
        normalization_options = ["difference"]

    base_output_dir = (config_path.parent / cfg["experiment"]["output_dir"]).resolve()

    for idx, csv_path in enumerate(csv_paths):
        for norm in normalization_options:
            cfg_run = deepcopy(cfg)
            cfg_run["data"]["csv_path"] = csv_path
            cfg_run["data"]["normalization"] = norm
            run_label = Path(csv_path).stem or f"run_{idx}"
            norm_label = str(norm).lower().replace(" ", "_")
            output_dir = base_output_dir / f"{run_label}__{norm_label}"
            _run_single_experiment(cfg_run, output_dir)


if __name__ == "__main__":
    main()
