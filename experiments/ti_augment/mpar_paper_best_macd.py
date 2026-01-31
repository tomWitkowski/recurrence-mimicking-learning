import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

CONFIG_PATH = BASE_DIR / "ti_augment_config.yaml"

from src.rml_model import Encoder, Decoder, Agent


PAIR_LIST = [
    "EURUSD",
    "GBPUSD",
    "USDCHF",
    "USDJPY",
    "GBPJPY",
    "GBPAUD",
    "CADJPY",
    "EURCHF",
]


def add_ma(data: pd.DataFrame, period_short: int, period_long: int, d_prev: int, ema: bool = True):
    if ema:
        data["ma_short"] = data["bid"].ewm(span=period_short).mean()
        data["ma_long"] = data["bid"].ewm(span=period_long).mean()
    else:
        data["ma_short"] = data["bid"].rolling(window=period_short).mean()
        data["ma_long"] = data["bid"].rolling(window=period_long).mean()
    data["diff"] = data["ma_short"] - data["ma_long"]
    for i in range(1, d_prev + 1):
        data[f"diff_{i}"] = data["diff"].shift(i)
    return data.dropna(inplace=False)


def ma(x, revert, threshold):
    revert = np.sign(revert)
    return revert * (1 if x > threshold else (-1 if x < -threshold else 0))


def threshold_grid(diff_series: pd.Series, n_points: int):
    diff_abs = diff_series.abs().replace(0, np.nan).dropna()
    if diff_abs.empty:
        return np.array([0.0])
    low = diff_abs.quantile(0.1)
    high = diff_abs.quantile(0.9)
    if low == high:
        low = diff_abs.min()
        high = diff_abs.max()
    low = max(low, 1e-6)
    high = max(high, low)
    return np.linspace(low, high, n_points)


def objective(params, X_train, BA_train, agent):
    threshold, revert = params
    dec_ma = X_train["diff"].map(lambda x: ma(x, threshold=threshold, revert=revert))
    sr_train = agent.utility_function([BA_train.values, dec_ma.values]).numpy()
    return sr_train


def optimize_ma(X_train, BA_train, agent, n_thresholds: int):
    best_objective_value = float("-inf")
    best_params = None
    thresholds = threshold_grid(X_train["diff"], n_thresholds)
    reverts = [-1, 1]
    for threshold in thresholds:
        for revert in reverts:
            params = (threshold, revert)
            objective_value = objective(params, X_train, BA_train, agent)
            if objective_value > best_objective_value:
                best_objective_value = objective_value
                best_params = params
    zero_decisions = np.zeros(len(X_train))
    zero_sr = agent.utility_function([BA_train.values, zero_decisions]).numpy()
    if not np.isfinite(best_objective_value) or zero_sr >= best_objective_value:
        return (float("inf"), 1), float(zero_sr)
    return best_params, float(best_objective_value)


def load_pair_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    if "time" in data.columns:
        data["time"] = pd.to_datetime(data["time"])
    return data


def parse_quarter(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError("Quarter must be [year, quarter].")


def quarter_start(year: int, quarter: int):
    if quarter not in (1, 2, 3, 4):
        raise ValueError("Quarter must be 1-4.")
    month = 1 + 3 * (quarter - 1)
    return pd.Timestamp(year=year, month=month, day=1)


def add_quarters(dt: pd.Timestamp, n: int):
    return dt + pd.DateOffset(months=3 * n)


def build_static_window(start_q, end_q):
    start_year, start_quarter = parse_quarter(start_q)
    end_year, end_quarter = parse_quarter(end_q)
    train_start = quarter_start(start_year, start_quarter)
    train_end = add_quarters(quarter_start(end_year, end_quarter), 1)
    label = f"train_{train_start.date()}_to_{train_end.date()}"
    return [(train_start, train_end, label)]


def build_roll_windows(start_q, end_q, end_test_q, train_mode, train_quarters, test_quarters):
    windows = []
    start_year, start_quarter = parse_quarter(start_q)
    end_year, end_quarter = parse_quarter(end_q)
    end_test_year, end_test_quarter = parse_quarter(end_test_q)

    train_start_base = quarter_start(start_year, start_quarter)
    train_end_base = add_quarters(quarter_start(end_year, end_quarter), 1)
    test_start = train_end_base
    test_end_limit = add_quarters(quarter_start(end_test_year, end_test_quarter), 1)

    i = 0
    while True:
        test_start_i = add_quarters(test_start, i)
        test_end_i = add_quarters(test_start_i, test_quarters)
        if test_end_i > test_end_limit:
            break
        if train_mode == "fixed":
            train_start_i = add_quarters(test_start_i, -train_quarters)
            if train_start_i < train_start_base:
                train_start_i = train_start_base
            train_end_i = test_start_i
        else:
            train_start_i = train_start_base
            train_end_i = test_start_i
        label = f"train_{train_start_i.date()}_to_{train_end_i.date()}"
        windows.append((train_start_i, train_end_i, label))
        i += 1
    return windows


def ga_optimize_ma(
    data: pd.DataFrame,
    agent,
    n_thresholds: int,
    s_range=(6, 60),
    l_range=(30, 240),
    n_generations=5,
    pop_size=12,
    parents=6,
):
    try:
        import pygad
    except ImportError as exc:
        raise ImportError("pygad is required for GA-based MACD optimization.") from exc

    low = [s_range[0], l_range[0]]
    high = [s_range[1], l_range[1]]

    def fitness_func(ga, sol, idx):
        s, l = map(int, sol)
        if s >= l:
            return -1e9
        df = add_ma(data.copy(), s, l, 0, ema=True)
        X = df[[c for c in df.columns if "diff" in c]]
        BA = df[["bid", "ask"]]
        (threshold, revert), score = optimize_ma(X, BA, agent, n_thresholds)
        return float(score)

    ga = pygad.GA(
        num_generations=n_generations,
        sol_per_pop=pop_size,
        num_parents_mating=parents,
        num_genes=2,
        fitness_func=fitness_func,
        init_range_low=low,
        init_range_high=high,
        gene_type=int,
        mutation_percent_genes=25,
        parallel_processing=4,
    )
    ga.run()
    best_sol, best_fit, _ = ga.best_solution()
    s_best, l_best = int(best_sol[0]), int(best_sol[1])
    df_best = add_ma(data.copy(), s_best, l_best, 0, ema=True)
    X_best = df_best[[c for c in df_best.columns if "diff" in c]]
    BA_best = df_best[["bid", "ask"]]
    (threshold, revert), ma_train = optimize_ma(X_best, BA_best, agent, n_thresholds)
    return {
        "short": s_best,
        "long": l_best,
        "ma_train": float(ma_train),
        "threshold": float(threshold),
        "revert": float(revert),
    }


def load_yaml(path: Path):
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to read YAML config files.") from exc
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a YAML mapping.")
    return data


def main():
    parser = argparse.ArgumentParser(description="Grid search best MACD params per pair.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--data-dir", default=str(BASE_DIR / "data" / "processed"))
    parser.add_argument("--pairs", nargs="*", default=PAIR_LIST)
    parser.add_argument("--output", default=str(BASE_DIR / "macd_params.yaml"))
    parser.add_argument("--short-min", type=int, default=6)
    parser.add_argument("--short-max", type=int, default=60)
    parser.add_argument("--short-step", type=int, default=2)
    parser.add_argument("--long-min", type=int, default=30)
    parser.add_argument("--long-max", type=int, default=240)
    parser.add_argument("--long-step", type=int, default=4)
    parser.add_argument("--thresholds", type=int, default=30)
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    split_cfg = config.get("split", {})
    roll_cfg = split_cfg.get("roll", {})

    data_dir = Path(config.get("data_dir", args.data_dir))
    if not data_dir.is_absolute():
        data_dir = BASE_DIR / data_dir
    short_windows = list(range(args.short_min, args.short_max + 1, args.short_step))
    long_windows = list(range(args.long_min, args.long_max + 1, args.long_step))

    model_config_e = [{"type": "dense", "units": 5, "activation": "softplus", "dropout": 0.3}]
    model_config_d = [{"type": "dense", "units": 5, "activation": "softplus", "dropout": 0.3}]
    encoder = Encoder(2, model_config_e)
    decoder = Decoder(encoder.output.shape[1], model_config_d)
    agent = Agent(encoder, decoder)

    pairs = config.get("pairs", args.pairs)
    output_path = config.get("macd_params_path", args.output)
    split_mode = split_cfg.get("mode", "static")
    start_quarter = split_cfg.get("start_quarter", [2019, 1])
    end_quarter = split_cfg.get("end_quarter", [2020, 4])
    end_test_quarter = split_cfg.get("end_test_quarter", [2021, 3])
    roll_train_mode = roll_cfg.get("train_mode", "expand")
    roll_train_quarters = int(roll_cfg.get("train_quarters", 8))
    roll_test_quarters = int(roll_cfg.get("test_quarters", 2))

    output = {}
    for pair in pairs:
        path = data_dir / f"{pair}_15min_expanded.csv"
        if not path.exists():
            print(f"missing data for {pair}: {path}")
            continue
        data = load_pair_data(path)
        if split_mode == "roll":
            windows = build_roll_windows(
                start_quarter,
                end_quarter,
                end_test_quarter,
                roll_train_mode,
                roll_train_quarters,
                roll_test_quarters,
            )
            for train_start, train_end, label in windows:
                train_data = data
                if "time" in data.columns:
                    train_data = data[(data["time"] >= train_start) & (data["time"] < train_end)]
                best = ga_optimize_ma(
                    train_data,
                    agent,
                    args.thresholds,
                    s_range=(short_windows[0], short_windows[-1]),
                    l_range=(long_windows[0], long_windows[-1]),
                )
                output.setdefault(label, {})[pair] = best
                print(pair, label, best)
        else:
            windows = build_static_window(start_quarter, end_quarter)
            for train_start, train_end, label in windows:
                train_data = data
                if "time" in data.columns:
                    train_data = data[(data["time"] >= train_start) & (data["time"] < train_end)]
                best = ga_optimize_ma(
                    train_data,
                    agent,
                    args.thresholds,
                    s_range=(short_windows[0], short_windows[-1]),
                    l_range=(long_windows[0], long_windows[-1]),
                )
                output.setdefault(label, {})[pair] = best
                print(pair, label, best)

    if not output:
        raise SystemExit("No results produced.")

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to write macd_params.yaml.") from exc

    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = BASE_DIR / out_path
    out_path.write_text(yaml.safe_dump(output, sort_keys=False))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
