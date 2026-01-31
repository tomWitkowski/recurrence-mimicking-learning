import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

CONFIG_PATH = BASE_DIR / "ti_augment_config.yaml"


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


config = load_yaml(CONFIG_PATH)
DATA_DIR = BASE_DIR / config.get("data_dir", "data/processed")
MACD_CONFIG_PATH = BASE_DIR / config.get("macd_params_path", "macd_params.yaml")

def load_macd_cfg(path: Path):
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load macd_params.yaml.") from exc
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("macd_params.yaml must contain a mapping of pair -> params.")
    return data


def save_macd_cfg(path: Path, cfg: dict):
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to save macd_params.yaml.") from exc
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def update_macd_cfg(path: Path, label: str, pair: str, params: dict):
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to save macd_params.yaml.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:
        fcntl = None
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "a+") as lockf:
        if fcntl:
            fcntl.flock(lockf, fcntl.LOCK_EX)
        content = path.read_text() if path.exists() else ""
        data = yaml.safe_load(content) if content.strip() else {}
        if not isinstance(data, dict):
            data = {}

        is_flat = bool(data) and all(isinstance(v, dict) and "short" in v for v in data.values())
        if is_flat:
            data[pair] = params
        else:
            data.setdefault(label, {})[pair] = params

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(yaml.safe_dump(data, sort_keys=False))
        os.replace(tmp_path, path)
        if fcntl:
            fcntl.flock(lockf, fcntl.LOCK_UN)


macd_cfg = load_macd_cfg(MACD_CONFIG_PATH)
if all(isinstance(v, dict) and "short" in v for v in macd_cfg.values()):
    data_list = list(macd_cfg.keys())
else:
    first_bucket = next(iter(macd_cfg.values()), {})
    data_list = list(first_bucket.keys()) if isinstance(first_bucket, dict) else []
data_list = config.get("pairs", data_list) or ["GBPUSD", "EURUSD", "USDJPY", "USDCHF"]
import copy

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings; warnings.filterwarnings('ignore')
from math import log, factorial
from tqdm import tqdm
import matplotlib as mpl
from tqdm import tqdm
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import seaborn as sns
from scipy.optimize import minimize

plt.rcParams['figure.figsize'] = 10,8
plt.rcParams['font.size'] = 35
plt.style.use(['ggplot','fivethirtyeight'])
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf


from src.rml_model import Encoder, Decoder, Agent

# model_config_e = [   
#     {'type': 'dense', 'units': 5, 'activation': 'silu'},
# ]
# model_config_d = [   
#     {'type': 'dense', 'units': 5, 'activation': 'relu', 'dropout': 0.2},
#     {'type': 'dense', 'units': 3, 'activation': 'tanh'},
# ]
model_cfg = config.get("model", {})
model_config_e = model_cfg.get(
    "encoder_layers",
    [{'type': 'dense', 'units': 5, 'activation': 'softplus', 'dropout': 0.3}],
)
model_config_d = model_cfg.get(
    "decoder_layers",
    [{'type': 'dense', 'units': 5, 'activation': 'softplus', 'dropout': 0.3}],
)

def _coerce_list(val):
    return val if isinstance(val, list) else [val]


def expand_schedule(cfg):
    schedule = cfg.get("schedule", {})
    exp_cfg = cfg.get("experiment", {})
    model_cfg = cfg.get("model", {})
    schedule_items = dict(schedule)
    for key, val in exp_cfg.items():
        if isinstance(val, list) and key not in schedule_items:
            schedule_items[key] = val
    if not schedule_items:
        return [cfg]
    keys = list(schedule_items.keys())
    values = [_coerce_list(schedule_items[k]) for k in keys]
    combos = []

    def rec(i, current):
        if i == len(keys):
            combos.append(current.copy())
            return
        for v in values[i]:
            current[keys[i]] = v
            rec(i + 1, current)

    rec(0, {})
    variants = []

    def set_path(dct, path, value):
        parts = path.split(".")
        cur = dct
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value

    for combo in combos:
        new_cfg = dict(cfg)
        new_exp = dict(exp_cfg)
        new_model = dict(model_cfg)
        for k, v in combo.items():
            if k.startswith("experiment."):
                set_path(new_cfg, k, v)
            elif k.startswith("model."):
                set_path(new_cfg, k, v)
            elif k in exp_cfg:
                new_exp[k] = v
            elif k in model_cfg:
                new_model[k] = v
            else:
                new_exp[k] = v
        new_cfg["experiment"] = new_exp
        new_cfg["model"] = new_model
        new_cfg["schedule"] = {}
        variants.append(new_cfg)
    return variants


def snapshot_config(cfg, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to save YAML config files.") from exc
    (out_dir / "ti_augment_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def next_results_root(base_dir: Path):
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    for p in base_dir.iterdir():
        if p.is_dir() and p.name.startswith("results_"):
            try:
                existing.append(int(p.name.split("_", 1)[1]))
            except Exception:
                continue
    next_idx = max(existing, default=0) + 1
    return base_dir / f"results_{next_idx}"


def apply_experiment_config(cfg):
    exp_cfg = cfg.get("experiment", {})
    model_cfg = cfg.get("model", {})
    globals()["model_config_e"] = model_cfg.get(
        "encoder_layers",
        [{'type': 'dense', 'units': 5, 'activation': 'softplus', 'dropout': 0.3}],
    )
    globals()["model_config_d"] = model_cfg.get(
        "decoder_layers",
        [{'type': 'dense', 'units': 5, 'activation': 'softplus', 'dropout': 0.3}],
    )
    globals()["PREV_HOURS"] = int(exp_cfg.get("prev_hours", 12))
    globals()["n_runs"] = int(exp_cfg.get("runs", 2))
    globals()["epochs"] = int(exp_cfg.get("epochs", 2000))
    globals()["lr"] = float(exp_cfg.get("lr", 0.0002))
    os.environ['train_log_name'] = str(exp_cfg.get("train_log_name", "20"))
    globals()["MULTIPROCESS"] = bool(exp_cfg.get("multiprocess", True))
    globals()["N_PROCESSES"] = int(exp_cfg.get("processes", 8))
    globals()["RESULTS_ROOT"] = BASE_DIR / "results"
    globals()["data_list"] = cfg.get("pairs", data_list)


MACD_SHORT_RANGE = list(range(6, 61, 2))
MACD_LONG_RANGE = list(range(30, 241, 4))
split_cfg = config.get("split", {})
SPLIT_MODE = split_cfg.get("mode", "static")  # static | roll
START_QUARTER = split_cfg.get("start_quarter", [2019, 1])
END_QUARTER = split_cfg.get("end_quarter", [2020, 4])
END_TEST_QUARTER = split_cfg.get("end_test_quarter", [2021, 3])
ROLL_CFG = split_cfg.get("roll", {})
ROLL_TRAIN_MODE = ROLL_CFG.get("train_mode", "expand")  # expand | fixed
ROLL_TRAIN_QUARTERS = int(ROLL_CFG.get("train_quarters", 8))
ROLL_TEST_QUARTERS = int(ROLL_CFG.get("test_quarters", 2))


def add_ma(data: pd.DataFrame, period_short: int, period_long: int, d_prev: int, ema:bool=True):
    if ema:
        data['ma_short'] = data['bid'].ewm(span=period_short).mean()
        data['ma_long'] = data['bid'].ewm(span=period_long).mean()
    else:
        data['ma_short'] = data['bid'].rolling(window=period_short).mean()
        data['ma_long'] = data['bid'].rolling(window=period_long).mean()
    data['diff'] = data['ma_short'] - data['ma_long']
    for i in range(1, d_prev+1):
        data[f'diff_{i}'] = data['diff'].shift(i)
    return data.dropna(inplace=False)

def ma(x, revert, threshold):
    revert = np.sign(revert)
    return revert*(1 if x > threshold else (-1 if x < -threshold else 0))

# Define the objective function to minimize (negative Sharpe Ratio)
def objective(params, X_train, BA_train, agent):
    threshold, revert = params
    dec_ma = X_train['diff'].map(lambda x: ma(x, threshold=threshold, revert=revert))
    sr_train = agent.utility_function([BA_train.values, dec_ma.values]).numpy()
    return sr_train


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


def optimize_ma(X_train, BA_train, agent, n_thresholds: int = 30):
    best_objective_value = float('-inf')
    best_params = None

    # Define the range of values to test
    thresholds = threshold_grid(X_train['diff'], n_thresholds)
    reverts = [-1, 1]

    # Iterate over all combinations of parameters
    for threshold in thresholds:
        for revert in reverts:
            params = (threshold, revert)
            objective_value = objective(params, X_train, BA_train, agent)

            # Check if this is the best set of parameters found so far
            if objective_value > best_objective_value:
                best_objective_value = objective_value
                best_params = params

    zero_decisions = np.zeros(len(X_train))
    zero_sr = agent.utility_function([BA_train.values, zero_decisions]).numpy()
    if not np.isfinite(best_objective_value) or zero_sr >= best_objective_value:
        return (float("inf"), 1)
    return best_params


results = []


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


def build_static_window():
    start_year, start_q = parse_quarter(START_QUARTER)
    end_year, end_q = parse_quarter(END_QUARTER)
    end_test_year, end_test_q = parse_quarter(END_TEST_QUARTER)
    train_start = quarter_start(start_year, start_q)
    train_end = add_quarters(quarter_start(end_year, end_q), 1)
    test_start = train_end
    test_end = add_quarters(quarter_start(end_test_year, end_test_q), 1)
    label = f"{test_start.date()}_{test_end.date()}"
    macd_label = f"train_{train_start.date()}_to_{train_end.date()}"
    return [(train_start, train_end, test_start, test_end, label, macd_label)]


def build_roll_windows():
    windows = []
    start_year, start_q = parse_quarter(START_QUARTER)
    end_year, end_q = parse_quarter(END_QUARTER)
    end_test_year, end_test_q = parse_quarter(END_TEST_QUARTER)
    train_start_base = quarter_start(start_year, start_q)
    train_end_base = add_quarters(quarter_start(end_year, end_q), 1)
    test_start = train_end_base
    test_end_limit = add_quarters(quarter_start(end_test_year, end_test_q), 1)

    i = 0
    while True:
        test_start_i = add_quarters(test_start, i)
        test_end_i = add_quarters(test_start_i, ROLL_TEST_QUARTERS)
        if test_end_i > test_end_limit:
            break
        if ROLL_TRAIN_MODE == "fixed":
            train_start_i = add_quarters(test_start_i, -ROLL_TRAIN_QUARTERS)
            if train_start_i < train_start_base:
                train_start_i = train_start_base
            train_end_i = test_start_i
        else:
            train_start_i = train_start_base
            train_end_i = test_start_i

        label = f"{test_start_i.date()}_{test_end_i.date()}"
        macd_label = f"train_{train_start_i.date()}_to_{train_end_i.date()}"
        windows.append((train_start_i, train_end_i, test_start_i, test_end_i, label, macd_label))
        i += 1
    return windows


def select_macd_params(cfg, label, pair):
    if not isinstance(cfg, dict):
        raise ValueError("macd_params.yaml must be a mapping.")
    if all(isinstance(v, dict) and "short" in v for v in cfg.values()):
        return cfg.get(pair)
    if label in cfg and isinstance(cfg[label], dict) and pair in cfg[label]:
        return cfg[label][pair]
    return None


def compute_best_macd_params(data_train: pd.DataFrame, asset_name: str, verbose: bool = True):
    try:
        import pygad
    except ImportError as exc:
        raise ImportError("pygad is required for GA-based MACD optimization.") from exc

    encoder = Encoder(2, {})
    decoder = Decoder(encoder.output.shape[1], {})
    agent = Agent(encoder, decoder)

    low = [MACD_SHORT_RANGE[0], MACD_LONG_RANGE[0]]
    high = [MACD_SHORT_RANGE[-1], MACD_LONG_RANGE[-1]]

    def fitness_func(ga, sol, idx):
        s, l = map(int, sol)
        if s >= l:
            return -1e9
        df_tmp = add_ma(data_train.copy(), s, l, 0, ema=True)
        X_train = df_tmp[[col for col in df_tmp.columns if 'diff' in col]]
        BA_train = df_tmp[['bid', 'ask']]
        threshold, revert = optimize_ma(X_train, BA_train, agent=agent)
        dec_ma = X_train['diff'].map(lambda x: ma(x, threshold=threshold, revert=revert))
        sr_train = agent.utility_function([BA_train.values, dec_ma.values]).numpy()
        return float(sr_train)

    ga = pygad.GA(
        num_generations=5,
        sol_per_pop=12,
        num_parents_mating=6,
        num_genes=2,
        fitness_func=fitness_func,
        init_range_low=low,
        init_range_high=high,
        gene_type=int,
        mutation_percent_genes=25,
        parallel_processing=4,
    )
    if verbose:
        total = 5
        def _on_gen(ga):
            if ga.generations_completed % 3 != 0 and ga.generations_completed != total:
                return
            pct = int(100 * ga.generations_completed / total)
            print(
                f"[{asset_name}] GA {ga.generations_completed}/{total} ({pct}%) "
                f"best={ga.best_solution()[1]:.6f}"
            )
        ga.on_generation = _on_gen
    ga.run()
    best_sol, _, _ = ga.best_solution()
    s_best, l_best = int(best_sol[0]), int(best_sol[1])
    df_best = add_ma(data_train.copy(), s_best, l_best, 0, ema=True)
    X_best = df_best[[col for col in df_best.columns if 'diff' in col]]
    BA_best = df_best[['bid', 'ask']]
    threshold, revert = optimize_ma(X_best, BA_best, agent=agent)
    dec_ma = X_best['diff'].map(lambda x: ma(x, threshold=threshold, revert=revert))
    sr_train = agent.utility_function([BA_best.values, dec_ma.values]).numpy()
    return {
        "short": int(s_best),
        "long": int(l_best),
        "ma_train": float(sr_train),
        "threshold": float(threshold),
        "revert": float(revert),
    }


def build_windows():
    if SPLIT_MODE == "roll":
        return build_roll_windows()
    return build_static_window()


def save_quarter_diagnostics(
    folder,
    times,
    BA_test,
    dec_ma_test,
    decs,
    decs_test_best,
    agent,
):
    if times is None or times.empty:
        return {}, []
    diag_dir = os.path.join(folder, "quarterly")
    os.makedirs(diag_dir, exist_ok=True)
    times = pd.Series(pd.to_datetime(times)).reset_index(drop=True)
    quarter = times.dt.to_period("Q")

    summary_rows = []
    for qtr in sorted(quarter.unique()):
        mask = quarter == qtr
        if mask.sum() == 0:
            continue
        ba_q = BA_test.loc[mask]
        ma_q = dec_ma_test[mask.values]
        rrl_q = np.array(decs)[mask.values]
        rrl_best_q = np.array(decs_test_best)[mask.values]

        sr_ma = agent.utility_function([ba_q.values, ma_q]).numpy()
        sr_rrl = agent.utility_function([ba_q.values, rrl_q]).numpy()
        sr_best = agent.utility_function([ba_q.values, rrl_best_q]).numpy()

        summary_rows.append(
            {
                "quarter": str(qtr),
                "rows": int(mask.sum()),
                "sr_ma": float(sr_ma),
                "sr_rrl": float(sr_rrl),
                "sr_rrl_best": float(sr_best),
            }
        )

        df_q = pd.DataFrame(
            {
                "time": times.loc[mask].values,
                "MA": ma_q,
                "RRL": rrl_q,
                "RRL_best": rrl_best_q,
                "bid": ba_q["bid"].values,
                "ask": ba_q["ask"].values,
            }
        )
        df_q.to_csv(os.path.join(diag_dir, f"positions_test_{qtr}.csv"), index=False)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(diag_dir, "summary.csv"), index=False
        )
    summary_map = {}
    for row in summary_rows:
        q = row["quarter"]
        summary_map[f"{q}_sr_ma"] = row["sr_ma"]
        summary_map[f"{q}_sr_rrl"] = row["sr_rrl"]
        summary_map[f"{q}_sr_rrl_best"] = row["sr_rrl_best"]
        summary_map[f"{q}_rows"] = row["rows"]
    return summary_map, summary_rows


def main(name):
    print(name)
    path = DATA_DIR / f"{name}_15min_expanded.csv"
    data = pd.read_csv(path)

    # Split into 90% training and 10% testing.
    # Ensure 'date' column is in datetime format
    data['time'] = pd.to_datetime(data['time'])
    windows = build_windows()
    is_rolling = bool(windows)
    split_sets = windows or []

    for split in split_sets:
        print(split)
        train_start, train_end, test_start, test_end, label, macd_label = split
        macd_params = select_macd_params(macd_cfg, macd_label, name)
        if not macd_params:
            print(f"Start computing GA MACD for {name} ({macd_label})")
            train_mask = (data["time"] >= train_start) & (data["time"] < train_end)
            train_data_raw = data.loc[train_mask, ["bid", "ask"]].copy()
            macd_params = compute_best_macd_params(train_data_raw, asset_name=name, verbose=True)
            if not macd_params:
                raise ValueError(f"Unable to compute MACD params for {name}")
            update_macd_cfg(MACD_CONFIG_PATH, macd_label, name, macd_params)
            return
        else:
            print(f"Got parameters for {name} ({macd_label})")
        s = macd_params.get("short")
        l = macd_params.get("long")
        if s is None or l is None:
            raise ValueError(f"Missing short/long for {name} ({macd_label})")
        df = add_ma(data.copy(), s, l, PREV_HOURS)
        df["time"] = data.loc[df.index, "time"]
        X = df[[col for col in df.columns if 'diff' in col]]
        BA = df[['bid', 'ask']]

        train_mask = (df["time"] >= train_start) & (df["time"] < train_end)
        test_mask = (df["time"] >= test_start) & (df["time"] < test_end)
        idx_train = df[train_mask].index
        idx_test = df[test_mask].index
        
        X_train, X_test = X.loc[idx_train], X.loc[idx_test]
        BA_train, BA_test = BA.loc[idx_train], BA.loc[idx_test]
        X_eval = pd.DataFrame()
        BA_eval = pd.DataFrame()

        encoder = Encoder(2, {}); decoder = Decoder(encoder.output.shape[1], {})
        agent = dummy_agent = Agent(encoder, decoder)

        threshold = macd_params.get("threshold")
        revert = macd_params.get("revert")
        if threshold is None or revert is None:
            raise ValueError(f"Missing threshold/revert for {name} ({macd_label})")
        
        dec_ma_train = X_train['diff'].map(lambda x: ma(x, threshold=threshold, revert=revert))
        sr_ma_train = agent.utility_function([BA_train.values, dec_ma_train.values]).numpy()

        dec_ma_test = X_test['diff'].map(lambda x: ma(x, threshold=threshold, revert=revert))
        sr_ma_test = agent.utility_function([BA_test.values, dec_ma_test.values]).numpy()

        if is_rolling:
            X_eval = pd.DataFrame()
            BA_eval = pd.DataFrame()
        else:
            dec_ma_eval = X_eval['diff'].map(lambda x: ma(x, threshold=threshold, revert=revert))
            sr_ma_eval = agent.utility_function([BA_eval.values, dec_ma_eval.values]).numpy()

        X_local = X.copy()
        X_local['ma'] = X_local['diff'].map(lambda x: ma(x, threshold=threshold, revert=revert))
        if is_rolling:
            X_eval = pd.DataFrame()
        else:
            X_eval = X_eval.copy()
            X_eval['ma'] = X_eval['diff'].map(lambda x: ma(x, threshold=threshold, revert=revert))


        multiplier = 1/X_local['diff'].std() # 1 if name == 'USDJPY' else 100 # 100
        X_local *= multiplier
        X_local['ma'] /= multiplier

        X_train, X_test = X_local.loc[X_train.index], X_local.loc[X_test.index]

        if not is_rolling and not X_eval.empty:
            X_eval *= multiplier
            X_eval['ma'] /= multiplier

        sr_test=0
        rml_train_srs = []
        rml_test_srs  = []
        finished = 0
        while finished < n_runs:
            broken=False
            encoder = Encoder(X_train.shape[1], model_config_e)
            decoder = Decoder(encoder.output.shape[1], model_config_d)
            agent = Agent(encoder, decoder)
            agent.set_lr(lr)

            # if n_runs<=2:
            #     finished += 1
            #     continue

            srs = [ ]
            srs_train = []
            srs_eval = []
            var = []
            var_train = []
            var_eval = []

            for i in tqdm(range(int(epochs/20))):
                agent.fit(X_train, BA_train, epochs=20, verbose=False)
                _, decs, sr, _, dec_pred = agent.test_iteration(X_train, BA_train) 
                var_train.append((np.array(decs)**2).mean()**0.5)
                srs_train.append(sr.numpy())
                _, decs, sr, _, dec_pred = agent.test_iteration(X_test, BA_test) 
                # Store checkpoint if this is the best SR so far

                if len(srs) == 0 or sr.numpy() > max(srs):
                    best_encoder_we = encoder.get_weights()
                    best_decoder_we = decoder.get_weights()

                var.append((np.array(decs)**2).mean()**0.5)
                srs.append(sr.numpy())

                if not is_rolling and not X_eval.empty:
                    _, decs_eval, sr_eval, _, dec_pred_eval = agent.test_iteration(X_eval, BA_eval) 
                    var_eval.append((np.array(decs_eval)**2).mean()**0.5)
                    srs_eval.append(sr_eval.numpy())

                if tf.math.is_nan(sr):
                    broken=True
                    break

                # if sr>sr_ma_test and i>250:
                #     break

            if broken:
                continue
            finished += 1

            if not np.isnan(sr_test): 
                rml_test_srs.append(sr_test)


            # Evaluate on training set.
            _, decs, sr_train, _, _ = agent.test_iteration(X_train, BA_train)
            sr_train = sr_train.numpy()
            var_train_scalar = (np.array(decs)**2).mean()**0.5

            if not np.isnan(sr_train):
                rml_train_srs.append(sr_train)
                
            _, decs, sr_test, _, _ = agent.test_iteration(X_test, BA_test)
            sr_test = sr_test.numpy()

            sr_recomputed = agent.utility_function([BA_test.values, decs]).numpy()

            if not is_rolling and not X_eval.empty:
                _, decs_eval, sr_eval, _, _ = agent.test_iteration(X_eval, BA_eval)
                sr_eval = sr_eval.numpy()


            best_encoder = Encoder(X_train.shape[1], model_config_e)
            best_decoder = Decoder(best_encoder.output.shape[1], model_config_d)
            best_encoder.set_weights(best_encoder_we)
            best_decoder.set_weights(best_decoder_we)
            best_agent = Agent(best_encoder, best_decoder)

            _, decs_test_best, sr_test_best, _, _ = best_agent.test_iteration(X_test, BA_test)
            sr_test_best = sr_test_best.numpy()
            sr_recomputed_best = best_agent.utility_function([BA_test.values, decs_test_best]).numpy()

            if not is_rolling and not X_eval.empty:
                _, decs_eval_best, sr_eval_best, _, _ = best_agent.test_iteration(X_eval, BA_eval)
                sr_eval_best = sr_eval_best.numpy()

            # if sr_test<sr_ma_test:
            #     continue

            # save in catalogue in results/ in catalogue per s, l sr_test weights in npy and var, srs for train and test
            base_catalogue = os.path.join(str(RESULTS_ROOT), f"results_15min_{name}")
            if not os.path.exists(base_catalogue):
                os.makedirs(base_catalogue)
            catalogue_folder = os.path.join(
                base_catalogue, f"{label}_{s}_{l}_{round(100*sr_recomputed,5)}_{finished}"
            )
            if not os.path.exists(catalogue_folder):
                os.makedirs(catalogue_folder)

            # Save decoder and encoder weights into separate .npy files.
            np.save(os.path.join(catalogue_folder, "decoder_weights.npy"), decoder.get_weights())
            np.save(os.path.join(catalogue_folder, "encode_weights.npy"), encoder.get_weights())

            df_save = pd.DataFrame({
                'MA': dec_ma_test.values,
                'RRL': decs,
                'RRL_best': decs_test_best,
                'bid': BA_test.bid.values,
                'ask': BA_test.ask.values,
            })
            df_save.to_csv(os.path.join(catalogue_folder, f'positions_test.csv'), index=False)

            test_times = df.loc[idx_test, "time"] if "time" in df.columns else None
            quarter_summary, quarter_rows = save_quarter_diagnostics(
                catalogue_folder,
                test_times,
                BA_test.reset_index(drop=True),
                dec_ma_test.values,
                decs,
                decs_test_best,
                agent,
            )

            eval_series = srs_eval if srs_eval else [np.nan] * len(srs)
            df = pd.DataFrame({
                "var": var,
                "var_train": var_train,
                "srs": srs,
                "srs_train": srs_train,
                "srs_eval": eval_series,
            })
            for key, value in quarter_summary.items():
                df[key] = value
            df.to_csv(os.path.join(catalogue_folder, 'df.csv'))

            sr_rows = [
                {
                    "quarter": "overall",
                    "rows": int(len(BA_test)),
                    "sr_ma": float(sr_ma_test),
                    "sr_rrl": float(sr_recomputed),
                    "sr_rrl_best": float(sr_recomputed_best),
                }
            ] + quarter_rows
            pd.DataFrame(sr_rows).to_csv(
                os.path.join(catalogue_folder, "sr_recomputed.csv"), index=False
            )

            if not is_rolling and not X_eval.empty:
                df_eval = pd.DataFrame({
                    'MA': dec_ma_eval.values,
                    'RRL': decs_eval,
                    'RRL_best': decs_eval_best,
                    'bid': BA_eval.bid.values,
                    'ask': BA_eval.ask.values,
                })
                df_eval.to_csv(os.path.join(catalogue_folder, f'positions_eval.csv'), index=False)




if __name__ == '__main__':
    variants = expand_schedule(config)
    for _, cfg_variant in enumerate(variants, start=1):
        apply_experiment_config(cfg_variant)
        RESULTS_ROOT = next_results_root(BASE_DIR / "results")
        snapshot_config(cfg_variant, RESULTS_ROOT)

        if not MULTIPROCESS:
            for name in data_list:
                main(name)
        else:
            import multiprocessing

            with multiprocessing.Pool(processes=N_PROCESSES) as pool:
                pool.map(main, data_list)
