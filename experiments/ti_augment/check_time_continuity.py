import argparse
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DIR = BASE_DIR / "data" / "processed"


def load_time_series(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if "time" in df.columns:
        times = pd.to_datetime(df["time"])
    else:
        times = pd.to_datetime(df.iloc[:, 0])
    return times.sort_values().reset_index(drop=True)


def check_continuity(times: pd.Series, expected_minutes: int):
    diffs = times.diff().dropna()
    expected = pd.Timedelta(minutes=expected_minutes)
    gaps = diffs[diffs != expected]
    return gaps


def main():
    parser = argparse.ArgumentParser(description="Check continuity of time columns in data files.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DIR))
    parser.add_argument("--minutes", type=int, default=15)
    parser.add_argument("--gap-days", type=int, default=7)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Missing directory: {data_dir}")

    for path in sorted(data_dir.glob("*.csv")):
        times = load_time_series(path)
        if times.empty:
            print(f"{path.name}: empty")
            continue
        gaps = check_continuity(times, args.minutes)
        if gaps.empty:
            print(f"{path.name}: OK")
            continue
        threshold = pd.Timedelta(days=args.gap_days)
        big_gaps = gaps[gaps >= threshold]
        longest = gaps.max()
        span = times.iloc[-1] - times.iloc[0]
        summary = (
            f"{path.name}: span {span} | rows {len(times)} | "
            f"{len(gaps)} gaps | longest {longest} | "
            f">= {args.gap_days}d: {len(big_gaps)}"
        )
        print(summary)
        for idx, delta in big_gaps.items():
            print(f"  {times.iloc[idx - 1]} -> {times.iloc[idx]} ({delta})")


if __name__ == "__main__":
    main()
