import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"

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


ASK_FACTOR = 1.00015


def normalize_pair(raw: str) -> Optional[str]:
    cleaned = re.sub(r"[^A-Z]", "", raw.upper())
    if len(cleaned) >= 6:
        return cleaned[:6]
    return None


def parse_histdata_name(name: str) -> Optional[Tuple[str, str]]:
    """
    Returns (pair, yyyymm) if a known pair is found in the filename.
    """
    upper = name.upper()
    cleaned = re.sub(r"[^A-Z]", "", upper)
    for pair in PAIR_LIST:
        if pair in upper or pair in cleaned:
            match = re.search(r"(\\d{4,6})", upper)
            return pair, match.group(1) if match else ""
    match = re.search(r"([A-Z]{6}).*?(\\d{4,6})", upper)
    if not match:
        return None
    pair = normalize_pair(match.group(1))
    if not pair:
        return None
    return pair, match.group(2)


def read_histdata_zip(zip_path: Path) -> pd.DataFrame:
    import zipfile

    cols = ["date", "time", "open", "high", "low", "close", "vol"]
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(zf.namelist()[0]) as f:
            df = pd.read_csv(f, sep=",", names=cols, engine="python")
            if df.shape[1] == 1:
                f.seek(0)
                df = pd.read_csv(f, sep=";", names=cols, engine="python")
    dt = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M")
    return df.assign(time=dt).set_index("time")[["close"]]


def build_15min_data(pair: str, raw_files: List[Path]) -> pd.DataFrame:
    dfs = [read_histdata_zip(p) for p in raw_files]
    if not dfs:
        raise ValueError(f"No raw files for {pair}")
    minute = pd.concat(dfs).sort_index()
    sampled = minute.resample("15min").last().dropna().rename(columns={"close": "bid"})
    sampled["ask"] = sampled["bid"] * ASK_FACTOR
    return sampled


def collect_raw_files(raw_dir: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for zip_path in sorted(raw_dir.glob("*.zip")):
        parsed = parse_histdata_name(zip_path.name)
        if not parsed:
            continue
        pair, _ = parsed
        groups.setdefault(pair, []).append(zip_path)
    return groups


def main():
    parser = argparse.ArgumentParser(description="Convert histdata zips to 15-minute FX data.")
    parser.add_argument("--pairs", nargs="*", default=PAIR_LIST)
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--output-dir", default=str(PROC_DIR))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = collect_raw_files(raw_dir)
    if not grouped:
        print(f"No zip files found in {raw_dir}")
        return

    for pair in args.pairs:
        raw_files = grouped.get(pair)
        if not raw_files:
            print(f"No raw data found for {pair}")
            continue
        processed = build_15min_data(pair, raw_files)
        out_path = output_dir / f"{pair}_15min_expanded.csv"
        processed.to_csv(out_path, float_format="%.5f")
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
