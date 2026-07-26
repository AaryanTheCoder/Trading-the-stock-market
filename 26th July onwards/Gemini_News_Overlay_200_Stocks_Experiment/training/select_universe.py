"""Select 200 liquid stocks using only information available through 2024."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(MODEL_ROOT / "source"))

from price_engine import SOURCE_PRICE_DIR, TRAINING_DATA_DIR, _read_one


REPO_ROOT = MODEL_ROOT.parents[1]
SECTOR_FILE = (
    REPO_ROOT
    / "Before 26th July"
    / "Models"
    / "RL_Codex_3_V2_34.63pct"
    / "data"
    / "cache"
    / "rl3_cache"
    / "sp500_current_sectors.csv"
)
UNIVERSE_SIZE = 200
LIQUIDITY_START = "2024-01-01"
LIQUIDITY_END = "2024-12-31"
MINIMUM_2024_SESSIONS = 240
MINIMUM_PRICE = 5.0

# Avoid nearly duplicate share classes consuming two portfolio slots.
EXCLUDED_DUPLICATE_CLASSES = {"GOOG", "FOX", "NWS"}


def yahoo_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


def main() -> None:
    members = pd.read_csv(SECTOR_FILE)
    rows: list[dict] = []
    skipped: dict[str, str] = {}

    for record in members.to_dict(orient="records"):
        ticker = yahoo_symbol(record["Symbol"])
        if ticker in EXCLUDED_DUPLICATE_CLASSES:
            skipped[ticker] = "duplicate share class"
            continue
        if not (SOURCE_PRICE_DIR / f"{ticker}.csv.gz").exists():
            skipped[ticker] = "no cached price file"
            continue
        try:
            prices = _read_one(ticker)
        except (OSError, ValueError, KeyError) as exc:
            skipped[ticker] = f"unreadable price data: {type(exc).__name__}"
            continue
        sample = prices.loc[LIQUIDITY_START:LIQUIDITY_END]
        if len(sample) < MINIMUM_2024_SESSIONS:
            skipped[ticker] = f"only {len(sample)} sessions in 2024"
            continue
        median_close = float(sample["close"].median())
        if median_close < MINIMUM_PRICE:
            skipped[ticker] = f"median 2024 price below ${MINIMUM_PRICE:g}"
            continue
        average_dollar_volume = float((sample["close"] * sample["volume"]).mean())
        rows.append(
            {
                "ticker": ticker,
                "company": str(record["Security"]),
                "sector": str(record["GICS Sector"]),
                "sub_industry": str(record["GICS Sub-Industry"]),
                "average_2024_dollar_volume": average_dollar_volume,
                "median_2024_close": median_close,
                "first_cached_date": prices.index[0].date().isoformat(),
                "last_cached_date": prices.index[-1].date().isoformat(),
                "sessions_2024": len(sample),
            }
        )

    eligible = pd.DataFrame(rows).sort_values(
        ["average_2024_dollar_volume", "ticker"],
        ascending=[False, True],
    )
    if len(eligible) < UNIVERSE_SIZE:
        raise RuntimeError(
            f"Only {len(eligible)} eligible stocks; need {UNIVERSE_SIZE}."
        )
    selected = eligible.head(UNIVERSE_SIZE).copy()
    if selected["ticker"].duplicated().any():
        raise AssertionError("Selected universe contains duplicate tickers.")

    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TRAINING_DATA_DIR / "universe_200.csv"
    selected.to_csv(csv_path, index=False)
    canonical = selected.to_csv(index=False).encode("utf-8")
    report = {
        "selection_cutoff": LIQUIDITY_END,
        "universe_size": len(selected),
        "selection_method": (
            "Top current S&P 500 members by average adjusted close times "
            "volume during 2024, after price/data-quality filters"
        ),
        "survivorship_bias_warning": (
            "The candidate list uses constituents known in 2026, so historical "
            "results exclude companies that left the index."
        ),
        "excluded_duplicate_share_classes": sorted(EXCLUDED_DUPLICATE_CLASSES),
        "eligible_count": len(eligible),
        "skipped_count": len(skipped),
        "universe_sha256": hashlib.sha256(canonical).hexdigest(),
        "sector_counts": {
            str(key): int(value)
            for key, value in selected["sector"].value_counts().items()
        },
    }
    json_path = TRAINING_DATA_DIR / "universe_200_metadata.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(selected[["ticker", "company", "sector", "average_2024_dollar_volume"]].to_string(index=False))
    print(f"\nSaved {len(selected)} stocks to {csv_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

