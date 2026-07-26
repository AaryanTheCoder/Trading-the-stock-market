"""Download only the latest overlay into this isolated experiment directory."""

from __future__ import annotations

from datetime import date, timedelta
import json

import pandas as pd
import yfinance as yf

from research_engine import LOCAL_PRICE_DIR, STOCKS, WORK_DIR


REFRESH_START = "2026-07-09"


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).lower() for column in result.columns]
    needed = ["open", "high", "low", "close", "volume"]
    if not set(needed).issubset(result.columns):
        return pd.DataFrame(columns=needed)
    result = result[needed].apply(pd.to_numeric, errors="coerce")
    result.index = pd.DatetimeIndex(result.index).tz_localize(None)
    return result.loc[~result.index.duplicated(keep="last")].sort_index().dropna()


def download_one(ticker: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        ticker,
        start=REFRESH_START,
        end=end,
        auto_adjust=True,
        progress=False,
        timeout=30,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return clean(raw)


def main() -> None:
    end = (date.today() + timedelta(days=1)).isoformat()
    print(f"Downloading {len(STOCKS)} tickers from {REFRESH_START} to {end}...")
    downloaded = yf.download(
        STOCKS,
        start=REFRESH_START,
        end=end,
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
        timeout=30,
    )
    LOCAL_PRICE_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    for ticker in STOCKS:
        try:
            frame = clean(downloaded[ticker])
        except (KeyError, TypeError):
            frame = pd.DataFrame()
        if frame.empty:
            print(f"Retrying {ticker} individually...")
            frame = download_one(ticker, end)
        if frame.empty:
            manifest[ticker] = {"status": "missing"}
            continue
        frame.to_csv(LOCAL_PRICE_DIR / f"{ticker}.csv.gz", compression="gzip")
        manifest[ticker] = {
            "status": "ok",
            "first_date": frame.index[0].date().isoformat(),
            "last_date": frame.index[-1].date().isoformat(),
            "rows": len(frame),
        }

    missing = [ticker for ticker, item in manifest.items() if item["status"] != "ok"]
    last_dates = sorted(
        {item["last_date"] for item in manifest.values() if item["status"] == "ok"}
    )
    report = {
        "requested_start": REFRESH_START,
        "requested_end_exclusive": end,
        "ticker_count": len(STOCKS),
        "successful_tickers": len(STOCKS) - len(missing),
        "missing_tickers": missing,
        "observed_last_dates": last_dates,
        "tickers": manifest,
    }
    path = WORK_DIR / "refresh_manifest.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "tickers"}, indent=2))
    if missing:
        raise RuntimeError(f"Latest overlay is incomplete: {', '.join(missing)}")


if __name__ == "__main__":
    main()

