"""Leakage-aware, compact cross-sectional research engine.

All signals use information available at a session close. Direct-comparison
backtests use the RL Codex 2 convention: rebalance at that close and apply the
next close-to-close return. A separate next-open stress test is added by the
final experiment script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


WORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORK_DIR.parent
SOURCE_PRICE_DIR = REPO_ROOT / "CODEX NOT ME" / "rl3_cache" / "prices"
LOCAL_PRICE_DIR = WORK_DIR / "latest_prices"

STOCKS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AMAT", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BKNG", "BLK", "BMY", "BRK-B", "C", "CAT",
    "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX", "DE",
    "DHR", "DIS", "DUK", "EMR", "FDX", "GD", "GE", "GILD", "GM", "GOOGL",
    "GS", "HD", "HON", "IBM", "INTC", "INTU", "ISRG", "JNJ", "JPM", "KO",
    "LIN", "LLY", "LMT", "LOW", "LRCX", "MA", "MCD", "MDLZ", "MDT", "META",
    "MMM", "MO", "MRK", "MS", "MSFT", "MU", "NEE", "NFLX", "NKE", "NOW",
    "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "QCOM", "RTX", "SBUX", "SCHW",
    "SO", "SPG", "T", "TMO", "TMUS", "TSLA", "TXN", "UNH", "UNP", "UPS",
    "USB", "V", "VZ", "WFC", "WMT", "XOM", "ADP", "TGT", "PYPL", "O",
]

FEE = 0.001
FEATURE_NAMES = (
    "ret_5",
    "ret_20",
    "ret_60",
    "ret_120",
    "ret_252",
    "skip_252_20",
    "trend_20",
    "trend_50",
    "trend_200",
    "vol_20",
    "vol_60",
    "drawdown_252",
    "range_20",
    "volume_20",
)


@dataclass
class MarketData:
    dates: pd.DatetimeIndex
    tickers: tuple[str, ...]
    close: np.ndarray
    open: np.ndarray
    features: np.ndarray

    def date_positions(self, start: str, end: str) -> np.ndarray:
        mask = (self.dates >= pd.Timestamp(start)) & (self.dates <= pd.Timestamp(end))
        return np.flatnonzero(mask)


def _read_one(ticker: str) -> pd.DataFrame:
    path = SOURCE_PRICE_DIR / f"{ticker}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing cached prices for {ticker}: {path}")
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    local_path = LOCAL_PRICE_DIR / f"{ticker}.csv.gz"
    if local_path.exists():
        local = pd.read_csv(local_path, index_col=0, parse_dates=True)
        frame = pd.concat([frame, local])
    frame.columns = [str(column).lower() for column in frame.columns]
    needed = ["open", "high", "low", "close", "volume"]
    frame = frame[needed].apply(pd.to_numeric, errors="coerce")
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
    return frame.loc[~frame.index.duplicated(keep="last")].sort_index().dropna()


def _percent_change(values: np.ndarray, periods: int) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=np.float64)
    result[periods:] = values[periods:] / values[:-periods] - 1.0
    return result


def _rolling_mean(frame: pd.DataFrame, window: int) -> np.ndarray:
    return frame.rolling(window, min_periods=window).mean().to_numpy()


def _rolling_std(frame: pd.DataFrame, window: int) -> np.ndarray:
    return frame.rolling(window, min_periods=window).std(ddof=1).to_numpy()


def load_market_data(tickers: Iterable[str] = STOCKS) -> MarketData:
    """Load cached adjusted OHLCV data without writing outside this directory."""
    ticker_list = tuple(tickers)
    raw = {ticker: _read_one(ticker) for ticker in ticker_list}
    dates: pd.DatetimeIndex | None = None
    for frame in raw.values():
        dates = frame.index if dates is None else dates.intersection(frame.index)
    if dates is None:
        raise RuntimeError("No price data loaded.")
    dates = dates.sort_values()

    close_frame = pd.DataFrame(
        {ticker: raw[ticker].loc[dates, "close"] for ticker in ticker_list},
        index=dates,
    )
    open_frame = pd.DataFrame(
        {ticker: raw[ticker].loc[dates, "open"] for ticker in ticker_list},
        index=dates,
    )
    high_frame = pd.DataFrame(
        {ticker: raw[ticker].loc[dates, "high"] for ticker in ticker_list},
        index=dates,
    )
    low_frame = pd.DataFrame(
        {ticker: raw[ticker].loc[dates, "low"] for ticker in ticker_list},
        index=dates,
    )
    volume_frame = pd.DataFrame(
        {ticker: raw[ticker].loc[dates, "volume"] for ticker in ticker_list},
        index=dates,
    )

    close = close_frame.to_numpy(dtype=np.float64)
    daily = close_frame.pct_change(fill_method=None)
    mean20 = _rolling_mean(close_frame, 20)
    mean50 = _rolling_mean(close_frame, 50)
    mean200 = _rolling_mean(close_frame, 200)
    rolling_high = close_frame.rolling(252, min_periods=252).max().to_numpy()
    dollar_volume = close_frame * volume_frame

    pieces = [
        _percent_change(close, 5),
        _percent_change(close, 20),
        _percent_change(close, 60),
        _percent_change(close, 120),
        _percent_change(close, 252),
        np.divide(
            close,
            np.roll(close, 252, axis=0),
            out=np.full_like(close, np.nan),
            where=np.roll(close, 252, axis=0) != 0,
        )
        / np.divide(
            close,
            np.roll(close, 20, axis=0),
            out=np.full_like(close, np.nan),
            where=np.roll(close, 20, axis=0) != 0,
        )
        - 1,
        close / mean20 - 1,
        close / mean50 - 1,
        close / mean200 - 1,
        _rolling_std(daily, 20) * np.sqrt(252),
        _rolling_std(daily, 60) * np.sqrt(252),
        close / rolling_high - 1,
        ((high_frame - low_frame) / close_frame)
        .rolling(20, min_periods=20)
        .mean()
        .to_numpy(),
        (dollar_volume / dollar_volume.rolling(20, min_periods=20).mean() - 1)
        .to_numpy(),
    ]
    # np.roll wraps at the beginning; explicitly invalidate all warm-up rows.
    features = np.stack(pieces, axis=2)
    features[:252, :, :] = np.nan
    return MarketData(
        dates=dates,
        tickers=ticker_list,
        close=close,
        open=open_frame.to_numpy(dtype=np.float64),
        features=features,
    )


def cross_sectional_ranks(values: np.ndarray) -> np.ndarray:
    """Map each date/feature cross-section to approximately [-1, 1]."""
    ranked = np.full_like(values, np.nan, dtype=np.float64)
    for date_number in range(values.shape[0]):
        for feature_number in range(values.shape[2]):
            column = values[date_number, :, feature_number]
            valid = np.isfinite(column)
            count = int(valid.sum())
            if count < 2:
                continue
            order = np.argsort(column[valid], kind="mergesort")
            ranks = np.empty(count, dtype=np.float64)
            ranks[order] = np.arange(count, dtype=np.float64)
            scaled = 2.0 * ranks / (count - 1) - 1.0
            ranked[date_number, valid, feature_number] = scaled
    return ranked


def rebalance_positions(
    data: MarketData, start: str, end: str, holding_days: int
) -> np.ndarray:
    available = data.date_positions(start, end)
    if len(available) < 2:
        raise ValueError("Backtest period has fewer than two sessions.")
    return available[::holding_days]


def score_from_weights(ranked_features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("dtf,f->dt", ranked_features, weights)


def simulate_scores(
    data: MarketData,
    scores: np.ndarray,
    *,
    start: str,
    end: str,
    holding_days: int = 20,
    top_k: int = 10,
    keep_rank: int | None = None,
    fee: float = FEE,
    execution: str = "close",
    rebalance_offset: int = 0,
) -> dict:
    """Simulate a long-only equal-weight selector with drifting weights."""
    positions = data.date_positions(start, end)
    if len(positions) < 2:
        raise ValueError("Backtest period has fewer than two sessions.")
    first, last = int(positions[0]), int(positions[-1])
    if not 0 <= rebalance_offset < holding_days:
        raise ValueError("rebalance_offset must be in [0, holding_days).")
    rebalance_set = set(positions[rebalance_offset::holding_days].tolist())
    weights = np.zeros(len(data.tickers), dtype=np.float64)
    value = 100_000.0
    values = [value]
    value_dates = [data.dates[first]]
    turnover_total = 0.0
    trade_rows: list[dict] = []
    holding_counts: list[int] = []

    for today in range(first, last):
        rebalance_today = today in rebalance_set
        target = weights.copy()
        if rebalance_today:
            ranking = np.argsort(np.nan_to_num(scores[today], nan=-np.inf))[::-1]
            valid_ranking = [int(i) for i in ranking if np.isfinite(scores[today, i])]
            keep_cutoff = top_k if keep_rank is None else keep_rank
            keep_zone = set(valid_ranking[:keep_cutoff])
            held = np.flatnonzero(weights > 1e-8)
            selected = [int(i) for i in held if int(i) in keep_zone][:top_k]
            for number in valid_ranking:
                if len(selected) >= top_k:
                    break
                if number not in selected:
                    selected.append(number)
            target = np.zeros_like(weights)
            if selected:
                target[selected] = 1.0 / len(selected)

        if execution == "close":
            if rebalance_today:
                changes = target - weights
                for number in np.flatnonzero(np.abs(changes) > 1e-9):
                    trade_rows.append(
                        {
                            "signal_date": data.dates[today].date().isoformat(),
                            "execution_date": data.dates[today].date().isoformat(),
                            "ticker": data.tickers[number],
                            "action": "BUY" if changes[number] > 0 else "SELL",
                            "weight_before": float(weights[number]),
                            "weight_after": float(target[number]),
                            "weight_change": float(changes[number]),
                            "score": float(scores[today, number]),
                        }
                    )
                turnover = float(np.abs(changes).sum())
                turnover_total += turnover
                value *= max(0.0, 1.0 - fee * turnover)
            returns = data.close[today + 1] / data.close[today] - 1.0
            multiplier = float(1.0 + np.sum(target * returns))
            value *= multiplier
            weights = target * (1.0 + returns) / multiplier
        elif execution == "next_open":
            # Existing positions first earn the close-to-next-open move.
            overnight_returns = data.open[today + 1] / data.close[today] - 1.0
            overnight_multiplier = float(1.0 + np.sum(weights * overnight_returns))
            value *= overnight_multiplier
            pretrade = weights * (1.0 + overnight_returns) / overnight_multiplier

            if rebalance_today:
                changes = target - pretrade
                for number in np.flatnonzero(np.abs(changes) > 1e-9):
                    trade_rows.append(
                        {
                            "signal_date": data.dates[today].date().isoformat(),
                            "execution_date": data.dates[today + 1].date().isoformat(),
                            "ticker": data.tickers[number],
                            "action": "BUY" if changes[number] > 0 else "SELL",
                            "weight_before": float(pretrade[number]),
                            "weight_after": float(target[number]),
                            "weight_change": float(changes[number]),
                            "score": float(scores[today, number]),
                        }
                    )
                turnover = float(np.abs(changes).sum())
                turnover_total += turnover
                value *= max(0.0, 1.0 - fee * turnover)
                posttrade = target
            else:
                posttrade = pretrade
            intraday_returns = data.close[today + 1] / data.open[today + 1] - 1.0
            multiplier = float(1.0 + np.sum(posttrade * intraday_returns))
            value *= multiplier
            weights = posttrade * (1.0 + intraday_returns) / multiplier
        else:
            raise ValueError(f"Unknown execution mode: {execution}")

        values.append(value)
        value_dates.append(data.dates[today + 1])
        holding_counts.append(int(np.count_nonzero(target > 1e-8)))

    equity = pd.Series(values, index=pd.DatetimeIndex(value_dates), name="equity")
    daily_returns = equity.pct_change(fill_method=None).dropna()
    running_high = equity.cummax()
    max_drawdown = float((equity / running_high - 1.0).min())
    annual_vol = float(daily_returns.std(ddof=1) * np.sqrt(252))
    sharpe = (
        float(daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(252))
        if daily_returns.std(ddof=1) > 0
        else 0.0
    )
    return {
        "first_date": equity.index[0],
        "last_date": equity.index[-1],
        "final_balance": float(value),
        "return": float(value / 100_000.0 - 1.0),
        "max_drawdown": max_drawdown,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "turnover": turnover_total,
        "average_holdings": float(np.mean(holding_counts)),
        "equity": equity,
        "trades": pd.DataFrame(trade_rows),
        "final_weights": {
            data.tickers[i]: float(weight)
            for i, weight in enumerate(weights)
            if weight > 1e-8
        },
    }


def equal_weight_benchmark(data: MarketData, start: str, end: str) -> dict:
    positions = data.date_positions(start, end)
    first, last = int(positions[0]), int(positions[-1])
    multipliers = data.close[last] / data.close[first]
    final = 100_000.0 * (1.0 - FEE) * float(np.mean(multipliers))
    return {"final_balance": final, "return": final / 100_000.0 - 1.0}
