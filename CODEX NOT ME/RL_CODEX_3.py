"""RL CODEX 3: train a portfolio-aware PPO over the historical S&P 500.

This is a new experiment.  RL CODEX 1 and RL CODEX 2 are not imported or
changed.

The important design ideas are:

* Point-in-time S&P 500 membership decides which companies are eligible on
  each historical date, reducing survivorship bias.
* Roughly 500 liquid stocks are examined at every decision date.
* PPO chooses a set of understandable factor weights and a cash exposure.
  Those weights rank all eligible stocks.  We do NOT call PPO's action
  probability a prediction confidence.
* Training and testing use the same long-only portfolio builder.
* Signals use today's close, but trades occur at the next market open.
* Spread, slippage, turnover, drawdown, position, sector and volatility limits
  are included.
* Several expanding walk-forward tests select a seed before the final model is
  trained through July 10, 2025.  Later prices are never used for training.

Free-data limitation
--------------------
Yahoo may not return every ticker. Missing tickers are excluded, and the
program writes a data-quality report. Historical membership and free prices
can still contain errors, so results must not be treated as proof of future
performance.

This remains educational research, not a promise of profit or a real-money
trading system.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.request import Request, urlopen

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed


# ---------------------------------------------------------------------------
# Reproducible experiment settings
# ---------------------------------------------------------------------------

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
CACHE_DIRECTORY = SCRIPT_DIRECTORY / "rl3_cache"
MODEL_PATH = SCRIPT_DIRECTORY / "rl_codex_3_portfolio_policy"
METADATA_PATH = SCRIPT_DIRECTORY / "RL_CODEX_3_METADATA.json"
WALK_FORWARD_PATH = SCRIPT_DIRECTORY / "RL_CODEX_3_WALK_FORWARD.csv"
DATA_QUALITY_PATH = SCRIPT_DIRECTORY / "RL_CODEX_3_DATA_QUALITY.json"

MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/hanshof/sp500_constituents/"
    "main/sp_500_historical_components.csv"
)
SECTOR_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
    "main/data/constituents.csv"
)

DOWNLOAD_START = "2009-01-01"  # Warm-up for 200-session indicators.
DOWNLOAD_END = "2025-07-11"  # Exclusive: includes prices through July 10.
TRAIN_START = "2010-01-01"
FINAL_TRAIN_END = "2025-07-10"

WALK_FORWARD_YEARS = (2021, 2022, 2023, 2024)
CANDIDATE_SEEDS = (11, 22, 33)
WALK_FORWARD_STEPS = 50_000
FINAL_TRAINING_STEPS = 400_000

HOLDING_PERIOD_DAYS = 20
EPISODE_DECISIONS = 36
MAX_STOCKS_EXAMINED = 500
TOP_STOCKS_TO_HOLD = 20
KEEP_HOLDING_WITHIN_TOP_RANK = 40

MINIMUM_PRICE = 5.0
MINIMUM_DOLLAR_VOLUME = 5_000_000.0
# Ordinary missing tickers are skipped, but stop if less than 60% of the
# point-in-time universe works.
MINIMUM_DATA_COVERAGE = 0.60

MAX_POSITION_WEIGHT = 0.075
MAX_SECTOR_WEIGHT = 0.25
TARGET_ANNUAL_VOLATILITY = 0.18
MINIMUM_TRADE_WEIGHT = 0.01
MAX_TOTAL_WEIGHT_CHANGE = 0.60

# Eight basis points represents a simple spread/ordinary slippage estimate.
# Market impact grows with the square root of trade size / dollar volume.
BASE_EXECUTION_COST = 0.0008
MARKET_IMPACT_COEFFICIENT = 0.0015
MAX_MARKET_IMPACT = 0.005
MISSING_EXIT_HAIRCUT = 0.10

SECTOR_ETFS = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

# These are interpretable cross-sectional factors.  PPO chooses how strongly
# positive or negative each factor should be in the current market regime.
FACTOR_COLUMNS = [
    "factor_return_1d",
    "factor_return_5d",
    "factor_return_20d",
    "factor_return_60d",
    "factor_return_120d",
    "factor_trend_20d",
    "factor_trend_50d",
    "factor_trend_200d",
    "factor_volatility_20d",
    "factor_rsi_14",
    "factor_volume_20d",
    "factor_daily_range",
    "factor_relative_market_20d",
    "factor_relative_market_60d",
    "factor_relative_sector_20d",
]


def yahoo_symbol(symbol: str) -> str:
    """Convert common index notation such as BRK.B into Yahoo notation."""
    return str(symbol).strip().upper().replace(".", "-")


def _safe_filename(symbol: str) -> str:
    return symbol.replace("/", "_").replace("^", "INDEX_")


def _download_text(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "RL-Codex-3/1.0"})
    with urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())


@dataclass(frozen=True)
class MembershipHistory:
    dates: pd.DatetimeIndex
    members: tuple[frozenset[str], ...]

    def members_on(self, when: pd.Timestamp | str) -> frozenset[str]:
        timestamp = pd.Timestamp(when).tz_localize(None)
        position = int(self.dates.searchsorted(timestamp, side="right") - 1)
        if position < 0:
            raise ValueError(f"No membership snapshot exists before {timestamp.date()}.")
        return self.members[position]

    def union(self, start: str, end: str) -> set[str]:
        start_time = pd.Timestamp(start)
        end_time = pd.Timestamp(end)
        selected: set[str] = set()
        for snapshot_date, snapshot in zip(self.dates, self.members):
            if start_time <= snapshot_date <= end_time:
                selected.update(snapshot)
        selected.update(self.members_on(start_time))
        selected.update(self.members_on(end_time))
        return selected


def load_membership_history(*, refresh: bool = False) -> MembershipHistory:
    """Load and cache point-in-time S&P 500 membership snapshots."""
    path = CACHE_DIRECTORY / "sp500_membership_history.csv"
    if refresh or not path.exists():
        print("Downloading point-in-time S&P 500 membership history...")
        _download_text(MEMBERSHIP_URL, path)

    frame = pd.read_csv(path)
    lowered = {str(column).lower(): column for column in frame.columns}
    if "date" not in lowered or "tickers" not in lowered:
        raise RuntimeError("Membership CSV must contain date and tickers columns.")

    frame = frame.rename(
        columns={lowered["date"]: "date", lowered["tickers"]: "tickers"}
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "tickers"]).sort_values("date")
    frame = frame.drop_duplicates("date", keep="last")

    dates: list[pd.Timestamp] = []
    snapshots: list[frozenset[str]] = []
    for row in frame.itertuples(index=False):
        tickers = frozenset(
            yahoo_symbol(value)
            for value in str(row.tickers).split(",")
            if str(value).strip()
        )
        if len(tickers) < 400:
            continue
        dates.append(pd.Timestamp(row.date).tz_localize(None))
        snapshots.append(tickers)

    if not dates:
        raise RuntimeError("No valid S&P 500 membership snapshots were loaded.")
    return MembershipHistory(pd.DatetimeIndex(dates), tuple(snapshots))


def load_sector_map(*, refresh: bool = False) -> dict[str, str]:
    """Load today's sector labels; old unavailable tickers remain Unknown."""
    path = CACHE_DIRECTORY / "sp500_current_sectors.csv"
    if refresh or not path.exists():
        print("Downloading current S&P 500 sector labels...")
        _download_text(SECTOR_URL, path)
    frame = pd.read_csv(path)
    lowered = {str(column).lower(): column for column in frame.columns}
    symbol_column = lowered.get("symbol")
    sector_column = lowered.get("gics sector")
    if symbol_column is None or sector_column is None:
        raise RuntimeError("Sector CSV must contain Symbol and GICS Sector.")
    return {
        yahoo_symbol(symbol): str(sector)
        for symbol, sector in zip(frame[symbol_column], frame[sector_column])
        if pd.notna(symbol) and pd.notna(sector)
    }


def _clean_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).lower() for column in result.columns]
    needed = ["open", "high", "low", "close", "volume"]
    if not set(needed).issubset(result.columns):
        return pd.DataFrame(columns=needed)
    result = result[needed].apply(pd.to_numeric, errors="coerce")
    result.index = pd.to_datetime(result.index, errors="coerce")
    result = result.loc[~result.index.isna()]
    if result.index.tz is not None:
        result.index = result.index.tz_localize(None)
    result = result.loc[~result.index.duplicated(keep="last")].sort_index()
    return result.dropna(subset=["open", "high", "low", "close"])


def _read_price_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    return _clean_price_frame(frame)


def _write_price_cache(symbol: str, frame: pd.DataFrame) -> None:
    directory = CACHE_DIRECTORY / "prices"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_safe_filename(symbol)}.csv.gz"
    if path.exists():
        try:
            previous = _read_price_csv(path)
            frame = pd.concat([previous, frame])
            frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
        except (OSError, ValueError, pd.errors.ParserError):
            pass
    frame.to_csv(path, compression="gzip")


def download_prices(
    symbols: Iterable[str],
    start: str,
    end: str,
    *,
    refresh: bool = False,
    external_directory: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Download adjusted OHLCV data in chunks and cache one CSV per ticker."""
    requested = sorted({yahoo_symbol(symbol) for symbol in symbols})
    # Yahoo's end date is exclusive. A seven-day allowance covers weekends and
    # short market holidays without accepting a cache that is months out of date.
    latest_acceptable_cache_date = pd.Timestamp(end) - pd.Timedelta(days=7)
    prices: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    stale_cache: dict[str, pd.DataFrame] = {}
    cache_directory = CACHE_DIRECTORY / "prices"
    cache_directory.mkdir(parents=True, exist_ok=True)

    to_download: list[str] = []
    for symbol in requested:
        external = (
            external_directory / f"{_safe_filename(symbol)}.csv"
            if external_directory is not None
            else None
        )
        cache = cache_directory / f"{_safe_filename(symbol)}.csv.gz"
        try:
            if external is not None and external.exists():
                frame = _read_price_csv(external)
                selected = frame.loc[start:end].copy()
                if len(selected) >= 20:
                    # A properly supplied delisted history is expected to end
                    # before the overall request end.
                    prices[symbol] = frame
                    continue
            elif cache.exists() and not refresh:
                frame = _read_price_csv(cache)
            else:
                to_download.append(symbol)
                continue
            selected = frame.loc[start:end].copy()
            cache_reaches_request = (
                not frame.empty
                and pd.Timestamp(frame.index.max()) >= latest_acceptable_cache_date
            )
            if len(selected) >= 20 and cache_reaches_request:
                prices[symbol] = frame
            else:
                if len(selected) >= 20:
                    stale_cache[symbol] = frame
                to_download.append(symbol)
        except (OSError, ValueError, pd.errors.ParserError):
            to_download.append(symbol)

    for offset in range(0, len(to_download), 50):
        chunk = to_download[offset : offset + 50]
        print(
            f"Downloading prices {offset + 1}-{offset + len(chunk)} "
            f"of {len(to_download)}..."
        )
        try:
            downloaded = yf.download(
                chunk,
                start=start,
                end=end,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
                timeout=30,
            )
        except Exception as error:  # yfinance raises several network types.
            print(f"Price chunk failed: {error}")
            downloaded = pd.DataFrame()

        for symbol in chunk:
            try:
                if downloaded.empty:
                    raise KeyError(symbol)
                if isinstance(downloaded.columns, pd.MultiIndex):
                    frame = downloaded[symbol].copy()
                elif len(chunk) == 1:
                    frame = downloaded.copy()
                else:
                    raise KeyError(symbol)
                frame = _clean_price_frame(frame)
                if len(frame.loc[start:end]) < 20:
                    raise ValueError("too little price history")
                prices[symbol] = frame
                _write_price_cache(symbol, frame)
            except (KeyError, ValueError):
                # A delisted company may correctly have no recent prices. Keep
                # its older cached history so earlier simulation dates remain
                # accurate; it still cannot be bought after its prices stop.
                fallback = stale_cache.get(symbol)
                if fallback is not None:
                    prices[symbol] = fallback
                else:
                    missing.append(symbol)

    missing.extend(symbol for symbol in requested if symbol not in prices)
    missing = [symbol for symbol in set(missing) if symbol not in prices]
    return prices, sorted(missing)


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    change = close.diff()
    gains = change.clip(lower=0).rolling(window).mean()
    losses = (-change.clip(upper=0)).rolling(window).mean()
    relative_strength = gains / losses.replace(0, np.nan)
    rsi = 100 - 100 / (1 + relative_strength)
    rsi = rsi.mask((losses == 0) & (gains > 0), 100)
    rsi = rsi.mask((losses == 0) & (gains == 0), 50)
    return rsi


def add_asset_features(
    frame: pd.DataFrame,
    spy_close: pd.Series,
    sector_close: pd.Series | None,
) -> pd.DataFrame:
    """Create scale-limited features using current and previous data only."""
    featured = frame.copy()
    close = featured["close"]
    daily_return = close.pct_change(fill_method=None)

    featured["factor_return_1d"] = (daily_return / 0.10).clip(-2, 2)
    featured["factor_return_5d"] = (
        close.pct_change(5, fill_method=None) / 0.20
    ).clip(-2, 2)
    featured["factor_return_20d"] = (
        close.pct_change(20, fill_method=None) / 0.40
    ).clip(-2, 2)
    featured["factor_return_60d"] = (
        close.pct_change(60, fill_method=None) / 0.70
    ).clip(-2, 2)
    featured["factor_return_120d"] = (
        close.pct_change(120, fill_method=None) / 1.20
    ).clip(-2, 2)

    for window, scale in ((20, 0.20), (50, 0.30), (200, 0.50)):
        average = close.rolling(window).mean()
        featured[f"factor_trend_{window}d"] = (
            (close / average - 1) / scale
        ).clip(-2, 2)

    annual_volatility = daily_return.rolling(20).std() * math.sqrt(252)
    featured["annual_volatility_20d"] = annual_volatility
    featured["factor_volatility_20d"] = (annual_volatility / 0.80).clip(0, 2)
    featured["factor_rsi_14"] = ((calculate_rsi(close) - 50) / 50).clip(-1, 1)
    featured["factor_volume_20d"] = (
        (featured["volume"] / featured["volume"].rolling(20).mean() - 1) / 2
    ).clip(-1, 1)
    featured["factor_daily_range"] = (
        ((featured["high"] - featured["low"]) / close) / 0.10
    ).clip(0, 2)

    spy_aligned = spy_close.reindex(featured.index).ffill()
    featured["factor_relative_market_20d"] = (
        (
            close.pct_change(20, fill_method=None)
            - spy_aligned.pct_change(20, fill_method=None)
        )
        / 0.30
    ).clip(-2, 2)
    featured["factor_relative_market_60d"] = (
        (
            close.pct_change(60, fill_method=None)
            - spy_aligned.pct_change(60, fill_method=None)
        )
        / 0.50
    ).clip(-2, 2)

    comparison = spy_aligned if sector_close is None else sector_close.reindex(
        featured.index
    ).ffill()
    featured["factor_relative_sector_20d"] = (
        (
            close.pct_change(20, fill_method=None)
            - comparison.pct_change(20, fill_method=None)
        )
        / 0.30
    ).clip(-2, 2)

    featured["dollar_volume_20d"] = (
        close * featured["volume"]
    ).rolling(20).mean()
    featured = featured.replace([np.inf, -np.inf], np.nan)
    return featured


def make_market_features(
    spy: pd.DataFrame, sector_prices: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    close = spy["close"]
    returns = close.pct_change(fill_method=None)
    result = pd.DataFrame(index=spy.index)
    result["market_return_5d"] = (close.pct_change(5) / 0.15).clip(-2, 2)
    result["market_return_20d"] = (close.pct_change(20) / 0.30).clip(-2, 2)
    result["market_return_60d"] = (close.pct_change(60) / 0.50).clip(-2, 2)
    for window, scale in ((20, 0.15), (50, 0.25), (200, 0.40)):
        result[f"market_trend_{window}d"] = (
            (close / close.rolling(window).mean() - 1) / scale
        ).clip(-2, 2)
    result["market_volatility_20d"] = (
        returns.rolling(20).std() * math.sqrt(252) / 0.60
    ).clip(0, 2)
    result["market_drawdown_200d"] = (
        close / close.rolling(200).max() - 1
    ).clip(-1, 0)

    for sector, etf in SECTOR_ETFS.items():
        key = "sector_" + sector.lower().replace(" ", "_") + "_return_20d"
        data = sector_prices.get(etf)
        if data is None:
            result[key] = 0.0
        else:
            aligned = data["close"].reindex(result.index).ffill()
            result[key] = (aligned.pct_change(20) / 0.30).clip(-2, 2)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


@dataclass
class MarketBundle:
    membership: MembershipHistory
    prices: dict[str, pd.DataFrame]
    sectors: dict[str, str]
    market_features: pd.DataFrame
    dates: pd.DatetimeIndex
    missing_tickers: list[str]
    _raw_cross_section_cache: dict[pd.Timestamp, pd.DataFrame] = field(
        default_factory=dict, init=False, repr=False
    )
    _ranked_cross_section_cache: dict[pd.Timestamp, pd.DataFrame] = field(
        default_factory=dict, init=False, repr=False
    )
    _ranked_components_cache: dict[
        pd.Timestamp, tuple[tuple[str, ...], np.ndarray]
    ] = field(default_factory=dict, init=False, repr=False)
    _observation_static_cache: dict[pd.Timestamp, tuple[float, ...]] = field(
        default_factory=dict, init=False, repr=False
    )
    _market_row_cache: dict[pd.Timestamp, pd.Series] = field(
        default_factory=dict, init=False, repr=False
    )
    _date_positions: dict[pd.Timestamp, int] = field(
        default_factory=dict, init=False, repr=False
    )
    _open_arrays: dict[str, np.ndarray] = field(
        default_factory=dict, init=False, repr=False
    )
    _close_arrays: dict[str, np.ndarray] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Pre-align prices once instead of doing pandas lookups every step."""
        self._date_positions = {
            pd.Timestamp(when): position for position, when in enumerate(self.dates)
        }
        for ticker, data in self.prices.items():
            self._open_arrays[ticker] = (
                data["open"].reindex(self.dates).to_numpy(dtype=float)
            )
            self._close_arrays[ticker] = (
                data["close"].reindex(self.dates).to_numpy(dtype=float)
            )

    @property
    def observation_names(self) -> list[str]:
        names = list(self.market_features.columns)
        names += [f"cross_mean_{name}" for name in FACTOR_COLUMNS]
        names += [f"cross_std_{name}" for name in FACTOR_COLUMNS]
        names += [
            "breadth_positive_20d",
            "breadth_above_50d",
            "breadth_above_200d",
            "data_coverage",
            "portfolio_exposure",
            "portfolio_holdings",
            "portfolio_drawdown",
            "portfolio_recent_return",
            "portfolio_previous_turnover",
        ]
        return names

    def members_on(self, when: pd.Timestamp) -> frozenset[str]:
        return self.membership.members_on(when)

    def market_row(self, when: pd.Timestamp) -> pd.Series:
        key = pd.Timestamp(when)
        cached = self._market_row_cache.get(key)
        if cached is None:
            cached = self.market_features.loc[:key].iloc[-1]
            self._market_row_cache[key] = cached
        return cached

    def price(self, ticker: str, when: pd.Timestamp, column: str) -> float | None:
        """Read an aligned open/close price with constant-time array access."""
        position = self._date_positions.get(pd.Timestamp(when))
        arrays = self._open_arrays if column == "open" else self._close_arrays
        values = arrays.get(ticker)
        if position is None or values is None:
            return None
        numeric = float(values[position])
        return numeric if np.isfinite(numeric) and numeric > 0 else None

    def dates_after_through(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DatetimeIndex:
        """Return market dates after start through end without a full mask scan."""
        start_position = self._date_positions[pd.Timestamp(start)]
        end_position = self._date_positions[pd.Timestamp(end)]
        return self.dates[start_position + 1 : end_position + 1]

    def raw_cross_section(self, when: pd.Timestamp) -> pd.DataFrame:
        key = pd.Timestamp(when)
        cached = self._raw_cross_section_cache.get(key)
        if cached is not None:
            return cached
        members = self.members_on(when)
        rows: list[dict[str, Any]] = []
        for ticker in members:
            data = self.prices.get(ticker)
            if data is None or when not in data.index:
                continue
            row = data.loc[when]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            if (
                not np.isfinite(row.get("close", np.nan))
                or float(row["close"]) < MINIMUM_PRICE
                or not np.isfinite(row.get("dollar_volume_20d", np.nan))
                or float(row["dollar_volume_20d"]) < MINIMUM_DOLLAR_VOLUME
                or not np.isfinite(row[FACTOR_COLUMNS].to_numpy(dtype=float)).all()
            ):
                continue
            values = {name: float(row[name]) for name in FACTOR_COLUMNS}
            values.update(
                {
                    "ticker": ticker,
                    "sector": self.sectors.get(ticker, "Unknown"),
                    "annual_volatility_20d": float(row["annual_volatility_20d"]),
                    "dollar_volume_20d": float(row["dollar_volume_20d"]),
                }
            )
            rows.append(values)

        if not rows:
            result = pd.DataFrame(
                columns=FACTOR_COLUMNS
                + ["sector", "annual_volatility_20d", "dollar_volume_20d"]
            )
        else:
            result = pd.DataFrame(rows).set_index("ticker")
            result = result.sort_values("dollar_volume_20d", ascending=False).head(
                MAX_STOCKS_EXAMINED
            )
        self._raw_cross_section_cache[key] = result
        return result

    def ranked_cross_section(self, when: pd.Timestamp) -> pd.DataFrame:
        key = pd.Timestamp(when)
        cached = self._ranked_cross_section_cache.get(key)
        if cached is not None:
            return cached
        raw = self.raw_cross_section(when)
        if raw.empty:
            return raw
        ranked = raw.copy()
        ranked[FACTOR_COLUMNS] = (
            raw[FACTOR_COLUMNS].rank(pct=True, method="average") * 2 - 1
        )
        self._ranked_cross_section_cache[key] = ranked
        return ranked

    def ranked_components(
        self, when: pd.Timestamp
    ) -> tuple[pd.DataFrame, tuple[str, ...], np.ndarray]:
        """Return cached ranked data and its NumPy scoring matrix."""
        key = pd.Timestamp(when)
        frame = self.ranked_cross_section(key)
        cached = self._ranked_components_cache.get(key)
        if cached is None:
            cached = (
                tuple(str(ticker) for ticker in frame.index),
                frame[FACTOR_COLUMNS].to_numpy(dtype=float),
            )
            self._ranked_components_cache[key] = cached
        return frame, cached[0], cached[1]

    def observation(
        self,
        when: pd.Timestamp,
        current_weights: dict[str, float],
        *,
        drawdown: float,
        recent_return: float,
        previous_turnover: float,
    ) -> np.ndarray:
        key = pd.Timestamp(when)
        static = self._observation_static_cache.get(key)
        if static is None:
            raw = self.raw_cross_section(key)
            market_row = self.market_row(key)
            values = [
                float(market_row[name]) for name in self.market_features.columns
            ]

            if raw.empty:
                values.extend([0.0] * (2 * len(FACTOR_COLUMNS) + 4))
            else:
                means = raw[FACTOR_COLUMNS].mean().clip(-2, 2)
                standard_deviations = raw[FACTOR_COLUMNS].std().fillna(0).clip(0, 2)
                values.extend(float(means[name]) for name in FACTOR_COLUMNS)
                values.extend(
                    float(standard_deviations[name]) for name in FACTOR_COLUMNS
                )
                values.extend(
                    [
                        float((raw["factor_return_20d"] > 0).mean()),
                        float((raw["factor_trend_50d"] > 0).mean()),
                        float((raw["factor_trend_200d"] > 0).mean()),
                        min(1.0, len(raw) / MAX_STOCKS_EXAMINED),
                    ]
                )
            static = tuple(values)
            self._observation_static_cache[key] = static

        values = list(static)

        exposure = float(sum(max(0.0, value) for value in current_weights.values()))
        values.extend(
            [
                exposure,
                min(1.0, len(current_weights) / TOP_STOCKS_TO_HOLD),
                float(np.clip(drawdown, -1, 0)),
                float(np.clip(recent_return / 0.30, -2, 2)),
                float(np.clip(previous_turnover, 0, 2)),
            ]
        )
        observation = np.asarray(values, dtype=np.float32)
        if len(observation) != len(self.observation_names):
            raise AssertionError("RL3 observation length changed unexpectedly.")
        return observation


def build_market_bundle(
    start: str,
    end: str,
    *,
    refresh: bool = False,
    allow_incomplete: bool = False,
    price_data_directory: Path | None = None,
    quality_report_path: Path = DATA_QUALITY_PATH,
) -> MarketBundle:
    membership = load_membership_history(refresh=refresh)
    sectors = load_sector_map(refresh=refresh)
    union = membership.union(start, end)
    required = union | {"SPY"} | set(SECTOR_ETFS.values())
    prices, missing = download_prices(
        required,
        start,
        end,
        refresh=refresh,
        external_directory=price_data_directory,
    )
    if "SPY" not in prices:
        raise RuntimeError("SPY prices are required to create the market calendar.")

    spy = prices["SPY"]
    sector_prices = {ticker: prices[ticker] for ticker in SECTOR_ETFS.values() if ticker in prices}
    featured_prices: dict[str, pd.DataFrame] = {}
    for number, ticker in enumerate(sorted(union), start=1):
        data = prices.get(ticker)
        if data is None:
            continue
        sector = sectors.get(ticker)
        sector_etf = SECTOR_ETFS.get(sector or "")
        sector_data = sector_prices.get(sector_etf or "")
        sector_close = None if sector_data is None else sector_data["close"]
        featured_prices[ticker] = add_asset_features(
            data, spy["close"], sector_close
        )
        if number % 100 == 0:
            print(f"Prepared features for {number}/{len(union)} historical tickers...")

    # Market ETFs remain available for benchmarks and market observations.
    featured_prices["SPY"] = add_asset_features(spy, spy["close"], spy["close"])
    market_features = make_market_features(spy, sector_prices)
    dates = spy.loc[start:end].index.sort_values()
    bundle = MarketBundle(
        membership=membership,
        prices=featured_prices,
        sectors=sectors,
        market_features=market_features,
        dates=dates,
        missing_tickers=missing,
    )

    coverage_dates = dates[min(252, max(0, len(dates) - 1)) :]
    sample_dates = coverage_dates[:: max(1, len(coverage_dates) // 24)]
    coverage_rows = []
    for sample_date in sample_dates:
        member_count = len(membership.members_on(sample_date))
        eligible_count = len(bundle.raw_cross_section(sample_date))
        coverage_rows.append(
            {
                "date": sample_date.date().isoformat(),
                "members": member_count,
                "eligible": eligible_count,
                "coverage": eligible_count / max(1, min(MAX_STOCKS_EXAMINED, member_count)),
            }
        )
    minimum_coverage = min((row["coverage"] for row in coverage_rows), default=0.0)
    quality = {
        "generated_at": date.today().isoformat(),
        "start": start,
        "end": end,
        "historical_tickers_requested": len(union),
        "historical_tickers_with_prices": len(featured_prices) - 1,
        "missing_tickers": missing,
        "sampled_coverage": coverage_rows,
        "minimum_sampled_coverage": minimum_coverage,
        "warning": (
            "Point-in-time membership reduces survivorship bias, but the free "
            "membership and Yahoo price histories may still be incomplete."
        ),
    }
    quality_report_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print(
        f"Historical-universe coverage: minimum sampled eligible coverage "
        f"{minimum_coverage:.1%}; missing price histories: {len(missing)}"
    )
    if minimum_coverage < MINIMUM_DATA_COVERAGE and not allow_incomplete:
        raise RuntimeError(
            f"Historical-universe price coverage fell to {minimum_coverage:.1%}. "
            "Too much data is missing for a useful training run."
        )
    return bundle


# ---------------------------------------------------------------------------
# Portfolio construction and realistic next-open execution
# ---------------------------------------------------------------------------

def _capped_weights(
    raw_values: pd.Series, total_weight: float, cap: float
) -> pd.Series:
    if raw_values.empty or total_weight <= 0:
        return pd.Series(0.0, index=raw_values.index)
    raw = raw_values.clip(lower=0).replace([np.inf, -np.inf], np.nan).fillna(0)
    if raw.sum() <= 0:
        raw[:] = 1.0
    weights = raw / raw.sum() * total_weight
    for _ in range(10):
        over = weights > cap
        if not over.any():
            break
        weights.loc[over] = cap
        remaining = total_weight - float(weights.loc[over].sum())
        under = ~over
        if remaining <= 0 or not under.any() or raw.loc[under].sum() <= 0:
            break
        weights.loc[under] = raw.loc[under] / raw.loc[under].sum() * remaining
    return weights.clip(upper=cap)


def construct_target_weights(
    bundle: MarketBundle,
    when: pd.Timestamp,
    current_weights: dict[str, float],
    action: np.ndarray,
) -> tuple[dict[str, float], dict[str, float], float]:
    """Turn PPO factor preferences into a risk-controlled stock portfolio."""
    cross_section, tickers, factor_matrix = bundle.ranked_components(when)
    if cross_section.empty:
        return {}, {}, 0.0

    action = np.asarray(action, dtype=float).reshape(-1)
    expected_size = len(FACTOR_COLUMNS) + 1
    if len(action) != expected_size:
        raise ValueError(f"Expected {expected_size} action values, got {len(action)}.")
    factor_preferences = action[:-1]
    magnitude = float(np.abs(factor_preferences).sum())
    if magnitude < 1e-6:
        factor_preferences = np.zeros(len(FACTOR_COLUMNS), dtype=float)
        factor_preferences[FACTOR_COLUMNS.index("factor_return_60d")] = 1.0
    else:
        factor_preferences = factor_preferences / magnitude

    scores_array = factor_matrix @ factor_preferences
    order = np.argsort(-scores_array, kind="stable")
    ordered_tickers = [tickers[position] for position in order]
    keep_zone = set(ordered_tickers[:KEEP_HOLDING_WITHIN_TOP_RANK])
    kept = [
        ticker
        for ticker in current_weights
        if ticker in keep_zone and current_weights[ticker] > 0.0001
    ][:TOP_STOCKS_TO_HOLD]
    selected = list(kept)
    for ticker in ordered_tickers:
        if len(selected) >= TOP_STOCKS_TO_HOLD:
            break
        if ticker not in selected:
            selected.append(ticker)

    desired_exposure = float(np.clip((action[-1] + 1) / 2, 0, 1))
    market = bundle.market_row(when)
    if float(market["market_trend_200d"]) < 0:
        desired_exposure = min(desired_exposure, 0.50)
    if float(market["market_volatility_20d"]) > 0.60:
        desired_exposure = min(desired_exposure, 0.40)

    selected_frame = cross_section.loc[selected]
    inverse_volatility = 1 / selected_frame["annual_volatility_20d"].clip(lower=0.10)
    weights = _capped_weights(
        inverse_volatility, desired_exposure, MAX_POSITION_WEIGHT
    )

    # A simple correlation-aware volatility estimate.  It is intentionally
    # conservative compared with pretending all stocks are independent.
    weighted_volatility = weights * selected_frame["annual_volatility_20d"]
    average_correlation = 0.30
    estimated_variance = (
        (1 - average_correlation) * float((weighted_volatility**2).sum())
        + average_correlation * float(weighted_volatility.abs().sum()) ** 2
    )
    estimated_volatility = math.sqrt(max(0.0, estimated_variance))
    if estimated_volatility > TARGET_ANNUAL_VOLATILITY:
        weights *= TARGET_ANNUAL_VOLATILITY / estimated_volatility

    # Sector limits leave the excess in cash instead of secretly reallocating
    # it into another concentrated group.
    for sector in selected_frame["sector"].unique():
        if sector == "Unknown":
            continue
        names = selected_frame.index[selected_frame["sector"] == sector]
        sector_total = float(weights.reindex(names).fillna(0).sum())
        if sector_total > MAX_SECTOR_WEIGHT:
            weights.loc[names] *= MAX_SECTOR_WEIGHT / sector_total

    target = {
        ticker: float(weight)
        for ticker, weight in weights.items()
        if weight > 0.0001
    }
    scores = {
        tickers[position]: float(scores_array[position]) for position in order
    }
    return target, scores, desired_exposure


def apply_trade_controls(
    current: dict[str, float], target: dict[str, float]
) -> tuple[dict[str, float], float]:
    names = sorted(set(current) | set(target))
    controlled = {ticker: float(target.get(ticker, 0.0)) for ticker in names}
    for ticker in names:
        before = float(current.get(ticker, 0.0))
        after = float(controlled.get(ticker, 0.0))
        if abs(after - before) < MINIMUM_TRADE_WEIGHT:
            controlled[ticker] = before

    total = sum(max(0.0, value) for value in controlled.values())
    if total > 1:
        controlled = {ticker: value / total for ticker, value in controlled.items()}
    turnover = sum(
        abs(controlled.get(ticker, 0.0) - current.get(ticker, 0.0))
        for ticker in names
    )
    if turnover > MAX_TOTAL_WEIGHT_CHANGE:
        fraction = MAX_TOTAL_WEIGHT_CHANGE / turnover
        controlled = {
            ticker: current.get(ticker, 0.0)
            + fraction * (controlled.get(ticker, 0.0) - current.get(ticker, 0.0))
            for ticker in names
        }
        turnover = MAX_TOTAL_WEIGHT_CHANGE
    controlled = {
        ticker: float(weight)
        for ticker, weight in controlled.items()
        if weight > 0.0001
    }
    return controlled, float(turnover)


def _drift_weights(
    weights: dict[str, float], returns: dict[str, float]
) -> tuple[dict[str, float], float]:
    cash = max(0.0, 1 - sum(weights.values()))
    grown = {
        ticker: weight * (1 + returns.get(ticker, 0.0))
        for ticker, weight in weights.items()
    }
    multiplier = cash + sum(grown.values())
    if multiplier <= 0:
        return {}, 0.0
    return (
        {
            ticker: value / multiplier
            for ticker, value in grown.items()
            if value / multiplier > 0.0001
        },
        float(multiplier),
    )


@dataclass
class TransitionResult:
    final_value: float
    final_weights: dict[str, float]
    turnover: float
    fees: float
    daily_values: list[tuple[pd.Timestamp, float]]
    trades: list[dict[str, Any]]


def execute_next_open_and_hold(
    bundle: MarketBundle,
    signal_date: pd.Timestamp,
    next_signal_date: pd.Timestamp,
    portfolio_value: float,
    current_weights: dict[str, float],
    desired_target: dict[str, float],
    scores: dict[str, float],
    *,
    record_trades: bool = False,
    enforce_trade_controls: bool = True,
) -> TransitionResult:
    """Move old holdings overnight, trade next open, then hold to next signal."""
    period_dates = bundle.dates_after_through(signal_date, next_signal_date)
    if len(period_dates) == 0:
        return TransitionResult(
            portfolio_value, current_weights, 0.0, 0.0, [], []
        )
    execution_date = period_dates[0]

    # Existing positions experience the overnight move before the new order can
    # fill.  A missing next-open price receives a conservative haircut.
    overnight_returns: dict[str, float] = {}
    for ticker in current_weights:
        data = bundle.prices.get(ticker)
        if data is None:
            overnight_returns[ticker] = -MISSING_EXIT_HAIRCUT
            continue
        old_close = bundle.price(ticker, signal_date, "close")
        next_open = bundle.price(ticker, execution_date, "open")
        overnight_returns[ticker] = (
            -MISSING_EXIT_HAIRCUT
            if old_close is None or next_open is None
            else next_open / old_close - 1
        )
    drifted, overnight_multiplier = _drift_weights(
        current_weights, overnight_returns
    )
    portfolio_value *= overnight_multiplier

    target, turnover = (
        apply_trade_controls(drifted, desired_target)
        if enforce_trade_controls
        else (
            {ticker: weight for ticker, weight in desired_target.items() if weight > 0},
            sum(
                abs(desired_target.get(ticker, 0.0) - drifted.get(ticker, 0.0))
                for ticker in set(drifted) | set(desired_target)
            ),
        )
    )

    # A newly selected stock without a real next-open price cannot be bought.
    # Its intended allocation simply remains in cash.
    target = {
        ticker: weight
        for ticker, weight in target.items()
        if ticker in bundle.prices
        and bundle.price(ticker, execution_date, "open") is not None
    }
    turnover = sum(
        abs(target.get(ticker, 0.0) - drifted.get(ticker, 0.0))
        for ticker in set(drifted) | set(target)
    )

    trades: list[dict[str, Any]] = []
    total_fees = 0.0
    cross_section = bundle.raw_cross_section(signal_date)
    for ticker in sorted(set(drifted) | set(target)):
        change = target.get(ticker, 0.0) - drifted.get(ticker, 0.0)
        if abs(change) < 1e-8:
            continue
        dollars = abs(change) * portfolio_value
        dollar_volume = (
            float(cross_section.loc[ticker, "dollar_volume_20d"])
            if ticker in cross_section.index
            else MINIMUM_DOLLAR_VOLUME
        )
        participation = dollars / max(dollar_volume, 1.0)
        impact = min(MAX_MARKET_IMPACT, MARKET_IMPACT_COEFFICIENT * math.sqrt(participation))
        fee = dollars * (BASE_EXECUTION_COST + impact)
        total_fees += fee
        if record_trades:
            trades.append(
                {
                    "signal_date": signal_date.date().isoformat(),
                    "execution_date": execution_date.date().isoformat(),
                    "ticker": ticker,
                    "sector": bundle.sectors.get(ticker, "Unknown"),
                    "action": "BUY" if change > 0 else "SELL",
                    "weight_before": drifted.get(ticker, 0.0),
                    "weight_after": target.get(ticker, 0.0),
                    "weight_change": change,
                    "estimated_dollars_traded": dollars,
                    "estimated_execution_cost": fee,
                    "factor_score": scores.get(ticker, np.nan),
                    "portfolio_before_trade": portfolio_value,
                }
            )
    portfolio_value = max(0.0, portfolio_value - total_fees)

    # The first daily return is execution-open to execution-close.  Later
    # returns are previous-close to current-close.  Removed/missing holdings are
    # moved to cash instead of silently disappearing from the calculation.
    weights = target.copy()
    last_prices: dict[str, float] = {
        ticker: bundle.price(ticker, execution_date, "open") or np.nan
        for ticker in weights
        if ticker in bundle.prices
    }
    daily_values: list[tuple[pd.Timestamp, float]] = []
    for current_date in period_dates:
        members = bundle.members_on(current_date)
        day_returns: dict[str, float] = {}
        forced_out: list[str] = []
        for ticker in list(weights):
            data = bundle.prices.get(ticker)
            current_close = (
                None if data is None else bundle.price(ticker, current_date, "close")
            )
            previous = last_prices.get(ticker)
            if current_close is None or previous is None or not np.isfinite(previous):
                day_returns[ticker] = -MISSING_EXIT_HAIRCUT
                forced_out.append(ticker)
            else:
                day_returns[ticker] = current_close / previous - 1
                last_prices[ticker] = current_close
                if ticker not in members:
                    forced_out.append(ticker)

        weights, multiplier = _drift_weights(weights, day_returns)
        portfolio_value *= multiplier
        forced_weight = sum(weights.get(ticker, 0.0) for ticker in forced_out)
        if forced_weight > 0:
            value_before_forced = portfolio_value
            forced_fee = value_before_forced * forced_weight * BASE_EXECUTION_COST
            portfolio_value = max(0.0, portfolio_value - forced_fee)
            total_fees += forced_fee
            turnover += forced_weight
            if record_trades:
                for ticker in forced_out:
                    weight = weights.get(ticker, 0.0)
                    if weight <= 0:
                        continue
                    trades.append(
                        {
                            "signal_date": signal_date.date().isoformat(),
                            "execution_date": current_date.date().isoformat(),
                            "ticker": ticker,
                            "sector": bundle.sectors.get(ticker, "Unknown"),
                            "action": "SELL_FORCED_EXIT",
                            "weight_before": weight,
                            "weight_after": 0.0,
                            "weight_change": -weight,
                            "estimated_dollars_traded": weight * value_before_forced,
                            "estimated_execution_cost": (
                                weight * value_before_forced * BASE_EXECUTION_COST
                            ),
                            "factor_score": scores.get(ticker, np.nan),
                            "portfolio_before_trade": value_before_forced,
                        }
                    )
        for ticker in forced_out:
            weights.pop(ticker, None)
        daily_values.append((current_date, portfolio_value))

    return TransitionResult(
        final_value=float(portfolio_value),
        final_weights=weights,
        turnover=float(turnover),
        fees=float(total_fees),
        daily_values=daily_values,
        trades=trades,
    )


def decision_dates(
    dates: pd.DatetimeIndex, start: str, end: str
) -> pd.DatetimeIndex:
    selected = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    if len(selected) < HOLDING_PERIOD_DAYS + 2:
        raise RuntimeError(f"Too few market sessions between {start} and {end}.")
    positions = list(range(0, len(selected), HOLDING_PERIOD_DAYS))
    if positions[-1] != len(selected) - 1:
        positions.append(len(selected) - 1)
    return selected[positions]


# ---------------------------------------------------------------------------
# PPO environment: the action controls the entire portfolio
# ---------------------------------------------------------------------------

class PortfolioFactorEnvironment(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        bundle: MarketBundle,
        start: str,
        end: str,
        *,
        episode_decisions: int = EPISODE_DECISIONS,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.signals = decision_dates(bundle.dates, start, end)
        self.episode_decisions = min(episode_decisions, len(self.signals) - 1)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(FACTOR_COLUMNS) + 1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(len(bundle.observation_names),),
            dtype=np.float32,
        )
        self.index = 0
        self.last_index = self.episode_decisions
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.weights: dict[str, float] = {}
        self.recent_return = 0.0
        self.previous_turnover = 0.0

    def _observation(self) -> np.ndarray:
        drawdown = self.portfolio_value / self.peak_value - 1
        return self.bundle.observation(
            self.signals[self.index],
            self.weights,
            drawdown=drawdown,
            recent_return=self.recent_return,
            previous_turnover=self.previous_turnover,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        maximum_start = len(self.signals) - self.episode_decisions - 1
        self.index = int(self.np_random.integers(0, max(1, maximum_start + 1)))
        self.last_index = self.index + self.episode_decisions
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.weights = {}
        self.recent_return = 0.0
        self.previous_turnover = 0.0
        return self._observation(), {"signal_date": str(self.signals[self.index].date())}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        signal_date = self.signals[self.index]
        next_signal_date = self.signals[self.index + 1]
        old_value = self.portfolio_value
        old_drawdown = old_value / self.peak_value - 1
        target, scores, _ = construct_target_weights(
            self.bundle, signal_date, self.weights, action
        )
        transition = execute_next_open_and_hold(
            self.bundle,
            signal_date,
            next_signal_date,
            self.portfolio_value,
            self.weights,
            target,
            scores,
        )
        self.portfolio_value = transition.final_value
        self.weights = transition.final_weights
        self.recent_return = self.portfolio_value / max(old_value, 1e-12) - 1
        self.previous_turnover = transition.turnover
        self.peak_value = max(self.peak_value, self.portfolio_value)
        new_drawdown = self.portfolio_value / self.peak_value - 1

        net_log_return = math.log(max(self.portfolio_value, 1e-12) / max(old_value, 1e-12))
        spy_start = self.bundle.price("SPY", signal_date, "close")
        spy_end = self.bundle.price("SPY", next_signal_date, "close")
        benchmark_log_return = (
            0.0
            if spy_start is None or spy_end is None
            else math.log(spy_end / spy_start)
        )
        downside = max(0.0, -net_log_return)
        drawdown_worsening = max(0.0, old_drawdown - new_drawdown)
        reward = (
            net_log_return
            + 0.15 * (net_log_return - benchmark_log_return)
            - 0.05 * transition.turnover
            - 0.50 * downside**2
            - 0.25 * drawdown_worsening
        )

        self.index += 1
        terminated = self.portfolio_value <= 0
        truncated = self.index >= self.last_index or self.index >= len(self.signals) - 1
        observation = self._observation()
        info = {
            "signal_date": str(signal_date.date()),
            "portfolio_value": self.portfolio_value,
            "period_return": self.recent_return,
            "turnover": transition.turnover,
            "fees": transition.fees,
            "holdings": len(self.weights),
            "drawdown": new_drawdown,
        }
        return observation, float(reward), terminated, truncated, info


class TrainingProgressCallback(BaseCallback):
    """Print training progress at each 10% milestone."""

    def __init__(self, total_steps: int, label: str):
        super().__init__(verbose=0)
        self.total_steps = max(1, int(total_steps))
        self.label = label
        self.report_every = max(1, self.total_steps // 10)
        self.next_report = self.report_every

    def _on_step(self) -> bool:
        completed = int(self.model.num_timesteps)
        if completed >= self.next_report:
            shown = min(completed, self.total_steps)
            percentage = min(100, round(100 * shown / self.total_steps))
            print(
                f"{self.label}: {shown:,} / {self.total_steps:,} steps "
                f"({percentage}%)",
                flush=True,
            )
            while self.next_report <= completed:
                self.next_report += self.report_every
        return True


def make_model(environment: gym.Env, seed: int) -> PPO:
    set_random_seed(seed)
    return PPO(
        "MlpPolicy",
        environment,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.20,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
        seed=seed,
        device="cpu",
        verbose=0,
    )


# ---------------------------------------------------------------------------
# Shared backtester used by walk-forward validation and Simulation 3
# ---------------------------------------------------------------------------

def performance_metrics(
    daily_values: pd.Series, starting_value: float, turnover: float, fees: float
) -> dict[str, float]:
    values = daily_values.dropna().astype(float)
    if len(values) < 2:
        raise RuntimeError("At least two portfolio values are required.")
    daily_returns = values.pct_change(fill_method=None).dropna()
    total_return = float(values.iloc[-1] / starting_value - 1)
    years = max((values.index[-1] - values.index[0]).days / 365.25, 1 / 252)
    cagr = float((values.iloc[-1] / starting_value) ** (1 / years) - 1)
    volatility = float(daily_returns.std() * math.sqrt(252))
    sharpe = (
        float(daily_returns.mean() / daily_returns.std() * math.sqrt(252))
        if daily_returns.std() > 1e-12
        else 0.0
    )
    downside = daily_returns[daily_returns < 0].std()
    sortino = (
        float(daily_returns.mean() / downside * math.sqrt(252))
        if pd.notna(downside) and downside > 1e-12
        else 0.0
    )
    running_high = values.cummax()
    max_drawdown = float((values / running_high - 1).min())
    calmar = cagr / abs(max_drawdown) if max_drawdown < -1e-12 else 0.0
    return {
        "starting_balance": float(starting_value),
        "final_balance": float(values.iloc[-1]),
        "return": total_return,
        "cagr": cagr,
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": float(calmar),
        "turnover": float(turnover),
        "estimated_execution_costs": float(fees),
    }


def simulate_strategy(
    bundle: MarketBundle,
    start: str,
    end: str,
    *,
    starting_cash: float,
    action_provider: Callable[[np.ndarray, pd.Timestamp], np.ndarray] | None = None,
    equal_weight_all: bool = False,
    record_trades: bool = False,
) -> dict[str, Any]:
    signals = decision_dates(bundle.dates, start, end)
    value = float(starting_cash)
    peak = value
    weights: dict[str, float] = {}
    recent_return = 0.0
    previous_turnover = 0.0
    total_turnover = 0.0
    total_fees = 0.0
    all_trades: list[dict[str, Any]] = []
    value_rows: list[tuple[pd.Timestamp, float]] = [(signals[0], value)]
    holding_counts: list[int] = []

    for index in range(len(signals) - 1):
        signal_date = signals[index]
        next_signal_date = signals[index + 1]
        old_value = value
        if equal_weight_all:
            eligible = bundle.raw_cross_section(signal_date)
            target = (
                {ticker: 1 / len(eligible) for ticker in eligible.index}
                if len(eligible)
                else {}
            )
            scores = {ticker: 0.0 for ticker in target}
        else:
            if action_provider is None:
                raise ValueError("action_provider is required for an AI/factor strategy.")
            observation = bundle.observation(
                signal_date,
                weights,
                drawdown=value / peak - 1,
                recent_return=recent_return,
                previous_turnover=previous_turnover,
            )
            action = action_provider(observation, signal_date)
            target, scores, _ = construct_target_weights(
                bundle, signal_date, weights, action
            )

        transition = execute_next_open_and_hold(
            bundle,
            signal_date,
            next_signal_date,
            value,
            weights,
            target,
            scores,
            record_trades=record_trades,
            enforce_trade_controls=not equal_weight_all,
        )
        value = transition.final_value
        weights = transition.final_weights
        recent_return = value / max(old_value, 1e-12) - 1
        previous_turnover = transition.turnover
        peak = max(peak, value)
        total_turnover += transition.turnover
        total_fees += transition.fees
        all_trades.extend(transition.trades)
        value_rows.extend(transition.daily_values)
        holding_counts.append(len(weights))

    equity = (
        pd.DataFrame(value_rows, columns=["date", "portfolio_value"])
        .drop_duplicates("date", keep="last")
        .set_index("date")["portfolio_value"]
        .sort_index()
    )
    metrics = performance_metrics(equity, starting_cash, total_turnover, total_fees)
    metrics.update(
        {
            "first_date": equity.index[0],
            "last_date": equity.index[-1],
            "trading_days": len(equity),
            "average_holdings": float(np.mean(holding_counts)) if holding_counts else 0.0,
            "trade_log": all_trades,
            "equity_curve": equity,
            "final_weights": weights,
        }
    )
    return metrics


def simulate_model(
    model: PPO,
    bundle: MarketBundle,
    start: str,
    end: str,
    *,
    starting_cash: float = 100_000,
    record_trades: bool = False,
) -> dict[str, Any]:
    def provider(observation: np.ndarray, _: pd.Timestamp) -> np.ndarray:
        action, _state = model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    return simulate_strategy(
        bundle,
        start,
        end,
        starting_cash=starting_cash,
        action_provider=provider,
        record_trades=record_trades,
    )


def simulate_fixed_factor(
    bundle: MarketBundle,
    start: str,
    end: str,
    factor_name: str,
    *,
    starting_cash: float = 100_000,
) -> dict[str, Any]:
    action = np.zeros(len(FACTOR_COLUMNS) + 1, dtype=np.float32)
    action[FACTOR_COLUMNS.index(factor_name)] = 1.0
    action[-1] = 1.0
    return simulate_strategy(
        bundle,
        start,
        end,
        starting_cash=starting_cash,
        action_provider=lambda _observation, _date: action,
    )


def buy_and_hold_spy(
    bundle: MarketBundle, start: str, end: str, starting_cash: float
) -> dict[str, Any]:
    spy = bundle.prices["SPY"].loc[start:end]
    if len(spy) < 2:
        raise RuntimeError("SPY benchmark has too little data.")
    first = float(spy.iloc[0]["close"])
    curve = starting_cash * (1 - BASE_EXECUTION_COST) * spy["close"] / first
    return performance_metrics(
        curve.rename("portfolio_value"),
        starting_cash,
        turnover=1.0,
        fees=starting_cash * BASE_EXECUTION_COST,
    ) | {"equity_curve": curve}


def walk_forward_score(result: dict[str, Any]) -> float:
    return float(
        result["return"]
        + 0.25 * result["sharpe"]
        + 0.50 * result["max_drawdown"]
        - 0.02 * result["turnover"]
    )


def experiment_fingerprint(settings: dict[str, Any]) -> str:
    encoded = json.dumps(settings, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--allow-incomplete-data", action="store_true")
    parser.add_argument(
        "--price-data-dir",
        type=Path,
        default=(Path(os.environ["RL3_PRICE_DATA_DIR"]) if os.getenv("RL3_PRICE_DATA_DIR") else None),
        help="Optional licensed CSV directory; one DATE/OHLCV file per ticker.",
    )
    parser.add_argument("--walk-forward-steps", type=int, default=WALK_FORWARD_STEPS)
    parser.add_argument("--final-steps", type=int, default=FINAL_TRAINING_STEPS)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast plumbing test: fewer folds/steps. Do not use its model live.",
    )
    args = parser.parse_args()

    folds = WALK_FORWARD_YEARS
    walk_steps = args.walk_forward_steps
    final_steps = args.final_steps
    if args.quick:
        folds = WALK_FORWARD_YEARS[-2:]
        walk_steps = min(walk_steps, 5_000)
        final_steps = min(final_steps, 20_000)

    print("=" * 84)
    print("RL CODEX 3 - HISTORICAL S&P 500 PORTFOLIO PPO")
    print("=" * 84)
    print("Training cutoff: 2025-07-10. Later data is forbidden here.")
    print(f"Walk-forward years: {folds}")
    print(f"Factor action size: {len(FACTOR_COLUMNS)} factors + cash exposure")
    print(f"Maximum stocks examined: {MAX_STOCKS_EXAMINED}")
    print(f"Maximum holdings: {TOP_STOCKS_TO_HOLD}")
    print("Signals use the close; all simulated orders execute next open.")

    bundle = build_market_bundle(
        DOWNLOAD_START,
        DOWNLOAD_END,
        refresh=args.refresh_data,
        allow_incomplete=args.allow_incomplete_data,
        price_data_directory=args.price_data_dir,
    )
    if (bundle.dates > pd.Timestamp(FINAL_TRAIN_END)).any():
        raise AssertionError("Post-training-cutoff data leaked into RL Codex 3 training.")

    rows: list[dict[str, Any]] = []
    for seed in CANDIDATE_SEEDS:
        for validation_year in folds:
            training_end = f"{validation_year - 1}-12-31"
            print(
                f"\nSeed {seed}: train {TRAIN_START} to {training_end}; "
                f"test untouched {validation_year} ({walk_steps:,} steps)"
            )
            environment = PortfolioFactorEnvironment(
                bundle, TRAIN_START, training_end
            )
            model = make_model(environment, seed)
            model.learn(
                total_timesteps=walk_steps,
                callback=TrainingProgressCallback(
                    walk_steps, f"Seed {seed}, validation {validation_year}"
                ),
            )
            environment.close()
            result = simulate_model(
                model,
                bundle,
                f"{validation_year}-01-01",
                f"{validation_year}-12-31",
                starting_cash=100_000,
            )
            score = walk_forward_score(result)
            row = {
                "seed": seed,
                "validation_year": validation_year,
                "return": result["return"],
                "sharpe": result["sharpe"],
                "max_drawdown": result["max_drawdown"],
                "turnover": result["turnover"],
                "score": score,
            }
            rows.append(row)
            print(
                f"Return {result['return']:+.2%} | Sharpe {result['sharpe']:.2f} | "
                f"drawdown {result['max_drawdown']:.2%} | score {score:.3f}"
            )

    walk_forward = pd.DataFrame(rows)
    WALK_FORWARD_PATH.write_text(walk_forward.to_csv(index=False), encoding="utf-8")
    seed_scores = walk_forward.groupby("seed")["score"].median()
    winning_seed = int(seed_scores.idxmax())
    print("\nMedian walk-forward scores:")
    for seed, score in seed_scores.items():
        print(f"  seed {seed}: {score:.3f}")
    print(f"Winning seed: {winning_seed}")

    print(
        f"\nFinal training on {TRAIN_START} through {FINAL_TRAIN_END} "
        f"for {final_steps:,} steps..."
    )
    final_environment = PortfolioFactorEnvironment(
        bundle, TRAIN_START, FINAL_TRAIN_END
    )
    final_model = make_model(final_environment, winning_seed)
    final_model.learn(
        total_timesteps=final_steps,
        callback=TrainingProgressCallback(
            final_steps, f"Final seed {winning_seed}"
        ),
    )
    final_environment.close()
    final_model.save(MODEL_PATH)

    settings = {
        "version": 3,
        "training_start": TRAIN_START,
        "training_end": FINAL_TRAIN_END,
        "walk_forward_years": list(folds),
        "candidate_seeds": list(CANDIDATE_SEEDS),
        "winning_seed": winning_seed,
        "walk_forward_steps": walk_steps,
        "final_training_steps": final_steps,
        "quick_model": bool(args.quick),
        "factor_columns": FACTOR_COLUMNS,
        "maximum_stocks_examined": MAX_STOCKS_EXAMINED,
        "maximum_holdings": TOP_STOCKS_TO_HOLD,
        "holding_period_days": HOLDING_PERIOD_DAYS,
        "membership_source": MEMBERSHIP_URL,
        "universe_method": "point-in-time historical S&P 500 membership",
        "price_source": "RL3_PRICE_DATA_DIR plus yfinance fallback",
        "membership_file_sha256": file_sha256(
            CACHE_DIRECTORY / "sp500_membership_history.csv"
        ),
        "sector_file_sha256": file_sha256(
            CACHE_DIRECTORY / "sp500_current_sectors.csv"
        ),
        "model_file_sha256": file_sha256(Path(f"{MODEL_PATH}.zip")),
    }
    settings["fingerprint"] = experiment_fingerprint(settings)
    METADATA_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    print("\n" + "=" * 84)
    print(f"Saved model: {MODEL_PATH}.zip")
    print(f"Saved walk-forward results: {WALK_FORWARD_PATH}")
    print(f"Saved frozen metadata: {METADATA_PATH}")
    print(f"Experiment fingerprint: {settings['fingerprint']}")
    if args.quick:
        print("WARNING: --quick was used. This model is only a plumbing test.")
    print("Run CODEX_SIMULATION_3.py next. Do not tune RL3 using its holdout result.")
    print("=" * 84)


if __name__ == "__main__":
    main()
