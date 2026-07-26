"""Utilities for turning audited Gemini caches into portfolio scores."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def load_news_scores(
    cache_root: Path,
    as_of: date,
    tickers: Iterable[str],
    expected_tickers: Iterable[str] | None = None,
) -> tuple[np.ndarray, dict]:
    """Load one date, require candidate coverage, and return universe vector."""
    ticker_list = list(tickers)
    expected = set(ticker_list if expected_tickers is None else expected_tickers)
    directory = cache_root / as_of.isoformat()
    paths = sorted(directory.glob("batch_*.json"))
    if not paths:
        raise FileNotFoundError(
            f"No Gemini caches for {as_of}; run tools/collect_gemini_scores.py."
        )
    by_ticker: dict[str, dict] = {}
    prompt_versions: set[str] = set()
    models: set[str] = set()
    token_count = 0
    audit_pass_count = 0
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("as_of") != as_of.isoformat():
            raise ValueError(f"Cache date mismatch in {path}.")
        prompt_versions.add(str(document.get("prompt_version", "")))
        models.add(str(document.get("model", "")))
        usage = document.get("metadata", {}).get("usage_metadata", {})
        token_count += int(usage.get("totalTokenCount", 0) or 0)
        for row in document.get("records", []):
            ticker = str(row.get("ticker", "")).upper()
            if ticker in by_ticker:
                raise ValueError(f"Duplicate Gemini score for {ticker} on {as_of}.")
            by_ticker[ticker] = row
            audit_pass_count += int(bool(row.get("audit_passed")))
    missing = sorted(expected - set(by_ticker))
    unexpected = sorted(set(by_ticker) - expected)
    if missing or unexpected:
        raise ValueError(
            f"Gemini cache coverage mismatch for {as_of}: "
            f"{len(missing)} missing, {len(unexpected)} unexpected. "
            f"Missing sample={missing[:5]}, unexpected sample={unexpected[:5]}"
        )
    values = np.asarray(
        [
            float(by_ticker[ticker]["effective_news_score"])
            if ticker in by_ticker
            else 0.0
            for ticker in ticker_list
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite Gemini score for {as_of}.")
    return values, {
        "as_of": as_of.isoformat(),
        "stocks": len(by_ticker),
        "universe_size": len(ticker_list),
        "audit_pass_count": audit_pass_count,
        "audit_pass_rate": audit_pass_count / len(by_ticker),
        "total_tokens": token_count,
        "prompt_versions": sorted(prompt_versions),
        "models": sorted(models),
        "batch_files": len(paths),
    }


def tie_aware_rank(values: np.ndarray) -> np.ndarray:
    """Map a one-dimensional vector to [-1, 1] with tied values tied."""
    result = np.full(len(values), np.nan, dtype=np.float64)
    valid = np.isfinite(values)
    count = int(valid.sum())
    if count < 2:
        return result
    valid_values = values[valid]
    order = np.argsort(valid_values, kind="mergesort")
    sorted_values = valid_values[order]
    positions = np.empty(count, dtype=np.float64)
    start = 0
    while start < count:
        end = start + 1
        while end < count and sorted_values[end] == sorted_values[start]:
            end += 1
        positions[order[start:end]] = (start + end - 1) / 2.0
        start = end
    result[valid] = 2.0 * positions / (count - 1) - 1.0
    return result


def combined_scores(
    price_scores: np.ndarray,
    data_dates,
    rebalance_positions: np.ndarray,
    cache_root: Path,
    tickers: Iterable[str],
    news_alpha: float,
    candidate_count: int = 20,
) -> tuple[np.ndarray, list[dict]]:
    """Add ``news_alpha * news_rank`` at each rebalance date."""
    ticker_list = list(tickers)
    result = price_scores.copy()
    audit_rows = []
    for position in rebalance_positions:
        as_of = data_dates[int(position)].date()
        price_row = price_scores[int(position)]
        ranking = np.argsort(np.nan_to_num(price_row, nan=-np.inf))[::-1]
        candidate_indices = [
            int(number)
            for number in ranking
            if np.isfinite(price_row[number])
        ][:candidate_count]
        candidate_tickers = [ticker_list[number] for number in candidate_indices]
        news_values, audit = load_news_scores(
            cache_root, as_of, ticker_list, candidate_tickers
        )
        candidate_news_rank = tie_aware_rank(news_values[candidate_indices])
        combined = np.full(len(ticker_list), np.nan, dtype=np.float64)
        combined[candidate_indices] = (
            price_row[candidate_indices] + news_alpha * candidate_news_rank
        )
        result[int(position)] = combined
        audit_rows.append(audit)
    return result, audit_rows


def news_tilt_targets(
    price_scores: np.ndarray,
    data_dates,
    rebalance_positions: np.ndarray,
    cache_root: Path,
    tickers: Iterable[str],
    news_alpha: float,
    candidate_count: int = 10,
) -> tuple[np.ndarray, list[dict]]:
    """Select top price candidates, then tilt their weights with news ranks."""
    ticker_list = list(tickers)
    targets = np.zeros_like(price_scores, dtype=np.float64)
    audit_rows = []
    for position in rebalance_positions:
        price_row = price_scores[int(position)]
        ranking = np.argsort(np.nan_to_num(price_row, nan=-np.inf))[::-1]
        candidates = [
            int(number)
            for number in ranking
            if np.isfinite(price_row[number])
        ][:candidate_count]
        if len(candidates) != candidate_count:
            raise ValueError(f"Only {len(candidates)} candidates at position {position}.")
        candidate_tickers = [ticker_list[number] for number in candidates]
        news, audit = load_news_scores(
            cache_root,
            data_dates[int(position)].date(),
            ticker_list,
            candidate_tickers,
        )
        news_rank = tie_aware_rank(news[candidates])
        logits = news_alpha * news_rank
        logits -= float(np.max(logits))
        allocation = np.exp(logits)
        allocation /= allocation.sum()
        targets[int(position), candidates] = allocation
        audit_rows.append(audit)
    return targets, audit_rows
