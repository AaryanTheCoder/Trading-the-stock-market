"""Select a compact factor model on pre-2025 validation periods.

This is a research sweep, not the final frozen-test runner. It writes only
inside this directory and never fits or selects anything on 2025+ returns.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_engine import (
    FEATURE_NAMES,
    WORK_DIR,
    cross_sectional_ranks,
    load_market_data,
    score_from_weights,
    simulate_scores,
)


VALIDATION_YEARS = (2021, 2022, 2023, 2024)
HOLDING_DAYS = (10, 20, 40)
TOP_K_VALUES = (5, 8, 10, 15)


def named_weights(**values: float) -> np.ndarray:
    weights = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    for name, value in values.items():
        weights[FEATURE_NAMES.index(name)] = value
    return weights


def candidate_factors() -> list[tuple[str, np.ndarray]]:
    candidates: list[tuple[str, np.ndarray]] = []
    for name in ("ret_20", "ret_60", "ret_120", "ret_252", "skip_252_20"):
        candidates.append((name, named_weights(**{name: 1.0})))

    horizons = ("ret_20", "ret_60", "ret_120", "skip_252_20")
    values = (0.0, 0.5, 1.0)
    risk_values = (0.0, -0.25, -0.5)
    for combo in itertools.product(values, repeat=len(horizons)):
        if not any(combo):
            continue
        for risk in risk_values:
            mapping = dict(zip(horizons, combo))
            mapping["vol_60"] = risk
            mapping["drawdown_252"] = 0.25
            label = "_".join(f"{key}={value:g}" for key, value in mapping.items())
            candidates.append((label, named_weights(**mapping)))

    # De-duplicate scale-equivalent vectors.
    unique: dict[tuple[float, ...], tuple[str, np.ndarray]] = {}
    for label, weights in candidates:
        scale = float(np.max(np.abs(weights)))
        normalized = weights / scale if scale else weights
        key = tuple(np.round(normalized, 8))
        unique.setdefault(key, (label, weights))
    return list(unique.values())


def main() -> None:
    data = load_market_data()
    ranked = cross_sectional_ranks(data.features)
    print(
        f"Loaded {len(data.tickers)} stocks, {data.dates[0].date()} to "
        f"{data.dates[-1].date()} ({len(data.dates)} common sessions)."
    )
    rows: list[dict] = []
    for label, weights in candidate_factors():
        scores = score_from_weights(ranked, weights)
        for holding_days in HOLDING_DAYS:
            for top_k in TOP_K_VALUES:
                yearly = []
                drawdowns = []
                sharpes = []
                turnovers = []
                for year in VALIDATION_YEARS:
                    result = simulate_scores(
                        data,
                        scores,
                        start=f"{year}-01-01",
                        end=f"{year}-12-31",
                        holding_days=holding_days,
                        top_k=top_k,
                        keep_rank=top_k * 2,
                    )
                    yearly.append(result["return"])
                    drawdowns.append(result["max_drawdown"])
                    sharpes.append(result["sharpe"])
                    turnovers.append(result["turnover"])
                compounded = float(np.prod(np.asarray(yearly) + 1.0) - 1.0)
                # Prefer repeatability: all four years matter, with only a
                # modest drawdown penalty and no access to the final test.
                objective = (
                    compounded
                    + 0.50 * float(np.min(yearly))
                    + 0.10 * float(np.mean(sharpes))
                    + 0.25 * float(np.mean(drawdowns))
                )
                rows.append(
                    {
                        "label": label,
                        "weights": json.dumps(
                            {
                                name: float(value)
                                for name, value in zip(FEATURE_NAMES, weights)
                                if value
                            },
                            sort_keys=True,
                        ),
                        "holding_days": holding_days,
                        "top_k": top_k,
                        "return_2021": yearly[0],
                        "return_2022": yearly[1],
                        "return_2023": yearly[2],
                        "return_2024": yearly[3],
                        "compounded_return": compounded,
                        "mean_sharpe": float(np.mean(sharpes)),
                        "worst_drawdown": float(np.min(drawdowns)),
                        "mean_turnover": float(np.mean(turnovers)),
                        "objective": objective,
                    }
                )

    results = pd.DataFrame(rows).sort_values("objective", ascending=False)
    output = WORK_DIR / "factor_sweep_validation.csv"
    results.to_csv(output, index=False)
    print(f"Wrote {len(results):,} pre-2025 validation results to {output.name}")
    print(results.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

