"""Select and freeze the 200-stock price model using pre-2025 data only."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(MODEL_ROOT / "source"))

from price_engine import (
    FEATURE_NAMES,
    MODEL_ARTIFACT_DIR,
    TRAINING_DATA_DIR,
    cross_sectional_ranks,
    load_market_data,
    score_from_weights,
    simulate_scores,
)


VALIDATION_YEARS = (2021, 2022, 2023, 2024)
HOLDING_DAYS = (10, 20, 40)
TOP_K_VALUES = (10, 15, 20)


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
    for combo in itertools.product((0.0, 0.5, 1.0), repeat=len(horizons)):
        if not any(combo):
            continue
        for risk in (0.0, -0.25, -0.5):
            mapping = dict(zip(horizons, combo))
            mapping["vol_60"] = risk
            mapping["drawdown_252"] = 0.25
            label = "_".join(f"{key}={value:g}" for key, value in mapping.items())
            candidates.append((label, named_weights(**mapping)))

    unique: dict[tuple[float, ...], tuple[str, np.ndarray]] = {}
    for label, weights in candidates:
        scale = float(np.max(np.abs(weights)))
        normalized = weights / scale if scale else weights
        unique.setdefault(tuple(np.round(normalized, 8)), (label, weights))
    return list(unique.values())


def main() -> None:
    universe = pd.read_csv(TRAINING_DATA_DIR / "universe_200.csv")
    tickers = universe["ticker"].astype(str).tolist()
    if len(tickers) != 200:
        raise AssertionError(f"Expected 200 tickers, found {len(tickers)}.")
    data = load_market_data(tickers)
    ranked = cross_sectional_ranks(data.features)
    print(
        f"Loaded {len(data.tickers)} stocks, {data.dates[0].date()} to "
        f"{data.dates[-1].date()} ({len(data.dates)} union-calendar sessions)."
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
    sweep_path = TRAINING_DATA_DIR / "price_factor_sweep_validation.csv"
    results.to_csv(sweep_path, index=False)
    winner = results.iloc[0]
    model = {
        "name": "Gemini experiment 200-stock price foundation",
        "universe_file": "../data/training/universe_200.csv",
        "universe_size": 200,
        "selection_data_start": "2021-01-01",
        "selection_data_end": "2024-12-31",
        "test_data_start": "2025-01-01",
        "weights": json.loads(winner["weights"]),
        "holding_days": int(winner["holding_days"]),
        "top_k": int(winner["top_k"]),
        "keep_rank": int(winner["top_k"]) * 2,
        "fee": 0.001,
        "validation": {
            key: float(winner[key])
            for key in (
                "return_2021",
                "return_2022",
                "return_2023",
                "return_2024",
                "compounded_return",
                "mean_sharpe",
                "worst_drawdown",
                "objective",
            )
        },
        "survivorship_bias_warning": (
            "Universe selected from constituents known in 2026."
        ),
    }
    canonical = json.dumps(model, sort_keys=True, separators=(",", ":"))
    model["fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_ARTIFACT_DIR / "price_model_200.json"
    model_path.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(results):,} validation rows to {sweep_path}")
    print("\nTop 20:")
    print(results.head(20).to_string(index=False))
    print(f"\nFrozen price model: {model_path}")
    print(json.dumps(model, indent=2))


if __name__ == "__main__":
    main()

