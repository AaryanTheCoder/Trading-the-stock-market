# 26th July experiment

This directory is self-contained and organized by purpose: source, training,
simulations, model artifact, tests, trading data, and caches.

## Outcome

The frozen model beat both saved RL Codex 2 policies from January 2, 2025
through July 24, 2026:

| Portfolio | Execution | Return | Final $100k | Max drawdown |
|---|---:|---:|---:|---:|
| July26 model | RL2-comparable close | **+69.30%** | **$169,302.77** | -22.30% |
| July26 model | next market open | **+65.76%** | **$165,763.69** | -22.18% |
| RL Codex 2 original | RL2 close convention | +45.95% | $145,949.40 | -19.20% |
| 100-stock equal weight | buy and hold | +43.56% | $143,558.73 | n/a |
| RL Codex 2 long-term | RL2 close convention | +28.79% | $128,794.10 | -25.60% |

The directly comparable improvement over the stronger saved RL2 policy is
23.35 percentage points. These are development-backtest results, not evidence
of future return or a recommendation to trade.

## Model

The winning model is intentionally simpler than another PPO:

- Rank the same 100 stocks used by RL Codex 2.
- Main signal: return from 252 sessions ago to 20 sessions ago ("12-1"
  momentum), avoiding the most recent month.
- Secondary signal: proximity to the trailing 252-session high.
- Score = rank(12-1 momentum) + 0.25 * rank(proximity to high).
- Hold 10 equal-weight stocks, reconsider every 40 common trading sessions,
  and keep an existing holding while it remains in the top 20.
- Charge 0.1% on every dollar bought or sold.

The feature blend, holding period, and concentration were selected using only
2021-2024 validation results. The frozen model file explicitly records a
2024-12-31 selection cutoff. No 2025+ return was used to choose its parameters.

## Reproduce

Run from this model directory with the repository virtual environment:

```bash
# Optional network refresh; writes only to data/cache/latest_prices
../../.venv/bin/python tools/refresh_prices.py

# Re-run the pre-2025 selection, freeze its winner, and evaluate 2025+
../../.venv/bin/python training/sweep_factors.py
../../.venv/bin/python training/freeze_model.py
../../.venv/bin/python simulations/evaluate_frozen.py

# Reproduce both saved RL2 policies on the identical price panel
../../.venv/bin/python simulations/benchmark_rl2.py

# Accounting and leakage-cutoff checks
../../.venv/bin/python tests/test_research_engine.py -v
```

The factor sweep takes roughly two minutes in this environment. If the latest
overlay is absent, evaluation falls back to the repository's read-only RL3
price cache.

## Key artifacts

- `REPORT.md`: experiment design, comparison, robustness, and limitations.
- `source/research_engine.py`: feature, ranking, and accounting engine.
- `model/july26_model.json`: frozen parameters and fingerprint.
- `data/training/factor_sweep_validation.csv`: all 2,940 pre-2025 candidates.
- `data/simulation/july26_summary.json`: final metrics and holdings.
- `data/simulation/rl2_benchmark.json`: identical-panel saved-RL2 results.
- `data/simulation/robustness_checks.csv`: cost, schedule, and concentration tests.
- `data/simulation/july26_equity.csv`: candidate equity curves.
- `data/simulation/trades/`: candidate and reproduced-RL2 trade audit trails.
- `data/cache/`: refreshed 100-stock overlay and its manifest.

## Important limitations

- The fixed list of today's large companies creates survivorship bias.
- The 2025-2026 period is now a development backtest, not an untouched test.
- The close-execution result is included only for direct RL2 comparability;
  next-open execution is more credible but still omits bid/ask spread and
  market impact beyond the explicit fee stress.
- Forty schedule offsets were checked after the frozen test. The next-open
  median was +66.35%, but the worst offset returned +24.44%; timing matters.
- The result is concentrated in 10 stocks and has a roughly 22% drawdown.
- Historical performance can fail abruptly and should not be treated as a
  real-money trading recommendation.
