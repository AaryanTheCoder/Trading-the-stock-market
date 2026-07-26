# July26 model research report

## Result

The experiment succeeded on the requested period. On the exact same 100-stock
panel and close-execution convention as RL Codex 2, the July26 model returned
69.30% from January 2, 2025 through July 24, 2026. The stronger saved RL2 policy
(the original policy) returned 45.95%. Starting from $100,000, the difference
was $23,353.37.

The model also returned 65.76% when signals formed at the close were delayed
until the next market open. That version ended at $165,763.69 with a 22.18%
maximum drawdown and a 1.33 daily-return Sharpe ratio.

## Fair comparison

| Portfolio | Return | Final balance | Max drawdown | Turnover |
|---|---:|---:|---:|---:|
| July26, close execution | **69.30%** | **$169,302.77** | -22.30% | 6.45x |
| July26, next-open execution | **65.76%** | **$165,763.69** | -22.18% | 6.43x |
| RL2 original | 45.95% | $145,949.40 | -19.20% | 2.72x |
| Equal-weight 100-stock buy/hold | 43.56% | $143,558.73 | n/a | 1.00x entry |
| RL2 long-term | 28.79% | $128,794.10 | -25.60% | 21.33x |

All rows use the same adjusted local price panel, 0.1% trading fee convention,
January 2, 2025 start, and July 24, 2026 end. The exact RL2 models were loaded
from their existing saved ZIP files; they were not retrained or altered.

## Why this model

RL2's strongest useful structural idea is portfolio construction rather than
PPO itself: score a moderate universe, hold a small equal-weight portfolio,
rebalance slowly, and use a keep zone to reduce churn. The new model retains
that structure while replacing PPO's saturated action probabilities with two
transparent cross-sectional ranks:

1. 252-to-20-session return (classic momentum excluding the newest month).
2. Distance from the trailing 252-session high, weighted at 0.25.

Every signal is calculated from values known by that session's close. The
portfolio holds the highest-scoring 10 names, checks every 40 common sessions,
and retains an owned name while it remains among the top 20.

This is a trained model in the model-selection sense: 2,940 factor/holding/
concentration candidates were evaluated only across calendar years 2021-2024.
The objective rewarded compounded return, the worst calendar year, Sharpe,
and drawdown. The selected rule was then serialized before the 2025+ run.

## What changed from RL Codex 2

The strongest improvement was not a larger neural network. It was better
alignment between the signal and the final portfolio decision.

RL2 trains PPO on one randomly selected stock per episode. PPO learns whether
that individual stock should be short, cash, or long over the next 20 sessions.
The portfolio simulator then repurposes the policy's long-action probability
to rank 100 stocks. Those are related tasks, but they are not identical: a
classifier can strongly prefer "long" for many stocks without learning useful
relative ordering between them. In the saved RL2 trade logs, many long
probabilities are extremely close to 1.0, so tiny numerical differences can
decide which names enter the top 10.

The July26 model directly solves the portfolio's actual problem:

1. It ranks every stock against the other 99 stocks on the same date.
2. Its main feature is 12-to-1-month momentum, a slow signal suited to holding
   periods measured in months rather than next-day direction.
3. It omits the newest 20 sessions from the main return signal. That reduces
   sensitivity to short-lived jumps and one-month reversals.
4. Proximity to the trailing yearly high confirms that momentum remains
   structurally intact instead of relying on one old price jump.
5. It checks every 40 sessions instead of every 20. That let persistent trends
   compound and cut directly comparable turnover from RL2 long-term's 21.33x
   to 6.45x, although RL2 original itself had lower turnover at 2.72x.
6. The top-20 keep zone avoids replacing a holding because of a small rank
   change, while the actual portfolio remains concentrated in 10 equal-weight
   names.
7. Factor weights, concentration, and cadence were selected across four
   pre-2025 calendar regimes instead of relying on one PPO seed or one year's
   return alone.

The requested period happened to contain strong, persistent cross-sectional
trends. The final portfolio included semiconductor and cyclical names such as
AMAT, LRCX, MU, CAT, and GM, which is consistent with the momentum rule doing
what it was designed to do. This period-specific fit is also the main risk:
across alternate rebalance schedules, next-open return ranged from +24.44% to
+96.25%. The headline result therefore reflects both a better-aligned model
and favorable realized trend/timing conditions.

## Pre-2025 validation

The selected configuration produced:

| Year | Return |
|---|---:|
| 2021 | +26.92% |
| 2022 | -0.19% |
| 2023 | +28.38% |
| 2024 | +62.01% |

Compounded across the four separately evaluated years, that is +163.49%.
The strong 2024 result and use of a current-company universe make
survivorship/selection bias material concerns.

## Robustness

Higher one-way cost, using next-open execution:

| Cost on value traded | Return |
|---|---:|
| 0.10% | +65.76% |
| 0.20% | +64.70% |
| 0.50% | +61.55% |

Concentration check, next-open and 0.1% cost:

| Stocks held | Return | Max drawdown |
|---:|---:|---:|
| 5 | +112.97% | -26.21% |
| 8 | +73.38% | -24.05% |
| 10 (frozen) | +65.76% | -22.18% |
| 12 | +63.46% | -22.01% |
| 15 | +63.64% | -23.30% |
| 20 | +65.11% | -19.07% |

The frozen 10-stock configuration was not changed to the post-test 5-stock
winner. Doing so would be test-period overfitting.

Across all 40 possible rebalance offsets, next-open returns ranged from
+24.44% to +96.25%, with a +66.35% median. Thirty-six of 40 offsets beat 45%.
The wide range is the clearest warning that realized timing contributed
meaningfully to the headline result.

## Leakage and accounting checks

- Model selection end is asserted to be earlier than test start.
- Feature calculations use rolling or lagged values only.
- The final model's parameters and selection cutoff are fingerprinted.
- The latest overlay successfully downloaded all 100 tickers and has one
  common last session, July 24, 2026.
- The close engine matches RL2's convention.
- The separate next-open engine accounts for existing holdings overnight,
  rebalances at the following open, then applies open-to-close returns.
- Five deterministic unit tests cover ranking, close accounting, next-open
  accounting, delayed first rebalance, and the date cutoff.

## Limitations

This is an educational development backtest. The 100-stock list is fixed using
companies known today, causing survivorship bias in both training and testing.
Hyperparameter exploration across 2,940 candidates creates selection bias even
though all selection dates precede 2025. The strategy lacks point-in-time
constituents, taxes, borrowing constraints, bid/ask spread, detailed slippage,
market impact, corporate-action verification, and live-forward evidence.

The correct conclusion is narrow: under the repository's simulation framework
and the documented stress tests, this simple pre-2025-selected model beat RL
Codex 2 over the requested historical period. It does not establish a
repeatable future edge.
