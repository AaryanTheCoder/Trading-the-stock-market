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
