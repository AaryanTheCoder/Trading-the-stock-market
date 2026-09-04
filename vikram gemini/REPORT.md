# 200-stock Gemini experiment report

## Outcome

The 200-stock price work succeeded, but the complete Gemini holdout is not
available under the supplied project's current 20-request allowance.

| Model | Execution | 2025–2026 return | Max drawdown |
|---|---|---:|---:|
| API-efficient 200-stock momentum | Close | +148.31% | -41.55% |
| API-efficient 200-stock momentum | Next open | +144.69% | -41.51% |
| General 200-stock momentum | Close | +127.69% | -39.53% |
| General 200-stock momentum | Next open | +125.81% | -39.26% |
| Earlier July26 100-stock model | Close | +69.30% | -22.30% |
| Earlier July26 100-stock model | Next open | +65.76% | -22.18% |
| RL Codex 2 original | Close convention | +45.95% | -19.20% |

The new return is much higher, but so is risk. The 200-stock model is heavily
exposed to technology, semiconductors, data-center infrastructure, and recent
momentum winners. It also has material present-day constituent/survivorship
bias. These are development simulations, not evidence of a live trading edge.

## Gemini result that is actually proven

The API-efficient protocol ranks all 200 stocks by 12-to-1 momentum, selects
ten every 40 sessions, and asks Gemini 2.5 Flash to score those ten using
Google Search with a strict 90-day point-in-time window. Unsupported,
post-cutoff, missing-ticker, and malformed responses are forced to zero.

All seven 2024 validation dates were collected before the news weight was
frozen. The allocation is:

```
weight_i = softmax(news_alpha * audited_news_rank_i)
```

| 2024 validation alpha | Close return | Next-open return |
|---:|---:|---:|
| 0.00 | +190.09% | +190.38% |
| 0.10 | +191.47% | +191.82% |
| 0.20 | +192.97% | +193.37% |
| 0.35 | +195.42% | +195.91% |
| 0.50 | +198.18% | +198.76% |
| 0.75 | +203.45% | +204.18% |
| 1.00 | +209.54% | +210.42% |
| 1.50 | +223.76% | +224.90% |
| **2.00 (frozen)** | **+239.13%** | **+240.48%** |

Only three of ten holdout dates were collected before quota exhaustion.
Therefore no Gemini 2025–2026 return is reported. The final runner raises a
`FileNotFoundError` on the first missing date rather than backfilling or
neutralizing absent calls.

The operational caches contain 100 stock-date outputs across validation and
partial holdout. Only 17 passed the source audit. Thirty-six requested tickers
were omitted across responses, and one ten-stock batch failed JSON three times
and was neutralized. Accepted scores averaged strongly positive. The large
validation improvement therefore comes from concentrated tilts based on sparse,
optimistic evidence and should be treated cautiously.

## Why the earlier +69.30% model worked

The +69.30% model improved on RL Codex 2 mostly by aligning the signal with the
portfolio decision:

1. It replaced PPO's often-saturated “long” probabilities with direct
   cross-sectional ranks across the whole universe.
2. Its main signal was 12-to-1 momentum: return from 252 sessions ago to 20
   sessions ago. Omitting the newest month reduced short-term reversal noise.
3. A 0.25-weight proximity-to-52-week-high factor confirmed that momentum was
   still intact.
4. It held only ten equal-weight names, checked every 40 sessions, and kept an
   existing holding while it remained in the top 20. Persistent trends could
   compound without excessive churn.
5. Factor blend, concentration, and cadence were selected only on 2021–2024.
   The 2025–2026 rule was frozen before evaluation.

The period favored semiconductors and cyclicals, so AMAT, LRCX, MU, CAT, GM,
and similar trends contributed strongly. Timing still mattered: the earlier
model's next-open result ranged from +24.44% to +96.25% across 40 rebalance
offsets. Its +69.30% headline was a combination of better portfolio alignment,
favorable persistent trends, and a favorable schedule—not proof that the rule
will repeat.

## Integrity evidence

- Universe and factor selection end no later than December 31, 2024.
- Price refresh ends July 24, 2026 and is isolated under this experiment.
- API keys are read only from the environment and are absent from repository
  files.
- Raw Gemini responses, search queries, grounding metadata, dated citations,
  token counts, normalized scores, and audit decisions are cached atomically.
- Model-specific caches prevent Flash pilot responses from mixing with the
  API-efficient protocol.
- Nine deterministic tests cover cutoff construction, parsing, future-source
  rejection, cache coverage, rank/weight blending, target normalization, and
  accounting.
- The frozen price and news artifacts have fingerprints.
- Missing holdout coverage is enumerated in
  `data/training/gemini_api_efficient_coverage.json`.

## Resume

After the project quota resets or billing is enabled:

```bash
export GEMINI_API_KEY='your-key'

../../.venv/bin/python tools/collect_gemini_scores.py \
  --model gemini-2.5-flash \
  --price-model api_efficient_price_model_200.json \
  --cache-tag gemini_2_5_flash_api_efficient \
  --period test \
  --batch-size 10

../../.venv/bin/python simulations/run_news_allocation_holdout.py
```

The collector reuses all completed dates. The frozen alpha must not be
reselected after the holdout is collected.

