# Gemini grounded-news overlay — 200-stock experiment

This is an isolated July 26 experiment. It analyzes a 200-stock universe with a
pre-2025 price model, then uses grounded Gemini news scores to rerank the price
model's top 20 candidates. Nothing here modifies code outside this directory.

## Current verified result

The general 20-session price-only control returned **+127.69%** close and
**+125.81%** next-open from January 2, 2025 through July 24, 2026. The
API-efficient 40-session foundation used by the final Gemini experiment
returned **+148.31%** close and **+144.69%** next-open. Its maximum drawdown
was roughly 41.5%, so the return came with much more risk than the earlier
+69.30% model.

Across all 20 possible rebalance offsets, next-open return ranged from +88.78%
to +143.70%, with a +121.41% median; every offset beat +69.30%. Raising the
simulated cost from 0.1% to an extreme 1.0% per dollar traded reduced the
return to +111.79%. A post-hoc five-stock variant returned +259.74% but had a
45.75% drawdown and was not substituted for the frozen pre-2025 top-10 choice.

The supplied project's live 20-request limit allowed a complete seven-date
2024 validation collection and three of ten holdout dates. Validation froze
`news_alpha = 2.0`: it improved 2024 from +190.09% to +239.13% close, and from
+190.38% to +240.48% next-open. The Gemini holdout result is intentionally not
claimed because seven 2025-current responses remain missing. The holdout
runner fails closed rather than treating those missing dates as neutral.

## Frozen price foundations

- Universe: 200 liquid stocks selected using 2024 dollar volume.
- Factor selection: calendar years 2021–2024 only.
- Winner: cross-sectional 252-session total-return rank.
- Portfolio: top 10, equal weight, rebalanced every 20 sessions.
- Keep zone: retain a holding while it remains in the top 20.
- Trading cost: 0.1% on absolute weight changed.
- Test: January 2025 onward, never used to select the saved rule.

The universe comes from constituents known in 2026. This creates material
survivorship and membership look-ahead bias despite the factor/date cutoff.
The +127.69% result is a repository development simulation, not live evidence.

The API-efficient foundation was selected in a separate pre-2025 sweep under a
40-session cadence, exactly ten holdings/candidates, and no keep zone. Its
winner was 12-to-1 momentum (`skip_252_20`). This protocol needs seven
validation plus ten holdout requests—17 total—while still ranking all 200
stocks at every rebalance.

## Gemini overlay

The pilot ranked all 200 stocks and sent its top 20 to Gemini in five-stock
batches. The API-efficient final protocol ranks all 200, selects ten, and sends
those ten in one call every 40 sessions. Gemini 2.5 Flash uses Google Search
grounding and a 90-calendar-day search window ending at the simulated cutoff.
For every stock it returns a -100 to +100 score, confidence, catalyst, risk,
and dated sources.

The effective news score is:

```
(raw score / 100) * confidence
```

It is forcibly set to zero if Gemini reports insufficient information, omits
the ticker, provides no dated source, or cites a source outside the allowed
point-in-time window. The final planned score is:

```
price rank + news_alpha * cross-sectional news rank
```

The pre-registered alpha sweep is 0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0,
1.5, and 2.0. `training/select_news_weight.py` selects the alpha using only
2024 close and next-open results. `simulations/run_holdout.py` then reports
2025-current results without changing that frozen winner.

The API-efficient design keeps the same ten stocks and tilts their allocations:

```
weight_i = softmax(news_alpha * audited_news_rank_i)
```

Its 2024-only sweep tested alpha 0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.5,
and 2.0. Alpha 2.0 won and was fingerprinted before any holdout simulation.

## What the live pilot showed

The first 70 stock-date responses used 100 grounded search queries and about
121,300 reported tokens:

- 31/70 passed the source-date audit.
- 30/70 produced a non-zero effective score.
- Accepted scores had mean confidence 0.823.
- Accepted effective scores averaged +0.545.
- 26 accepted scores were strongly positive (>0.25); only two were negative.

This reveals a pronounced optimism/selection bias in retrospective financial
news search. The provisional sweep on only three fully covered rebalances found
that small weights (0.05–0.75) hurt 2024 return, while alpha 1.0–2.0 improved it
from +181.50% to +186.95% close / +195.17% next-open. Those numbers are not a
model-selection result: Gemini was active for only 3 of 13 rebalances.

The separate API-efficient validation is complete:

- 7/7 validation date caches completed.
- Alpha 0: +190.09% close / +190.38% next-open in 2024.
- Frozen alpha 2: +239.13% close / +240.48% next-open in 2024.
- 3/10 holdout date caches completed before the project limit stopped calls.
- No 2025-current Gemini return is reported from partial coverage.

## Reproduce and resume

Run from this directory with the repository environment:

```bash
# Complete price-only control and trading spreadsheets
../../.venv/bin/python simulations/run_price_baseline.py

# Export the credential only in the shell; never save it in this repository
export GEMINI_API_KEY='your-key'

# Resume atomically; existing batch caches are reused
../../.venv/bin/python tools/collect_gemini_scores.py --period both --batch-size 5

# Resume the API-efficient holdout; existing dates are reused atomically
../../.venv/bin/python tools/collect_gemini_scores.py \
  --model gemini-2.5-flash \
  --price-model api_efficient_price_model_200.json \
  --cache-tag gemini_2_5_flash_api_efficient \
  --period test --batch-size 10

# Inspect response quality at any point
../../.venv/bin/python tools/audit_gemini_cache.py --model gemini-2.5-flash
../../.venv/bin/python training/explore_partial_news.py

# Only after complete 2024 and 2025-current cache coverage:
../../.venv/bin/python training/select_news_weight.py
../../.venv/bin/python simulations/run_holdout.py

# API-efficient frozen model; final runner requires all holdout dates
../../.venv/bin/python training/select_news_allocation_weight.py
../../.venv/bin/python simulations/run_news_allocation_holdout.py

# Tests
../../.venv/bin/python -m unittest discover -s tests -v
```

The cache writer is atomic, retries transient HTTP and malformed-JSON
responses, and never persists the API key. The `.gitignore` excludes credential
files.

## Directory map

- `model/`: frozen price model; final news model appears only after validation.
- `source/`: price engine, Gemini client, cache audit, and score combination.
- `training/`: universe/model selection and pre-2025 news-weight selection.
- `simulations/`: price control and untouched holdout runners.
- `data/training/`: factor sweeps and Gemini-response audit spreadsheets.
- `data/simulation/`: summaries, equity curves, and `trades/`.
- `data/cache/latest_prices/`: price refresh local to this experiment.
- `data/cache/gemini_2_5_flash/`: Flash pilot responses by as-of date.
- `data/cache/gemini_2_5_flash_lite/`: separate Flash-Lite fallback caches.
- `data/cache/gemini_2_5_flash_api_efficient/`: complete validation and partial
  holdout responses for the frozen allocation model.
- `tests/`: cutoff, parsing, source audit, coverage, and blending tests.
