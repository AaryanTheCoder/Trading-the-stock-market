# Best verified 200-stock momentum model

This is a self-contained copy of the strongest **complete, reproducible
price-only** model in this repository. Its frozen rules were chosen using data
through 2024 only, then measured from January 2, 2025 to July 24, 2026.

- Close-to-close test return: **+148.31%** ($100,000 became $248,309).
- More conservative next-open test return: **+144.69%** ($100,000 became
  $244,686).
- Maximum drawdown: about **-41.5%**. High return does not mean low risk or a
  guarantee of future profit.

The model ranks 200 stocks by 12-to-1 momentum, holds the top 10 equally, and
rebalances every 40 market sessions. It charges 0.1% trading cost in the
simulation. The 2025--2026 test period was not used to pick those settings.

## Everything needed is in this folder

- `model/`: the frozen model settings.
- `source/`: the pricing and portfolio logic.
- `data/cache/base_prices/`: all 200 historical price files the model reads.
- `data/training/`: stock universe and the pre-2025 selection results.
- `data/simulation/`: saved return, equity, and robustness results.
- `simulations/`: programs that run the backtest.
- `tests/`: checks for the important calculation rules.

`data/reference/july26_summary.json` is only a copied comparison result for an
older model; it is not required for the price-only baseline.

The included Gemini-news scripts are preserved from the original experiment,
but they do **not** have a complete 2025--2026 news-result claim. The
`api_efficient_price_model_200.json` price model is the model responsible for
the +148.31% result.

## Run it

From this folder:

```bash
python3 -m pip install -r requirements.txt
python3 simulations/run_price_baseline.py
python3 -m unittest discover -s tests -v
```

The runner recreates the price-only saved outputs in `data/simulation/`. No
internet connection is needed for that baseline because its historical prices
are included. `tools/refresh_prices.py` is optional and needs internet access
to fetch newer prices.
