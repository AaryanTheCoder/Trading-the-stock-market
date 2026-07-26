# Best pre-July26 RL model: RL Codex 2

The original saved PPO policy returned +45.95% from January 2, 2025 through
July 24, 2026 on the 100-stock panel later used for the July26 comparison. The
long-term PPO policy returned +28.79% over that same refreshed comparison.

## Layout

- `source/RL_CODEX_2.py`: training environment and shared simulation logic.
- `model/`: original and long-term PPO policy ZIP files.
- `simulations/CODEX_SIMULATION_2.py`: current 2025-2026 evaluation.
- `data/trades/`: the existing policy trade spreadsheets.
- `data/cache/python_bytecode/`: historical generated Python caches.

Run from this model directory:

```bash
../../../.venv/bin/python simulations/CODEX_SIMULATION_2.py
```

Training data ends in 2024, but the long-term design followed an earlier 2025
result. The current result is therefore a development backtest.

