# RL Codex 3 V2 — 34.63%

RL3 revision 2 returned +34.63% from July 11, 2025 through July 10, 2026. Its
simple 60-day momentum comparison returned +54.11% in the same report.

## Layout

- `source/RL_CODEX_3.py`: portfolio-aware PPO trainer.
- `model/`: V2 policy, checkpoint, and frozen metadata.
- `simulations/CODEX_SIMULATION_3.py`: V2 evaluation program.
- `data/training/`: walk-forward and training data-quality reports.
- `data/simulation/`: trades, equity, holdings, summary, and simulation audit.
- `data/cache/rl3_cache/`: 639 cached market series plus membership/sector data.
- `data/cache/python_bytecode/`: historical generated Python caches.

Run from this model directory:

```bash
../../../.venv/bin/python simulations/CODEX_SIMULATION_3.py
```

This version trained through July 10, 2025 and is not directly comparable to
models trained only through 2024.

