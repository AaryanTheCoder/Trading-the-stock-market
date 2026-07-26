# Trading model repository

The repository is split into two chronological eras. Model code, serialized
policies, simulations, trading spreadsheets, and caches are kept together by
model instead of being mixed at the repository root.

## Repository map

```text
Before 26th July/
├── Models/
│   ├── BEST_RL_Codex_2_45.95pct/
│   ├── RL_Codex_1_Performance_Not_Recorded/
│   ├── RL_Codex_3_V1_0.00pct/
│   └── RL_Codex_3_V2_34.63pct/
└── Legacy_Staged_Prototypes/

26th July onwards/
└── BEST_July26_Momentum_69.30pct/
```

Every maintained model uses the same internal vocabulary where applicable:

- `source/`: training and shared strategy code.
- `model/`: serialized policies and frozen metadata.
- `simulations/`: executable evaluation programs.
- `data/training/`: walk-forward and model-selection output.
- `data/simulation/`: summaries, equity curves, holdings, and trades.
- `data/cache/`: market-price or Python bytecode caches.

The repository `.venv`, `.git`, `.vscode`, `.agents`, and `.codex` directories
remain at the root because they configure the whole workspace rather than one
model.

See each model's README for its date range, performance meaning, limitations,
and run command.

