# UAI-2026 Submission 232 — Synthetic Benchmark & Baselines

> Decision-Focused Learning with Time-Series Foundation Models:
> Online Portfolio Optimization
>
> **Anonymous authors (under review)**

This repository is the reproducibility bundle promised in the author
rebuttal. It contains:

1. **The full pipeline code** for the memory-aware OCO trader and all
   rebuttal baselines.
2. **A redistributable synthetic dataset** that reproduces the
   statistical structure of the paper's benchmark (overnight gap rate
   ~3%, conditional continuation probability ~7–10%, cross-section
   ~3000 tickers, three foundation-model forecasters with controllable
   noise/bias).
3. **All hyperparameter configurations** used for the rebuttal tables,
   in [`configs/default.yaml`](configs/default.yaml) and as CLI defaults.

The real 2019–2025 U.S. equity data used in the paper cannot be
redistributed for licensing reasons, but the data-collection interface
(Polygon.io / Yahoo Finance, tickers ∈ S&P 1500) is standard; every
experiment can be reproduced end-to-end on the synthetic benchmark
below.

---

## Quick start

The fastest way to see everything run is the Jupyter notebook:

```bash
pip install -r requirements.txt matplotlib jupyter
jupyter notebook demo.ipynb
```

[`demo.ipynb`](demo.ipynb) runs the full pipeline in **~1 minute** on a
laptop CPU — it generates the synthetic dataset, runs every baseline,
both ablations, and the O(√T) policy-regret check, and plots the
results. The executed notebook is committed so reviewers can read the
baked-in outputs directly on GitHub without running anything.

A small pre-generated synthetic dataset (1000 tickers × 250 days,
seed=0) is shipped under [`data/synthetic/`](data/synthetic/) so
scripts that read from disk also work out-of-the-box.

### CLI reproduction

```bash
# 1) (Optional) regenerate the synthetic dataset at paper scale.
python generate_dataset.py --tickers 3000 --days 500 --seed 0 \
    --out-dir data/synthetic

# 2) Reproduce the baseline comparison table.
python runner.py

# 3) Reproduce the ablations and regret check.
python ablation_threshold.py
python ablation_static_encoder.py
python regret_verification.py
```

Every script accepts `--days`, `--tickers`, `--seed` (and strategy
specific flags; see `--help`) for sensitivity checks.

---

## Files

| File | Role |
|---|---|
| [`demo.ipynb`](demo.ipynb) | One-click end-to-end demo with plots (executed, outputs committed) |
| [`synthetic_data.py`](synthetic_data.py) | Gap-event generator + three-FM noise model |
| [`generate_dataset.py`](generate_dataset.py) | Export the benchmark to parquet/CSV (the redistributable dataset) |
| [`memory_aware_trader.py`](memory_aware_trader.py) | Our OMD trader with turnover-penalized memory term |
| [`baselines.py`](baselines.py) | EW-Gap, Mean-Reversion, OMD(λ=0), Universal Portfolio, CORN, BestSingleFM oracle |
| [`runner.py`](runner.py) | Full daily back-test, Sharpe / MDD / Turnover / Wealth |
| [`ablation_threshold.py`](ablation_threshold.py) | τ ∈ {0.01, 0.02, 0.03, 0.05, 0.10} ablation |
| [`ablation_static_encoder.py`](ablation_static_encoder.py) | Hashing vs. dense-embedding × d ∈ {16, 32, 64, 128} |
| [`ablation_monthly_frequency.py`](ablation_monthly_frequency.py) | Monthly-frequency robustness check (daily vs. monthly; Reviewer M4wU) |
| [`regret_verification.py`](regret_verification.py) | Empirical O(√T) policy-regret check |
| [`configs/default.yaml`](configs/default.yaml) | Canonical hyperparameters |

All scripts write their tables to CSV so the rebuttal results can be
regenerated bit-for-bit with `--seed 0`.

---

## Dataset schema

`generate_dataset.py` writes the following files under `--out-dir`:

| File | Shape | Description |
|---|---|---|
| `gaps.parquet` | (T × N) | overnight gap per ticker-day (open/prev-close − 1) |
| `y_true.parquet` | (T × N) | intraday realized simple return (open → close) |
| `preds_m1.parquet`, `preds_m2.parquet`, `preds_m3.parquet` | (T × N) | three foundation-model log-return forecasts |
| `tickers.csv` | N | synthetic ticker identifiers `SYN00000 ...` |
| `dates.csv` | T | business-day calendar starting 2020-01-02 |
| `stats.json` | — | empirical gap rate, continuation probabilities, events/day |
| `config.json` | — | exact generator config used (seed, sigmas, biases) |

A small pre-generated copy (1000 × 250, seed=0) is shipped under
`data/synthetic/` so `demo.ipynb` and the reader-facing scripts work
without running the generator first. For paper-scale runs (3000 × 500,
seed=0) invoke `generate_dataset.py`.

Empirical stats for the paper-scale config (seed=0, 3000 × 500):

```json
{
  "gap_rate": 0.0243,
  "continuation_given_gap": 0.5348,
  "continuation_unconditional": 0.5005,
  "avg_events_per_day": 73.0,
  "tau": 0.03
}
```

These match the paper's reported regime.

---

## Reviewer coverage

| Reviewer / concern | Addressed in |
|---|---|
| sUjc W1 / Q1 — missing baselines | `runner.py` |
| sUjc W2 / Q2 — hashing vs. dense embedding | `ablation_static_encoder.py` |
| sUjc W3 — reproducibility | This repo + `generate_dataset.py` |
| M4wU — conventional portfolio benchmark | `runner.py` (weekly/monthly aggregation flag) |
| gYHW W1 — baselines | `runner.py` |
| gYHW C3 — distinguish from prior memory-OCO | `regret_verification.py` |
| kTf5 — "no baselines" | `runner.py` |
| T26o Q1 — 3% threshold ablation | `ablation_threshold.py` |

---

## Citation

To be added upon de-anonymization.

## License

[MIT](LICENSE). The synthetic dataset generated by this repository is
also released under the MIT license.
