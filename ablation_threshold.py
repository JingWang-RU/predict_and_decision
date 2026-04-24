"""
Ablation for the 3% gap threshold (Reviewer T26o Q1).

Reports, per tau in {0.01, 0.02, 0.03, 0.05, 0.10}:
  - number of eligible events per year
  - conditional continuation probability
  - downstream Sharpe of MemoryAwareTrader

Run:
    python ablation_threshold.py --out ablation_threshold.csv
"""

from __future__ import annotations

import argparse
import pandas as pd

from synthetic_data import SyntheticConfig, generate, empirical_stats
from memory_aware_trader import MemoryAwareTrader
from runner import run_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=500)
    ap.add_argument("--tickers", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ablation_threshold.csv")
    args = ap.parse_args()

    cfg = SyntheticConfig(n_tickers=args.tickers, n_days=args.days, seed=args.seed)
    data = generate(cfg)
    universe = data["tickers"].tolist()

    rows = []
    for tau in [0.01, 0.02, 0.03, 0.05, 0.10]:
        stats = empirical_stats(data, tau)
        trader = MemoryAwareTrader(universe, eta=0.01, lambda_tp=0.001, beta=1.0)
        trader.name = f"MemOMD@tau={tau}"
        result = run_one(trader, data, tau, universe)
        rows.append({
            "tau": tau,
            "events_per_day": stats["avg_events_per_day"],
            "events_per_year": stats["avg_events_per_day"] * 252,
            "continuation_given_gap": stats["continuation_given_gap"],
            "sharpe": result["sharpe"],
            "mdd": result["mdd"],
            "avg_turnover": result["avg_turnover"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
