"""
Monthly-frequency robustness check (Reviewer M4wU).

The paper is deliberately positioned at *daily* frequency with an
event-conditioned candidate set A_t (|overnight gap| > tau = 3%).
Monthly asset-allocation literature assumes a very different regime.
This script runs the *same* MemoryAwareTrader under a monthly variant
so the two regimes can be compared on identical accounting, and
quantifies how much the decision-focused loss degrades once the
event-conditioning signal weakens at lower frequency.

Monthly construction
--------------------
1. Aggregate the daily synthetic dataset into ~month-long chunks of M
   trading days each (default M = 21).
2. Monthly gap for ticker i, month k: compound of daily overnight
   returns over the month, i.e. prod_{t in month_k}(1+gap_{i,t}) - 1.
3. Event set A_k = {i : |monthly_gap| > tau_monthly}. Because monthly
   returns are much larger in magnitude, we report a sweep over
   tau_monthly in {0.03, 0.10, 0.20}.
4. Monthly realized "intraday" label = compound of daily intraday
   labels over the month. This matches how a monthly strategy would
   realize PnL once a position is taken.
5. Monthly forecast for each FM = mean of daily log-return forecasts
   over the month (i.e., a naive temporal pooling of the daily FM
   signal). This is the most charitable reading of what a monthly
   deployment of the same TSFMs could produce without retraining.

Run:
    python ablation_monthly_frequency.py --out ablation_monthly.csv
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from synthetic_data import SyntheticConfig, generate, empirical_stats
from memory_aware_trader import MemoryAwareTrader
from runner import run_one


def _aggregate_monthly(data: dict, M: int) -> dict:
    """Compound daily series into month-long blocks of M trading days."""
    gaps_d = data["gaps"]       # (T, N) daily overnight simple returns
    y_d = data["y_true"]        # (T, N) daily intraday simple returns
    preds_d = data["preds_log"]  # list of 3 arrays (T, N) log-return forecasts

    T, N = gaps_d.shape
    K = T // M
    if K < 3:
        raise ValueError(f"Need at least 3 monthly blocks; got K={K} with T={T}, M={M}")
    gaps_m = np.zeros((K, N))
    y_m = np.zeros((K, N))
    preds_m_list = [np.zeros((K, N)) for _ in preds_d]
    for k in range(K):
        sl = slice(k * M, (k + 1) * M)
        gaps_m[k] = np.prod(1.0 + gaps_d[sl], axis=0) - 1.0
        y_m[k] = np.prod(1.0 + y_d[sl], axis=0) - 1.0
        for m, pa in enumerate(preds_d):
            preds_m_list[m][k] = pa[sl].mean(axis=0)

    return {
        "tickers": data["tickers"],
        "dates": np.array([f"M{k:03d}" for k in range(K)]),
        "gaps": gaps_m,
        "y_true": y_m,
        "preds_log": preds_m_list,
        "cfg": data["cfg"],
    }


def _stats(g, r, tau):
    mask = np.abs(g) > tau
    return {
        "gap_rate": float(mask.mean()),
        "events_per_period": float(mask.sum(axis=1).mean()),
        "P(continuation|gap)": float((np.sign(g[mask]) == np.sign(r[mask])).mean()) if mask.any() else float("nan"),
    }


def _sharpe_annualized(value_history, periods_per_year):
    v = np.asarray(value_history, dtype=float)
    if len(v) < 3:
        return 0.0
    r = np.diff(v) / v[:-1]
    sd = r.std()
    return float(np.sqrt(periods_per_year) * r.mean() / sd) if sd > 1e-12 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=500)
    ap.add_argument("--tickers", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--M", type=int, default=21, help="trading days per month")
    ap.add_argument("--out", default="ablation_monthly.csv")
    args = ap.parse_args()

    cfg = SyntheticConfig(n_tickers=args.tickers, n_days=args.days, seed=args.seed)
    data_d = generate(cfg)
    universe = data_d["tickers"].tolist()

    # --- Daily reference (matches the paper's regime) ---
    tau_d = 0.03
    stats_d = _stats(data_d["gaps"], data_d["y_true"], tau_d)
    trader_d = MemoryAwareTrader(universe, eta=0.01, lambda_tp=0.001, beta=1.0)
    trader_d.name = "MemOMD-daily"
    res_d = run_one(trader_d, data_d, tau_d, universe)
    # Re-annualize using trading-day convention
    res_d["sharpe"] = _sharpe_annualized(trader_d.value_history, periods_per_year=252)
    res_d.update({
        "frequency": "daily",
        "tau": tau_d,
        "periods": len(data_d["dates"]),
        "events_per_period": stats_d["events_per_period"],
        "P(cont|gap)": stats_d["P(continuation|gap)"],
    })

    # --- Monthly variants at several reasonable thresholds ---
    data_m = _aggregate_monthly(data_d, args.M)
    rows = [res_d]
    for tau_m in [0.03, 0.10, 0.20]:
        stats_m = _stats(data_m["gaps"], data_m["y_true"], tau_m)
        trader_m = MemoryAwareTrader(universe, eta=0.01, lambda_tp=0.001, beta=1.0)
        trader_m.name = f"MemOMD-monthly@tau={tau_m}"
        res_m = run_one(trader_m, data_m, tau_m, universe)
        res_m["sharpe"] = _sharpe_annualized(trader_m.value_history, periods_per_year=12)
        res_m.update({
            "frequency": f"monthly (M={args.M})",
            "tau": tau_m,
            "periods": len(data_m["dates"]),
            "events_per_period": stats_m["events_per_period"],
            "P(cont|gap)": stats_m["P(continuation|gap)"],
        })
        rows.append(res_m)

    cols = [
        "strategy", "frequency", "tau", "periods",
        "events_per_period", "P(cont|gap)",
        "sharpe", "mdd", "avg_turnover", "final_wealth", "avg_return_bps",
    ]
    df = pd.DataFrame(rows)[cols]
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
