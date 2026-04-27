"""
Daily back-test runner that evaluates every baseline + MemoryAwareTrader
on the synthetic gap-event dataset.

Reports the comparison table promised in the rebuttal:
  strategy, annualized Sharpe, Max Drawdown, average daily turnover, final wealth

Run:
    python runner.py                      # default synthetic run
    python runner.py --days 500 --tickers 3000 --tau 0.03
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from synthetic_data import SyntheticConfig, generate, day_prediction_frames, empirical_stats
from baselines import (
    EqualWeightGap, MeanReversion, StandardOMD,
    UniversalPortfolio, CORN, BestSingleFMOracle,
    LightGBMRanker, EconReversal,
)
from memory_aware_trader import MemoryAwareTrader


def _portfolio_return(w: np.ndarray, r_full: np.ndarray) -> float:
    return float(np.dot(w, r_full))


def _to_r_full(tickers_At, y_true_At, universe, u_size):
    r_full = np.zeros(u_size + 1)
    t_to_i = {t: i + 1 for i, t in enumerate(universe)}
    for t, y in zip(tickers_At, y_true_At):
        if t in t_to_i:
            r_full[t_to_i[t]] = y
    return r_full


def _sharpe(value_history, freq=252):
    v = np.asarray(value_history, dtype=float)
    if len(v) < 3:
        return 0.0
    r = np.diff(v) / v[:-1]
    sd = r.std()
    return float(np.sqrt(freq) * r.mean() / sd) if sd > 1e-12 else 0.0


def _max_drawdown(value_history):
    v = np.asarray(value_history, dtype=float)
    peak = np.maximum.accumulate(v)
    dd = (v - peak) / peak
    return float(dd.min())


def run_one(strategy, data, tau, universe):
    u_size = len(universe)
    turnovers, rets = [], []
    prev_w = np.zeros(u_size + 1); prev_w[0] = 1.0
    for t in range(len(data["dates"])):
        frames = day_prediction_frames(data, t, tau)
        if not frames:
            strategy.value_history.append(strategy.value_history[-1])
            continue

        w = strategy.decide(frames)
        tickers_At = frames[0]["ticker"].values
        y_true_At = frames[0]["y_true"].values
        r_full = _to_r_full(tickers_At, y_true_At, universe, u_size)

        pr = _portfolio_return(w, r_full)
        rets.append(pr)
        turnovers.append(float(np.abs(w - prev_w).sum()))
        strategy.value_history.append(strategy.value_history[-1] * (1 + pr))
        strategy.w_history.append(w)
        prev_w = w

        if hasattr(strategy, "update"):
            strategy.update(r_full)
        if hasattr(strategy, "observe"):
            strategy.observe(r_full[1:])

    return {
        "strategy": strategy.name,
        "sharpe": _sharpe(strategy.value_history),
        "mdd": _max_drawdown(strategy.value_history),
        "avg_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "final_wealth": float(strategy.value_history[-1]),
        "avg_return_bps": 1e4 * float(np.mean(rets)) if rets else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=500)
    ap.add_argument("--tickers", type=int, default=3000)
    ap.add_argument("--tau", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="baseline_comparison.csv")
    args = ap.parse_args()

    cfg = SyntheticConfig(n_tickers=args.tickers, n_days=args.days, gap_threshold=args.tau, seed=args.seed)
    data = generate(cfg)
    print("Dataset stats:", empirical_stats(data, args.tau))

    universe = data["tickers"].tolist()
    gap_lookup = {d: data["gaps"][t] for t, d in enumerate(data["dates"])}

    def gaps_today_fn_factory(t_ref):
        def f(tickers):
            t = t_ref[0]
            row = data["gaps"][t]
            t_to_i = {tk: i for i, tk in enumerate(universe)}
            return np.array([row[t_to_i[x]] for x in tickers])
        return f

    # MeanReversion needs per-day gaps; we patch via closure updated inside run_one.
    # Simpler: compute gaps lazily from the day index held in a mutable cell.
    # Here we instead give MeanReversion the full matrix and reconstruct within runner.
    class _MR(MeanReversion):
        def __init__(self, universe_):
            # gaps_today_fn replaced later
            super().__init__(universe_, gaps_today_fn=lambda _t: None)
            self._t = 0
        def decide(self, preds_list, **_):
            self.gaps_today_fn = lambda tickers: np.array(
                [data["gaps"][self._t][list(universe).index(t)] for t in tickers]
            )
            w = super().decide(preds_list)
            self._t += 1
            return w

    # Rolling realized-return history for EconReversal (per ticker).
    t_to_i = {tk: i for i, tk in enumerate(universe)}
    realized_hist: list[np.ndarray] = []  # appended each day inside run_one wrappers

    def past_returns_fn(tickers, k):
        if not realized_hist:
            return np.zeros(len(tickers))
        arr = np.array(realized_hist[-k:])  # (<=k, u_size)
        cum = arr.sum(axis=0)
        return np.array([cum[t_to_i[t]] if t in t_to_i else 0.0 for t in tickers])

    class _EconRev(EconReversal):
        def __init__(self, universe_):
            super().__init__(universe_, lookback=1, bottom_frac=0.2,
                             past_returns_fn=past_returns_fn)
        def observe(self, r_full_active):
            realized_hist.append(np.asarray(r_full_active))

    strategies = [
        EqualWeightGap(universe),
        _MR(universe),
        StandardOMD(universe, eta=0.01, beta=1.0),
        UniversalPortfolio(universe),
        CORN(universe),
        BestSingleFMOracle(universe, top_k=20, chosen_model=0),
        LightGBMRanker(universe, top_k=20),
        _EconRev(universe),
        MemoryAwareTrader(universe, eta=0.01, lambda_tp=0.001, beta=1.0),
    ]

    rows = []
    for s in strategies:
        print(f"\nRunning {s.name} ...", flush=True)
        rows.append(run_one(s, data, args.tau, universe))
        print(rows[-1])

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("\n=== Baseline Comparison ===")
    print(df.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
