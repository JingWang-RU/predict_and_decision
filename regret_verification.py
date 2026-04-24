"""
Empirical O(sqrt(T)) policy-regret check for the rebuttal (Reviewer gYHW C3).

Runs MemoryAwareTrader vs. the best-fixed-policy-in-hindsight (BFPH) over
the synthetic gap benchmark and plots cumulative regret against c*sqrt(T).

Run:
    python regret_verification.py --out regret.csv
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from synthetic_data import SyntheticConfig, generate, day_prediction_frames
from memory_aware_trader import MemoryAwareTrader
from baselines import _merge_preds


def _r_full(tickers_At, y, universe, u_size):
    r = np.zeros(u_size + 1)
    t_to_i = {t: i + 1 for i, t in enumerate(universe)}
    for t, v in zip(tickers_At, y):
        if t in t_to_i:
            r[t_to_i[t]] = v
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=500)
    ap.add_argument("--tickers", type=int, default=1000)
    ap.add_argument("--tau", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="regret.csv")
    args = ap.parse_args()

    cfg = SyntheticConfig(n_tickers=args.tickers, n_days=args.days, gap_threshold=args.tau, seed=args.seed)
    data = generate(cfg)
    universe = data["tickers"].tolist()
    u_size = len(universe)

    trader = MemoryAwareTrader(universe, eta=0.01, lambda_tp=0.001, beta=1.0)

    # Best-fixed-policy-in-hindsight over {cash, uniform-over-A_t}: the better
    # of the two across the full horizon. This is the comparator class used
    # implicitly in the existing uai_trade_real.ipynb BFPH analysis.
    losses_algo = []
    losses_cash = []
    losses_unif = []

    prev_w = np.zeros(u_size + 1); prev_w[0] = 1.0
    for t in range(len(data["dates"])):
        frames = day_prediction_frames(data, t, args.tau)
        if not frames:
            continue
        w = trader.decide(frames)
        tickers_At = frames[0]["ticker"].values
        y_true_At = frames[0]["y_true"].values
        r_full = _r_full(tickers_At, y_true_At, universe, u_size)

        # algorithm loss: -return + lambda * turnover
        turnover = float(np.abs(w - prev_w).sum())
        l_algo = -float(np.dot(w, r_full)) + trader.lambda_tp * turnover
        losses_algo.append(l_algo)

        # cash loss: always 100% cash
        losses_cash.append(0.0)

        # uniform-over-A_t loss (constant strategy -> zero turnover between days
        # where A_t doesn't change; we approximate by mean return on event day)
        losses_unif.append(-float(np.mean(y_true_At)))

        prev_w = w
        trader.w_history.append(w)
        trader.update(r_full)

    L_algo = np.cumsum(losses_algo)
    L_bfph = np.minimum(np.cumsum(losses_cash), np.cumsum(losses_unif))
    regret = L_algo - L_bfph
    T = np.arange(1, len(regret) + 1)

    df = pd.DataFrame({
        "t": T,
        "cum_loss_algo": L_algo,
        "cum_loss_bfph": L_bfph,
        "cum_regret": regret,
        "sqrt_t_bound_c1": np.sqrt(T),
    })
    df.to_csv(args.out, index=False)

    # Fit c such that regret ~ c * sqrt(T) on the latter half.
    half = len(T) // 2
    c = float(np.mean(regret[half:] / np.sqrt(T[half:]))) if half > 0 else 0.0
    print(f"Estimated constant c in regret ~ c*sqrt(T): {c:.4f}")
    print(df.tail().to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
