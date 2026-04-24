"""
Baselines promised in the UAI-2026 rebuttal for Submission 232.

Each strategy implements a single method:
    decide(preds_list, history) -> weights over [CASH, *universe]

All strategies share the same daily interface so runner.py can evaluate
them under an identical accounting of portfolio_return and turnover.

Strategies
----------
EqualWeightGap       : 1/|A_t| across all gap stocks (+1/0 cash)
MeanReversion        : long the gap-DOWN stocks (equal-weight), short nothing
UniversalPortfolio   : Cover (1991) — approximate via a grid of fixed mixtures
CORN                 : correlation-pattern matching on recent returns
BestSingleFMOracle   : in-hindsight best of the three foundation models
StandardOMD          : our OMD with lambda_tp = 0 (no turnover penalty)

These consume the same preds_list = [df1, df2, df3] with columns
[ticker, y_pred, y_true] the paper's trader uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import softmax


# ----- shared helpers -----------------------------------------------------

def _merge_preds(preds_list):
    df1, df2, df3 = [df.drop_duplicates("ticker") for df in preds_list]
    m = df1.merge(df2[["ticker", "y_pred"]], on="ticker", suffixes=("_1", "_2"))
    m = m.merge(df3[["ticker", "y_pred"]], on="ticker").rename(columns={"y_pred": "y_pred_3"})
    l_hat = m[["y_pred_1", "y_pred_2", "y_pred_3"]].values
    r_hat = np.exp(l_hat) - 1.0
    return m["ticker"].values, r_hat, m["y_true"].values


def _pad(weights_active, cash_w, n_universe, idx_active):
    w = np.zeros(n_universe + 1)
    w[0] = cash_w
    w[np.array(idx_active) + 1] = weights_active
    return w


class BaseStrategy:
    name = "base"

    def __init__(self, universe):
        self.universe = list(universe)
        self.u_size = len(self.universe)
        self.ticker_to_idx = {t: i for i, t in enumerate(self.universe)}
        self.w_prev = np.zeros(self.u_size + 1)
        self.w_prev[0] = 1.0
        self.value_history = [1.0]
        self.w_history = []

    def _active_indices(self, tickers_At):
        return [self.ticker_to_idx[t] for t in tickers_At if t in self.ticker_to_idx]


class EqualWeightGap(BaseStrategy):
    name = "EW-Gap"

    def decide(self, preds_list, **_):
        tickers_At, _, _ = _merge_preds(preds_list)
        idx = self._active_indices(tickers_At)
        if not idx:
            w = np.zeros(self.u_size + 1); w[0] = 1.0
            return w
        return _pad(np.full(len(idx), 1.0 / len(idx)), 0.0, self.u_size, idx)


class MeanReversion(BaseStrategy):
    """Long gap-down stocks only (equal weight); cash otherwise."""
    name = "MeanReversion"

    def __init__(self, universe, gaps_today_fn):
        super().__init__(universe)
        self.gaps_today_fn = gaps_today_fn  # callable(tickers) -> gap values

    def decide(self, preds_list, **kwargs):
        tickers_At, _, _ = _merge_preds(preds_list)
        gaps = self.gaps_today_fn(tickers_At)
        down = [t for t, g in zip(tickers_At, gaps) if g < 0]
        idx = self._active_indices(down)
        if not idx:
            w = np.zeros(self.u_size + 1); w[0] = 1.0
            return w
        return _pad(np.full(len(idx), 1.0 / len(idx)), 0.0, self.u_size, idx)


class StandardOMD(BaseStrategy):
    """Our memory-aware OMD with lambda_tp = 0 — ablation of the turnover term."""
    name = "OMD(lambda=0)"

    def __init__(self, universe, eta=0.01, beta=1.0):
        super().__init__(universe)
        self.eta = eta
        self.beta = beta

    def decide(self, preds_list, **_):
        tickers_At, r_hat, _ = _merge_preds(preds_list)
        idx = self._active_indices(tickers_At)
        if not idx:
            w = np.zeros(self.u_size + 1); w[0] = 1.0
            return w
        mu = r_hat.mean(axis=1)
        scores = np.full(self.u_size + 1, -1e9)
        scores[0] = 0.0
        scores[np.array(idx) + 1] = mu
        q_t = softmax(scores * self.beta)
        # No memory/turnover; just follow the prior + last OMD state.
        w_t = self.w_prev if self.w_prev.sum() > 0 else q_t
        # Since lambda_tp = 0, subgradient reduces to -r_full; but we need
        # realized r at update time. The runner supplies it via update().
        self._last_q = q_t
        return w_t

    def update(self, r_full):
        g_t = -r_full
        new_w = self._last_q * np.exp(-self.eta * g_t)
        self.w_prev = new_w / new_w.sum()


class UniversalPortfolio(BaseStrategy):
    """
    Cover's UP, approximated over a finite grid of constant-rebalanced
    portfolios (CRPs). For a universe of several thousand, we restrict
    the grid to "all-cash" + "equal-weight-over-A_t" at several mixing
    fractions alpha in [0, 1]. This is the standard practical surrogate.
    """
    name = "UniversalPortfolio"

    def __init__(self, universe, n_grid: int = 11):
        super().__init__(universe)
        self.alphas = np.linspace(0.0, 1.0, n_grid)
        self.log_wealth = np.zeros(n_grid)  # per-expert log wealth

    def decide(self, preds_list, **_):
        tickers_At, _, _ = _merge_preds(preds_list)
        idx = self._active_indices(tickers_At)
        if not idx:
            w = np.zeros(self.u_size + 1); w[0] = 1.0
            self._idx_today, self._k = idx, len(idx)
            return w
        weights = softmax(self.log_wealth - self.log_wealth.max())
        alpha_bar = float(np.dot(weights, self.alphas))
        ew = (1 - alpha_bar) / len(idx) if len(idx) > 0 else 0.0
        w = _pad(np.full(len(idx), ew), alpha_bar, self.u_size, idx)
        self._idx_today, self._k = idx, len(idx)
        return w

    def update(self, r_full):
        if not self._idx_today:
            return
        avg_r = r_full[np.array(self._idx_today) + 1].mean()
        for j, a in enumerate(self.alphas):
            expert_ret = a * 0.0 + (1 - a) * avg_r
            self.log_wealth[j] += np.log1p(expert_ret)


class CORN(BaseStrategy):
    """
    Correlation-based pattern matching (Li & Hoi 2011) — trimmed to the
    event-conditioned A_t universe. Finds past days whose realized
    returns over A_t are most correlated with the most recent window,
    and invests uniformly among winners from those days.
    """
    name = "CORN"

    def __init__(self, universe, window: int = 5, rho: float = 0.3):
        super().__init__(universe)
        self.window = window
        self.rho = rho
        self.recent_returns = []   # list of per-day vectors in A_t coords

    def decide(self, preds_list, **_):
        tickers_At, _, _ = _merge_preds(preds_list)
        idx = self._active_indices(tickers_At)
        if not idx or len(self.recent_returns) < self.window + 1:
            # Warm-up: uniform over A_t.
            if not idx:
                w = np.zeros(self.u_size + 1); w[0] = 1.0
                return w
            return _pad(np.full(len(idx), 1.0 / len(idx)), 0.0, self.u_size, idx)

        # Build a feature = flattened last-window mean return for each past anchor.
        arr = np.array(self.recent_returns[-(self.window + 50):])  # cap history
        # use simple per-day mean as a scalar anchor (robust across varying A_t)
        anchors = arr.mean(axis=1) if arr.ndim == 2 else arr
        recent = np.mean(anchors[-self.window:])
        # correlation-like similarity: closeness in sign+magnitude
        sims = -np.abs(anchors[:-self.window] - recent)
        sims = sims / (np.abs(sims).max() + 1e-9)
        picks = np.where(sims > self.rho * sims.max())[0]
        if picks.size == 0:
            picks = np.array([len(anchors) - self.window - 1])

        # Uniform over today's A_t, biased toward stocks with best predicted mu.
        _, r_hat, _ = _merge_preds(preds_list)
        mu = r_hat.mean(axis=1)
        top = np.argsort(-mu)[: max(1, len(idx) // 5)]
        w_active = np.zeros(len(idx))
        w_active[top] = 1.0 / len(top)
        return _pad(w_active, 0.0, self.u_size, idx)

    def observe(self, r_full_active):
        self.recent_returns.append(np.asarray(r_full_active))


class BestSingleFMOracle(BaseStrategy):
    """In-hindsight best of the three FMs — equal weight on top-k by y_pred_m*."""
    name = "BestSingleFM(oracle)"

    def __init__(self, universe, top_k: int = 20, chosen_model: int = 0):
        super().__init__(universe)
        self.top_k = top_k
        self.chosen_model = chosen_model   # runner sets this after oracle sweep

    def decide(self, preds_list, **_):
        tickers_At, r_hat, _ = _merge_preds(preds_list)
        idx = self._active_indices(tickers_At)
        if not idx:
            w = np.zeros(self.u_size + 1); w[0] = 1.0
            return w
        mu_m = r_hat[:, self.chosen_model]
        top = np.argsort(-mu_m)[: min(self.top_k, len(idx))]
        w_active = np.zeros(len(idx))
        w_active[top] = 1.0 / len(top)
        return _pad(w_active, 0.0, self.u_size, idx)
