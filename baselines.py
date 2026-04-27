"""
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
LightGBMRanker       : LightGBM regressor on FM preds, refit on rolling buffer
EconReversal         : short-horizon contrarian (Lehmann 1990 / Jegadeesh 1990)

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


class LightGBMRanker(BaseStrategy):
    """
    LightGBM regressor trained online on a rolling buffer of
        X = [y_pred_1, y_pred_2, y_pred_3]   (per-stock per-day FM predictions)
        y = realized log return on that gap day
    At decision time we score today's A_t and equal-weight the top-k.
    Refits every `refit_every` days once `min_train` samples are available.
    """
    name = "LightGBM"

    def __init__(self, universe, top_k: int = 20,
                 min_train: int = 500, refit_every: int = 20,
                 buffer_cap: int = 50_000, params: dict | None = None):
        super().__init__(universe)
        self.top_k = top_k
        self.min_train = min_train
        self.refit_every = refit_every
        self.buffer_cap = buffer_cap
        self.params = params or {
            "objective": "regression",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "verbose": -1,
        }
        self.X_buf: list[np.ndarray] = []
        self.y_buf: list[float] = []
        self.model = None
        self._since_refit = 0
        self._last_X = None

    def _maybe_refit(self):
        if len(self.y_buf) < self.min_train:
            return
        if self.model is not None and self._since_refit < self.refit_every:
            return
        try:
            import lightgbm as lgb
        except ImportError:
            return
        X = np.vstack(self.X_buf)
        y = np.asarray(self.y_buf)
        dset = lgb.Dataset(X, label=y)
        self.model = lgb.train(self.params, dset, num_boost_round=200)
        self._since_refit = 0

    def decide(self, preds_list, **_):
        tickers_At, r_hat, _ = _merge_preds(preds_list)
        idx = self._active_indices(tickers_At)
        if not idx:
            w = np.zeros(self.u_size + 1); w[0] = 1.0
            self._last_X, self._last_idx = None, idx
            return w

        l_hat = np.log1p(r_hat)  # back to log-return scale used as features
        self._last_X = l_hat
        self._last_idx = idx

        if self.model is None:
            score = r_hat.mean(axis=1)
        else:
            score = self.model.predict(l_hat)
        top = np.argsort(-score)[: min(self.top_k, len(idx))]
        w_active = np.zeros(len(idx))
        w_active[top] = 1.0 / len(top)
        return _pad(w_active, 0.0, self.u_size, idx)

    def observe(self, r_full_active):
        """Runner passes the universe-sized realized-return vector (cash stripped)."""
        if self._last_X is None or not self._last_idx:
            return
        r = np.asarray(r_full_active)
        y = r[np.array(self._last_idx)]
        for row, target in zip(self._last_X, y):
            self.X_buf.append(row[None, :])
            self.y_buf.append(float(target))
        if len(self.y_buf) > self.buffer_cap:
            drop = len(self.y_buf) - self.buffer_cap
            self.X_buf = self.X_buf[drop:]
            self.y_buf = self.y_buf[drop:]
        self._since_refit += 1
        self._maybe_refit()


class EconReversal(BaseStrategy):
    """
    Econometric short-horizon reversal (Lehmann 1990; Jegadeesh 1990;
    Jegadeesh & Titman 1995). Long the losers of the prior k-day window
    in the current event universe A_t, equal-weighted; cash if A_t empty.

    The classic dollar-neutral long/short isn't expressible in this
    long-only simplex framework, so we use the long-leg-only variant
    used in event-study replications (e.g., Da, Liu & Schaumburg 2014).
    """
    name = "EconReversal"

    def __init__(self, universe, lookback: int = 1, bottom_frac: float = 0.2,
                 past_returns_fn=None):
        super().__init__(universe)
        self.lookback = lookback
        self.bottom_frac = bottom_frac
        # callable(tickers, k) -> array of k-day cumulative log returns
        self.past_returns_fn = past_returns_fn

    def decide(self, preds_list, **_):
        tickers_At, _, _ = _merge_preds(preds_list)
        idx = self._active_indices(tickers_At)
        if not idx:
            w = np.zeros(self.u_size + 1); w[0] = 1.0
            return w
        if self.past_returns_fn is None:
            # No past-return source available: fall back to equal weight on A_t.
            return _pad(np.full(len(idx), 1.0 / len(idx)), 0.0, self.u_size, idx)
        past = np.asarray(self.past_returns_fn(tickers_At, self.lookback))
        # Pick the worst-performing fraction (the "losers").
        n_pick = max(1, int(np.ceil(self.bottom_frac * len(idx))))
        order = np.argsort(past)  # ascending: most negative first
        losers = order[:n_pick]
        w_active = np.zeros(len(idx))
        w_active[losers] = 1.0 / n_pick
        return _pad(w_active, 0.0, self.u_size, idx)


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
