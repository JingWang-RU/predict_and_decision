"""
A standalone copy of the MemoryAwareTrader so the rebuttal comparison
can import it without opening the notebook. Mirrors the logic of
uai_trade_real.ipynb::MemoryAwareTrader (the active, non-commented variant).

Decision rule:
    w_t = (1 - alpha) * w_{t-1} + alpha * q_t
where q_t concentrates on the top-k tickers by confidence-weighted predicted
return (high mean mu across FMs, low cross-FM disagreement). The OMD-style
multiplicative update on realized returns provides the memory step;
alpha is set inversely to lambda_tp so a stronger turnover penalty produces
stickier weights. Constructor signature is preserved (eta, lambda_tp, beta) —
existing callers in runner.py / ablation_*.py / regret_verification.py
continue to work without changes.
"""

from __future__ import annotations

import numpy as np


class MemoryAwareTrader:
    def __init__(self, ticker_universe, eta=0.01, lambda_tp=0.001, beta=1.0,
                 top_k=20, mu_floor=0.0, alpha_blend=None):
        self.universe = list(ticker_universe)
        self.u_size = len(self.universe)
        self.eta = eta
        self.lambda_tp = lambda_tp
        self.beta = beta
        self.top_k = int(top_k)
        self.mu_floor = float(mu_floor)
        # alpha_blend: fraction of the new prediction blended into the action.
        # alpha=1 → fully trust today's prediction; alpha=0 → keep yesterday's weights.
        # Default scales inversely with lambda_tp so a stronger turnover penalty
        # (= more memory) produces a stickier policy without separate tuning.
        self.alpha_blend = (1.0 / (1.0 + 50.0 * lambda_tp)
                            if alpha_blend is None else float(alpha_blend))
        self.theta = np.array([1.0, -0.1])
        self.w_prev = np.zeros(self.u_size + 1)
        self.w_prev[0] = 1.0
        self.value_history = [1.0]
        self.w_history = []
        self.budget = 1.0
        self.name = f"MemOMD(lambda={lambda_tp})"

    # -- prediction features ---------------------------------------------------

    def _features(self, preds_list):
        df1, df2, df3 = [df.drop_duplicates("ticker") for df in preds_list]
        m = df1.merge(df2[["ticker", "y_pred"]], on="ticker", suffixes=("_1", "_2"))
        m = m.merge(df3[["ticker", "y_pred"]], on="ticker").rename(columns={"y_pred": "y_pred_3"})
        l_hat = m[["y_pred_1", "y_pred_2", "y_pred_3"]].values
        r_hat = np.exp(l_hat) - 1.0
        return (
            m["ticker"].values,
            r_hat.mean(axis=1),
            l_hat.std(axis=1),
            np.clip(m["y_true"].values, -3.0, 3.0),
        )

    # -- decision --------------------------------------------------------------

    def _all_cash(self):
        w = np.zeros(self.u_size + 1)
        w[0] = 1.0
        return w

    def decide(self, preds_list, **_):
        tickers_At, mu_At, d_At, _ = self._features(preds_list)
        if len(tickers_At) == 0:
            w = self._all_cash()
            self._last_q = w.copy()
            return w

        t_to_i = {t: i + 1 for i, t in enumerate(self.universe)}
        keep = [i for i, t in enumerate(tickers_At) if t in t_to_i]
        if not keep:
            w = self._all_cash()
            self._last_q = w.copy()
            return w

        keep = np.array(keep)
        full_idx = np.array([t_to_i[tickers_At[i]] for i in keep])
        mu = mu_At[keep]
        disagreement = d_At[keep]

        # Confidence-weighted predicted excess return.
        # High mean-mu × low cross-FM disagreement → high score; clamp at mu_floor
        # so we never allocate to negative-expected-return names.
        confidence = 1.0 / (1.0 + self.beta * disagreement)
        score = confidence * np.maximum(mu - self.mu_floor, 0.0)

        q_t = np.zeros(self.u_size + 1)
        if score.sum() <= 1e-12:
            # No positive signal → stay in cash
            q_t[0] = 1.0
        else:
            k = min(self.top_k, len(keep))
            order = np.argsort(-score)[:k]
            w_active = score[order]
            w_active = w_active / w_active.sum()
            q_t[full_idx[order]] = w_active

        # Memory: blend with previous weights for turnover control
        a = float(np.clip(self.alpha_blend, 0.0, 1.0))
        w_t = a * q_t + (1.0 - a) * self.w_prev
        s = w_t.sum()
        if s > 0:
            w_t = w_t / s

        self._last_q = q_t
        return w_t

    # -- OMD memory update -----------------------------------------------------

    def update(self, r_full):
        prev_w = self.w_history[-1] if self.w_history else self.w_prev
        # Subgradient: -realized return + turnover penalty around q_t
        g_t = -r_full + self.lambda_tp * np.sign(self._last_q - prev_w)
        if self.eta > 0:
            new_w = self._last_q * np.exp(-self.eta * g_t)
            s = new_w.sum()
            if s > 0:
                self.w_prev = new_w / s
            else:
                self.w_prev = self._last_q.copy()
