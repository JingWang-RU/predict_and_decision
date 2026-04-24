"""
Synthetic gap-event dataset for UAI-2026 Submission 232 rebuttal.

Reproduces the statistical structure claimed in the paper:
  - U ~= 3000 tickers
  - Overnight gap rate ~ 3% per ticker-day  (|gap| > tau)
  - Continuation probability ~ 7-10% conditional on a >3% gap
  - Three foundation-model "forecasters" with controllable noise and bias,
    so downstream predictions differ enough to produce a meaningful
    disagreement signal d_{i,t} = std(l_hat_{i,t}).

The generator yields, for each trading day, three per-ticker prediction
frames with columns [ticker, y_pred, y_true] exactly like the real
pipeline in uai_trade_real.ipynb consumes via load_predictions().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class SyntheticConfig:
    n_tickers: int = 3000
    n_days: int = 500
    overnight_sigma: float = 0.0085       # tuned so P(|gap|>3%) ~= 3% under t_4
    gap_threshold: float = 0.03           # tau = 3%
    continuation_prob: float = 0.08       # P(same-sign intraday | gap) ~ 8%
    continuation_mean: float = 0.015      # mean intraday return when continuation fires
    continuation_sigma: float = 0.02
    normal_intraday_sigma: float = 0.01
    fm_noise: tuple = (0.010, 0.015, 0.012)   # per-FM noise stdev around the truth
    fm_bias: tuple = (0.000, 0.002, -0.001)   # per-FM mean bias
    seed: int = 0


def _sample_overnight_gaps(cfg: SyntheticConfig, rng: np.random.Generator) -> np.ndarray:
    """Heavy-tailed overnight returns so |g| > tau happens ~3% of the time."""
    # Student-t with df=4 scaled to hit ~3% tail mass beyond tau.
    # Empirically df=4 with scale=overnight_sigma produces ~3% tail beyond 3%.
    df = 4.0
    t = rng.standard_t(df, size=(cfg.n_days, cfg.n_tickers))
    return t * cfg.overnight_sigma


def _sample_intraday_returns(
    cfg: SyntheticConfig, gaps: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Intraday returns: mostly noise, but on gap days there is a small
    excess probability of same-sign continuation."""
    n_days, n = gaps.shape
    r = rng.normal(0.0, cfg.normal_intraday_sigma, size=(n_days, n))

    gap_mask = np.abs(gaps) > cfg.gap_threshold
    sign = np.sign(gaps)

    # Base continuation rate implicit in Gaussian noise crossing zero is ~50%.
    # We *add* continuation_prob extra mass on the same-sign side.
    fire = rng.uniform(size=gaps.shape) < cfg.continuation_prob
    fire &= gap_mask
    boost = rng.normal(cfg.continuation_mean, cfg.continuation_sigma, size=gaps.shape)
    r = np.where(fire, sign * np.abs(boost), r)
    return r


def _make_forecasts(
    cfg: SyntheticConfig, y_true: np.ndarray, rng: np.random.Generator
) -> list[np.ndarray]:
    """Three forecasters that see the truth through different noise/bias.
    Returns log-space predictions (l_hat) matching the pipeline's convention."""
    # Convert simple returns -> log returns as the real pipeline does.
    log_true = np.log1p(y_true)
    preds = []
    for s, b in zip(cfg.fm_noise, cfg.fm_bias):
        preds.append(log_true + b + rng.normal(0.0, s, size=log_true.shape))
    return preds


def generate(cfg: SyntheticConfig | None = None) -> dict:
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)

    gaps = _sample_overnight_gaps(cfg, rng)
    y_true = _sample_intraday_returns(cfg, gaps, rng)
    preds_log = _make_forecasts(cfg, y_true, rng)

    tickers = np.array([f"SYN{i:05d}" for i in range(cfg.n_tickers)])
    dates = pd.bdate_range("2020-01-02", periods=cfg.n_days).strftime("%Y-%m-%d").values

    return {
        "tickers": tickers,
        "dates": dates,
        "gaps": gaps,               # (T, N) overnight gap
        "y_true": y_true,           # (T, N) intraday realized simple return
        "preds_log": preds_log,     # list of 3 arrays (T, N) log-return forecasts
        "cfg": cfg,
    }


def eligible_indices(gaps_row: np.ndarray, tau: float) -> np.ndarray:
    return np.where(np.abs(gaps_row) > tau)[0]


def day_prediction_frames(data: dict, t: int, tau: float) -> list[pd.DataFrame]:
    """Return the three per-FM DataFrames the trader expects for day t,
    restricted to the event-conditioned universe A_t (|gap| > tau)."""
    idx = eligible_indices(data["gaps"][t], tau)
    if idx.size == 0:
        return []
    tickers = data["tickers"][idx]
    y_true = data["y_true"][t, idx]
    frames = []
    for m in range(3):
        y_pred = data["preds_log"][m][t, idx]
        frames.append(pd.DataFrame({"ticker": tickers, "y_pred": y_pred, "y_true": y_true}))
    return frames


def empirical_stats(data: dict, tau: float = 0.03) -> dict:
    g = data["gaps"]
    r = data["y_true"]
    gap_mask = np.abs(g) > tau
    gap_rate = gap_mask.mean()
    if gap_mask.any():
        continuation = (np.sign(g[gap_mask]) == np.sign(r[gap_mask])).mean()
    else:
        continuation = float("nan")
    baseline_cont = (np.sign(g) == np.sign(r)).mean()
    return {
        "gap_rate": float(gap_rate),
        "continuation_given_gap": float(continuation),
        "continuation_unconditional": float(baseline_cont),
        "avg_events_per_day": float(gap_mask.sum(axis=1).mean()),
        "tau": tau,
    }


if __name__ == "__main__":
    data = generate()
    print("Shape:", data["gaps"].shape)
    for tau in [0.01, 0.02, 0.03, 0.05, 0.10]:
        print(tau, empirical_stats(data, tau))
