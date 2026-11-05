"""Intraday market simulator: U-shaped volume and volatility, time-varying spread.

The output is the no-trade price path. Alpha (orders.apply_alpha) and impact
(execution.execute) are layered on top, so the counterfactual stays available.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAY_COLUMNS = ["minute", "mid", "index_mid", "volume", "spread", "sigma_min"]


@dataclass(frozen=True)
class MarketParams:
    n_minutes: int = 390
    s0: float = 100.0
    daily_vol: float = 0.02
    daily_volume: float = 1_000_000
    spread_open_bps: float = 10.0
    spread_close_bps: float = 4.0
    u_shape: float = 2.0
    volume_noise: float = 0.3
    beta: float = 1.0
    index_vol: float = 0.012


def u_profile(n: int, ratio: float) -> np.ndarray:
    """Symmetric U-shaped profile of length n with mean 1 and max/min == ratio."""
    if n < 3:
        raise ValueError("n must be >= 3")
    x = np.linspace(-1.0, 1.0, n)
    base = x**2
    base = (base - base.min()) / (base.max() - base.min()) * (ratio - 1.0) + 1.0
    return base / base.mean()


def _validate(p: MarketParams) -> None:
    if p.n_minutes < 3:
        raise ValueError("n_minutes must be >= 3")
    for field in ("s0", "daily_vol", "daily_volume", "spread_open_bps", "spread_close_bps",
                  "u_shape", "index_vol"):
        if getattr(p, field) <= 0:
            raise ValueError(f"{field} must be > 0")
    if p.volume_noise < 0:
        raise ValueError("volume_noise must be >= 0")
    if p.spread_close_bps > p.spread_open_bps:
        raise ValueError("spread_close_bps must be <= spread_open_bps")
    if p.daily_vol**2 <= (p.beta * p.index_vol) ** 2:
        raise ValueError("daily_vol must exceed beta * index_vol")


def _index_returns(rng: np.random.Generator, p: MarketParams) -> np.ndarray:
    profile = u_profile(p.n_minutes, p.u_shape)
    sigma = p.index_vol * np.sqrt(profile / p.n_minutes)
    r = sigma * rng.standard_normal(p.n_minutes)
    r[0] = 0.0
    return r


def _stock_bars(rng: np.random.Generator, p: MarketParams, index_ret: np.ndarray,
                beta: float) -> pd.DataFrame:
    n = p.n_minutes
    profile = u_profile(n, p.u_shape)
    sigma_min = p.daily_vol * np.sqrt(profile / n)
    idio_var = p.daily_vol**2 - (beta * p.index_vol) ** 2
    if idio_var <= 0:
        raise ValueError(f"beta {beta} too large: |beta| * index_vol must be < daily_vol")
    idio = np.sqrt(idio_var * profile / n) * rng.standard_normal(n)
    r = beta * index_ret + idio
    r[0] = 0.0
    mid = p.s0 * np.exp(np.cumsum(r))
    index_mid = 100.0 * np.exp(np.cumsum(index_ret))
    noise = rng.normal(-0.5 * p.volume_noise**2, p.volume_noise, n)
    volume = p.daily_volume / n * profile * np.exp(noise)
    spread = p.s0 * np.linspace(p.spread_open_bps, p.spread_close_bps, n) / 1e4
    return pd.DataFrame({
        "minute": np.arange(n, dtype=np.int64),
        "mid": mid,
        "index_mid": index_mid,
        "volume": volume,
        "spread": spread,
        "sigma_min": sigma_min,
    })


def simulate_day(rng: np.random.Generator, params: MarketParams | None = None) -> pd.DataFrame:
    """One no-trade day for a single name; see SPEC 3.1 for the schema."""
    params = params or MarketParams()
    _validate(params)
    return _stock_bars(rng, params, _index_returns(rng, params), params.beta)


def simulate_universe(rng: np.random.Generator, n_names: int = 50,
                      params: MarketParams | None = None,
                      beta_range: tuple[float, float] = (0.6, 1.4),
                      betas: np.ndarray | None = None) -> pd.DataFrame:
    """n_names correlated names sharing one index path (one-factor model).

    betas fixes the per-name betas (length n_names) instead of drawing them from
    beta_range, so a fixed basket can be re-simulated on new market days.
    """
    params = params or MarketParams()
    _validate(params)
    if n_names < 1:
        raise ValueError("n_names must be >= 1")
    if betas is not None:
        # beta_range is not used when betas are given, so it is not validated either
        betas = np.asarray(betas, dtype=float)
        if betas.shape != (n_names,):
            raise ValueError(f"betas must have shape ({n_names},), got {betas.shape}")
        if not np.isfinite(betas).all():
            raise ValueError("betas must be finite")
        if (np.abs(betas) * params.index_vol >= params.daily_vol).any():
            raise ValueError("betas too large: max|beta| * index_vol must be < daily_vol")
    else:
        if len(beta_range) != 2:
            raise ValueError("beta_range must have exactly 2 elements")
        if beta_range[0] > beta_range[1]:
            raise ValueError(f"beta_range is inverted: {beta_range}")
        if max(abs(beta_range[0]), abs(beta_range[1])) * params.index_vol >= params.daily_vol:
            raise ValueError("beta_range too wide: max|beta| * index_vol must be < daily_vol")
    index_ret = _index_returns(rng, params)
    if betas is None:
        betas = rng.uniform(beta_range[0], beta_range[1], n_names)
    frames = []
    for name, beta in enumerate(betas):
        bars = _stock_bars(rng, params, index_ret, float(beta))
        bars["name"] = np.int64(name)
        bars["beta"] = float(beta)
        frames.append(bars)
    return pd.concat(frames, ignore_index=True)
