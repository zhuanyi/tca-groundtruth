"""Child-order scheduling, propagator impact, passive fills with adverse selection.

The affected mid is mid + impact_shift; the no-trade mid stays in `mid`, so the
TCA layer can compute true impact per fill.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

FILL_COLUMNS = ["minute", "qty", "price", "passive", "mid_affected", "mid"]
ALGOS = ("VWAP", "TWAP", "POV")


@dataclass(frozen=True)
class ImpactParams:
    """Propagator impact with a linear permanent part.

    Transient: child traded at minute t with n shares shifts the mid at lag l by
    (Y / 2) * sigma_min[t] * (n / volume[t])**delta * (1 + l)**(-beta_kernel) * mid[t].
    With delta = beta_kernel = 0.5 a constant-rate metaorder peaks at about
    Y * sigma_day * sqrt(Q / V_day): the square-root law (Toth et al. 2011), on the
    Gatheral (2010) no-arbitrage boundary delta + beta_kernel = 1.
    Permanent: gamma_perm * sigma_day * (n / V_day) * mid[t], never decays (linear in Q,
    as in Almgren et al. 2005). gamma_perm is never read by the TCA layer.

    temporary="almgren2005" swaps the propagator for a rate-dependent temporary impact
    that hits only the minute it is traded in: eta * sigma_day * (n / volume[t])**beta_temp
    * mid[t], with eta = 0.142 and beta_temp = 0.6 from Almgren et al. (2005). The
    permanent part is unchanged. Y, delta and beta_kernel are then unused.
    """

    Y: float = 0.5
    delta: float = 0.5
    beta_kernel: float = 0.5
    gamma_perm: float = 0.3
    temporary: str = "propagator"
    eta: float = 0.142
    beta_temp: float = 0.6


@dataclass(frozen=True)
class FillParams:
    passive_frac: float = 0.5
    p_fill_adverse: float = 0.9
    p_fill_benign: float = 0.3
    max_participation: float = 0.5


TEMPORARY_MODELS = ("propagator", "almgren2005")


def _validate_impact_params(ip: ImpactParams) -> None:
    if ip.temporary not in TEMPORARY_MODELS:
        raise ValueError(f"unknown temporary {ip.temporary!r}; expected one of {TEMPORARY_MODELS}")
    if not ip.eta >= 0:
        raise ValueError("eta must be >= 0")
    if not ip.beta_temp > 0:
        raise ValueError("beta_temp must be > 0")


def _validate_fill_params(fp: FillParams) -> None:
    if not 0 <= fp.passive_frac <= 1:
        raise ValueError("passive_frac must be in [0, 1]")
    if not 0 <= fp.p_fill_adverse <= 1:
        raise ValueError("p_fill_adverse must be in [0, 1]")
    if not 0 <= fp.p_fill_benign <= 1:
        raise ValueError("p_fill_benign must be in [0, 1]")
    if not 0 < fp.max_participation <= 1:
        raise ValueError("max_participation must be in (0, 1]")


def _window(order: pd.Series, n: int) -> tuple[int, int]:
    t0, t1 = int(order["arrival_minute"]), int(order["end_minute"])
    if not 0 <= t0 < t1 <= n:
        raise ValueError(f"invalid trading window [{t0}, {t1}) for {n} bars")
    return t0, t1


def _integer_split(weights: np.ndarray, total: int) -> np.ndarray:
    """Integer shares per slot proportional to weights, summing exactly to total."""
    cum = np.round(np.cumsum(weights) / weights.sum() * total)
    return np.diff(np.concatenate([[0.0], cum])).astype(np.int64)


def schedule(bars: pd.DataFrame, order: pd.Series) -> np.ndarray:
    """Integer shares targeted per minute; zero outside [arrival, end).

    POV targets at most participation x volume per minute and may finish short of qty.
    """
    n = len(bars)
    t0, t1 = _window(order, n)
    qty = int(order["qty"])
    algo = order["algo"]
    vol = bars["volume"].to_numpy()[t0:t1]
    target = np.zeros(n, dtype=np.int64)
    if algo == "VWAP":
        target[t0:t1] = _integer_split(vol, qty)
    elif algo == "TWAP":
        target[t0:t1] = _integer_split(np.ones(t1 - t0), qty)
    elif algo == "POV":
        per_min = np.floor(float(order["participation"]) * vol)
        target[t0:t1] = np.diff(np.concatenate([[0.0], np.minimum(np.cumsum(per_min), qty)]))
    else:
        raise ValueError(f"unknown algo {algo!r}; expected one of {ALGOS}")
    return target


def impact_kernel(lags: np.ndarray, p: ImpactParams) -> np.ndarray:
    """Transient kernel G(l) = (1 + l)**(-beta_kernel); G(0) == 1."""
    return (1.0 + np.asarray(lags, dtype=float)) ** (-p.beta_kernel)


def execute(rng: np.random.Generator, bars: pd.DataFrame, order: pd.Series,
            impact: ImpactParams | None = None,
            fills: FillParams | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the order through the day. Returns (fills_df, bars_out); see SPEC 3.3."""
    ip = impact or ImpactParams()
    fp = fills or FillParams()
    _validate_impact_params(ip)
    _validate_fill_params(fp)
    n = len(bars)
    t0, t1 = _window(order, n)
    side = int(order["side"])
    target = schedule(bars, order)
    mid = bars["mid"].to_numpy()
    vol = bars["volume"].to_numpy()
    spread = bars["spread"].to_numpy()
    sigma = bars["sigma_min"].to_numpy()
    sigma_day = float(np.sqrt((sigma**2).sum()))
    v_day = float(vol.sum())
    shift = np.zeros(n)
    rows = []
    pending = 0
    for t in range(t0, t1):
        pending += int(target[t])
        if pending <= 0:
            continue
        passive_q = int(round(fp.passive_frac * pending))
        aggressive_q = min(pending - passive_q, int(np.floor(fp.max_participation * vol[t])))
        if passive_q > 0:
            next_move = side * (mid[t + 1] - mid[t]) if t + 1 < n else 0.0
            p_fill = fp.p_fill_adverse if next_move < 0 else fp.p_fill_benign
            if rng.uniform() >= p_fill:
                passive_q = 0
        traded = passive_q + aggressive_q
        if traded <= 0:
            continue
        permanent = ip.gamma_perm * sigma_day * traded / v_day
        if ip.temporary == "propagator":
            transient = 0.5 * ip.Y * sigma[t] * (traded / vol[t]) ** ip.delta
            shift[t:] += side * mid[t] * (transient * impact_kernel(np.arange(n - t), ip)
                                          + permanent)
        else:
            shift[t:] += side * mid[t] * permanent
            shift[t] += side * mid[t] * ip.eta * sigma_day * (traded / vol[t]) ** ip.beta_temp
        mid_aff = mid[t] + shift[t]
        if passive_q > 0:
            rows.append((t, passive_q, mid_aff - side * spread[t] / 2, True, mid_aff, mid[t]))
        if aggressive_q > 0:
            rows.append((t, aggressive_q, mid_aff + side * spread[t] / 2, False, mid_aff, mid[t]))
        pending -= traded
    fills_df = pd.DataFrame(rows, columns=FILL_COLUMNS)
    fills_df = fills_df.astype({"minute": np.int64, "qty": np.int64, "passive": bool})
    out = bars.copy()
    out["impact_shift"] = shift
    out["mid_affected"] = out["mid"] + shift
    return fills_df, out
