"""PM parent orders: size, side, decaying alpha, PM-to-desk delay, flow flag."""

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

ORDER_COLUMNS = [
    "order_id", "side", "qty", "decision_minute", "arrival_minute", "end_minute",
    "alpha_bps", "alpha_halflife_min", "flow", "algo", "participation", "fee_per_share",
]


@dataclass(frozen=True)
class OrderParams:
    qty_pct_adv: tuple[float, float] = (0.005, 0.10)
    alpha_bps: tuple[float, float] = (5.0, 30.0)
    alpha_halflife_min: tuple[float, float] = (30.0, 240.0)
    flow_share: float = 0.3
    delay_min: tuple[int, int] = (0, 60)
    decision_min: tuple[int, int] = (5, 200)
    duration_min: tuple[int, int] = (30, 180)
    participation: tuple[float, float] = (0.05, 0.25)
    algos: tuple[str, ...] = ("VWAP", "TWAP", "POV")
    fee_per_share: float = 0.005


def _validate(p: OrderParams) -> None:
    for f in fields(p):
        v = getattr(p, f.name)
        if isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, int | float) for x in v):
            if v[0] > v[1]:
                raise ValueError(f"{f.name} range is inverted: {v}")
    if not 0 <= p.flow_share <= 1:
        raise ValueError("flow_share must be in [0, 1]")
    if not p.qty_pct_adv[0] > 0:
        raise ValueError("qty_pct_adv must start above 0")
    if not 0 < p.participation[0] or not p.participation[1] <= 1:
        raise ValueError("participation must lie inside (0, 1]")
    if not p.alpha_halflife_min[0] > 0:
        raise ValueError("alpha_halflife_min must start above 0")
    if not p.duration_min[0] >= 1:
        raise ValueError("duration_min must start at 1 or more")
    if not p.decision_min[0] >= 0:
        raise ValueError("decision_min must start at 0 or more")
    if not p.delay_min[0] >= 0:
        raise ValueError("delay_min must start at 0 or more")
    if not p.fee_per_share >= 0:
        raise ValueError("fee_per_share must be >= 0")


def generate_orders(rng: np.random.Generator, n_orders: int, daily_volume: float,
                    params: OrderParams | None = None, n_minutes: int = 390) -> pd.DataFrame:
    """n_orders parent orders; see SPEC 3.2 for the schema.

    qty is sampled log-uniformly in qty_pct_adv x daily_volume, not uniform in level.
    """
    p = params or OrderParams()
    _validate(p)
    if n_orders < 1:
        raise ValueError("n_orders must be >= 1")
    if n_minutes < 3:
        raise ValueError("n_minutes must be >= 3")
    lo, hi = np.log(p.qty_pct_adv)
    qty = np.maximum(1, np.round(daily_volume * np.exp(rng.uniform(lo, hi, n_orders)))).astype(
        np.int64
    )
    flow = rng.uniform(size=n_orders) < p.flow_share
    alpha = np.where(flow, 0.0, rng.uniform(*p.alpha_bps, n_orders))
    decision = rng.integers(p.decision_min[0], p.decision_min[1] + 1, n_orders)
    delay = rng.integers(p.delay_min[0], p.delay_min[1] + 1, n_orders)
    arrival = decision + delay
    duration = rng.integers(p.duration_min[0], p.duration_min[1] + 1, n_orders)
    end = np.minimum(arrival + duration, n_minutes - 1)
    # keep a window of at least one minute for orders arriving at the last bar
    arrival = np.minimum(arrival, end - 1)
    decision = np.minimum(decision, arrival)
    if ((decision < 0).any() or (decision > arrival).any() or (arrival >= end).any()
            or (end > n_minutes - 1).any()):
        raise RuntimeError("window invariant violated")
    return pd.DataFrame({
        "order_id": np.arange(n_orders, dtype=np.int64),
        "side": rng.choice(np.array([1, -1], dtype=np.int64), n_orders),
        "qty": qty,
        "decision_minute": decision.astype(np.int64),
        "arrival_minute": arrival.astype(np.int64),
        "end_minute": end.astype(np.int64),
        "alpha_bps": alpha,
        "alpha_halflife_min": rng.uniform(*p.alpha_halflife_min, n_orders),
        "flow": flow,
        "algo": rng.choice(np.array(p.algos, dtype=object), n_orders),
        "participation": rng.uniform(*p.participation, n_orders),
        "fee_per_share": np.full(n_orders, p.fee_per_share),
    })


def apply_alpha(bars: pd.DataFrame, order: pd.Series) -> pd.DataFrame:
    """Add the order's decaying alpha drift to the no-trade mid from the decision minute."""
    out = bars.copy()
    t = out["minute"].to_numpy() if "minute" in out else np.arange(len(out))
    t_d = int(order["decision_minute"])
    pos = np.flatnonzero(t == t_d)
    if pos.size == 0:
        raise ValueError(f"decision_minute {t_d} not in bars")
    elapsed = np.clip(t - t_d, 0, None).astype(float)
    frac = 1.0 - np.exp(-elapsed * np.log(2.0) / float(order["alpha_halflife_min"]))
    anchor = out["mid"].to_numpy()[pos[0]]
    shift = int(order["side"]) * float(order["alpha_bps"]) / 1e4 * anchor * frac
    out["mid_base"] = out["mid"]
    out["alpha_shift"] = shift
    out["mid"] = out["mid_base"] + out["alpha_shift"]
    return out
