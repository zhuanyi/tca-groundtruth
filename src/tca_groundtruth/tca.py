"""Post-trade TCA: benchmarks, Perold implementation-shortfall split, market adjustment.

Signs: positive = cost. IS components are in bps of intended value Q * P_d;
benchmark slippages are in bps of the benchmark price.

tca_row also reports a performance view, positive = beat the benchmark: each
`*_perf_bps` column is exactly -1 times its cost-positive counterpart.
`tcost_perf_bps` = -true_total, `<component>_perf_bps` = -true_<component> and
`<bench>_perf_bps` = -slip_<bench>. Net alpha = gross alpha + tcost_perf_bps.
"""

import numpy as np
import pandas as pd

BENCHMARKS = ("decision", "arrival", "interval_vwap", "close")
COMPONENTS = ("delay", "spread", "impact", "timing", "opportunity", "fees", "total")
POST_TRADE = ("post_5", "post_30", "post_close")


def to_perf(x):
    """Cost-positive bps -> performance bps (positive = beat the benchmark)."""
    return -x


def _check_fills(bars: pd.DataFrame, fills: pd.DataFrame) -> None:
    if len(fills) and not fills["minute"].between(0, len(bars) - 1).all():
        raise ValueError("fills contain a minute outside the bars")


def _anchors(order: pd.Series, bars: pd.DataFrame) -> tuple[int, float, float, float, float]:
    """side, Q, P_d, P_0, P_c."""
    s = int(order["side"])
    Q = float(order["qty"])
    P_d = float(bars["mid"].iloc[int(order["decision_minute"])])
    P_0 = float(bars["mid_affected"].iloc[int(order["arrival_minute"])])
    P_c = float(bars["mid_affected"].iloc[-1])
    return s, Q, P_d, P_0, P_c


def _trading_terms(order, bars, fills) -> tuple[float, float, float, float, float]:
    """q, spread, impact, timing, trading (all in currency, signed, positive = cost)."""
    s, _, _, P_0, _ = _anchors(order, bars)
    q = float(fills["qty"].sum())
    qty = fills["qty"].to_numpy(dtype=float)
    spread = s * float((qty * (fills["price"] - fills["mid_affected"])).sum())
    impact = s * float((qty * (fills["mid_affected"] - fills["mid"])).sum())
    timing = s * float((qty * (fills["mid"] - P_0)).sum())
    return q, spread, impact, timing, spread + impact + timing


def _assemble(order, bars, fills, spread, impact, timing,
              decision_price: float | None = None) -> pd.Series:
    s, Q, P_d, P_0, P_c = _anchors(order, bars)
    if decision_price is not None:
        P_d = float(decision_price)
        if not P_d > 0:  # also rejects NaN
            raise ValueError("decision_price must be > 0")
    q = float(fills["qty"].sum())
    delay = s * Q * (P_0 - P_d)
    opportunity = s * (Q - q) * (P_c - P_0)
    fees = q * float(order["fee_per_share"])
    parts = np.array([delay, spread, impact, timing, opportunity, fees])
    scale = 1e4 / (Q * P_d)
    return pd.Series(np.append(parts, parts.sum()) * scale, index=list(COMPONENTS))


def shortfall_true(order: pd.Series, bars: pd.DataFrame, fills: pd.DataFrame,
                   decision_price: float | None = None) -> pd.Series:
    """Perold split with the simulator's true impact (SPEC 3.4).

    decision_price replaces P_d, the way a TCA system with a wrong or missing decision
    stamp would (P_d := P_0 is the order-creation fallback). It changes only delay and
    the bps denominator; the other components are measured from P_0 and the fills.
    """
    _check_fills(bars, fills)
    _, spread, impact, timing, _ = _trading_terms(order, bars, fills)
    return _assemble(order, bars, fills, spread, impact, timing, decision_price)


def shortfall_observed(order: pd.Series, bars: pd.DataFrame, fills: pd.DataFrame,
                       method: str = "naive") -> pd.Series:
    """Perold split as a TCA system without the counterfactual would report it.

    naive: everything in trading cost beyond spread is labelled impact.
    reversion: impact = the part of the price move that comes back, i.e. executed
    quantity times (affected mid at the last fill minus affected mid 30 minutes later);
    the rest of trading cost is timing. Alpha that keeps realising after the order
    suppresses the reversion, so this understates impact on informed orders.
    """
    _check_fills(bars, fills)
    s = int(order["side"])
    _, spread, _, _, trading = _trading_terms(order, bars, fills)
    if method == "naive":
        impact = trading - spread
    elif method == "reversion":
        if len(fills) == 0:
            impact = 0.0
        else:
            t_last = int(fills["minute"].max())
            t_post = min(t_last + 30, len(bars) - 1)
            m = bars["mid_affected"].to_numpy()
            impact = s * float(fills["qty"].sum()) * (m[t_last] - m[t_post])
    else:
        raise ValueError(f"unknown method {method!r}")
    return _assemble(order, bars, fills, spread, impact, trading - spread - impact)


def benchmarks(order: pd.Series, bars: pd.DataFrame, fills: pd.DataFrame) -> pd.Series:
    """Decision mid, arrival affected mid, interval VWAP including own fills, close."""
    _check_fills(bars, fills)
    _, _, P_d, P_0, P_c = _anchors(order, bars)
    if len(fills) == 0:
        vwap = np.nan
    else:
        t0, t1 = int(order["arrival_minute"]), int(fills["minute"].max())
        win = bars.iloc[t0:t1 + 1]
        mkt_value = float((win["volume"] * win["mid_affected"]).sum())
        mkt_vol = float(win["volume"].sum())
        own_value = float((fills["qty"] * fills["price"]).sum())
        vwap = (mkt_value + own_value) / (mkt_vol + float(fills["qty"].sum()))
    return pd.Series([P_d, P_0, vwap, P_c], index=list(BENCHMARKS))


def slippage_bps(order: pd.Series, fills: pd.DataFrame, bench: pd.Series) -> pd.Series:
    if len(fills) == 0:
        return pd.Series(np.nan, index=bench.index)
    p_bar = float((fills["qty"] * fills["price"]).sum() / fills["qty"].sum())
    return int(order["side"]) * (p_bar - bench) / bench * 1e4


def post_trade_moves_bps(order: pd.Series, bars: pd.DataFrame, fills: pd.DataFrame,
                         horizons: tuple[int, ...] = (5, 30)) -> pd.Series:
    """Signed move of the affected mid after the last fill, bps of P_d.

    Positive = price kept moving in the trade direction (information);
    negative = reverted (impact). NaN if no fills.
    """
    _check_fills(bars, fills)
    idx = [f"post_{h}" for h in horizons] + ["post_close"]
    if len(fills) == 0:
        return pd.Series(np.nan, index=idx)
    s, _, P_d, _, _ = _anchors(order, bars)
    m = bars["mid_affected"].to_numpy()
    t_last = int(fills["minute"].max())
    targets = [min(t_last + h, len(bars) - 1) for h in horizons] + [len(bars) - 1]
    return pd.Series([s * (m[t] - m[t_last]) / P_d * 1e4 for t in targets], index=idx)


def market_adjustment_bps(order: pd.Series, bars: pd.DataFrame, fills: pd.DataFrame,
                          beta: float) -> float:
    """Beta-implied move of the intended position, bps of Q * P_d.

    Each executed share carries the index return from decision to its fill minute; each
    unfilled share carries the index return from decision to the last bar. This is
    Perold's total with the index-implied price in place of the stock price, so the
    adjustment matches the windows of the components it is removed from.
    """
    s, Q = int(order["side"]), float(order["qty"])
    idx = bars["index_mid"].to_numpy()
    idx_d = idx[int(order["decision_minute"])]
    q = float(fills["qty"].sum())
    executed = float((fills["qty"] * (idx[fills["minute"].to_numpy()] / idx_d - 1.0)).sum())
    unfilled = (Q - q) * (idx[-1] / idx_d - 1.0)
    return s * beta * (executed + unfilled) / Q * 1e4


def tca_row(order: pd.Series, bars: pd.DataFrame, fills: pd.DataFrame, beta: float) -> pd.Series:
    """One row of TCA output per order (SPEC 3.4)."""
    _check_fills(bars, fills)
    true = shortfall_true(order, bars, fills).add_prefix("true_")
    obs = shortfall_observed(order, bars, fills, "naive").add_prefix("obs_")
    rev = shortfall_observed(order, bars, fills, "reversion").add_prefix("rev_")
    bench = benchmarks(order, bars, fills)
    slip = slippage_bps(order, fills, bench).add_prefix("slip_")
    post = post_trade_moves_bps(order, bars, fills)
    q = float(fills["qty"].sum())
    if len(fills):
        win = bars.iloc[int(order["arrival_minute"]):int(fills["minute"].max()) + 1]
        part = q / float(win["volume"].sum())
    else:
        part = 0.0
    extra = pd.Series({
        "q": q,
        "fill_rate": q / float(order["qty"]),
        "participation_realised": part,
        "market_adj": market_adjustment_bps(order, bars, fills, beta),
    })
    extra["is_adj"] = true["true_total"] - extra["market_adj"]
    perf = to_perf(pd.concat([
        pd.Series({"tcost": true["true_total"]}),
        true.drop("true_total").rename(lambda c: c.removeprefix("true_")),
        slip.rename(lambda c: c.removeprefix("slip_")),
    ])).add_suffix("_perf_bps")
    return pd.concat([true, obs, rev, slip, post, extra, perf])
