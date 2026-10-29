import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tca_groundtruth.execution import FillParams, ImpactParams, execute
from tca_groundtruth.market import simulate_day
from tca_groundtruth.orders import apply_alpha
from tca_groundtruth.tca import (
    BENCHMARKS,
    COMPONENTS,
    benchmarks,
    market_adjustment_bps,
    post_trade_moves_bps,
    shortfall_observed,
    shortfall_true,
    slippage_bps,
    tca_row,
)


def _order(**kw) -> pd.Series:
    base = {
        "order_id": 0, "side": 1, "qty": 20_000, "decision_minute": 10, "arrival_minute": 30,
        "end_minute": 200, "alpha_bps": 0.0, "alpha_halflife_min": 60.0, "flow": True,
        "algo": "TWAP", "participation": 0.1, "fee_per_share": 0.005,
    }
    base.update(kw)
    return pd.Series(base)


def _run(seed, order, impact=None, fills=None):
    bars = apply_alpha(simulate_day(np.random.default_rng(seed)), order)
    fills_df, out = execute(np.random.default_rng(seed), bars, order, impact, fills)
    return fills_df, out


# --- property: components sum to Perold's total on every order ----------------------------

price = st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False)


@st.composite
def random_case(draw):
    n = 40
    side = draw(st.sampled_from([1, -1]))
    qty = draw(st.integers(min_value=1, max_value=100_000))
    t_d = draw(st.integers(min_value=0, max_value=10))
    t_0 = draw(st.integers(min_value=t_d, max_value=20))
    mids = np.array(draw(st.lists(price, min_size=n, max_size=n)))
    shift = np.array(draw(st.lists(st.floats(-5, 5, allow_nan=False), min_size=n, max_size=n)))
    bars = pd.DataFrame({
        "minute": np.arange(n), "mid": mids, "index_mid": np.linspace(100, 101, n),
        "volume": np.full(n, 1000.0), "spread": np.full(n, 0.05), "sigma_min": np.full(n, 1e-3),
        "impact_shift": shift, "mid_affected": mids + shift,
    })
    n_fills = draw(st.integers(min_value=0, max_value=6))
    minutes = draw(st.lists(st.integers(min_value=t_0, max_value=n - 1), min_size=n_fills,
                            max_size=n_fills))
    qtys = draw(st.lists(st.integers(min_value=1, max_value=10_000), min_size=n_fills,
                         max_size=n_fills))
    prices = draw(st.lists(price, min_size=n_fills, max_size=n_fills))
    fills = pd.DataFrame({
        "minute": np.array(minutes, dtype=np.int64), "qty": np.array(qtys, dtype=np.int64),
        "price": np.array(prices), "passive": np.zeros(n_fills, dtype=bool),
        "mid_affected": bars["mid_affected"].to_numpy()[minutes],
        "mid": bars["mid"].to_numpy()[minutes],
    })
    if fills["qty"].sum() > qty:
        fills = fills.iloc[:0]
    order = _order(side=side, qty=qty, decision_minute=t_d, arrival_minute=t_0, end_minute=n)
    return order, bars, fills


def _perold_total_bps(order, bars, fills):
    s, Q = order["side"], order["qty"]
    P_d = bars["mid"].iloc[order["decision_minute"]]
    P_c = bars["mid_affected"].iloc[-1]
    q = fills["qty"].sum()
    paid = (fills["qty"] * fills["price"]).sum()
    F = q * order["fee_per_share"]
    return (s * (paid - q * P_d) + s * (Q - q) * (P_c - P_d) + F) / (Q * P_d) * 1e4


@settings(max_examples=300, deadline=None)
@given(random_case())
def test_components_sum_to_total(case):
    order, bars, fills = case
    expected = _perold_total_bps(order, bars, fills)
    for sf in (shortfall_true(order, bars, fills),
               shortfall_observed(order, bars, fills, "naive"),
               shortfall_observed(order, bars, fills, "reversion")):
        assert list(sf.index) == list(COMPONENTS)
        assert abs(sf[list(COMPONENTS[:-1])].sum() - sf["total"]) < 1e-6
        assert abs(sf["total"] - expected) < 1e-6


# --- zero alpha, zero impact recovers fees + half spread -----------------------------------

def test_zero_alpha_zero_impact_is_fees_plus_half_spread():
    order = _order(decision_minute=30, arrival_minute=30, alpha_bps=0.0)
    fills, out = _run(5, order, ImpactParams(Y=0.0, gamma_perm=0.0),
                      FillParams(passive_frac=0.0, max_participation=1.0))
    assert fills["qty"].sum() == order["qty"]
    sf = shortfall_true(order, out, fills)
    P_d = out["mid"].iloc[30]
    half_spread_bps = (fills["qty"] * out["spread"].to_numpy()[fills["minute"]] / 2).sum() / (
        order["qty"] * P_d) * 1e4
    fees_bps = order["qty"] * order["fee_per_share"] / (order["qty"] * P_d) * 1e4
    assert sf["impact"] == 0 and sf["delay"] == 0 and sf["opportunity"] == 0
    assert abs(sf["spread"] - half_spread_bps) < 1e-6
    assert abs(sf["fees"] - fees_bps) < 1e-6
    # timing is market noise during the order; the identity still holds
    assert abs(sf["total"] - (sf["fees"] + sf["spread"] + sf["timing"])) < 1e-6


def test_true_impact_matches_simulator_shift():
    order = _order(arrival_minute=30, end_minute=100)
    fills, out = _run(6, order, fills=FillParams(passive_frac=0.0, max_participation=1.0))
    sf = shortfall_true(order, out, fills)
    P_d = out["mid"].iloc[10]
    expect = (fills["qty"] * (fills["mid_affected"] - fills["mid"])).sum() / (20_000 * P_d) * 1e4
    assert abs(sf["impact"] - expect) < 1e-9
    assert sf["impact"] > 0


# --- benchmarks, slippage, post-trade moves ------------------------------------------------

def test_benchmarks_single_fill_vwap_between_mid_and_price():
    order = _order(arrival_minute=50, end_minute=51)
    bars = apply_alpha(simulate_day(np.random.default_rng(8)), order)
    bars["impact_shift"] = 0.0
    bars["mid_affected"] = bars["mid"]
    mid = bars["mid"].iloc[50]
    px = mid + bars["spread"].iloc[50] / 2
    fills = pd.DataFrame({"minute": [50], "qty": [1000], "price": [px], "passive": [False],
                          "mid_affected": [mid], "mid": [mid]})
    b = benchmarks(order, bars, fills)
    assert list(b.index) == ["decision", "arrival", "interval_vwap", "close"]
    assert mid < b["interval_vwap"] < px
    assert b["decision"] == bars["mid"].iloc[10]
    assert b["arrival"] == mid
    slip = slippage_bps(order, fills, b)
    assert slip["arrival"] == pytest.approx((px - mid) / mid * 1e4)
    assert 0 < slip["interval_vwap"] < slip["arrival"]


def test_post_trade_moves_signs_and_clipping():
    order = _order(arrival_minute=30, end_minute=380)
    fills, out = _run(9, order)
    pt = post_trade_moves_bps(order, out, fills)
    assert list(pt.index) == ["post_5", "post_30", "post_close"]
    t_last = fills["minute"].max()
    m = out["mid_affected"].to_numpy()
    P_d = out["mid"].iloc[10]
    assert pt["post_close"] == pytest.approx((m[-1] - m[t_last]) / P_d * 1e4)
    assert pt["post_30"] == pytest.approx((m[min(t_last + 30, 389)] - m[t_last]) / P_d * 1e4)


def test_market_adjustment_sign_and_windows():
    order = _order(side=-1)
    fills, out = _run(10, order)
    idx = out["index_mid"].to_numpy()
    r_close = idx[-1] / idx[10] - 1
    # no fills: the whole intended quantity is marked decision -> close
    empty = fills.iloc[:0]
    assert market_adjustment_bps(order, out, empty, 1.2) == pytest.approx(-1.2 * r_close * 1e4)
    # with fills: executed shares carry the index move to their fill minute
    q = fills["qty"].sum()
    r_fill = (fills["qty"] * (idx[fills["minute"]] / idx[10] - 1)).sum() / q
    expect = -1.2 * (q * r_fill + (20_000 - q) * r_close) / 20_000 * 1e4
    assert market_adjustment_bps(order, out, fills, 1.2) == pytest.approx(expect)


def test_tca_row_no_fills():
    order = _order(arrival_minute=30, end_minute=31, algo="POV", participation=0.0)
    fills, out = _run(12, order)
    assert len(fills) == 0
    row = tca_row(order, out, fills, beta=1.0)
    assert row["fill_rate"] == 0
    assert np.isnan(row["slip_arrival"])
    P_d, P_c = out["mid"].iloc[10], out["mid_affected"].iloc[-1]
    assert row["true_total"] == pytest.approx((P_c - P_d) / P_d * 1e4)
    assert row["true_impact"] == 0 and row["true_spread"] == 0 and row["true_fees"] == 0


def test_tca_row_rejects_fill_outside_bars():
    order = _order()
    fills, out = _run(13, order)
    bad = fills.copy()
    bad.loc[0, "minute"] = 999
    with pytest.raises(ValueError):
        tca_row(order, out, bad, beta=1.0)


def test_shortfall_true_decision_price_override():
    order = _order(alpha_bps=20.0, flow=False)
    fills, out = _run(14, order)
    true = shortfall_true(order, out, fills)
    P_d, P_0 = out["mid"].iloc[10], out["mid_affected"].iloc[30]
    fb = shortfall_true(order, out, fills, decision_price=P_0)
    assert fb["delay"] == 0.0
    # every other component keeps its currency value; only the bps denominator changes
    for c in ("spread", "impact", "timing", "opportunity", "fees"):
        assert fb[c] * P_0 == pytest.approx(true[c] * P_d)
    assert (fb["total"] * P_0 == pytest.approx((true["total"] - true["delay"]) * P_d))
    same = shortfall_true(order, out, fills, decision_price=P_d)
    assert np.allclose(same.to_numpy(), true.to_numpy())
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            shortfall_true(order, out, fills, decision_price=bad)


# --- performance view: positive = beat the benchmark ---------------------------------------

@settings(max_examples=200, deadline=None)
@given(random_case())
def test_perf_columns_are_negated_costs(case):
    order, bars, fills = case
    row = tca_row(order, bars, fills, beta=1.0)
    pairs = {"tcost_perf_bps": "true_total"}
    pairs.update({f"{c}_perf_bps": f"true_{c}" for c in COMPONENTS[:-1]})
    pairs.update({f"{b}_perf_bps": f"slip_{b}" for b in BENCHMARKS})
    assert sorted(c for c in row.index if c.endswith("_perf_bps")) == sorted(pairs)
    for perf, cost in pairs.items():
        assert np.array_equal(row[perf], -row[cost], equal_nan=True), perf
    comps = sum(row[f"{c}_perf_bps"] for c in COMPONENTS[:-1])
    assert abs(comps - row["tcost_perf_bps"]) < 1e-6
