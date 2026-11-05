import numpy as np
import pandas as pd
import pytest

from tca_groundtruth.market import simulate_day
from tca_groundtruth.orders import OrderParams, apply_alpha, generate_orders

ORDER_COLUMNS = [
    "order_id", "side", "qty", "decision_minute", "arrival_minute", "end_minute",
    "alpha_bps", "alpha_halflife_min", "flow", "algo", "participation", "fee_per_share",
]


def test_generate_orders_schema_and_invariants():
    orders = generate_orders(np.random.default_rng(0), 200, 1e6)
    assert list(orders.columns) == ORDER_COLUMNS
    assert len(orders) == 200
    assert orders["order_id"].tolist() == list(range(200))
    assert orders["side"].isin([1, -1]).all()
    assert (orders["qty"] >= 1).all()
    assert (orders["arrival_minute"] < orders["end_minute"]).all()
    assert (orders["arrival_minute"] >= orders["decision_minute"]).all()
    assert (orders["end_minute"] <= 389).all()
    assert (orders["flow"] == (orders["alpha_bps"] == 0)).all()
    assert abs(orders["flow"].mean() - 0.3) < 0.15
    assert orders["algo"].isin(["VWAP", "TWAP", "POV"]).all()


def test_generate_orders_rejects_bad_input():
    with pytest.raises(ValueError):
        generate_orders(np.random.default_rng(0), 0, 1e6)
    with pytest.raises(ValueError):
        generate_orders(np.random.default_rng(0), 5, 1e6, OrderParams(delay_min=(10, 5)))
    with pytest.raises(ValueError):
        generate_orders(np.random.default_rng(0), 5, 1e6, n_minutes=2)


def test_generate_orders_rejects_out_of_range_params():
    cases = [
        OrderParams(qty_pct_adv=(0.0, 0.1)),
        OrderParams(participation=(0.1, 1.5)),
        OrderParams(participation=(0.0, 0.5)),
        OrderParams(duration_min=(0, 30)),
        OrderParams(alpha_halflife_min=(0.0, 120.0)),
        OrderParams(decision_min=(-5, 200)),
        OrderParams(delay_min=(-5, 60)),
        OrderParams(fee_per_share=-0.001),
    ]
    for params in cases:
        with pytest.raises(ValueError):
            generate_orders(np.random.default_rng(0), 5, 1e6, params)


def test_generate_orders_window_invariant_holds_late():
    orders = generate_orders(
        np.random.default_rng(1), 500, 1e6,
        OrderParams(decision_min=(380, 400), delay_min=(0, 60), duration_min=(1, 5)),
        n_minutes=390,
    )
    decision = orders["decision_minute"].to_numpy()
    arrival = orders["arrival_minute"].to_numpy()
    end = orders["end_minute"].to_numpy()
    assert (decision >= 0).all()
    assert (decision <= arrival).all()
    assert (arrival < end).all()
    assert (end <= 390 - 1).all()


def test_apply_alpha_shape_and_sign():
    rng = np.random.default_rng(3)
    bars = simulate_day(rng)
    order = pd.Series({
        "side": -1, "decision_minute": 20, "alpha_bps": 20.0, "alpha_halflife_min": 10.0,
    })
    out = apply_alpha(bars, order)
    assert "alpha_shift" in out and "mid_base" in out
    assert (out["alpha_shift"].iloc[:20] == 0).all()
    assert (out["alpha_shift"].iloc[21:] < 0).all()
    np.testing.assert_allclose(out["mid"], out["mid_base"] + out["alpha_shift"])
    target = -1 * 20.0 / 1e4 * bars["mid"].iloc[20]
    assert abs(out["alpha_shift"].iloc[-1] / target - 1) < 0.01
    # input untouched
    assert "alpha_shift" not in bars


def test_apply_alpha_flow_order_is_identity():
    bars = simulate_day(np.random.default_rng(4))
    order = pd.Series({"side": 1, "decision_minute": 5, "alpha_bps": 0.0,
                       "alpha_halflife_min": 60.0})
    out = apply_alpha(bars, order)
    assert (out["alpha_shift"] == 0).all()
    np.testing.assert_array_equal(out["mid"], bars["mid"])


def test_apply_alpha_anchors_on_the_minute_column():
    bars = simulate_day(np.random.default_rng(5))
    order = pd.Series({"side": 1, "decision_minute": 50, "alpha_bps": 20.0,
                       "alpha_halflife_min": 60.0})
    with pytest.raises(ValueError):
        apply_alpha(bars.iloc[100:], order)
    order = order.copy()
    order["decision_minute"] = 150
    out = apply_alpha(bars.iloc[100:], order)
    assert out.loc[out["minute"] == 150, "alpha_shift"].iloc[0] == 0.0
    anchor = bars.loc[bars["minute"] == 150, "mid"].iloc[0]
    frac = 1.0 - np.exp(-np.log(2.0) / 60.0)
    expect = 20.0 / 1e4 * anchor * frac
    assert np.isclose(out.loc[out["minute"] == 151, "alpha_shift"].iloc[0], expect)
