import numpy as np
import pandas as pd
import pytest

from tca_groundtruth.execution import FillParams, ImpactParams, execute, impact_kernel, schedule
from tca_groundtruth.market import simulate_day
from tca_groundtruth.orders import apply_alpha

FILL_COLUMNS = ["minute", "qty", "price", "passive", "mid_affected", "mid"]


def _order(**kw) -> pd.Series:
    base = {
        "order_id": 0, "side": 1, "qty": 20_000, "decision_minute": 10, "arrival_minute": 30,
        "end_minute": 200, "alpha_bps": 0.0, "alpha_halflife_min": 60.0, "flow": True,
        "algo": "TWAP", "participation": 0.1, "fee_per_share": 0.005,
    }
    base.update(kw)
    return pd.Series(base)


@pytest.fixture
def bars():
    return apply_alpha(simulate_day(np.random.default_rng(11)), _order())


@pytest.mark.parametrize("algo", ["VWAP", "TWAP", "POV"])
def test_schedule_window_and_total(bars, algo):
    order = _order(algo=algo)
    tgt = schedule(bars, order)
    assert tgt.shape == (390,)
    assert tgt.sum() <= order["qty"]
    assert (tgt[:30] == 0).all() and (tgt[200:] == 0).all()
    assert (tgt >= 0).all()


def test_schedule_twap_uniform_and_vwap_proportional(bars):
    twap = schedule(bars, _order(algo="TWAP"))[30:200]
    assert twap.max() - twap.min() <= 1
    assert twap.sum() == 20_000
    vwap = schedule(bars, _order(algo="VWAP"))[30:200]
    vol = bars["volume"].to_numpy()[30:200]
    expected = 20_000 * vol / vol.sum()
    assert np.abs(vwap - expected).max() <= 1
    assert vwap.sum() == 20_000


def test_schedule_pov_tracks_volume(bars):
    pov = schedule(bars, _order(algo="POV", participation=0.1, qty=10**9))[30:200]
    vol = bars["volume"].to_numpy()[30:200]
    assert np.abs(pov - 0.1 * vol).max() <= 1


def test_schedule_errors(bars):
    with pytest.raises(ValueError):
        schedule(bars, _order(algo="SNIPER"))
    with pytest.raises(ValueError):
        schedule(bars, _order(arrival_minute=200, end_minute=200))


def test_impact_kernel_limits():
    lags = np.arange(0, 100)
    k = impact_kernel(lags, ImpactParams(beta_kernel=0.5))
    assert k[0] == 1.0 and k[-1] == pytest.approx(0.1) and (np.diff(k) < 0).all()


def test_execute_aggressive_full_fill(bars):
    rng = np.random.default_rng(0)
    order = _order(algo="TWAP", qty=20_000)
    fp = FillParams(passive_frac=0.0, max_participation=1.0)
    fills, out = execute(rng, bars, order, fills=fp)
    assert list(fills.columns) == FILL_COLUMNS
    assert fills["qty"].sum() == 20_000
    assert (~fills["passive"]).all()
    spread = bars["spread"].to_numpy()[fills["minute"]]
    np.testing.assert_allclose(fills["price"], fills["mid_affected"] + 0.5 * spread)
    np.testing.assert_allclose(out["mid_affected"], out["mid"] + out["impact_shift"])
    # a buy pushes the affected mid up
    assert (out["impact_shift"].iloc[30:] > 0).all()
    assert (out["impact_shift"].iloc[:30] == 0).all()


def test_execute_sell_passive_prices(bars):
    order = _order(side=-1, algo="TWAP")
    fills, _ = execute(np.random.default_rng(1), bars, order,
                       fills=FillParams(passive_frac=1.0, p_fill_adverse=1.0, p_fill_benign=1.0))
    assert (fills["passive"]).all()
    spread = bars["spread"].to_numpy()[fills["minute"]]
    np.testing.assert_allclose(fills["price"], fills["mid_affected"] + 0.5 * spread)
    assert fills["qty"].sum() <= 20_000


def test_execute_partial_fill_when_window_too_short(bars):
    order = _order(algo="POV", participation=0.05, arrival_minute=30, end_minute=35, qty=10**6)
    fills, _ = execute(np.random.default_rng(2), bars, order)
    assert 0 < fills["qty"].sum() < 10**6


def test_execute_transient_decays_permanent_does_not(bars):
    order = _order(algo="TWAP", arrival_minute=30, end_minute=60)
    fp = FillParams(passive_frac=0.0, max_participation=1.0)
    _, transient = execute(np.random.default_rng(0), bars, order, ImpactParams(gamma_perm=0.0), fp)
    _, permanent = execute(np.random.default_rng(0), bars, order,
                           ImpactParams(Y=0.0, gamma_perm=0.3), fp)
    s_t = transient["impact_shift"].to_numpy()
    s_p = permanent["impact_shift"].to_numpy()
    assert s_t[-1] < 0.5 * s_t.max()
    assert (np.diff(s_p) >= -1e-12).all()
    assert abs(s_p[-1] - s_p.max()) < 1e-12
    # linear permanent impact: gamma * sigma_day * Q / V_day, in price units at the fill mid
    sigma_day = np.sqrt((bars["sigma_min"] ** 2).sum())
    expect = 0.3 * sigma_day * 20_000 / bars["volume"].sum() * bars["mid"].iloc[30:60].mean()
    assert abs(s_p[-1] / expect - 1) < 0.02


def test_execute_metaorder_peak_matches_square_root_law(bars):
    """Constant-rate order, no permanent part: peak impact ~ Y * sigma_day * sqrt(Q / V_day)."""
    order = _order(algo="POV", participation=0.1, qty=10**9, arrival_minute=30, end_minute=330)
    fp = FillParams(passive_frac=0.0, max_participation=1.0)
    fills, out = execute(np.random.default_rng(0), bars, order, ImpactParams(gamma_perm=0.0), fp)
    q = fills["qty"].sum()
    sigma_day = np.sqrt((bars["sigma_min"] ** 2).sum())
    expect_bps = 0.5 * sigma_day * np.sqrt(q / bars["volume"].sum()) * 1e4
    t_last = fills["minute"].max()
    peak_bps = out["impact_shift"].iloc[t_last] / out["mid"].iloc[t_last] * 1e4
    assert abs(peak_bps / expect_bps - 1) < 0.25


def test_execute_no_impact_when_Y_zero(bars):
    fills, out = execute(np.random.default_rng(0), bars, _order(),
                         ImpactParams(Y=0.0, gamma_perm=0.0))
    assert (out["impact_shift"] == 0).all()
    assert len(fills) > 0


def test_execute_rejects_bad_fill_params(bars):
    rng = np.random.default_rng(0)
    order = _order()
    for fp in (FillParams(passive_frac=1.5), FillParams(passive_frac=-0.1),
               FillParams(p_fill_adverse=1.2), FillParams(p_fill_benign=-0.1),
               FillParams(max_participation=0.0), FillParams(max_participation=1.5)):
        with pytest.raises(ValueError):
            execute(rng, bars, order, fills=fp)


def test_execute_almgren2005_temporary_hits_only_the_traded_minute(bars):
    """Rate-dependent temporary impact: eta * sigma_day * (n / v)**0.6 at the fill minute only."""
    order = _order(algo="TWAP", arrival_minute=30, end_minute=60)
    fp = FillParams(passive_frac=0.0, max_participation=1.0)
    ip = ImpactParams(temporary="almgren2005", gamma_perm=0.0)
    fills, out = execute(np.random.default_rng(0), bars, order, ip, fp)
    shift = out["impact_shift"].to_numpy()
    traded = np.zeros(len(bars), dtype=bool)
    traded[fills["minute"].to_numpy()] = True
    assert (shift[~traded] == 0).all()
    sigma_day = np.sqrt((bars["sigma_min"] ** 2).sum())
    t = int(fills["minute"].iloc[0])
    n = float(fills["qty"].iloc[0])
    expect = 0.142 * sigma_day * (n / bars["volume"].iloc[t]) ** 0.6 * bars["mid"].iloc[t]
    assert abs(shift[t] / expect - 1) < 1e-9
    spread = bars["spread"].to_numpy()[fills["minute"].to_numpy()]
    assert np.allclose(fills["price"], fills["mid_affected"] + spread / 2)


def test_execute_default_impact_is_the_propagator(bars):
    order = _order(algo="POV", participation=0.1)
    a = execute(np.random.default_rng(3), bars, order)
    b = execute(np.random.default_rng(3), bars, order, ImpactParams(temporary="propagator"))
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])


def test_execute_rejects_bad_impact_params(bars):
    for ip in (ImpactParams(temporary="linear"), ImpactParams(temporary="almgren2005", eta=-0.1),
               ImpactParams(temporary="almgren2005", beta_temp=0.0)):
        with pytest.raises(ValueError):
            execute(np.random.default_rng(0), bars, _order(), ip)
