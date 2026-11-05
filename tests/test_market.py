import numpy as np
import pandas as pd
import pytest

from tca_groundtruth.market import MarketParams, simulate_day, simulate_universe, u_profile

DAY_COLUMNS = ["minute", "mid", "index_mid", "volume", "spread", "sigma_min"]


def test_u_profile_shape():
    p = u_profile(390, 2.0)
    assert p.shape == (390,)
    assert (p > 0).all()
    assert abs(p.mean() - 1.0) < 1e-9
    assert abs(p.max() / p.min() - 2.0) < 1e-9
    np.testing.assert_allclose(p, p[::-1])


def test_simulate_day_schema_and_values():
    bars = simulate_day(np.random.default_rng(0))
    assert list(bars.columns) == DAY_COLUMNS
    assert len(bars) == 390
    assert isinstance(bars.index, pd.RangeIndex)
    assert (bars["mid"] > 0).all()
    assert (bars["volume"] > 0).all()
    assert (np.diff(bars["spread"].to_numpy()) < 0).all()
    assert bars["index_mid"].iloc[0] == 100.0
    assert bars["minute"].tolist() == list(range(390))


def test_simulate_day_is_seeded():
    a = simulate_day(np.random.default_rng(7))
    b = simulate_day(np.random.default_rng(7))
    pd.testing.assert_frame_equal(a, b)


def test_simulate_day_rejects_bad_params():
    with pytest.raises(ValueError):
        simulate_day(np.random.default_rng(0), MarketParams(n_minutes=1))
    with pytest.raises(ValueError):
        simulate_day(np.random.default_rng(0), MarketParams(daily_vol=0))
    with pytest.raises(ValueError):
        simulate_day(np.random.default_rng(0), MarketParams(spread_open_bps=3, spread_close_bps=5))


def test_simulate_universe():
    uni = simulate_universe(np.random.default_rng(1), n_names=5)
    assert len(uni) == 5 * 390
    assert list(uni.columns) == DAY_COLUMNS + ["name", "beta"]
    pivot = uni.pivot(index="minute", columns="name", values="index_mid")
    assert (pivot.nunique(axis=1) == 1).all()
    assert uni["beta"].between(0.6, 1.4).all()
    assert uni.groupby("name")["beta"].nunique().eq(1).all()


def test_universe_rejects_beta_range_that_swamps_daily_vol():
    rng = np.random.default_rng(0)
    # 2.0 * 0.012 = 0.024 > daily_vol 0.02
    with pytest.raises(ValueError):
        simulate_universe(rng, 5, beta_range=(0.5, 2.0))
    with pytest.raises(ValueError):
        simulate_universe(rng, 5, beta_range=(1.4, 0.6))
    with pytest.raises(ValueError):
        simulate_universe(rng, 5, beta_range=(0.6,))
    df = simulate_universe(np.random.default_rng(2), 5)
    assert np.isfinite(df["mid"]).all()


def test_universe_sigma_min_matches_total_variance():
    df = simulate_universe(np.random.default_rng(3), 5)
    daily_var = MarketParams().daily_vol**2
    for _, bars in df.groupby("name"):
        assert np.isclose((bars["sigma_min"] ** 2).sum(), daily_var)


def test_simulate_day_needs_three_minutes():
    with pytest.raises(ValueError):
        simulate_day(np.random.default_rng(0), MarketParams(n_minutes=2))
    p = u_profile(3, 2.0)
    assert np.isfinite(p).all()
    assert abs(p.mean() - 1.0) < 1e-9


def test_simulate_day_rejects_negative_volume_noise():
    with pytest.raises(ValueError):
        simulate_day(np.random.default_rng(0), MarketParams(volume_noise=-0.1))
    bars = simulate_day(np.random.default_rng(0), MarketParams(volume_noise=0.0))
    assert (bars["volume"] > 0).all()
