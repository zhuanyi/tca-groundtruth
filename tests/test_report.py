import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tca_groundtruth.market import MarketParams, simulate_universe
from tca_groundtruth.report import (
    ALMGREN_2005,
    basket_frontier_summary,
    basket_frontier_table,
    chart_basket_frontier,
    delay_attribution_summary,
    delay_attribution_table,
    delay_slopes,
    make_report,
    orders_needed,
    power_table,
    run_orders,
)


def test_run_orders_schema():
    df = run_orders(seed=3, n_orders=4)
    assert len(df) == 4
    for col in ("true_total", "obs_total", "rev_total", "slip_arrival", "post_30", "is_adj",
                "fill_rate", "algo"):
        assert col in df.columns
    assert (df["fill_rate"].between(0, 1)).all()
    assert (df["true_total"] - df["obs_total"]).abs().max() < 1e-9


def test_orders_needed_scales_with_variance():
    assert orders_needed(10.0, 2.0) / orders_needed(5.0, 2.0) == 4.0
    assert orders_needed(10.0, 2.0) / orders_needed(10.0, 4.0) == 4.0


def test_make_report_smoke(tmp_path):
    paths = make_report(tmp_path, seed=1, quick=True)
    assert len(paths) >= 3
    pngs = [p for p in paths if p.suffix == ".png"]
    assert len(pngs) >= 3
    for p in pngs:
        assert p.exists()
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    summary = pd.read_csv(tmp_path / "summary.csv")
    assert len(summary) > 0


def test_power_table_accepts_market_params():
    tab = power_table(3, n_orders=20, n_boot=5, market=MarketParams(daily_volume=2e6))
    assert len(tab) == 3
    assert "market + difficulty adjusted" in tab.index


def test_delay_attribution_components_and_fallback():
    tab = delay_attribution_table(5, n_orders=60)
    parts = tab["delay"] + tab["trading"] + tab["opportunity"] + tab["fees"]
    assert (parts - tab["total"]).abs().max() < 1e-9
    assert (tab["fallback_delay"] == 0.0).all()
    flow = tab[tab["flow"]]
    assert len(flow) > 10
    assert (flow["alpha_delay"] == 0.0).all()
    # flow delay cost is pure market noise: mean within 3 standard errors of zero
    assert abs(flow["delay"].mean()) < 3 * flow["delay"].sem()
    alpha = tab[~tab["flow"]]
    assert (alpha["alpha_delay"] >= 0).all()
    assert (alpha.loc[alpha["delay_min"] > 0, "alpha_delay"] > 0).all()
    # the fallback stamp drops exactly the delay cost (up to the P_d -> P_0 rescaling)
    assert (tab["err_fallback"] + tab["delay"]).abs().max() < 0.05 * tab["delay"].abs().max()


def test_basket_frontier_cost_risk_tradeoff():
    tab = basket_frontier_table(7, rates=(0.02, 0.10, 0.30), n_runs=12, n_names=6)
    summ = basket_frontier_summary(tab)
    # faster execution leaves less of the basket exposed: residual risk falls, fills rise
    assert summ["risk"].is_monotonic_decreasing
    assert summ["fill_rate"].is_monotonic_increasing
    assert (summ["risk_adj"] <= summ["risk"]).all()
    assert (summ["n_runs"] == 12).all()
    # with delta + beta_kernel = 1 a completed order's impact barely depends on its rate,
    # so expected cost moves far less than risk across the speed grid
    assert summ["cost"].max() - summ["cost"].min() < 0.1 * (summ["risk"].max() - summ["risk"].min())
    # cost is the shortfall net of the no-trade move: the remainder averages out to market noise
    assert (summ["cost_se"] < 0.2).all()


def test_delay_true_series_zero_for_flow_and_rising_for_alpha():
    tab = delay_attribution_table(5, n_orders=600)
    summ = delay_attribution_summary(tab)
    # the true delay cost is the alpha realised during the delay: none on flow orders
    assert (tab.loc[tab["flow"], "alpha_delay"] == 0.0).all()
    assert (summ["alpha_delay_flow"] == 0.0).all()
    # and on informed orders it grows with the delay, bucket by bucket
    assert summ["alpha_delay_alpha"].is_monotonic_increasing
    assert summ["alpha_delay_alpha"].iloc[-1] > 3 * summ["alpha_delay_alpha"].iloc[0]
    sl = delay_slopes(tab)
    assert sl.loc["alpha", "true"] > 0 and sl.loc["flow", "true"] == 0.0
    # the measured slope is unbiased for the true one: inside 3 standard errors
    for name in ("alpha", "flow"):
        gap = abs(sl.loc[name, "measured"] - sl.loc[name, "true"])
        assert gap < 3 * sl.loc[name, "measured_se"]
    # the trader's noise-free cost does not depend on the delay
    assert (tab["spread_impact"] > 0).all()
    res = stats.linregress(tab["delay_min"], tab["spread_impact"])
    assert abs(res.slope) < 3 * res.stderr


def test_basket_rate_dependent_impact_cost_rises_with_speed():
    rates = (0.02, 0.10, 0.30)
    base = basket_frontier_summary(basket_frontier_table(7, rates=rates, n_runs=4, n_names=6))
    ac = basket_frontier_summary(basket_frontier_table(7, rates=rates, n_runs=4, n_names=6,
                                                       impact=ALMGREN_2005))
    # under Almgren et al. (2005) temporary impact, speed costs: a cost-risk trade-off
    assert ac["cost"].is_monotonic_increasing
    assert ac["cost"].iloc[-1] > 2 * ac["cost"].iloc[0]
    # same schedules and days: risk falls with speed under either impact model
    assert ac["risk"].is_monotonic_decreasing
    assert np.allclose(ac["fill_rate"], base["fill_rate"], atol=0.01)


def test_chart_basket_frontier_rejects_impact(tmp_path):
    with pytest.raises(TypeError):
        chart_basket_frontier(7, tmp_path / "d.png", rates=(0.1,), n_runs=1, n_names=2,
                              impact=ALMGREN_2005)


def test_simulate_universe_fixed_betas():
    betas = np.array([0.8, 1.2, 1.0])
    a = simulate_universe(np.random.default_rng(1), 3, betas=betas)
    b = simulate_universe(np.random.default_rng(2), 3, betas=betas)
    assert np.allclose(a.groupby("name")["beta"].first().to_numpy(), betas)
    assert np.allclose(b.groupby("name")["beta"].first().to_numpy(), betas)
    with pytest.raises(ValueError):
        simulate_universe(np.random.default_rng(1), 3, betas=np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        simulate_universe(np.random.default_rng(1), 3, betas=np.array([1.0, np.nan, 1.0]))
    with pytest.raises(ValueError):
        simulate_universe(np.random.default_rng(1), 3, betas=np.array([1.0, 2.0, 1.0]))
    # beta_range is unused, and not validated, when betas are given
    c = simulate_universe(np.random.default_rng(1), 3, betas=betas, beta_range=(1.4, 0.6))
    assert np.allclose(c["mid"].to_numpy(), a["mid"].to_numpy())

