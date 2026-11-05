"""Acceptance tests for the review fixes. The implementer must not edit this file."""

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from tca_groundtruth import report
from tca_groundtruth.execution import execute
from tca_groundtruth.market import MarketParams, simulate_day
from tca_groundtruth.orders import apply_alpha
from tca_groundtruth.tca import benchmarks, shortfall_true, tca_row

README = Path(__file__).resolve().parents[1] / "README.md"


def _order(**kw):
    base = {"order_id": 0, "side": 1, "qty": 1000, "decision_minute": 5, "arrival_minute": 5,
            "end_minute": 100, "alpha_bps": 0.0, "alpha_halflife_min": 60.0, "flow": True,
            "algo": "POV", "participation": 0.1, "fee_per_share": 0.0}
    base.update(kw)
    return pd.Series(base)


# --- M1: P_0 is the no-trade mid at arrival -------------------------------------------------

def test_arrival_price_excludes_own_first_child_impact_hand_built():
    # One fill at minute 5: mid 100, own impact 0.02, half spread 0.01.
    bars = pd.DataFrame({
        "mid": [100.0] * 10, "mid_affected": [100.0] * 5 + [100.02] * 5,
        "volume": [1000.0] * 10, "index_mid": [100.0] * 10})
    fills = pd.DataFrame({"minute": [5], "qty": [1000], "price": [100.03],
                          "mid": [100.0], "mid_affected": [100.02]})
    out = shortfall_true(_order(), bars, fills)
    assert out["delay"] == pytest.approx(0.0, abs=1e-9)
    assert out["timing"] == pytest.approx(0.0, abs=1e-9)
    assert out["spread"] == pytest.approx(1.0)      # 1000 * 0.01 / (1000 * 100) * 1e4
    assert out["impact"] == pytest.approx(2.0)      # 1000 * 0.02 / (1000 * 100) * 1e4
    assert out["total"] == pytest.approx(3.0)
    bench = benchmarks(_order(), bars, fills)
    assert bench["arrival"] == pytest.approx(100.0)


def test_delay_is_zero_with_impact_on_when_arrival_equals_decision():
    rng = np.random.default_rng(11)
    market = MarketParams()
    for _ in range(5):
        order = _order(qty=60_000, participation=0.1, decision_minute=20, arrival_minute=20)
        bars = apply_alpha(simulate_day(rng, market), order)
        fills, out = execute(rng, bars, order)
        assert len(fills) and fills["minute"].min() == 20
        row = tca_row(order, out, fills, beta=market.beta)
        assert row["true_delay"] == pytest.approx(0.0, abs=1e-9)


# --- M2: VWAP variants -----------------------------------------------------------------------

def test_interval_vwaps_hand_built():
    bars = pd.DataFrame({
        "mid": [100.0, 101.0, 102.0], "mid_affected": [100.5, 101.5, 102.5],
        "volume": [100.0, 300.0, 500.0], "index_mid": [100.0] * 3})
    fills = pd.DataFrame({"minute": [1], "qty": [100], "price": [101.6],
                          "mid": [101.0], "mid_affected": [101.5]})
    order = _order(qty=100, decision_minute=0, arrival_minute=0)
    v = report_tca().interval_vwaps(order, bars, fills)
    assert list(v.index) == ["incl_own", "market_only", "counterfactual"]
    assert v["incl_own"] == pytest.approx(50660 / 500)        # 101.32
    assert v["market_only"] == pytest.approx(40500 / 400)     # 101.25
    assert v["counterfactual"] == pytest.approx(40300 / 400)  # 100.75
    assert benchmarks(order, bars, fills)["interval_vwap"] == pytest.approx(v["incl_own"])
    assert np.isnan(report_tca().interval_vwaps(order, bars, fills.iloc[0:0])).all()


def report_tca():
    from tca_groundtruth import tca
    return tca


def test_vwap_trap_table_has_market_only_and_counterfactual_columns():
    tab = report.vwap_trap_table(7, rates=(0.05, 0.30), n_per_rate=15)
    for c in ("rate", "slip_vwap", "slip_arrival", "true_impact", "slip_vwap_mkt",
              "slip_vwap_cf"):
        assert c in tab.columns
    hi = tab[tab["rate"] == 0.30]
    # own fills move the benchmark by well under 1.5 bps; they are not the cause of the trap
    assert abs((hi["slip_vwap"] - hi["slip_vwap_mkt"]).mean()) < 1.5
    # the counterfactual benchmark sees the cost
    assert (hi["slip_vwap_cf"] - hi["slip_vwap_mkt"]).mean() > 0.5 * hi["true_impact"].mean()


def test_vwap_trap_does_not_build_billion_share_tca_row(monkeypatch):
    calls = []
    real = report.tca_row
    monkeypatch.setattr(report, "tca_row", lambda *a, **k: calls.append(1) or real(*a, **k))
    report.vwap_trap_table(7, rates=(0.1,), n_per_rate=2)
    assert calls == []


def test_chart_a_axis_label_and_lines(monkeypatch):
    figs = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: figs.append(self))
    report.chart_vwap_trap(1, "unused.png", rates=(0.05, 0.3), n_per_rate=3)
    ax = figs[0].axes[0]
    assert "own shares / market volume" in ax.get_xlabel()
    labels = " ".join(t.get_text() for t in ax.get_legend().get_texts())
    assert "market-only" in labels and "counterfactual" in labels


# --- M3 / m2: chart (c) --------------------------------------------------------------------

def test_waterfall_default_sample_size():
    assert inspect.signature(report.chart_waterfall).parameters["n_orders"].default == 1500


def test_waterfall_table_paired_control_and_standard_errors():
    tab = report.waterfall_table(5, n_orders=150)
    assert list(tab.index) == ["delay", "spread", "impact", "timing", "opportunity", "fees"]
    cols = {f"{m}_{k}" for m in ("true", "residual", "reversion")
            for k in ("informed", "informed_se", "control", "control_se", "alpha_effect",
                      "alpha_effect_se")}
    assert cols <= set(tab.columns)
    # control = same orders and days with alpha switched off; true impact differs only through
    # the second-order coupling of impact to the alpha-shifted mid (prototype: < 0.01 bps mean)
    assert tab.loc["impact", "true_informed"] == pytest.approx(
        tab.loc["impact", "true_control"], abs=0.1)
    assert abs(tab.loc["impact", "true_alpha_effect"]) < 0.1
    assert (tab[[c for c in tab.columns if c.endswith("_se")]] >= 0).all().all()
    assert tab.loc["impact", "true_informed_se"] > 0
    # structural understatement of the reversion estimator without any alpha
    assert tab.loc["impact", "reversion_control"] < 0.8 * tab.loc["impact", "true_control"]
    # alpha raises the residual-method impact (timing leaks into it)
    assert tab.loc["impact", "residual_alpha_effect"] > 0


def test_chart_waterfall_returns_table(monkeypatch):
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: None)
    tab = report.chart_waterfall(5, "unused.png", n_orders=20)
    assert "true_informed" in tab.columns


# --- m4: bootstrap CIs on chart (d) risk ratios ----------------------------------------------

def test_basket_summary_has_risk_ratio_cis():
    tab = report.basket_frontier_table(3, rates=(0.02, 0.30), n_runs=12, n_names=5)
    s = report.basket_frontier_summary(tab, n_boot=200, seed=3)
    for c in ("risk_ratio_vs_fastest", "risk_ratio_lo", "risk_ratio_hi",
              "hedge_ratio", "hedge_ratio_lo", "hedge_ratio_hi"):
        assert c in s.columns
    slow = s.loc[0.02]
    assert slow["risk_ratio_lo"] <= slow["risk_ratio_vs_fastest"] <= slow["risk_ratio_hi"]
    assert s.loc[0.30, "risk_ratio_vs_fastest"] == pytest.approx(1.0)
    assert slow["hedge_ratio"] == pytest.approx(slow["risk_adj"] / slow["risk"])


# --- m8: summary.csv column names -----------------------------------------------------------

def test_delay_summary_pooled_columns_renamed():
    tab = report.delay_attribution_table(5, n_orders=60)
    s = report.delay_attribution_summary(tab)
    assert "err_open_sd_pooled" in s.columns and "err_open_pooled" in s.columns
    assert "err_open_sd" not in s.columns and "err_open" not in s.columns
    assert "err_open_alpha" in s.columns


# --- m3 / README wording --------------------------------------------------------------------

@pytest.mark.parametrize("stale", [
    "the benchmark includes the order's own fills and moves with it",
    "so the measured estimate is unbiased",
    "because the alpha keeps realising and suppresses the reversion",
    "expected -3.0",
])
def test_readme_has_no_stale_claims(stale):
    assert stale not in README.read_text()


def test_readme_states_slope_not_level_is_unbiased():
    text = README.read_text()
    assert "slope is unbiased" in text
