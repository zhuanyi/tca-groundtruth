"""Golden test on summary.csv and README-to-summary.csv trace. The implementer must not edit it.

tests/golden/summary_quick.csv is generated once, by the fixed code, in the last commit:
  .venv/bin/python -m tca_groundtruth.report --quick --out /tmp/q
  mkdir -p tests/golden && cp /tmp/q/summary.csv tests/golden/summary_quick.csv
"""

from pathlib import Path

import pandas as pd
import pytest

from tca_groundtruth.report import DEFAULT_SEED, make_report

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "summary_quick.csv"


@pytest.fixture(scope="module")
def quick_summary(tmp_path_factory):
    out = tmp_path_factory.mktemp("quick")
    make_report(out, seed=DEFAULT_SEED, quick=True)
    return pd.read_csv(out / "summary.csv")


def test_quick_summary_matches_golden(quick_summary):
    golden = pd.read_csv(GOLDEN)
    assert list(quick_summary.columns) == list(golden.columns)
    assert quick_summary[["chart", "row"]].astype(str).equals(golden[["chart", "row"]].astype(str))
    pd.testing.assert_frame_equal(
        quick_summary.drop(columns=["chart", "row"]), golden.drop(columns=["chart", "row"]),
        rtol=1e-6, atol=1e-8, check_exact=False)


def test_summary_has_no_pooled_ambiguity_and_has_se_columns(quick_summary):
    cols = set(quick_summary.columns)
    assert {"err_open_sd_pooled", "err_open_pooled"} <= cols
    assert not {"err_open_sd", "err_open"} & cols
    assert {"true_informed_se", "reversion_control", "residual_alpha_effect",
            "slip_vwap_mkt", "slip_vwap_cf", "risk_ratio_lo", "risk_ratio_hi"} <= cols


def _cell(summary, chart, row, col):
    s = summary[(summary["chart"] == chart) & (summary["row"].astype(str) == str(row))]
    return float(s[col].iloc[0])


def test_readme_numbers_trace_to_committed_summary():
    """Every headline number in the README is the committed summary.csv value, rounded."""
    summ = pd.read_csv(ROOT / "reports" / "summary.csv")
    text = (ROOT / "README.md").read_text()
    checks = {
        "a slip_vwap @1%": f"{_cell(summ, 'a_vwap_trap', 0.01, 'slip_vwap'):.1f}",
        "a slip_vwap @30%": f"{_cell(summ, 'a_vwap_trap', 0.3, 'slip_vwap'):.1f}",
        "a true_impact @1%": f"{_cell(summ, 'a_vwap_trap', 0.01, 'true_impact'):.1f}",
        "a true_impact @30%": f"{_cell(summ, 'a_vwap_trap', 0.3, 'true_impact'):.1f}",
        "a slip_vwap_cf @30%": f"{_cell(summ, 'a_vwap_trap', 0.3, 'slip_vwap_cf'):.1f}",
        "c true impact": f"{_cell(summ, 'c_alpha_vs_impact', 'impact', 'true_informed'):.1f}",
        "c residual impact": f"{_cell(summ, 'c_alpha_vs_impact', 'impact', 'residual_informed'):.1f}",  # noqa: E501
        "c reversion impact": f"{_cell(summ, 'c_alpha_vs_impact', 'impact', 'reversion_informed'):.1f}",  # noqa: E501
        "c reversion control": f"{_cell(summ, 'c_alpha_vs_impact', 'impact', 'reversion_control'):.1f}",  # noqa: E501
        "c residual alpha effect": f"{_cell(summ, 'c_alpha_vs_impact', 'impact', 'residual_alpha_effect'):.1f}",  # noqa: E501
        "d risk @2%": f"{_cell(summ, 'd_basket_frontier', 0.02, 'risk'):.0f}",
        "d risk @30%": f"{_cell(summ, 'd_basket_frontier', 0.3, 'risk'):.0f}",
        "d risk ratio": f"{_cell(summ, 'd_basket_frontier', 0.02, 'risk_ratio_vs_fastest'):.1f}",
        "e raw n": f"{round(_cell(summ, 'e_orders_needed', 'raw IS', 'n_needed'), -2):,.0f}",
    }
    missing = {k: v for k, v in checks.items() if v not in text}
    assert not missing, f"README lacks these summary.csv values: {missing}"


def test_readme_percentages_match_summary():
    summ = pd.read_csv(ROOT / "reports" / "summary.csv")
    text = (ROOT / "README.md").read_text()
    t = _cell(summ, "c_alpha_vs_impact", "impact", "true_informed")
    for col in ("residual_informed", "reversion_informed"):
        pct = round((_cell(summ, "c_alpha_vs_impact", "impact", col) / t - 1) * 100)
        assert f"{pct:+d}%" in text, f"README lacks {pct:+d}% for {col}"
