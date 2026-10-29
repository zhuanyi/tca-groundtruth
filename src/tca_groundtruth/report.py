"""Validation report: rebuild every chart from a fixed seed.

python -m tca_groundtruth.report [--out reports] [--seed N] [--quick]
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

from tca_groundtruth.execution import FillParams, ImpactParams, execute
from tca_groundtruth.market import MarketParams, simulate_day, simulate_universe
from tca_groundtruth.orders import OrderParams, apply_alpha, generate_orders
from tca_groundtruth.tca import COMPONENTS, market_adjustment_bps, shortfall_true, tca_row

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_SEED = 20261028


def run_order(rng: np.random.Generator, order: pd.Series, market: MarketParams,
              impact: ImpactParams, fills: FillParams) -> pd.Series:
    """Simulate one day for one order and return its TCA row."""
    bars = apply_alpha(simulate_day(rng, market), order)
    fills_df, out = execute(rng, bars, order, impact, fills)
    return tca_row(order, out, fills_df, beta=market.beta)


def run_orders(seed: int, n_orders: int, market: MarketParams | None = None,
               orders: OrderParams | None = None, impact: ImpactParams | None = None,
               fills: FillParams | None = None) -> pd.DataFrame:
    """Orders joined with their TCA rows; one independent simulated day per order."""
    market = market or MarketParams()
    impact = impact or ImpactParams()
    fills = fills or FillParams()
    rng = np.random.default_rng(seed)
    od = generate_orders(rng, n_orders, market.daily_volume, orders, market.n_minutes)
    rows = [run_order(rng, o, market, impact, fills) for _, o in od.iterrows()]
    return pd.concat([od, pd.DataFrame(rows)], axis=1)


def _bps(x: float) -> str:
    return f"{x:+.1f} bps"


# --- (a) VWAP trap -------------------------------------------------------------------------

def vwap_trap_table(seed: int, rates=(0.01, 0.02, 0.05, 0.10, 0.20, 0.30),
                    n_per_rate: int = 100) -> pd.DataFrame:
    """Zero-alpha POV orders at each participation rate: VWAP slippage vs true impact paid.

    Costs are in bps of executed value so that orders of different fill size compare."""
    rng = np.random.default_rng(seed)
    market = MarketParams()
    out = []
    for rate in rates:
        for _ in range(n_per_rate):
            bars = simulate_day(rng, market)
            order = pd.Series({
                "order_id": 0, "side": int(rng.choice([1, -1])), "qty": 10**9,
                "decision_minute": 30, "arrival_minute": 30, "end_minute": 330,
                "alpha_bps": 0.0, "alpha_halflife_min": 60.0, "flow": True, "algo": "POV",
                "participation": rate, "fee_per_share": 0.0,
            })
            fills, bars_out = execute(rng, apply_alpha(bars, order), order)
            row = tca_row(order, bars_out, fills, beta=market.beta)
            q = fills["qty"].to_numpy(dtype=float)
            exec_value = float((q * fills["price"]).sum())
            if exec_value == 0:
                continue
            impact_exec = order["side"] * float(
                (q * (fills["mid_affected"] - fills["mid"])).sum()) / exec_value * 1e4
            out.append({"rate": rate, "slip_vwap": row["slip_interval_vwap"],
                        "slip_arrival": row["slip_arrival"], "true_impact": impact_exec})
    return pd.DataFrame(out)


def chart_vwap_trap(seed: int, out_path: str | Path, rates=(0.01, 0.02, 0.05, 0.10, 0.20, 0.30),
                    n_per_rate: int = 100) -> pd.DataFrame:
    tab = vwap_trap_table(seed, rates, n_per_rate)
    g = tab.groupby("rate").agg(["mean", "sem"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col, label in [("slip_vwap", "slippage vs interval VWAP (includes own fills)"),
                       ("true_impact", "true impact paid (simulator)"),
                       ("slip_arrival", "slippage vs arrival mid")]:
        ax.errorbar(g.index * 100, g[(col, "mean")], yerr=1.96 * g[(col, "sem")], marker="o",
                    capsize=3, label=label)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("participation rate (% of interval volume)")
    ax.set_ylabel("bps of executed value (positive = cost)")
    ax.set_title("(a) The VWAP trap: you become the benchmark; impact keeps growing")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return g


# --- (b) delay vs trader attribution -------------------------------------------------------

DELAY_BINS = (0, 10, 20, 30, 40, 50, 61)


def delay_attribution_table(seed: int, n_orders: int = 4000, flow_share: float = 0.5,
                            market: MarketParams | None = None) -> pd.DataFrame:
    """One row per order: true delay vs trading cost, and the error of two wrong stamps.

    trading = spread + impact + timing (the trader's part); delay is the PM-to-desk part.
    alpha_delay is the true delay cost (the alpha that realised during the delay), and
    spread_impact the trader's cost without market noise; only the simulator knows them.
    Wrong stamps: "fallback" sets P_d := P_0 (order creation time in the OMS), which
    zeroes delay; "open" sets P_d to the minute-0 mid, a stamp earlier than the decision
    (stand-in for the previous close). err_* = reported total - true total, bps of Q*P_d.
    """
    market = market or MarketParams()
    impact, fills = ImpactParams(), FillParams()
    rng = np.random.default_rng(seed)
    od = generate_orders(rng, n_orders, market.daily_volume,
                         OrderParams(flow_share=flow_share), market.n_minutes)
    rows = []
    for _, o in od.iterrows():
        bars = apply_alpha(simulate_day(rng, market), o)
        fills_df, out = execute(rng, bars, o, impact, fills)
        true = shortfall_true(o, out, fills_df)
        p_0 = float(out["mid_affected"].iloc[int(o["arrival_minute"])])
        fb = shortfall_true(o, out, fills_df, decision_price=p_0)
        op = shortfall_true(o, out, fills_df, decision_price=float(out["mid"].iloc[0]))
        p_d = float(out["mid"].iloc[int(o["decision_minute"])])
        # the alpha that realised during the delay, the part of delay the PM caused
        alpha_delay = (int(o["side"]) * float(out["alpha_shift"].iloc[int(o["arrival_minute"])])
                       / p_d * 1e4)
        rows.append({
            "delay_min": int(o["arrival_minute"] - o["decision_minute"]),
            "flow": bool(o["flow"]),
            "delay": true["delay"], "alpha_delay": alpha_delay,
            "trading": true["spread"] + true["impact"] + true["timing"],
            "spread_impact": true["spread"] + true["impact"],
            "opportunity": true["opportunity"], "fees": true["fees"], "total": true["total"],
            "fallback_delay": fb["delay"], "fallback_total": fb["total"],
            "err_fallback": fb["total"] - true["total"], "err_open": op["total"] - true["total"],
        })
    tab = pd.DataFrame(rows)
    labels = [f"{a}-{min(b, 60)}" for a, b in zip(DELAY_BINS[:-1], DELAY_BINS[1:], strict=True)]
    tab["bucket"] = pd.cut(tab["delay_min"], DELAY_BINS, right=False, labels=labels)
    return tab


def delay_attribution_summary(tab: pd.DataFrame) -> pd.DataFrame:
    """Per delay bucket: mean delay/trading cost by order type and mean stamp errors."""
    out = {}
    for flow, name in [(False, "alpha"), (True, "flow")]:
        g = tab[tab["flow"] == flow].groupby("bucket", observed=False)
        out[f"n_{name}"] = g.size()
        out[f"delay_{name}"] = g["delay"].mean()
        out[f"alpha_delay_{name}"] = g["alpha_delay"].mean()
        out[f"trading_{name}"] = g["trading"].mean()
        out[f"spread_impact_{name}"] = g["spread_impact"].mean()
        out[f"err_fallback_{name}"] = g["err_fallback"].mean()
        out[f"err_open_{name}"] = g["err_open"].mean()
    out["err_open_sd"] = tab.groupby("bucket", observed=False)["err_open"].std()
    out["err_open"] = tab.groupby("bucket", observed=False)["err_open"].mean()
    return pd.DataFrame(out)


def ols_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """OLS slope of y on x with an intercept, and its standard error."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    res = stats.linregress(x, y)
    return float(res.slope), float(res.stderr)


def delay_slopes(tab: pd.DataFrame) -> pd.DataFrame:
    """Slope (bps per minute of delay) of measured and true delay cost, alpha vs flow."""
    rows = {}
    for flow, name in [(False, "alpha"), (True, "flow")]:
        t = tab[tab["flow"] == flow]
        m, m_se = ols_slope(t["delay_min"], t["delay"])
        tr, _ = ols_slope(t["delay_min"], t["alpha_delay"])
        rows[name] = {"measured": m, "measured_se": m_se, "true": tr, "n": len(t)}
    return pd.DataFrame(rows).T


def chart_delay_attribution(seed: int, out_path: str | Path, n_orders: int = 4000,
                            flow_share: float = 0.5) -> pd.DataFrame:
    tab = delay_attribution_table(seed, n_orders, flow_share)
    summ = delay_attribution_summary(tab)
    sl = delay_slopes(tab)
    x = np.array([(a + min(b, 60)) / 2 for a, b in zip(DELAY_BINS[:-1], DELAY_BINS[1:],
                                                        strict=True)])
    alpha, flow = tab[~tab["flow"]], tab[tab["flow"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.8))

    # left: the true series carry the point; the measured means sit faintly behind them
    for t, dx, color, name in [(alpha, -0.8, "C3", "alpha"), (flow, 0.8, "0.5", "flow")]:
        g = t.groupby("bucket", observed=False)["delay"]
        ax1.errorbar(x + dx, g.mean(), yerr=1.96 * g.sem(), color=color, ls="none",
                     marker="o", ms=4, alpha=0.35, capsize=2,
                     label=f"measured delay cost, {name} (mean, 95% CI)")
    ax1.plot(x, summ["alpha_delay_alpha"], color="C3", lw=3, marker="o",
             label="true delay cost, alpha orders")
    ax1.plot(x, summ["alpha_delay_flow"], color="0.3", lw=3, ls="--",
             label="true delay cost, flow orders (= 0)")
    ax1.plot(x, summ["spread_impact_alpha"], color="C0", lw=2.5, marker="s",
             label="true trading cost (spread + impact), alpha")
    ax1.plot(x, summ["spread_impact_flow"], color="C0", lw=2.5, ls="--", marker="s",
             label="true trading cost (spread + impact), flow")
    ax1.axhline(0, color="k", lw=0.6)
    a_, f_ = sl.loc["alpha"], sl.loc["flow"]
    ax1.text(0.02, 0.97,
             "OLS slope of measured delay cost on delay minutes (95% CI), bps/min:\n"
             f"  alpha {a_['measured']:+.2f} [{a_['measured'] - 1.96 * a_['measured_se']:+.2f}, "
             f"{a_['measured'] + 1.96 * a_['measured_se']:+.2f}]   true {a_['true']:+.3f}\n"
             f"  flow  {f_['measured']:+.2f} [{f_['measured'] - 1.96 * f_['measured_se']:+.2f}, "
             f"{f_['measured'] + 1.96 * f_['measured_se']:+.2f}]   true {f_['true']:+.3f}\n"
             "unbiased, but per-order market noise "
             f"(sd {alpha['delay'].std():.0f} bps, alpha orders) hides the slope",
             transform=ax1.transAxes, fontsize=7.5, family="monospace", va="top",
             bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"})
    ax1.set_ylim(-12, 25)
    ax1.set_xticks(x, summ.index.astype(str))
    ax1.set_xlabel("PM-to-desk delay (minutes)")
    ax1.set_ylabel("mean cost, bps of intended value")
    ax1.set_title("True delay cost grows with delay and is the PM's;\n"
                  "the trader's cost does not move", fontsize=10)
    ax1.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

    # right: bias of each stamp on informed orders, dodged
    for col, dx, color, marker, label in [
            ("err_fallback", -1.2, "C1", "o", "late stamp P_d := P_0"),
            ("err_open", 1.2, "C2", "s", "early stamp P_d := open")]:
        g = alpha.groupby("bucket", observed=False)[col]
        ax2.errorbar(x + dx, g.mean(), yerr=1.96 * g.sem(), color=color, ls="none",
                     marker=marker, capsize=3,
                     label=f"{label} (OMS creation): error sd {alpha[col].std():.0f} bps/order"
                     if col == "err_fallback" else
                     f"{label}: error sd {alpha[col].std():.0f} bps/order")
    ax2.plot(x, -summ["alpha_delay_alpha"], color="C1", lw=2.5,
             label="late stamp, true bias (= minus true delay cost)")
    ax2.axhline(0, color="C2", lw=1.5, ls="--", label="early stamp, true bias (= 0)")
    ax2.set_xticks(x, summ.index.astype(str))
    ax2.set_xlabel("PM-to-desk delay (minutes)")
    ax2.set_ylabel("bias: reported total IS - true total IS, bps")
    ax2.set_title("Alpha orders: a late stamp under-reports IS by the delay cost;\n"
                  "an early stamp is unbiased but noisier", fontsize=10)
    ax2.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.suptitle(f"(b) Delay is the PM's cost, not the trader's; a late decision stamp "
                 f"erases it (n={len(tab):,} orders, half flow)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    for name in ("alpha", "flow"):
        summ[f"slope_measured_{name}"] = sl.loc[name, "measured"]
        summ[f"slope_measured_se_{name}"] = sl.loc[name, "measured_se"]
        summ[f"slope_true_{name}"] = sl.loc[name, "true"]
    summ["err_fallback_sd_alpha"] = alpha["err_fallback"].std()
    summ["err_open_sd_alpha"] = alpha["err_open"].std()
    return summ


# --- (c) alpha vs impact: observed vs true split -------------------------------------------

def chart_waterfall(seed: int, out_path: str | Path, n_orders: int = 300) -> pd.DataFrame:
    df = run_orders(seed, n_orders, orders=OrderParams(flow_share=0.0))
    comps = [c for c in COMPONENTS if c != "total"]
    tab = pd.DataFrame({
        "true (simulator)": [df[f"true_{c}"].mean() for c in comps],
        "observed: residual labelled impact": [df[f"obs_{c}"].mean() for c in comps],
        "observed: 30-min reversion": [df[f"rev_{c}"].mean() for c in comps],
    }, index=comps)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    tab.plot.bar(ax=ax, rot=0, width=0.8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("mean cost, bps of intended value")
    ax.set_title(f"(c) Alpha vs impact on informed orders (n={n_orders}): "
                 f"total IS {_bps(df['true_total'].mean())}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return tab


# --- (e) how many orders to detect 2 bps ---------------------------------------------------

def orders_needed(sd: float, delta_bps: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Per-arm sample size for a two-sample z-test of a mean difference delta_bps."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return 2 * (z * sd / delta_bps) ** 2


def _difficulty_residual(df: pd.DataFrame, y: str, adv: float) -> np.ndarray:
    """Residual of y on observable pre-trade difficulty sqrt(Q/ADV) (CUPED-style)."""
    x = np.sqrt(df["qty"].to_numpy(dtype=float) / adv)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, df[y].to_numpy(dtype=float), rcond=None)
    return df[y].to_numpy(dtype=float) - X @ beta


def power_table(seed: int, n_orders: int = 500, delta_bps: float = 2.0,
                n_boot: int = 500, market: MarketParams | None = None) -> pd.DataFrame:
    market = market or MarketParams()
    df = run_orders(seed, n_orders, market=market)
    series = {
        "raw IS": df["true_total"].to_numpy(dtype=float),
        "market-adjusted IS": df["is_adj"].to_numpy(dtype=float),
        "market + difficulty adjusted": _difficulty_residual(df, "is_adj", market.daily_volume),
    }
    rng = np.random.default_rng(seed + 1)
    rows = []
    for name, x in series.items():
        boots = np.array([orders_needed(np.std(rng.choice(x, len(x)), ddof=1), delta_bps)
                          for _ in range(n_boot)])
        rows.append({"metric": name, "sd_bps": np.std(x, ddof=1),
                     "n_needed": orders_needed(np.std(x, ddof=1), delta_bps),
                     "n_lo": np.percentile(boots, 2.5), "n_hi": np.percentile(boots, 97.5)})
    return pd.DataFrame(rows).set_index("metric")


def chart_power(seed: int, out_path: str | Path, n_orders: int = 500, delta_bps: float = 2.0,
                n_boot: int = 500) -> pd.DataFrame:
    tab = power_table(seed, n_orders, delta_bps, n_boot)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    err = np.array([tab["n_needed"] - tab["n_lo"], tab["n_hi"] - tab["n_needed"]])
    ax.bar(tab.index, tab["n_needed"], yerr=err, capsize=4, color=["C3", "C0", "C2"])
    for i, (n, hi, sd) in enumerate(zip(tab["n_needed"], tab["n_hi"], tab["sd_bps"], strict=True)):
        ax.text(i, hi * 1.03, f"n={n:,.0f}\nsd={sd:.0f} bps", ha="center", va="bottom",
                fontsize=8)
    ax.set_ylabel(f"orders per arm to detect {delta_bps:g} bps (80% power, 5% size)")
    ax.set_title(f"(e) TCA is mostly noise: sample size with 95% bootstrap CI (n={n_orders})")
    ax.set_ylim(0, tab["n_hi"].max() * 1.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return tab


# --- (d) basket residual risk vs cost ------------------------------------------------------

def _basket(seed: int, n_names: int, buy_share: float, qty_pct_adv: tuple[float, float],
            beta_range: tuple[float, float], market: MarketParams) -> pd.DataFrame:
    """Fixed basket: per-name beta, side and quantity, drawn once from the seed."""
    rng = np.random.default_rng([seed, 0])
    lo, hi = np.log(qty_pct_adv)
    return pd.DataFrame({
        "beta": rng.uniform(beta_range[0], beta_range[1], n_names),
        "side": np.where(rng.uniform(size=n_names) < buy_share, 1, -1),
        "qty": np.maximum(1, np.round(market.daily_volume * np.exp(
            rng.uniform(lo, hi, n_names)))).astype(np.int64),
    })


def basket_frontier_table(seed: int, rates=(0.02, 0.05, 0.10, 0.20, 0.30), n_runs: int = 40,
                          n_names: int = 20, buy_share: float = 0.7,
                          qty_pct_adv: tuple[float, float] = (0.005, 0.05),
                          start_minute: int = 30,
                          beta_range: tuple[float, float] = (0.6, 1.4),
                          market: MarketParams | None = None,
                          impact: ImpactParams | None = None) -> pd.DataFrame:
    """One row per (run, rate): basket true shortfall, unhedged and market-adjusted.

    Zero-alpha POV orders from start_minute to the last bar, no fees. Run r uses the
    same market day for every rate (common random numbers), so the frontier is paired.
    Shortfall is in bps of basket notional sum(Q * P_d); unfilled shares carry
    opportunity cost to the close. "cost" is the shortfall minus the no-trade move of the
    intended position (each filled share from decision to its fill minute, each unfilled
    share to the last bar): spread plus all of the order's own impact, including impact
    left on the unfilled remainder. Only the simulator knows it; its mean is the true
    expected cost, without the market noise that dominates the realised mean.
    impact selects the impact model (default: the square-root propagator).
    """
    market = market or MarketParams()
    basket = _basket(seed, n_names, buy_share, qty_pct_adv, beta_range, market)
    betas = basket["beta"].to_numpy()
    out = []
    for r in range(n_runs):
        day_rng = np.random.default_rng([seed, 1, r])
        uni = simulate_universe(day_rng, n_names, market, betas=betas)
        days = [d.reset_index(drop=True) for _, d in uni.groupby("name", sort=True)]
        for i_rate, rate in enumerate(rates):
            rng = np.random.default_rng([seed, 2, r, i_rate])
            cost = total = adj = notional = filled = 0.0
            for (_, b), bars in zip(basket.iterrows(), days, strict=True):
                order = pd.Series({
                    "order_id": 0, "side": int(b["side"]), "qty": int(b["qty"]),
                    "decision_minute": start_minute, "arrival_minute": start_minute,
                    "end_minute": market.n_minutes - 1, "alpha_bps": 0.0,
                    "alpha_halflife_min": 60.0, "flow": True, "algo": "POV",
                    "participation": rate, "fee_per_share": 0.0,
                })
                fills_df, bars_out = execute(rng, bars, order, impact)
                is_total = float(shortfall_true(order, bars_out, fills_df)["total"])
                is_adj = is_total - market_adjustment_bps(order, bars_out, fills_df,
                                                          float(b["beta"]))
                mid = bars["mid"].to_numpy()
                p_d = float(mid[start_minute])
                q = fills_df["qty"].to_numpy(dtype=float)
                filled_move = float((q * (mid[fills_df["minute"].to_numpy()] - p_d)).sum())
                unfilled_move = (float(b["qty"]) - q.sum()) * (mid[-1] - p_d)
                move = int(b["side"]) * (filled_move + unfilled_move)
                n = float(b["qty"]) * p_d
                total += is_total * n
                cost += is_total * n - move * 1e4
                adj += is_adj * n
                notional += n
                filled += q.sum() * p_d
            out.append({"run": r, "rate": rate, "cost": cost / notional,
                        "is": total / notional, "is_adj": adj / notional,
                        "fill_rate": filled / notional})
    return pd.DataFrame(out)


def basket_frontier_summary(tab: pd.DataFrame) -> pd.DataFrame:
    """Per rate: true expected cost and residual risk (sd of shortfall across runs)."""
    g = tab.groupby("rate")
    return pd.DataFrame({
        "cost": g["cost"].mean(), "cost_se": g["cost"].sem(),
        "is_mean": g["is"].mean(), "risk": g["is"].std(),
        "is_adj_mean": g["is_adj"].mean(), "risk_adj": g["is_adj"].std(),
        "fill_rate": g["fill_rate"].mean(), "n_runs": g.size(),
    })


ALMGREN_2005 = ImpactParams(temporary="almgren2005")


def chart_basket_frontier(seed: int, out_path: str | Path,
                          rates=(0.02, 0.05, 0.10, 0.20, 0.30), n_runs: int = 40,
                          n_names: int = 20, **basket_kw) -> pd.DataFrame:
    """basket_kw is forwarded to basket_frontier_table (buy_share, qty_pct_adv, ...).

    Runs the basket twice on the same days: the default square-root propagator, and
    the rate-dependent Almgren et al. (2005) temporary impact (ALMGREN_2005), so the
    impact model is fixed here and may not be passed."""
    if "impact" in basket_kw:
        raise TypeError("chart_basket_frontier always compares the propagator with "
                        "ALMGREN_2005; call basket_frontier_table to pick an impact model")
    summ = basket_frontier_summary(basket_frontier_table(seed, rates, n_runs, n_names,
                                                         **basket_kw))
    summ_ac = basket_frontier_summary(basket_frontier_table(seed, rates, n_runs, n_names,
                                                            impact=ALMGREN_2005, **basket_kw))
    x = summ.index.to_numpy() * 100
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.8))

    for s_, color, marker, label in [
            (summ, "C0", "o", "square-root propagator (this repo's model)"),
            (summ_ac, "C1", "s", "rate-dependent temporary impact\n"
                                 "(Almgren et al. 2005: 0.142 \u03c3 v^0.6)")]:
        ax1.errorbar(x, s_["cost"], yerr=1.96 * s_["cost_se"], color=color, marker=marker,
                     capsize=3, lw=2, label=label)
    ax1.annotate(f"{summ['fill_rate'].iloc[0]:.0%} filled", (x[0], summ["cost"].iloc[0]),
                 textcoords="offset points", xytext=(8, -12), fontsize=7.5, color="C0")
    min_fill = np.floor(summ["fill_rate"].iloc[1:].min() * 1000) / 10
    ax1.text(0.98, 0.03, f"{x[1]:g}% and faster: \u2265{min_fill:g}% filled",
             transform=ax1.transAxes, ha="right", fontsize=7.5, color="C0")
    ax1.set_xscale("log")
    ax1.set_xticks(x, [f"{v:g}%" for v in x])
    ax1.minorticks_off()
    ax1.set_ylim(0, None)
    ax1.set_xlabel("POV participation rate (log scale)")
    ax1.set_ylabel("true expected cost (spread + own impact), bps")
    ax1.set_title("Cost: flat under square-root impact,\nrising under rate-dependent impact",
                  fontsize=10)
    ax1.legend(fontsize=7, loc="upper left")

    for col, color, label in [("risk", "C3", "unhedged"),
                              ("risk_adj", "C2", "beta-hedged (known beta)")]:
        ax2.plot(x, summ[col], color=color, marker="o", lw=2, label=label)
    ax2.set_xscale("log")
    ax2.set_xticks(x, [f"{v:g}%" for v in x])
    ax2.minorticks_off()
    ax2.set_ylim(0, None)
    ax2.set_xlabel("POV participation rate (log scale)")
    ax2.set_ylabel("residual risk: sd of basket shortfall across days, bps")
    ax2.set_title(f"Risk: {summ['risk'].iloc[0] / summ['risk'].iloc[-1]:.1f}x lower from "
                  f"{x[0]:g}% to {x[-1]:g}%;\na beta hedge removes about half", fontsize=10)
    ax2.legend(fontsize=8)

    for s_, color, marker, label in [(summ, "C0", "o", "square-root propagator"),
                                     (summ_ac, "C1", "s", "rate-dependent (Almgren 2005)")]:
        ax3.plot(s_["risk"], s_["cost"], color=color, marker=marker, lw=2, label=label)
        for rate, xv, yv in zip(s_.index, s_["risk"], s_["cost"], strict=True):
            ax3.annotate(f"{rate:.0%}", (xv, yv), textcoords="offset points", xytext=(5, 3),
                         fontsize=7, color=color)
    ax3.set_ylim(0, None)
    ax3.set_xlabel("residual risk (unhedged sd), bps")
    ax3.set_ylabel("true expected cost, bps")
    ax3.set_title("Cost-risk frontier: a trade-off only\nwhen impact depends on the rate",
                  fontsize=10)
    ax3.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"(d) Faster trading cuts basket risk "
                 f"{summ['risk'].iloc[0] / summ['risk'].iloc[-1]:.1f}x; under square-root "
                 f"impact it barely costs more ({n_names} names, {n_runs} days per rate)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return pd.concat([summ, summ_ac.add_suffix("_ac2005")], axis=1)


# --- driver --------------------------------------------------------------------------------

def make_report(out_dir: str | Path = "reports", seed: int = DEFAULT_SEED,
                quick: bool = False) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    k = 0.05 if quick else 1.0
    paths = []
    p = out / "a_vwap_trap.png"
    g = chart_vwap_trap(seed, p, n_per_rate=max(3, int(100 * k)))
    paths.append(p)
    p = out / "c_alpha_vs_impact.png"
    tab_c = chart_waterfall(seed, p, n_orders=max(5, int(300 * k)))
    paths.append(p)
    p = out / "e_orders_needed.png"
    tab_e = chart_power(seed, p, n_orders=max(10, int(500 * k)), n_boot=max(20, int(500 * k)))
    paths.append(p)
    p = out / "b_delay_attribution.png"
    tab_b = chart_delay_attribution(seed, p, n_orders=max(24, int(4000 * k)))
    paths.append(p)
    p = out / "d_basket_frontier.png"
    tab_d = chart_basket_frontier(seed, p, n_runs=max(3, int(40 * k)),
                                  n_names=max(4, int(20 * k)))
    paths.append(p)
    summary = pd.concat({
        "a_vwap_trap": g.xs("mean", axis=1, level=1),
        "c_alpha_vs_impact": tab_c,
        "e_orders_needed": tab_e,
        "b_delay_attribution": tab_b,
        "d_basket_frontier": tab_d,
    }, names=["chart", "row"])
    p = out / "summary.csv"
    summary.to_csv(p)
    paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    for p in make_report(a.out, a.seed, a.quick):
        print(p)


if __name__ == "__main__":
    main()
