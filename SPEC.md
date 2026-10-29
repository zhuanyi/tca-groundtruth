# SPEC: tca-groundtruth MVP

Post-trade TCA on synthetic data where the counterfactual no-trade price path,
the true impact and the true alpha are recorded, so every estimate can be
compared with the truth.

Notation: side `s` (+1 buy, -1 sell), intended
quantity `Q`, executed quantity `q <= Q`, decision price `P_d`, arrival price
`P_0`, average fill price `P_bar`, horizon price `P_c`, fees `F`. All costs are
signed basis points, positive = cost (the `*_perf_bps` view in 3.4 flips the
sign: positive = beat the benchmark). Implementation-shortfall components are
in bps of intended value `Q * P_d`; benchmark slippages are in bps of the
benchmark price.

## 1. Assumptions

- One trading day of 390 one-minute bars (minute 0 = open, 389 = last bar).
  Each order lives on its own simulated day unless stated otherwise.
- The market mid is a driftless log random walk with a deterministic U-shaped
  intraday volatility profile. Volume follows a U-shaped profile with
  log-normal noise. The quoted spread is widest at the open and narrows
  through the day (linear in time between two bps levels).
- A single-name day is generated jointly with an index path through a one-factor
  model: `r_stock = beta * r_index + r_idio`, with `beta` an input. The
  universe generator reuses the same factor model for 50 names; chart (d)
  uses it for a 20-name basket.
- PM alpha is a deterministic, exponentially decaying drift added to the
  no-trade mid from the decision minute: the cumulative shift at minute `t` is
  `s * alpha_bps/1e4 * mid[t_d] * (1 - exp(-(t - t_d) / halflife * ln 2))`.
  Flow orders have `alpha_bps = 0`. Alpha is idiosyncratic; the index is not
  shifted.
- Impact is a propagator model on 1-minute child orders plus a linear permanent
  part. Child `j` traded at minute `t_j` with quantity `n_j` shifts the mid at
  every minute `t >= t_j` (lag 0 included, so a child pays its own impact) by
  `s * mid[t_j] * (f(n_j) * G(t - t_j) + h(n_j))` with
  `f(n) = (Y / 2) * sigma_min[t_j] * (n / volume[t_j]) ** delta`,
  `G(l) = (1 + l) ** (-beta_kernel)` and `h(n) = gamma_perm * sigma_day * n / V_day`
  (`sigma_day = sqrt(sum sigma_min**2)`, `V_day = sum volume`).
  With `delta = beta_kernel = 0.5` a constant-rate metaorder peaks at about
  `Y * sigma_day * sqrt(Q / V_day)`, the square-root law, on Gatheral's
  no-arbitrage boundary `delta + beta_kernel = 1`. `gamma_perm` is a simulator
  input that the TCA layer never reads. `mid_affected = mid + cumulative impact`.
- Fills: each minute the scheduler asks for `target[t]` shares. A fraction
  `passive_frac` is posted at the near touch (buy at bid). A passive child
  fills with probability `p_fill_adverse` if the next-minute unaffected return
  is against the order (price comes to the resting order), else
  `p_fill_benign`. Unfilled passive quantity rolls forward into the next
  minute's target. The remaining fraction crosses the spread and fills in full
  at `mid_affected + s * spread / 2`, capped at `max_participation` of that
  minute's volume; the excess rolls forward. Anything left at `end_minute`
  stays unfilled (opportunity cost). Fees are a flat per-share amount on `q`.
- Schedulers see the realised volume profile of the day (perfect volume
  forecast). This flatters VWAP/POV tracking and is stated as a limitation.
- Interval VWAP includes the order's own fills (own volume at own price added
  to market volume at the affected mid), which is what makes the VWAP trap
  possible.
- Market-adjusted IS uses the simulator's index beta as known. The beta choice
  is documented in the README; estimating it is out of scope.

## 2. Not done in the MVP

- No multi-day orders, no overnight, no auctions.
- No limit-order-book or queue model; passive fills are a probability model.
- No calibration of impact parameters from the simulated data (that is the
  `impact-lab` repo).
- No Almgren-Chriss optimal schedules.
- Benchmarks in a footnote only: far-touch arrival, participation-weighted
  price, full-day VWAP, TWAP.

## 3. Interface contract

All functions are pure given the `rng` argument (`numpy.random.Generator`).
DataFrames are returned with a `RangeIndex`. Parameter dataclasses are passed
as `params: X | None = None` (None means the defaults) because ruff forbids
calls in argument defaults.

### 3.1 `tca_groundtruth.market`

```python
@dataclass(frozen=True)
class MarketParams:
    n_minutes: int = 390
    s0: float = 100.0
    daily_vol: float = 0.02          # daily return volatility of the stock
    daily_volume: float = 1_000_000  # expected shares per day
    spread_open_bps: float = 10.0
    spread_close_bps: float = 4.0
    u_shape: float = 2.0             # peak/trough ratio of the volume and vol profiles
    volume_noise: float = 0.3        # log-normal sigma of per-minute volume
    beta: float = 1.0
    index_vol: float = 0.012         # daily return volatility of the index

def u_profile(n: int, ratio: float) -> np.ndarray
    # length-n positive array, mean 1.0, U-shaped, endpoints/minimum == ratio.

def simulate_day(rng, params: MarketParams = MarketParams()) -> pd.DataFrame
```

`simulate_day` returns the no-trade market with columns

| column | dtype | meaning |
|---|---|---|
| `minute` | int64 | 0..n_minutes-1 |
| `mid` | float64 | no-trade mid, no alpha |
| `index_mid` | float64 | index level, starts at 100.0 |
| `volume` | float64 | market shares traded this minute, > 0 |
| `spread` | float64 | quoted spread in price units, > 0 |
| `sigma_min` | float64 | per-minute return volatility used at this minute |

```python
def simulate_universe(rng, n_names: int = 50, params: MarketParams = MarketParams(),
                      beta_range: tuple[float, float] = (0.6, 1.4),
                      betas: np.ndarray | None = None) -> pd.DataFrame
```

Long frame with the `simulate_day` columns plus `name` (int64, 0..n_names-1)
and `beta` (float64). All names share one `index_mid` path. `betas` (length
`n_names`) fixes the per-name betas instead of drawing them from `beta_range`,
so one basket can be re-simulated on new days; `beta_range` is then ignored and
not validated. `ValueError` on a wrong length, a non-finite beta, or any
`|beta| * index_vol >= daily_vol`.

Errors: `ValueError` if `n_minutes < 3`, any volatility/volume/spread
parameter `<= 0`, `volume_noise < 0`, `spread_close_bps > spread_open_bps`,
or `|beta| * index_vol >= daily_vol`.

### 3.2 `tca_groundtruth.orders`

```python
@dataclass(frozen=True)
class OrderParams:
    qty_pct_adv: tuple[float, float] = (0.005, 0.10)  # log-uniform share of daily volume
    alpha_bps: tuple[float, float] = (5.0, 30.0)      # uniform, informed orders
    alpha_halflife_min: tuple[float, float] = (30.0, 240.0)
    flow_share: float = 0.3                            # fraction of orders with zero alpha
    delay_min: tuple[int, int] = (0, 60)               # uniform PM-to-desk delay
    decision_min: tuple[int, int] = (5, 200)
    duration_min: tuple[int, int] = (30, 180)
    participation: tuple[float, float] = (0.05, 0.25)
    algos: tuple[str, ...] = ("VWAP", "TWAP", "POV")
    fee_per_share: float = 0.005

def generate_orders(rng, n_orders: int, daily_volume: float,
                    params: OrderParams = OrderParams(), n_minutes: int = 390) -> pd.DataFrame
```

| column | dtype | meaning |
|---|---|---|
| `order_id` | int64 | 0..n-1 |
| `side` | int64 | +1 / -1 |
| `qty` | int64 | intended shares `Q`, >= 1 |
| `decision_minute` | int64 | PM decision |
| `arrival_minute` | int64 | `decision_minute + delay`, `< end_minute` |
| `end_minute` | int64 | hard stop, `<= n_minutes - 1` |
| `alpha_bps` | float64 | total expected favourable drift, 0 for flow |
| `alpha_halflife_min` | float64 | |
| `flow` | bool | True iff `alpha_bps == 0` |
| `algo` | object | "VWAP" / "TWAP" / "POV" |
| `participation` | float64 | POV rate, also the cap for VWAP/TWAP |
| `fee_per_share` | float64 | |

```python
def apply_alpha(bars: pd.DataFrame, order: pd.Series) -> pd.DataFrame
```

Returns a copy of `bars` with a new column `alpha_shift` (float64, price units,
0 before `decision_minute`) and `mid` replaced by `mid + alpha_shift`.
Original mid is kept as `mid_base`.

Errors: `ValueError` if `n_orders < 1` or any tuple range is inverted.

### 3.3 `tca_groundtruth.execution`

```python
@dataclass(frozen=True)
class ImpactParams:
    Y: float = 0.5              # square-root-law prefactor at metaorder level
    delta: float = 0.5
    beta_kernel: float = 0.5
    gamma_perm: float = 0.3     # linear permanent slope; unknown to the TCA layer
    temporary: str = "propagator"  # or "almgren2005" (chart (d) comparison only)
    eta: float = 0.142          # Almgren et al. (2005) temporary coefficient
    beta_temp: float = 0.6      # Almgren et al. (2005) temporary exponent

@dataclass(frozen=True)
class FillParams:
    passive_frac: float = 0.5
    p_fill_adverse: float = 0.9
    p_fill_benign: float = 0.3
    max_participation: float = 0.5

def schedule(bars, order) -> np.ndarray
    # length n_minutes, integer shares targeted per minute; zero outside [arrival, end);
    # VWAP: proportional to bars.volume over the window; TWAP: uniform;
    # POV: participation * volume each minute until qty is reached. Sum <= qty.

def impact_kernel(lags: np.ndarray, p: ImpactParams) -> np.ndarray
    # transient kernel G(l) = (1 + l) ** (-beta_kernel); G(0) == 1

def execute(rng, bars, order, impact: ImpactParams = ImpactParams(),
            fills: FillParams = FillParams()) -> tuple[pd.DataFrame, pd.DataFrame]
```

`execute` returns `(fills_df, bars_out)`.

With `temporary="almgren2005"` the propagator is replaced by a rate-dependent
temporary impact that hits only the minute it is traded in:
`impact_shift[t] += s * eta * sigma_day * (n_t / volume[t]) ** beta_temp * mid[t]`;
the permanent part is unchanged and `Y`, `delta`, `beta_kernel` are unused. The
default (`"propagator"`) is unchanged, so every chart except (d)'s comparison
line is unaffected.

`fills_df` columns (one row per child fill, possibly empty):

| column | dtype | meaning |
|---|---|---|
| `minute` | int64 | |
| `qty` | int64 | shares filled, > 0 |
| `price` | float64 | fill price |
| `passive` | bool | |
| `mid_affected` | float64 | affected mid at fill minute |
| `mid` | float64 | no-trade mid (with alpha) at fill minute |

`bars_out` is `bars` plus `impact_shift` (float64, price units) and
`mid_affected = mid + impact_shift`. Invariants: `fills.qty.sum() <= order.qty`;
aggressive fills have `price == mid_affected + side * spread / 2`; passive fills
have `price == mid_affected - side * spread / 2`.

Errors: `ValueError` for unknown `algo`, if `order.arrival_minute >=
order.end_minute`, for an unknown `temporary`, `eta < 0` or `beta_temp <= 0`.

### 3.4 `tca_groundtruth.tca`

```python
BENCHMARKS = ("decision", "arrival", "interval_vwap", "close")

def benchmarks(order, bars_out, fills_df) -> pd.Series
    # index: decision, arrival, interval_vwap, close (price units).
    # interval_vwap over [arrival_minute, last fill minute] including own fills;
    # NaN if there are no fills.

def slippage_bps(order, fills_df, bench: pd.Series) -> pd.Series
    # s * (P_bar - bench) / bench * 1e4 per benchmark; NaN if no fills.

def post_trade_moves_bps(order, bars_out, fills_df, horizons=(5, 30)) -> pd.Series
    # index: post_5, post_30, post_close. Value: s * (mid_affected[t_last + h]
    # - mid_affected[t_last]) / P_d * 1e4. Positive = price kept going in the
    # trade direction (information), negative = reverted (impact). Horizons
    # past the last bar are clipped to the last bar.

def shortfall_true(order, bars_out, fills_df, decision_price=None) -> pd.Series
    # index: delay, spread, impact, timing, opportunity, fees, total (bps of Q*P_d)
    # delay       = s * Q * (P_0 - P_d)
    # spread      = s * sum q_i (price_i - mid_affected_i)
    # impact      = s * sum q_i (mid_affected_i - mid_i)         # true, from simulator
    # timing      = s * sum q_i (mid_i - P_0)
    # opportunity = s * (Q - q) * (P_c - P_0)
    # fees        = q * fee_per_share
    # total       = sum of the six == s*q*(P_bar - P_d) + s*(Q-q)*(P_c - P_d) + F
    # P_d = bars.mid at decision_minute (no alpha, no impact: == mid_base);
    # P_0 = bars.mid_affected at arrival_minute; P_c = bars.mid_affected at the last bar.
    # decision_price (> 0) replaces P_d, as a TCA system with a wrong decision stamp
    # would; it changes delay and the bps denominator only. decision_price = P_0 gives
    # delay == 0 exactly.

def shortfall_observed(order, bars_out, fills_df, method="naive") -> pd.Series
    # same index. "naive": impact = trading - spread, timing = 0 (the trap of
    # labelling a residual as impact). "reversion": impact = s * q *
    # (mid_affected[t_last] - mid_affected[t_last + 30]) / (Q P_d), the part of
    # the move that came back; timing = trading - spread - impact. Other
    # components identical to shortfall_true.

def market_adjustment_bps(order, bars_out, fills_df, beta: float) -> float
    # s * beta * [sum q_i (idx[t_i]/idx[t_d] - 1) + (Q - q)(idx[last]/idx[t_d] - 1)] / Q * 1e4
    # i.e. Perold's total with the index-implied price, so each component's window matches.

def tca_row(order, bars_out, fills_df, beta: float) -> pd.Series
    # concat of: shortfall_true (prefix "true_"), shortfall_observed naive
    # (prefix "obs_") and reversion (prefix "rev_"), slippage_bps (prefix
    # "slip_"), post_trade_moves_bps, q, fill_rate = q/Q, participation_realised
    # = q / market volume over the window, market_adj, is_adj = true_total - market_adj,
    # and the performance view (positive = beat the benchmark), each exactly
    # to_perf(cost) = -cost: tcost_perf_bps = -true_total; <c>_perf_bps =
    # -true_<c> for c in delay, spread, impact, timing, opportunity, fees;
    # <b>_perf_bps = -slip_<b> for b in decision, arrival, interval_vwap, close.

def to_perf(x)  # -x: cost-positive bps -> performance bps, positive = beat the benchmark
```

Errors: `ValueError` if `fills_df` has a minute outside the bars.

### 3.5 `tca_groundtruth.report`

```python
def run_orders(seed: int, n_orders: int, market=MarketParams(), orders=OrderParams(),
               impact=ImpactParams(), fills=FillParams()) -> pd.DataFrame
    # orders joined with their tca_row; one simulated day per order.

def chart_vwap_trap(seed, out_path, rates=(0.01, 0.02, 0.05, 0.10, 0.20, 0.30), n_per_rate=100)
def chart_waterfall(seed, out_path, n_orders=300)
def chart_power(seed, out_path, n_orders=500, delta_bps=2.0, n_boot=500)
def delay_attribution_table(seed, n_orders=4000, flow_share=0.5, market=None) -> pd.DataFrame
def delay_slopes(tab) -> pd.DataFrame   # OLS slope of measured and true delay cost on minutes
def chart_delay_attribution(seed, out_path, n_orders=4000, flow_share=0.5)
def basket_frontier_table(seed, rates=(0.02, 0.05, 0.10, 0.20, 0.30), n_runs=40, n_names=20,
                          buy_share=0.7, qty_pct_adv=(0.005, 0.05), start_minute=30,
                          beta_range=(0.6, 1.4), market=None, impact=None) -> pd.DataFrame
ALMGREN_2005 = ImpactParams(temporary="almgren2005")
def chart_basket_frontier(seed, out_path, rates=..., n_runs=40, n_names=20, **basket_kw)
def make_report(out_dir="reports", seed=20261028, quick=False) -> list[str]
    # writes PNGs and a summary.csv; quick=True uses tiny sizes for the smoke test.
```

`python -m tca_groundtruth.report [--out reports] [--seed N] [--quick]` and
`make report` call `make_report`. Each chart draws from its own generator
seeded from `seed`, so adding a chart does not change the others.

### 3.6 Chart (b): delay vs trader attribution

Inputs: `n_orders` orders from `generate_orders` with default `OrderParams`
except `flow_share = 0.5` (so both groups are large), one simulated day each.
Per order (`delay_attribution_table`, one row per order):

- `delay_min = arrival_minute - decision_minute`, bucketed 0-10, ..., 50-60.
- `delay` = true delay component (the PM-to-desk part, owned by the investment
  process); `trading = spread + impact + timing` (owned by the trader);
  `opportunity`, `fees`, `total` as in `shortfall_true`, so
  `delay + trading + opportunity + fees == total`.
- `alpha_delay = s * alpha_shift[arrival] / P_d * 1e4`: the alpha that realised
  during the delay, known only to the simulator; 0 for flow orders. This is the
  true delay cost; `delay` is its measurement, plus market noise.
- `spread_impact = spread + impact`: the trader's cost without market noise.
- Wrong stamps via `shortfall_true(..., decision_price=...)`: `fallback` sets
  `P_d := P_0` (the order-creation time in the OMS), so `fallback_delay == 0`;
  `open` sets `P_d` to the minute-0 mid, a stamp earlier than the decision (a
  stand-in for the previous close, as there is no overnight).
  `err_x = x_total - total`, bps.

Output `reports/b_delay_attribution.png`, title stating the takeaway. Left:
the true series carry the point: mean `alpha_delay` per bucket (alpha orders;
exactly 0 for flow) and mean `spread_impact` (flat), with the measured mean
`delay` per bucket and its 95% CI drawn faintly behind them; a text box gives
the OLS slope of measured delay cost on delay minutes with its 95% CI, alpha vs
flow, next to the true slope. Right, alpha orders only: mean `err_fallback`
and `err_open` per bucket with 95% CIs, dodged on x, against their true bias
(`-alpha_delay` and 0); the legend states each stamp's per-order error sd.
Summary rows per bucket, slopes and error sds go to `summary.csv`.

### 3.7 Chart (d): basket residual risk vs cost

Inputs: a fixed 20-name basket drawn once from the seed (betas uniform 0.6 to
1.4, 70% buys, quantity log-uniform 0.5% to 5% of daily volume). For each of
`n_runs` days, `simulate_universe` with those betas gives one market day shared
by every speed (common random numbers); each name is traded by a zero-alpha POV
order from minute 30 to the last bar, no fees, at participation 2%, 5%, 10%,
20%, 30%. Per (run, rate), in bps of basket notional `sum(Q * P_d)`:

- `is`: basket true implementation shortfall (unfilled shares carry opportunity
  cost to the close).
- `is_adj`: basket market-adjusted shortfall with each name's known beta.
- `cost`: `is` minus the no-trade move of the intended position (filled shares
  from decision to their fill minute, unfilled shares to the last bar); this is
  spread plus all of the order's own impact, including impact left on the
  unfilled remainder. Its mean is the true expected cost.
- `fill_rate`: executed notional / intended notional.

The table is built twice on the same days: with the default propagator and with
`impact=ALMGREN_2005` (rate-dependent temporary impact, comparison only).

Output `reports/d_basket_frontier.png`, title stating the takeaway, three
panels with the POV rate on a log x axis in the first two. Left: mean `cost`
per rate with 95% CI for both impact models, fill rate annotated. Middle: sd
of `is` (unhedged) and of `is_adj` (beta-hedged) across runs, propagator model.
Right: the cost-risk frontier (x = unhedged sd, y = mean cost) for both
models. `summary.csv` holds both summaries; the comparison columns carry the
suffix `_ac2005`.

## 4. Acceptance tests (inputs -> expected outputs)

1. `u_profile(390, 2.0)`: length 390, all > 0, mean within 1e-9 of 1.0,
   `max/min` within 1e-9 of 2.0, symmetric.
2. `simulate_day(rng)`: 390 rows, columns exactly as the schema, `mid > 0`,
   `volume > 0`, `spread` strictly decreasing, `index_mid[0] == 100`.
   Same seed twice -> identical frames.
3. `simulate_universe(rng, 5)`: 5 * 390 rows, one `index_mid` path shared by
   all names, `beta` within range.
4. `generate_orders(rng, 200, 1e6)`: 200 rows, `arrival_minute < end_minute`,
   `end_minute <= 389`, `flow == (alpha_bps == 0)`, flow share within 0.15 of
   0.3, `qty >= 1`.
5. `apply_alpha`: `alpha_shift == 0` before decision; sign equals `side`;
   final shift within 1% of `side * alpha_bps/1e4 * mid[t_d]` when
   the remaining day is more than 10 half-lives.
6. `schedule`: sum of targets `<= qty`, zero outside window; TWAP targets
   equal within 1 share inside the window; VWAP targets proportional to volume.
7. `execute` with `passive_frac=0`, `max_participation=1` and a large window:
   fills sum to `qty`; every `price == mid_affected + side*spread/2`.
   `gamma_perm=0`: `impact_shift` at the last bar is below half its maximum.
   `Y=0, gamma_perm=0.3`: `impact_shift` is non-decreasing and ends within 2% of
   `gamma_perm * sigma_day * Q / V_day * mid`. Constant-rate POV order with
   `gamma_perm=0`: peak impact within 25% of `Y * sigma_day * sqrt(q / V_day)`.
8. **Property (hypothesis):** for random side, Q, fills (random minutes,
   quantities, prices), `shortfall_true` components sum to
   `s*q*(P_bar - P_d) + s*(Q-q)*(P_c - P_d) + F` within 1e-6 bps, and the same
   for `shortfall_observed` with both methods.
9. **Zero alpha, zero impact (`Y=0`), `passive_frac=0`, no delay, full fill:**
   `true_total` equals `fees + spread` and `spread` equals the fill-weighted
   half spread in bps of `Q*P_d`, within 1e-6; `impact == 0`, `delay == 0`,
   `opportunity == 0`.
10. `benchmarks`: a single aggressive full fill in one minute at
    `mid_affected + spread/2` gives `interval_vwap` strictly between the market
    mid and the fill price when market volume > 0.
11. `tca_row` on an order with no fills: `fill_rate == 0`, slippage NaN,
    `true_total == s*Q*(P_c - P_d)/(Q P_d)*1e4` (pure opportunity + delay).
12. Report smoke: `make_report(tmp_path, seed=1, quick=True)` returns at least
    three paths that exist, PNG magic bytes present, `summary.csv` readable.
13. Chart (b), small n: `delay + trading + opportunity + fees == total` per
    order; `fallback_delay == 0` exactly; flow orders have `alpha_delay == 0`
    and mean delay cost within 3 standard errors of 0; `err_fallback` equals
    `-delay` up to the denominator change.
14. Chart (d), small basket and coarse grid: residual risk decreases and fill
    rate increases with speed; market-adjusted risk <= unhedged risk at every
    speed; the range of expected cost across speeds is under a tenth of the
    range of risk (a completed order's impact barely depends on its rate when
    `delta + beta_kernel = 1`).
15. `shortfall_true(..., decision_price=P_0)`: delay == 0, other components
    keep their currency value; `decision_price <= 0` or NaN raises.
16. Chart (b), true series (seed 5, 600 orders): `alpha_delay == 0` on every
    flow order; the per-bucket mean `alpha_delay` on alpha orders increases
    bucket by bucket; the measured delay slope is within 3 standard errors of
    the true slope for both order types; `spread_impact` has no slope in the
    delay (within 3 standard errors).
17. Chart (d) with `impact=ALMGREN_2005`, small basket: expected cost increases
    with speed (more than doubling from 2% to 30%), risk still decreases, fill
    rates match the propagator run.
18. `execute` with `temporary="almgren2005"` and no permanent part: the impact
    shift is non-zero only at minutes with fills and equals
    `eta * sigma_day * (n/v)^beta_temp * mid`; the default equals
    `temporary="propagator"` exactly; unknown `temporary`, `eta < 0` or
    `beta_temp <= 0` raise.
19. **Property (hypothesis), performance view:** on the random orders of test 8,
    every `*_perf_bps` column of `tca_row` equals minus its cost column exactly
    (NaN where the cost is NaN), and the IS-component columns (`delay`,
    `spread`, `impact`, `timing`, `opportunity`, `fees` `_perf_bps`) sum to
    `tcost_perf_bps` within 1e-6 bps.

## 5. Success criteria (runnable)

```
.venv/bin/ruff check .            # no findings
.venv/bin/pytest -q               # all green
make report                       # writes reports/*.png and reports/summary.csv from the fixed seed
```

CI (`.github/workflows/ci.yml`) runs the first two on push and pull request.
