# Implied Volatility Surface

A 3D implied volatility surface built from real options market data. The project takes a live options chain, inverts the Black–Scholes formula contract by contract to recover each option's implied volatility, and renders the resulting **strike × maturity × IV** surface — exposing the volatility skew and term structure that a single Black–Scholes number cannot capture.

Built as an extension of a Black–Scholes / Monte Carlo option pricer, this project focuses on the inverse problem: given the *price*, recover the *volatility* the market is using.

<img width="832" height="698" alt="image" src="https://github.com/user-attachments/assets/b41f8248-df54-4b95-9ab5-c7d03cb7b4d4" />

---

## What is an implied volatility surface?

Black–Scholes takes volatility as an **input** and returns a price. In the real market the price is *observed* and volatility is the unknown. **Implied volatility (IV)** is the value of σ that, plugged into Black–Scholes, reproduces the option's market price.

If Black–Scholes were a perfect model, every option on the same underlying would share one IV. It doesn't. Each combination of strike `K` and maturity `T` yields a different IV, and plotting those values over `(K, T)` produces a surface with structure:

- **Skew** — variation across strikes at fixed maturity. Equity-index surfaces slope upward toward low strikes: OTM puts trade at higher IV than OTM calls, reflecting the market's demand for crash protection.
- **Term structure** — variation across maturities at fixed strike. Typically upward-sloping in calm markets as uncertainty accumulates over longer horizons.

---

## Methodology

The pipeline runs in five stages:

1. **Fetch** — download the full options chain (calls and puts, across maturities) for a ticker via `yfinance`, plus the spot price and a risk-free rate proxy (13-week T-bill, `^IRX`).
2. **Invert** — for each contract, solve `BlackScholes(σ) − market_price = 0` for σ using the **bisection method**.
3. **Clean** — filter on quote validity, liquidity, and moneyness; use the bid–ask **mid price** rather than the (potentially stale) last trade.
4. **Grid** — keep **OTM options on each side** (OTM puts below spot, OTM calls above), then pivot into a `maturity × strike` grid and interpolate interior gaps.
5. **Plot** — render the surface with `matplotlib` (static) and `plotly` (interactive).

### Solving for implied volatility

σ cannot be isolated algebraically from Black–Scholes (it sits inside `d1`, `d2`, and the normal CDF), so IV is found numerically. Because the Black–Scholes price is **monotonically increasing in σ**, the root is unique and bracketed, making bisection a robust choice: it is derivative-free and always converges given a sign change.

```python
def implied_vol(market_price, S, K, T, r, q, option_type='call', tol=1e-6):
    lo, hi = 1e-6, 5.0
    def f(sigma):
        return black_scholes(S, K, T, r, q, sigma, option_type) - market_price
    if f(lo) * f(hi) > 0:        # no sign change -> no root in bracket
        return np.nan            # return NaN instead of fabricating a value
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if f(mid) * f(lo) >= 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
```

### Why OTM options

OTM contracts carry more time value relative to their bid–ask spread, so their IV is more reliable than deep-ITM contracts at the same strike. Restricting the surface to OTM puts (below spot) and OTM calls (above spot) also sidesteps the region where the European Black–Scholes approximation is weakest for American-style equity options, since the early-exercise premium concentrates in the ITM wing.

---

## Reading the surface

- **Skew** is clearly visible: IV rises sharply toward low strikes (OTM puts) and flattens toward high strikes — the classic index "fear of the downside" pattern.
- **Term structure** slopes gently upward at the money as maturity increases, consistent with a calm-market regime.
- IV is least reliable at the shortest maturities, where a tiny `T` amplifies quoting noise; the steepest short-dated skew points should be read with that caveat.

---

## Technical notes

A few decisions and issues worth flagging:

- **Dividend yield.** Ignoring dividends biases IV systematically low, with the error growing toward ITM strikes (higher delta → more sensitivity to the forward level). `yfinance`'s `.info['dividendYield']` field returned a corrupted value, so `q` is set from the known SPY yield rather than trusting a fragile derived field.
- **Mid price over last trade.** The last trade can be stale for illiquid strikes; the bid–ask mid reflects live quotes and yields cleaner IVs.
- **Grid sparsity.** Maturities don't list identical strikes, so the pivoted grid is sparse. Strikes present in fewer than half the maturities are dropped; remaining interior gaps are linearly interpolated (no extrapolation past observed data).
---

## Usage

```python
# build the surface data (samples maturities from ~1 week to ~2 years)
surface_df = build_surface_data("SPY", max_expiries=8)

# pivot to a maturity × strike grid and clean gaps
grid = surface_df.pivot_table(index='T', columns='strike', values='iv', aggfunc='mean')
grid = grid.dropna(axis=1, thresh=grid.shape[0] // 2)
grid = grid.interpolate(axis=1, method='linear', limit_area='inside')

# plot (see notebook for matplotlib + plotly versions)
plot_surface(grid, "SPY")
```
---

## Possible extensions

- **SVI parametrization** — fit an arbitrage-free smile per maturity to smooth the surface and interpolate between listed strikes.
- **Daily snapshots** — capture the surface over time and track how skew and term structure shift around events, a natural bridge into regime-detection (HMM) work.
- **American pricing** — replace the European inversion with a binomial model for exact IVs on American-style equity options.
