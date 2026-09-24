# Methodology

## Expected returns and covariance

Expected returns are estimated from sample means in the current research implementation. Covariance is estimated with Ledoit-Wolf shrinkage to reduce sampling noise and improve numerical stability.

## Constrained optimization

The optimizer supports:

- long-only or bounded exposures
- minimum and maximum weights
- turnover constraints
- benchmark-relative active-weight limits
- tracking-error limits
- sector caps
- full-investment budget constraints

A feasible initial portfolio is constructed before SLSQP begins. Active-weight constraints are represented as vector inequalities rather than a non-smooth maximum-absolute-value constraint. Final weights are audited after optimization.

## Portfolio families

Minimum variance minimizes total portfolio variance.

Maximum Sharpe maximizes expected excess return per unit of volatility under the same constraints.

Mean-variance optimization trades expected return against quadratic risk aversion.

Risk parity minimizes dispersion in component risk contributions.

Black-Litterman begins from equilibrium implied returns and blends explicit views into a posterior expected-return vector before constrained optimization.

## Benchmark-relative analytics

Each portfolio is compared with the benchmark using:

- active return
- tracking error
- information ratio

Risk is decomposed into marginal and component contributions. Concentration is measured using a Herfindahl-style sum of squared weights.

## Attribution

Brinson-Fachler attribution separates:

- allocation effect
- selection effect
- interaction effect

at the chosen grouping level. Attribution is deliberately kept distinct from portfolio optimization.

## Live validation interpretation

In the current live run, the Black-Litterman portfolio has the strongest information ratio and lowest tracking error among the active portfolios, while risk parity is the least concentrated.

These results describe the behavior of the optimization framework over the observed sample. They should not be interpreted as a forecast that one portfolio will outperform in the future.
