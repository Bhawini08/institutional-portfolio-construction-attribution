# Institutional Portfolio Construction & Attribution Platform

An institutional-style portfolio research platform covering portfolio construction, benchmark-relative constraints, risk decomposition, and Brinson-Fachler attribution.

## Portfolio construction

- Mean-variance optimization
- Minimum variance
- Maximum Sharpe
- Risk parity
- Black-Litterman
- Ledoit-Wolf covariance shrinkage

## Constraints

- Long-only
- Maximum position weight
- Optional minimum position weight
- Sector limits
- Turnover constraints
- Benchmark-relative bounds
- Tracking-error constraints

## Analytics

- Active return and active risk
- Tracking error
- Information ratio
- Marginal and component risk contribution
- Concentration
- Rolling volatility
- Drawdown
- Brinson-Fachler allocation, selection, and interaction attribution

## Live validation snapshot

Using the live ETF universe and the constrained optimization framework:

| Portfolio | Annual return | Annual vol | Tracking error | Information ratio | HHI |
| --- | ---: | ---: | ---: | ---: | ---: |
| Minimum variance | 11.51% | 15.24% | 6.83% | -0.52 | 0.260 |
| Maximum Sharpe | 16.32% | 17.71% | 3.57% | 0.35 | 0.218 |
| Risk parity | 13.44% | 16.98% | 3.85% | -0.42 | 0.087 |
| Black-Litterman | 16.30% | 18.86% | 1.87% | 0.66 | 0.247 |

The Black-Litterman portfolio delivers the strongest benchmark-relative efficiency in this run, with the highest information ratio and the lowest tracking error among the active portfolios. Risk parity produces the lowest concentration by a wide margin.

These are in-sample portfolio-construction diagnostics, not forward-return forecasts.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
pytest -q
python scripts/run_portfolio_analysis.py --mode live
streamlit run dashboard/app.py
```

## Research discipline

Covariance uses Ledoit-Wolf shrinkage to reduce estimation noise. Portfolio constraints are explicitly audited after optimization so numerically successful solutions cannot silently violate budget, position, active-weight, or tracking-error limits.

Brinson-Fachler attribution is implemented as a separate allocation/selection/interaction layer rather than inferred from optimization output.
