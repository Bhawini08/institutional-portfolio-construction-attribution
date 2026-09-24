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

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
pytest -q
python scripts/run_portfolio_analysis.py
streamlit run dashboard/app.py
```

Synthetic mode is used for reproducible engineering validation. A live ETF/fund universe can be validated separately.
