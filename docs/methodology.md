# Methodology

Expected returns are estimated from the sample mean only for the deterministic engineering fixture. Covariance is estimated with Ledoit-Wolf shrinkage to reduce sampling noise. Portfolio optimizers use explicit budget, position, turnover, benchmark-relative, sector, and tracking-error constraints where requested.

Risk parity minimizes dispersion in component risk contribution. Black-Litterman starts from equilibrium implied returns and blends investor views through the standard Bayesian posterior construction.

Benchmark-relative evaluation reports active return, tracking error, and information ratio. Risk is decomposed into marginal and component contributions. Brinson-Fachler attribution separates allocation, selection, and interaction effects at the chosen grouping level.

The synthetic fixture validates architecture and accounting. Live portfolio conclusions require a separately validated market-data and benchmark dataset.
