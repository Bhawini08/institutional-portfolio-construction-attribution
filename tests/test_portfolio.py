import numpy as np
import pandas as pd
from portfolio.data import synthetic_returns, benchmark_weights, SECTORS
from portfolio.optimization import expected_returns,ledoit_wolf_cov,optimize,risk_parity,Constraints,black_litterman
from portfolio.risk import risk_contributions
from portfolio.attribution import brinson_fachler

def test_optimizers_respect_budget_and_bounds():
    r=synthetic_returns(400); mu=expected_returns(r); cov=ledoit_wolf_cov(r); b=benchmark_weights().reindex(r.columns).fillna(0)
    w=optimize(mu,cov,"min_variance",cfg=Constraints(max_weight=.35),bench=b,sectors=SECTORS)
    assert abs(w.sum()-1)<1e-6 and (w>=-1e-8).all() and (w<=.350001).all()
    rp=risk_parity(cov); assert abs(rp.sum()-1)<1e-6

def test_risk_contributions_sum_to_volatility():
    r=synthetic_returns(300); cov=ledoit_wolf_cov(r); w=pd.Series(1/len(r.columns),index=r.columns)
    rc=risk_contributions(w,cov); vol=np.sqrt(w.values@cov.values@w.values)
    assert abs(rc.component_risk.sum()-vol)<1e-8

def test_black_litterman_and_brinson():
    r=synthetic_returns(300); cov=ledoit_wolf_cov(r); b=benchmark_weights().reindex(r.columns).fillna(0)
    P=np.zeros((1,len(r.columns))); P[0,0]=1; Q=np.array([.07]); post=black_litterman(cov,b,P,Q)
    assert post.notna().all()
    x=brinson_fachler(pd.Series({"A":.6,"B":.4}),pd.Series({"A":.5,"B":.5}),pd.Series({"A":.1,"B":.05}),pd.Series({"A":.08,"B":.06}))
    assert np.isfinite(x.total_effect).all()
