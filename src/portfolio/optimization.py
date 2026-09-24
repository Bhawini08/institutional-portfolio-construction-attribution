from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

@dataclass
class Constraints:
    long_only: bool=True
    min_weight: float=0.0
    max_weight: float=0.25
    turnover_limit: float|None=None
    benchmark_active_limit: float|None=None
    tracking_error_limit: float|None=None
    sector_limits: dict[str,float]|None=None

def ledoit_wolf_cov(returns: pd.DataFrame, annualization=252):
    lw=LedoitWolf().fit(returns.dropna().values)
    return pd.DataFrame(lw.covariance_*annualization,index=returns.columns,columns=returns.columns)

def expected_returns(returns: pd.DataFrame, annualization=252):
    return returns.mean()*annualization

def portfolio_vol(w,cov):
    return float(np.sqrt(w@cov@w))

def _constraints(n,cov,cfg,prev,bench,sectors,assets):
    cons=[{"type":"eq","fun":lambda w: np.sum(w)-1}]
    if cfg.turnover_limit is not None and prev is not None:
        cons.append({"type":"ineq","fun":lambda w: cfg.turnover_limit-np.sum(np.abs(w-prev))})
    if cfg.benchmark_active_limit is not None and bench is not None:
        cons.append({"type":"ineq","fun":lambda w: cfg.benchmark_active_limit-np.max(np.abs(w-bench))})
    if cfg.tracking_error_limit is not None and bench is not None:
        cons.append({"type":"ineq","fun":lambda w: cfg.tracking_error_limit-np.sqrt((w-bench)@cov@(w-bench))})
    if cfg.sector_limits and sectors:
        for sec,limit in cfg.sector_limits.items():
            idx=np.array([i for i,a in enumerate(assets) if sectors.get(a)==sec],dtype=int)
            if len(idx):
                cons.append({"type":"ineq","fun":lambda w,idx=idx,limit=limit: limit-w[idx].sum()})
    return cons

def optimize(mu,cov,objective="max_sharpe",rf=0.0,cfg=None,prev=None,bench=None,sectors=None):
    cfg=cfg or Constraints(); assets=list(mu.index); n=len(assets)
    C=np.asarray(cov.loc[assets,assets],float); m=np.asarray(mu.loc[assets],float)
    p=None if prev is None else np.asarray(prev.reindex(assets).fillna(0),float)
    b=None if bench is None else np.asarray(bench.reindex(assets).fillna(0),float)
    bounds=[(cfg.min_weight if cfg.long_only else -cfg.max_weight,cfg.max_weight) for _ in range(n)]
    x0=b.copy() if b is not None and abs(b.sum()-1)<1e-8 else np.repeat(1/n,n)
    if objective=="min_variance": fun=lambda w: w@C@w
    elif objective=="max_sharpe": fun=lambda w: -(w@m-rf)/(np.sqrt(w@C@w)+1e-12)
    elif objective=="mean_variance": fun=lambda w: -(w@m-3.0*w@C@w)
    else: raise ValueError("unknown objective")
    res=minimize(fun,x0,method="SLSQP",bounds=bounds,constraints=_constraints(n,C,cfg,p,b,sectors,assets),
                 options={"maxiter":3000,"ftol":1e-12})
    if not res.success: raise RuntimeError("optimization failed: "+res.message)
    return pd.Series(res.x,index=assets,name=objective)

def risk_parity(cov,max_weight=0.35):
    assets=list(cov.index); C=np.asarray(cov,float); n=len(assets)
    def obj(w):
        vol=np.sqrt(w@C@w); mrc=C@w/(vol+1e-12); crc=w*mrc
        return np.sum((crc-crc.mean())**2)
    res=minimize(obj,np.repeat(1/n,n),method="SLSQP",bounds=[(1e-6,max_weight)]*n,
                 constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],options={"maxiter":3000,"ftol":1e-14})
    if not res.success: raise RuntimeError(res.message)
    return pd.Series(res.x,index=assets,name="risk_parity")

def black_litterman(cov,market_weights,P,Q,tau=0.05,risk_aversion=2.5,omega=None):
    C=np.asarray(cov,float); w=np.asarray(market_weights.reindex(cov.index),float)
    pi=risk_aversion*C@w; P=np.asarray(P,float); Q=np.asarray(Q,float)
    Omega=np.diag(np.diag(P@(tau*C)@P.T)) if omega is None else np.asarray(omega,float)
    middle=np.linalg.inv(np.linalg.inv(tau*C)+P.T@np.linalg.inv(Omega)@P)
    post=middle@(np.linalg.inv(tau*C)@pi+P.T@np.linalg.inv(Omega)@Q)
    return pd.Series(post,index=cov.index,name="bl_expected_return")
