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

def _feasible_start(n,cfg,bench=None):
    """Construct a budget-feasible starting point that respects box bounds.

    The benchmark itself may violate portfolio bounds, so it should not be used
    directly as an SLSQP starting point.
    """
    lo=cfg.min_weight if cfg.long_only else -cfg.max_weight
    hi=cfg.max_weight
    if bench is not None:
        x=np.clip(np.asarray(bench,float),lo,hi)
    else:
        x=np.repeat(1/n,n)

    # Redistribute any budget residual across names with available capacity.
    for _ in range(20):
        residual=1.0-x.sum()
        if abs(residual)<1e-10:
            break
        if residual>0:
            room=hi-x
            idx=np.where(room>1e-12)[0]
            if len(idx)==0: break
            add=np.minimum(room[idx],residual/len(idx))
            x[idx]+=add
        else:
            room=x-lo
            idx=np.where(room>1e-12)[0]
            if len(idx)==0: break
            sub=np.minimum(room[idx],(-residual)/len(idx))
            x[idx]-=sub

    if abs(x.sum()-1)>1e-6:
        raise RuntimeError("Unable to construct feasible starting weights under box constraints")
    return x

def _constraints(n,cov,cfg,prev,bench,sectors,assets):
    cons=[{"type":"eq","fun":lambda w: np.sum(w)-1}]

    if cfg.turnover_limit is not None and prev is not None:
        cons.append({
            "type":"ineq",
            "fun":lambda w: cfg.turnover_limit-np.sum(np.abs(w-prev))
        })

    if cfg.benchmark_active_limit is not None and bench is not None:
        # Vector inequalities avoid the non-smooth max(abs(.)) formulation.
        cons.append({
            "type":"ineq",
            "fun":lambda w: cfg.benchmark_active_limit-(w-bench)
        })
        cons.append({
            "type":"ineq",
            "fun":lambda w: cfg.benchmark_active_limit+(w-bench)
        })

    if cfg.tracking_error_limit is not None and bench is not None:
        cons.append({
            "type":"ineq",
            "fun":lambda w: cfg.tracking_error_limit**2-(w-bench)@cov@(w-bench)
        })

    if cfg.sector_limits and sectors:
        for sec,limit in cfg.sector_limits.items():
            idx=np.array([i for i,a in enumerate(assets) if sectors.get(a)==sec],dtype=int)
            if len(idx):
                cons.append({
                    "type":"ineq",
                    "fun":lambda w,idx=idx,limit=limit: limit-w[idx].sum()
                })
    return cons

def optimize(mu,cov,objective="max_sharpe",rf=0.0,cfg=None,prev=None,bench=None,sectors=None):
    cfg=cfg or Constraints()
    assets=list(mu.index)
    n=len(assets)
    C=np.asarray(cov.loc[assets,assets],float)
    m=np.asarray(mu.loc[assets],float)
    p=None if prev is None else np.asarray(prev.reindex(assets).fillna(0),float)
    b=None if bench is None else np.asarray(bench.reindex(assets).fillna(0),float)

    lo=cfg.min_weight if cfg.long_only else -cfg.max_weight
    bounds=[(lo,cfg.max_weight) for _ in range(n)]
    x0=_feasible_start(n,cfg,b)

    if objective=="min_variance":
        fun=lambda w: w@C@w
    elif objective=="max_sharpe":
        fun=lambda w: -(w@m-rf)/(np.sqrt(w@C@w)+1e-12)
    elif objective=="mean_variance":
        fun=lambda w: -(w@m-3.0*w@C@w)
    else:
        raise ValueError("unknown objective")

    cons=_constraints(n,C,cfg,p,b,sectors,assets)
    res=minimize(
        fun,x0,method="SLSQP",bounds=bounds,constraints=cons,
        options={"maxiter":5000,"ftol":1e-10,"disp":False}
    )

    if not res.success:
        # Retry from equal-weight if the benchmark-projected start lands close
        # to a difficult boundary.
        alt=_feasible_start(n,cfg,None)
        res=minimize(
            fun,alt,method="SLSQP",bounds=bounds,constraints=cons,
            options={"maxiter":5000,"ftol":1e-10,"disp":False}
        )

    if not res.success:
        raise RuntimeError("optimization failed: "+res.message)

    w=pd.Series(res.x,index=assets,name=objective)

    # Post-solve audit so a numerically "successful" solution cannot silently
    # violate institutional constraints.
    if abs(w.sum()-1)>1e-5:
        raise RuntimeError("optimized weights violate budget constraint")
    if (w<lo-1e-6).any() or (w>cfg.max_weight+1e-6).any():
        raise RuntimeError("optimized weights violate position bounds")
    if b is not None and cfg.benchmark_active_limit is not None:
        if np.max(np.abs(w.values-b))>cfg.benchmark_active_limit+1e-5:
            raise RuntimeError("optimized weights violate active-weight constraint")
    if b is not None and cfg.tracking_error_limit is not None:
        te=np.sqrt((w.values-b)@C@(w.values-b))
        if te>cfg.tracking_error_limit+1e-5:
            raise RuntimeError("optimized weights violate tracking-error constraint")

    return w

def risk_parity(cov,max_weight=0.35):
    assets=list(cov.index); C=np.asarray(cov,float); n=len(assets)
    def obj(w):
        vol=np.sqrt(w@C@w); mrc=C@w/(vol+1e-12); crc=w*mrc
        return np.sum((crc-crc.mean())**2)
    res=minimize(
        obj,np.repeat(1/n,n),method="SLSQP",
        bounds=[(1e-6,max_weight)]*n,
        constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],
        options={"maxiter":3000,"ftol":1e-14}
    )
    if not res.success:
        raise RuntimeError(res.message)
    return pd.Series(res.x,index=assets,name="risk_parity")

def black_litterman(cov,market_weights,P,Q,tau=0.05,risk_aversion=2.5,omega=None):
    C=np.asarray(cov,float); w=np.asarray(market_weights.reindex(cov.index),float)
    pi=risk_aversion*C@w; P=np.asarray(P,float); Q=np.asarray(Q,float)
    Omega=np.diag(np.diag(P@(tau*C)@P.T)) if omega is None else np.asarray(omega,float)
    middle=np.linalg.inv(np.linalg.inv(tau*C)+P.T@np.linalg.inv(Omega)@P)
    post=middle@(np.linalg.inv(tau*C)@pi+P.T@np.linalg.inv(Omega)@Q)
    return pd.Series(post,index=cov.index,name="bl_expected_return")
