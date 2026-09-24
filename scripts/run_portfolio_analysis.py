import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from portfolio.data import synthetic_returns, live_returns, benchmark_weights, SECTORS
from portfolio.optimization import *
from portfolio.risk import *
from portfolio.attribution import brinson_fachler
parser=argparse.ArgumentParser(); parser.add_argument("--mode",choices=["synthetic","live"],default="synthetic"); args=parser.parse_args()
out=Path("results"); out.mkdir(exist_ok=True)
r=synthetic_returns() if args.mode=="synthetic" else live_returns(); mu=expected_returns(r); cov=ledoit_wolf_cov(r); bench=benchmark_weights().reindex(r.columns).fillna(0)
cfg=Constraints(max_weight=0.30,benchmark_active_limit=0.20,tracking_error_limit=0.15,sector_limits={"Technology":0.25})
weights={
"min_variance":optimize(mu,cov,"min_variance",cfg=cfg,bench=bench,sectors=SECTORS),
"max_sharpe":optimize(mu,cov,"max_sharpe",cfg=cfg,bench=bench,sectors=SECTORS),
"risk_parity":risk_parity(cov)
}
P=np.zeros((2,len(mu))); P[0,mu.index.get_loc("QQQ")]=1; P[0,mu.index.get_loc("SPY")]=-1; P[1,mu.index.get_loc("IWM")]=1
Q=np.array([0.02,0.07]); bl_mu=black_litterman(cov,bench,P,Q); weights["black_litterman"]=optimize(bl_mu,cov,"max_sharpe",cfg=cfg,bench=bench,sectors=SECTORS)
pd.DataFrame(weights).to_csv(out/"portfolio_weights.csv")
rows=[]
bench_ret=r@bench
for name,w in weights.items():
    pr=r@w; am=active_metrics(pr,bench_ret); rc=risk_contributions(w,cov); rc.to_csv(out/(name+"_risk_contributions.csv"))
    rows.append({"portfolio":name,"annual_return":float(pr.mean()*252),"annual_vol":float(pr.std()*np.sqrt(252)),
                 "concentration_hhi":concentration(w),**am})
pd.DataFrame(rows).to_csv(out/"portfolio_summary.csv",index=False)
# illustrative one-period sector attribution fixture
pw=pd.Series({"Broad":0.35,"Growth":0.25,"Small Cap":0.10,"Value":0.10,"Technology":0.20})
bw=pd.Series({"Broad":0.40,"Growth":0.20,"Small Cap":0.10,"Value":0.15,"Technology":0.15})
pr=pd.Series({"Broad":0.08,"Growth":0.12,"Small Cap":0.05,"Value":0.06,"Technology":0.15})
br=pd.Series({"Broad":0.075,"Growth":0.10,"Small Cap":0.055,"Value":0.065,"Technology":0.13})
attr=brinson_fachler(pw,bw,pr,br); attr.to_csv(out/"brinson_fachler.csv")
print(pd.DataFrame(rows).to_json(orient="records",indent=2))
