import numpy as np
import pandas as pd

ASSETS=["SPY","QQQ","IWM","IWD","IWF","XLF","XLK","XLE","XLI","XLV","XLP","XLY"]
SECTORS={"SPY":"Broad","QQQ":"Growth","IWM":"Small Cap","IWD":"Value","IWF":"Growth",
"XLF":"Financials","XLK":"Technology","XLE":"Energy","XLI":"Industrials","XLV":"Health Care","XLP":"Staples","XLY":"Discretionary"}

def synthetic_returns(n=756,seed=12):
    rng=np.random.default_rng(seed); idx=pd.bdate_range("2023-01-02",periods=n)
    k=4; f=rng.normal([0.00025,0.00008,0.00003,0.00002],[0.009,0.005,0.004,0.003],size=(n,k))
    loads=rng.normal(0,0.35,size=(len(ASSETS),k)); loads[:,0]=rng.uniform(0.7,1.2,len(ASSETS))
    eps=rng.normal(0,0.006,size=(n,len(ASSETS)))
    r=f@loads.T+eps
    return pd.DataFrame(r,index=idx,columns=ASSETS)

def benchmark_weights():
    w=pd.Series(0.0,index=ASSETS)
    w.loc[["SPY","QQQ","IWM","IWD","IWF"]]=[0.40,0.15,0.10,0.15,0.20]
    return w/w.sum()
