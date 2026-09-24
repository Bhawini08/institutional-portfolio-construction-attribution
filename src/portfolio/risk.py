import numpy as np
import pandas as pd

def risk_contributions(weights: pd.Series,cov: pd.DataFrame):
    w=weights.reindex(cov.index).values; C=cov.values; vol=np.sqrt(w@C@w)
    mrc=C@w/(vol+1e-12); crc=w*mrc
    return pd.DataFrame({"weight":w,"marginal_risk":mrc,"component_risk":crc,
                         "component_risk_pct":crc/(crc.sum()+1e-12)},index=cov.index)

def active_metrics(portfolio_returns,benchmark_returns,annualization=252):
    active=portfolio_returns-benchmark_returns; te=active.std(ddof=1)*np.sqrt(annualization)
    ar=active.mean()*annualization; ir=ar/te if te>0 else np.nan
    return {"active_return":float(ar),"tracking_error":float(te),"information_ratio":float(ir)}

def concentration(weights):
    return float(np.sum(np.square(weights)))

def drawdown(returns):
    wealth=(1+returns.fillna(0)).cumprod(); return wealth/wealth.cummax()-1
