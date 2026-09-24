import pandas as pd

def brinson_fachler(port_w,bench_w,port_ret,bench_ret):
    idx=sorted(set(port_w.index)|set(bench_w.index)|set(port_ret.index)|set(bench_ret.index))
    wp=port_w.reindex(idx).fillna(0); wb=bench_w.reindex(idx).fillna(0)
    rp=port_ret.reindex(idx).fillna(0); rb=bench_ret.reindex(idx).fillna(0)
    total_bench=float((wb*rb).sum())
    allocation=(wp-wb)*(rb-total_bench)
    selection=wb*(rp-rb)
    interaction=(wp-wb)*(rp-rb)
    out=pd.DataFrame({"portfolio_weight":wp,"benchmark_weight":wb,"portfolio_return":rp,"benchmark_return":rb,
                      "allocation":allocation,"selection":selection,"interaction":interaction})
    out["total_effect"]=out[["allocation","selection","interaction"]].sum(axis=1)
    return out
