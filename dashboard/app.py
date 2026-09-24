import streamlit as st
import pandas as pd
from pathlib import Path
st.set_page_config(page_title="Institutional Portfolio Platform",layout="wide")
st.title("Institutional Portfolio Construction & Attribution Platform")
p=Path("results/portfolio_summary.csv")
if not p.exists(): st.info("Run scripts/run_portfolio_analysis.py first.")
else:
    st.subheader("Portfolio comparison"); st.dataframe(pd.read_csv(p),use_container_width=True)
    w=pd.read_csv("results/portfolio_weights.csv",index_col=0); st.subheader("Portfolio weights"); st.dataframe(w,use_container_width=True)
    st.bar_chart(w)
    a=Path("results/brinson_fachler.csv")
    if a.exists(): st.subheader("Brinson-Fachler attribution"); st.dataframe(pd.read_csv(a,index_col=0),use_container_width=True)
