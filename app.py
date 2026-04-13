import streamlit as st
import pandas as pd
import numpy as np
import io

# ─────────────────────────────────────────────
# PAGE CONFIG & CSS (VERSIÓN SEGURA)
# ─────────────────────────────────────────────
st.set_page_config(page_title="NBA PROPS & HUNTER", page_icon="🏀", layout="wide")

st.markdown("""
<style>
    /* Clases seguras solo para las cajas de resultados */
    .metric-box { background: #12151f; border: 1px solid #1e2a3a; padding: 15px; border-radius: 8px; text-align: center; }
    .metric-title { font-size: 0.8rem; color: #8890a0; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #3a8ef6; }
    .metric-green { color: #30d96a !important; }
    .metric-red { color: #f04545 !important; }
    .section-title { color: #3a8ef6; border-bottom: 1px solid #1e2a3a; padding-bottom: 5px; margin-top: 30px; margin-bottom: 15px; text-transform: uppercase; font-size: 1rem;}
</style>
""", unsafe_allow_html=True)
