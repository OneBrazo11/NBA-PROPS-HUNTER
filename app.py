import streamlit as st
import pandas as pd
import numpy as np
import io

# ─────────────────────────────────────────────
# PAGE CONFIG & CSS
# ─────────────────────────────────────────────
st.set_page_config(page_title="NBA PROPS & HUNTER", page_icon="🏀", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Mono', monospace; }
    .stApp { background: #0a0c10; color: #e8eaf0; }
    .metric-box { background: #12151f; border: 1px solid #1e2a3a; padding: 15px; border-radius: 8px; text-align: center; }
    .metric-title { font-size: 0.8rem; color: #8890a0; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #3a8ef6; }
    .metric-green { color: #30d96a !important; }
    .metric-red { color: #f04545 !important; }
    .section-title { color: #3a8ef6; border-bottom: 1px solid #1e2a3a; padding-bottom: 5px; margin-top: 30px; margin-bottom: 15px; text-transform: uppercase; font-size: 1rem;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA PROCESSING
# ─────────────────────────────────────────────
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[\s\-/]", "_", regex=True).str.replace(r"[^a-z0-9_]", "", regex=True)
    return df

def _col(df: pd.DataFrame, candidates: list) -> str | None:
    for c in candidates:
        if c in df.columns: return c
    return None

@st.cache_data(show_spinner=False)
def load_data(files_bytes: dict) -> dict:
    frames = {}
    for key, (name, raw) in files_bytes.items():
        try:
            if name.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(io.BytesIO(raw))
            else:
                for enc in ("utf-8", "latin-1", "cp1252"):
                    try:
                        df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                        break
                    except Exception: continue
            if df is not None: frames[key] = _normalize_cols(df)
        except Exception:
            frames[key] = pd.DataFrame()
    return frames

# ─────────────────────────────────────────────
# INJURY IMPACT ENGINE
# ─────────────────────────────────────────────
def calculate_absence_impact(team_name: str, out_players: list, impact_df: pd.DataFrame) -> float:
    total_impact = 0.0
    if impact_df.empty or not out_players: return 0.0
    
    p_col = _col(impact_df, ["player", "name"])
    diff_col = _col(impact_df, ["diff_pts_per_100_poss", "point_differential", "on_off_diff", "diff"])
    
    if p_col and diff_col:
