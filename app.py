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
    """Calcula el Net Rating perdido por ausencias."""
    total_impact = 0.0
    if impact_df.empty or not out_players: return 0.0
    
    p_col = _col(impact_df, ["player", "name"])
    diff_col = _col(impact_df, ["diff_pts_per_100_poss", "point_differential", "on_off_diff", "diff"])
    
    if p_col and diff_col:
        for player in out_players:
            val = impact_df[impact_df[p_col].astype(str).str.contains(player, na=False)][diff_col]
            if not val.empty:
                impact_val = pd.to_numeric(val.iloc[0], errors='coerce')
                if not np.isnan(impact_val):
                    total_impact -= impact_val 
    return total_impact

# ─────────────────────────────────────────────
# GAME HUNTER LOGIC
# ─────────────────────────────────────────────
def _team_momentum(summary_df: pd.DataFrame, team: str) -> dict:
    res = {"offrtg": np.nan, "defrtg": np.nan, "pace": np.nan}
    if summary_df.empty: return res
    tc = _col(summary_df, ["team", "tm", "team_name"])
    if not tc: return res
    mask = summary_df[tc].astype(str).str.lower().str.contains(team.lower(), na=False)
    tdf = summary_df[mask].copy()
    
    l10_col = _col(tdf, ["split", "last_n", "games"])
    if l10_col:
        l10_mask = tdf[l10_col].astype(str).str.lower().str.contains("l10|last", regex=True, na=False)
        if l10_mask.any(): tdf = tdf[l10_mask]

    oc = _col(tdf, ["offrtg", "ortg", "offensive_rating"])
    dc = _col(tdf, ["defrtg", "drtg", "defensive_rating"])
    pc = _col(tdf, ["pace"])
    
    if oc: res["offrtg"] = pd.to_numeric(tdf[oc], errors="coerce").mean()
    if dc: res["defrtg"] = pd.to_numeric(tdf[dc], errors="coerce").mean()
    if pc: res["pace"] = pd.to_numeric(tdf[pc], errors="coerce").mean()
    return res

def project_game(home: str, away: str, frames: dict, h_impact: float, a_impact: float) -> dict:
    sdf = frames.get("nbaleaguesumary", pd.DataFrame())
    hm = _team_momentum(sdf, home)
    am = _team_momentum(sdf, away)
    
    pace = np.nanmean([hm["pace"], am["pace"]]) if not np.isnan(np.nanmean([hm["pace"], am["pace"]])) else 98.5
    
    # Ajustar Ratings por lesiones
    h_ortg_adj = hm["offrtg"] + (h_impact / 2) if not np.isnan(hm["offrtg"]) else np.nan
    h_drtg_adj = hm["defrtg"] - (h_impact / 2) if not np.isnan(hm["defrtg"]) else np.nan
    
    a_ortg_adj = am["offrtg"] + (a_impact / 2) if not np.isnan(am["offrtg"]) else np.nan
    a_drtg_adj = am["defrtg"] - (a_impact / 2) if not np.isnan(am["defrtg"]) else np.nan

    h_pts = np.nanmean([h_ortg_adj, a_drtg_adj]) * (pace / 100) if not np.isnan(h_ortg_adj) else np.nan
    a_pts = np.nanmean([a_ortg_adj, h_drtg_adj]) * (pace / 100) if not np.isnan(a_ortg_adj) else np.nan
    
    # Ventaja de Localía
    if not np.isnan(h_pts): h_pts += 1.5
    if not np.isnan(a_pts): a_pts -= 1.5

    tot = h_pts + a_pts
    spread = a_pts - h_pts 
    
    winner = home if h_pts > a_pts else away
    win_margin = abs(h_pts - a_pts)

    return {"home_pts": h_pts, "away_pts": a_pts, "total": tot, "spread": spread, "winner": winner, "margin": win_margin}

# ─────────────────────────────────────────────
# PROP ASSASSIN LOGIC
# ─────────────────────────────────────────────
def _player_metrics(pdf: pd.DataFrame, idf: pd.DataFrame, player: str) -> dict:
    res = {"pts": np.nan, "reb": np.nan, "ast": np.nan, "usg": np.nan, "min": np.nan}
    if pdf.empty: return res
    
    nc = _col(pdf, ["player", "name"])
    if nc:
        mask = pdf[nc].astype(str).str.lower().str.contains(player.lower(), na=False)
        p1 = pdf[mask].copy()
        sc = _col(p1, ["split", "games"])
        if sc:
            l10 = p1[p1[sc].astype(str).str.lower().str.contains("l10|last", regex=True, na=False)]
            if not l10.empty: p1 = l10
        
        c_pts, c_reb, c_ast = _col(p1, ["pts", "points"]), _col(p1, ["reb", "trb"]), _col(p1, ["ast", "assists"])
        if c_pts: res["pts"] = pd.to_numeric(p1[c_pts], errors="coerce").mean()
        if c_reb: res["reb"] = pd.to_numeric(p1[c_reb], errors="coerce").mean()
        if c_ast: res["ast"] = pd.to_numeric(p1[c_ast], errors="coerce").mean()

    if not idf.empty:
        nic = _col(idf, ["player", "name"])
        if nic:
            imask = idf[nic].astype(str).str.lower().str.contains(player.lower(), na=False)
            i1 = idf[imask]
            c_usg, c_min = _col(i1, ["usg", "usage"]), _col(i1, ["min", "minutes"])
            if c_usg: res["usg"] = pd.to_numeric(i1[c_usg], errors="coerce").mean()
            if c_min: res["min"] = pd.to_numeric(i1[c_min], errors="coerce").mean()
            
    return res

def _dvp_factor(def_df: pd.DataFrame, opp: str, pos: str) -> float:
    if def_df.empty: return 1.0
    tc, pc = _col(def_df, ["team", "opp", "opponent", "tm"]), _col(def_df, ["pos", "position"])
    if not tc: return 1.0
    t_df = def_df[def_df[tc].astype(str).str.lower().str.contains(opp.lower(), na=False)]
    if pc and not t_df.empty:
        p_df = t_df[t_df[pc].astype(str).str.lower().str.contains(str(pos).lower()[:2], na=False)]
        if not p_df.empty: t_df = p_df
    
    rc = _col(t_df, ["rel", "diff", "vs_avg", "pct_diff"])
    if rc and not t_df.empty:
        val = pd.to_numeric(t_df[rc], errors="coerce").mean()
        if not np.isnan(val): return 1.0 + (val / 100.0)
    return 1.0

def evaluate_risk(dvp: float, usg: float, mins: float) -> tuple:
    if
