import streamlit as st
import pandas as pd
import numpy as np
import io

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(page_title="NBA PROPS & HUNTER", page_icon="🏀", layout="wide")

# ─────────────────────────────────────────────
# PROCESAMIENTO DE DATOS (FUNCIONES INTERNAS)
# ─────────────────────────────────────────────
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[\s\-/]", "_", regex=True).str.replace(r"[^a-z0-9_]", "", regex=True)
    return df

def _col(df: pd.DataFrame, candidates: list) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

@st.cache_data(show_spinner=False)
def load_data(files_bytes: dict) -> dict:
    frames = {}
    for key, (name, raw) in files_bytes.items():
        try:
            df = None
            if name.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(io.BytesIO(raw))
            else:
                for enc in ("utf-8", "latin-1", "cp1252"):
                    try:
                        df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                        break
                    except Exception:
                        continue
            
            if df is not None:
                frames[key] = _normalize_cols(df)
        except Exception:
            frames[key] = pd.DataFrame()
            
    return frames

# ─────────────────────────────────────────────
# MOTOR DE IMPACTO DE LESIONES
# ─────────────────────────────────────────────
def calculate_absence_impact(team_name: str, out_players: list, impact_df: pd.DataFrame) -> float:
    total_impact = 0.0
    if impact_df.empty or not out_players:
        return 0.0
    
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
# LÓGICA DE GAME HUNTER
# ─────────────────────────────────────────────
def _team_momentum(summary_df: pd.DataFrame, team: str) -> dict:
    res = {"offrtg": np.nan, "defrtg": np.nan, "pace": np.nan}
    if summary_df.empty:
        return res
        
    tc = _col(summary_df, ["team", "tm", "team_name"])
    if not tc:
        return res
        
    mask = summary_df[tc].astype(str).str.lower().str.contains(team.lower(), na=False)
    tdf = summary_df[mask].copy()
    
    l10_col = _col(tdf, ["split", "last_n", "games"])
    if l10_col:
        l10_mask = tdf[l10_col].astype(str).str.lower().str.contains("l10|last", regex=True, na=False)
        if l10_mask.any():
            tdf = tdf[l10_mask]

    oc = _col(tdf, ["offrtg", "ortg", "offensive_rating"])
    dc = _col(tdf, ["defrtg", "drtg", "defensive_rating"])
    pc = _col(tdf, ["pace"])
    
    if oc:
        res["offrtg"] = pd.to_numeric(tdf[oc], errors="coerce").mean()
    if dc:
        res["defrtg"] = pd.to_numeric(tdf[dc], errors="coerce").mean()
    if pc:
        res["pace"] = pd.to_numeric(tdf[pc], errors="coerce").mean()
        
    return res

def project_game(home: str, away: str, frames: dict, h_impact: float, a_impact: float) -> dict:
    sdf = frames.get("nbaleaguesumary", pd.DataFrame())
    hm = _team_momentum(sdf, home)
    am = _team_momentum(sdf, away)
    
    avg_pace = np.nanmean([hm["pace"], am["pace"]])
    pace = avg_pace if not np.isnan(avg_pace) else 98.5
    
    # Ajustar Ratings por lesiones (CORRECCIÓN DE VARIABLE DEFRTG)
    h_ortg_adj = hm["offrtg"] + (h_impact / 2) if not np.isnan(hm["offrtg"]) else np.nan
    h_drtg_adj = hm["defrtg"] - (h_impact / 2) if not np.isnan(hm["defrtg"]) else np.nan
    
    a_ortg_adj = am["offrtg"] + (a_impact / 2) if not np.isnan(am["offrtg"]) else np.nan
    a_drtg_adj = am["defrtg"] - (a_impact / 2) if not np.isnan(am["defrtg"]) else np.nan

    h_pts = np.nanmean([h_ortg_adj, a_drtg_adj]) * (pace / 100) if not np.isnan(h_ortg_adj) else np.nan
    a_pts = np.nanmean([a_ortg_adj, h_drtg_adj]) * (pace / 100) if not np.isnan(a_ortg_adj) else np.nan
    
    # Ventaja de Localía
    if not np.isnan(h_pts):
        h_pts += 1.5
    if not np.isnan(a_pts):
        a_pts -= 1.5

    tot = h_pts + a_pts
    spread = a_pts - h_pts 
    
    winner = home if h_pts > a_pts else away
    win_margin = abs(h_pts - a_pts)

    return {
        "home_pts": h_pts, 
        "away_pts": a_pts, 
        "total": tot, 
        "spread": spread, 
        "winner": winner, 
        "margin": win_margin
    }

# ─────────────────────────────────────────────
# LÓGICA DE PROP ASSASSIN
# ─────────────────────────────────────────────
def _player_metrics(pdf: pd.
