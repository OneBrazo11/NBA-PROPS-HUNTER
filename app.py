import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="NBA PROPS & HUNTER", page_icon="🏀", layout="wide")

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    c = df.columns.str.strip().str.lower()
    df.columns = c.str.replace(r"[\s\-/]", "_", regex=True).str.replace(r"[^a-z0-9_]", "", regex=True)
    return df

def _col(df: pd.DataFrame, candidates: list) -> str | None:
    return next((c for c in candidates if c in df.columns), None)

@st.cache_data(show_spinner=False)
def load_data(files_bytes: dict) -> dict:
    cats = ["roster", "nbaleaguesumary", "def", "players", "impact", "trends", "shooting", "fouls", "frequency", "accuracy"]
    frames, temp_dict = {}, {k: [] for k in cats}
    keys_dict = {"roster": ["roster", "plantilla", "ref"], "nbaleaguesumary": ["summary", "league", "resumen"], "def": ["def", "defense", "dvp"], "players": ["players", "jugadores", "overview"], "impact": ["impact", "impacto", "onoff"], "trends": ["trends", "tendencias"], "shooting": ["shooting", "tiros"], "fouls": ["foul", "faltas"], "frequency": ["frequency", "frecuencia"], "accuracy": ["accuracy", "precision"]}
    for _, (name, raw) in files_bytes.items():
        fname = name.lower()
        match = next((k for k, aliases in keys_dict.items() if any(a in fname for a in aliases)), None)
        if match:
            try:
                df = pd.read_excel(io.BytesIO(raw)) if fname.endswith((".xlsx", ".xls")) else pd.read_csv(io.BytesIO(raw), encoding="latin-1")
                temp_dict[match].append(_normalize_cols(df))
            except: pass
    for k, v in temp_dict.items():
        frames[k] = pd.concat(v, ignore_index=True) if v else pd.DataFrame()
    return frames

def calculate_absence_impact(team: str, out_pl: list, imp_df: pd.DataFrame) -> float:
    if imp_df.empty or not out_pl: return 0.0
    p_col, d_col = _col(imp_df, ["player", "name"]), _col(imp_df, ["diff_pts_per_100_poss", "point_differential", "on_off_diff", "diff"])
    if not (p_col and d_col): return 0.0
    tot = 0.0
    for p in out_pl:
        v = imp_df[imp_df[p_col].astype(str).str.contains(p, na=False)][d_col]
        if not v.empty:
            val = pd.to_numeric(v.iloc[0], errors='coerce')
            if not np.isnan(val): tot -= val
    return tot

def _team_momentum(sdf: pd.DataFrame, team: str) -> dict:
    res = {"offrtg": np.nan, "defrtg": np.nan, "pace": np.nan}
    tc = _col(sdf, ["team", "tm", "team_name"])
    if not tc or sdf.empty: return res
    tdf = sdf[sdf[tc].astype(str).str.lower().str.contains(team.lower(), na=False)].copy()
    if tdf.empty: return res
    l10 = _col(tdf, ["split", "last_n", "games"])
    if l10 and tdf[l10].astype(str).str.lower().str.contains("l10|last", regex=True, na=False).any():
        tdf = tdf[tdf[l10].astype(str).str.lower().str.contains("l10|last", regex=True, na=False)]
    oc = _col(tdf, ["last_2_weeks_offense", "offense", "offrtg", "ortg", "offensive_rating", "pts_per_100"])
    dc = _col(tdf, ["last_2_weeks_defense", "defense", "defrtg", "drtg", "defensive_rating", "opp_pts_per_100"])
    pc = _col(tdf, ["pace", "poss", "possessions", "ritmo"])
    if oc: res["offrtg"] = pd.to_numeric(tdf[oc], errors="coerce").mean()
    if dc: res["defrtg"] = pd.to_numeric(tdf[dc], errors="coerce").mean()
    if pc: res["pace"] = pd.to_numeric(tdf[pc], errors="coerce").mean()
    return res

def project_game(home: str, away: str, frames: dict, h_imp: float, a_imp: float) -> dict:
    hm, am = _team_momentum(frames.get("nbaleaguesumary", pd.DataFrame()), home), _team_momentum(frames.get("nbaleaguesumary", pd.DataFrame()), away)
    pace = np.nanmean([hm.get("pace", np.nan), am.get("pace", np.nan)])
    if np.isnan(pace): pace = 98.5
    ho, hd = hm.get("offrtg", np.nan), hm.get("defrtg", np.nan)
    ao, ad = am.get("offrtg", np.nan), am.get("defrtg", np.nan)
    ho_adj = ho + (h_imp/2) if not np.isnan(ho) else np.nan
    hd_adj = hd - (h_imp/2) if not np.isnan(hd) else np.nan
    ao_adj = ao + (a_imp/2) if not np.isnan(ao) else np.nan
    ad_adj = ad - (a_imp/2) if not np.isnan(ad) else np.nan
    h_pts = np.nanmean([ho_adj, ad_adj]) * (pace/100) if not np.isnan(ho_adj) else np.nan
    a_pts = np.nanmean([ao_adj, hd_adj]) * (pace/100) if not np.isnan(ao_adj) else np.nan
    if not np.isnan(h_pts): h_pts += 1.5
    if not np.isnan(a_pts): a_pts -= 1.5
    tot = h_pts + a_pts
    spread = a_pts - h_pts
    winner = home if (not np.isnan(h_pts) and not np.isnan(a_pts) and h_pts > a_pts) else away
    margin = abs(h_pts - a_pts) if (not np.isnan(h_pts) and not np.isnan(a_pts)) else 0.0
    return {"home_pts": h_pts, "away_pts": a_pts, "total": tot, "spread": spread, "winner": winner, "margin": margin}

def _player_metrics(pdf: pd.DataFrame, idf: pd.DataFrame, player: str) -> dict:
    res = {"pts": np.nan, "reb": np.nan, "ast": np.nan, "usg": np.nan, "min": np.nan}
    if pdf.empty: return res
    nc = _col(pdf, ["player", "name"])
    if nc:
        p1 = pdf[pdf[nc].astype(str).str.lower().str.contains(player.lower(), na=False)].copy()
        sc = _col(p1, ["split", "games"])
        if sc and not p1[p1[sc].astype(str).str.lower().str.contains("l10|last", regex=True, na=False)].empty:
            p1 = p1[p1[sc].astype(str).str.lower().str.contains("l10|last", regex=True, na=False)]
        c_pts = "pts" if "pts" in p1.columns else None
        c_reb = "trb" if "trb" in p1.columns else None
        c_ast = "ast" if "ast" in p1.columns else None
        c_min = "mp" if "mp" in p1.columns else None
        if c_pts: res["pts"] = pd.to_numeric(p1[c_pts], errors="coerce").mean()
        if c_reb: res["reb"] = pd.to_numeric(p1[c_reb], errors="coerce").mean()
        if c_ast: res["ast"] = pd.to_numeric(p1[c_ast], errors="coerce").mean()
        if c_min: res["min"] = pd.to_numeric(p1[c_min], errors="coerce").mean()
    if not idf.empty:
        nic = _col(idf, ["player", "name"])
        if nic:
            i1 = idf[idf[nic].astype(str).str.lower().str.contains(player.lower(), na=False)]
            c_usg = "usage" if "usage" in i1.columns else ("usg" if "usg" in i1.columns else None)
            if c_usg: res["usg"] = pd.to_numeric(i1[c_usg], errors="coerce").mean()
    return res

def _dvp_factor(def_df: pd.DataFrame, opp: str, pos: str) -> float:
    if def_df.empty: return 1.0
    tc, pc = _col(def_df, ["team", "opp", "opponent", "tm"]), _col(def_df, ["pos", "position"])
    if not tc: return 1.0
    t_df = def_df[def_df[tc].astype(str).str.lower().str.contains(opp.lower(), na=False)]
    if pc and not t_df.empty:
        p
