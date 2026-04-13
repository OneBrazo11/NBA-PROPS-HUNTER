import streamlit as st
import pandas as pd
import numpy as np
import io

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(page_title="NBA PROPS & HUNTER", page_icon="🏀", layout="wide")

# ─────────────────────────────────────────────
# PROCESAMIENTO DE DATOS (CONCATENACIÓN MASIVA)
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
    temp_dict = {
        "roster": [],
        "nbaleaguesumary": [],
        "def": [],
        "players": [],
        "impact": [],
        "trends": []
    }
    
    keys_dict = {
        "roster": ["roster", "plantilla"],
        "nbaleaguesumary": ["summary", "league", "resumen"],
        "def": ["def", "defense", "dvp"],
        "players": ["players", "jugadores"],
        "impact": ["impact", "impacto", "onoff"],
        "trends": ["trends", "tendencias", "trend"]
    }

    for _, (name, raw) in files_bytes.items():
        fname = name.lower()
        matched_category = None
        
        for main_key, aliases in keys_dict.items():
            if any(alias in fname for alias in aliases):
                matched_category = main_key
                break
                
        if matched_category:
            try:
                df = None
                if fname.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(io.BytesIO(raw))
                else:
                    for enc in ("utf-8", "latin-1", "cp1252"):
                        try:
                            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                            break
                        except Exception:
                            continue
                
                if df is not None:
                    temp_dict[matched_category].append(_normalize_cols(df))
            except Exception:
                pass

    for key, df_list in temp_dict.items():
        if df_list:
            try:
                frames[key] = pd.concat(df_list, ignore_index=True)
            except Exception:
                frames[key] = pd.DataFrame()
        else:
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
# LÓGICA DE GAME HUNTER (BLINDADA CONTRA KEYERROR)
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
    
    # USO SEGURO CON .GET() PARA EVITAR CUALQUIER KEYERROR
    h_pace = hm.get("pace", np.nan)
    a_pace = am.get("pace", np.nan)
    avg_pace = np.nanmean([h_pace, a_pace])
    pace = avg_pace if not np.isnan(avg_pace) else 98.5
    
    h_ortg = hm.get("offrtg", np.nan)
    h_drtg = hm.get("defrtg", np.nan)
    a_ortg = am.get("offrtg", np.nan)
    a_drtg = am.get("defrtg", np.nan)

    # Ajustar Ratings por lesiones
    h_ortg_adj = h_ortg + (h_impact / 2) if not np.isnan(h_ortg) else np.nan
    h_drtg_adj = h_drtg - (h_impact / 2) if not np.isnan(h_drtg) else np.nan
    
    a_ortg_adj = a_ortg + (a_impact / 2) if not np.isnan(a_ortg) else np.nan
    a_drtg_adj = a_drtg - (a_impact / 2) if not np.isnan(a_drtg) else np.nan

    h_pts = np.nanmean([h_ortg_adj, a_drtg_adj]) * (pace / 100) if not np.isnan(h_ortg_adj) else np.nan
    a_pts = np.nanmean([a_ortg_adj, h_drtg_adj]) * (pace / 100) if not np.isnan(a_ortg_adj) else np.nan
    
    if not np.isnan(h_pts):
        h_pts += 1.5
    if not np.isnan(a_pts):
        a_pts -= 1.5

    tot = h_pts + a_pts
    spread = a_pts - h_pts 
    
    winner = home if (not np.isnan(h_pts) and not np.isnan(a_pts) and h_pts > a_pts) else away
    win_margin = abs(h_pts - a_pts) if (not np.isnan(h_pts) and not np.isnan(a_pts)) else 0.0

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
def _player_metrics(pdf: pd.DataFrame, idf: pd.DataFrame, player: str) -> dict:
    res = {"pts": np.nan, "reb": np.nan, "ast": np.nan, "usg": np.nan, "min": np.nan}
    if pdf.empty:
        return res
    
    nc = _col(pdf, ["player", "name"])
    if nc:
        mask = pdf[nc].astype(str).str.lower().str.contains(player.lower(), na=False)
        p1 = pdf[mask].copy()
        sc = _col(p1, ["split", "games"])
        if sc:
            l10 = p1[p1[sc].astype(str).str.lower().str.contains("l10|last", regex=True, na=False)]
            if not l10.empty:
                p1 = l10
        
        c_pts = _col(p1, ["pts", "points"])
        c_reb = _col(p1, ["reb", "trb"])
        c_ast = _col(p1, ["ast", "assists"])
        
        if c_pts:
            res["pts"] = pd.to_numeric(p1[c_pts], errors="coerce").mean()
        if c_reb:
            res["reb"] = pd.to_numeric(p1[c_reb], errors="coerce").mean()
        if c_ast:
            res["ast"] = pd.to_numeric(p1[c_ast], errors="coerce").mean()

    if not idf.empty:
        nic = _col(idf, ["player", "name"])
        if nic:
            imask = idf[nic].astype(str).str.lower().str.contains(player.lower(), na=False)
            i1 = idf[imask]
            c_usg = _col(i1, ["usg", "usage"])
            c_min = _col(i1, ["min", "minutes"])
            
            if c_usg:
                res["usg"] = pd.to_numeric(i1[c_usg], errors="coerce").mean()
            if c_min:
                res["min"] = pd.to_numeric(i1[c_min], errors="coerce").mean()
            
    return res

def _dvp_factor(def_df: pd.DataFrame, opp: str, pos: str) -> float:
    if def_df.empty:
        return 1.0
        
    tc = _col(def_df, ["team", "opp", "opponent", "tm"])
    pc = _col(def_df, ["pos", "position"])
    
    if not tc:
        return 1.0
        
    t_df = def_df[def_df[tc].astype(str).str.lower().str.contains(opp.lower(), na=False)]
    
    if pc and not t_df.empty:
        p_df = t_df[t_df[pc].astype(str).str.lower().str.contains(str(pos).lower()[:2], na=False)]
        if not p_df.empty:
            t_df = p_df
    
    rc = _col(t_df, ["rel", "diff", "vs_avg", "pct_diff"])
    if rc and not t_df.empty:
        val = pd.to_numeric(t_df[rc], errors="coerce").mean()
        if not np.isnan(val):
            return 1.0 + (val / 100.0)
            
    return 1.0

def evaluate_risk(dvp: float, usg: float, mins: float) -> tuple:
    if dvp > 1.05:
        s_dvp, t_dvp = 3, "🟢 Favorable"
    elif dvp < 0.95:
        s_dvp, t_dvp = 1, "🔴 Difícil"
    else:
        s_dvp, t_dvp = 2, "🟡 Neutro"

    s_vol, t_vol = 2, "🟡 Rotación"
    if not np.isnan(usg):
        if usg > 22.0:
            s_vol, t_vol = 3, "🟢 Foco Ofensivo"
        elif usg < 15.0:
            s_vol, t_vol = 1, "🔴 Bajo Uso"
    elif not np.isnan(mins):
        if mins > 28.0:
            s_vol, t_vol = 3, "🟢 Titular Fijo"
        elif mins < 18.0:
            s_vol, t_vol = 1, "🔴 Pocos Minutos"

    total = s_dvp + s_vol
    if total == 6:
        risk = "🟢 ALTÍSIMA PROB."
    elif total == 5:
        risk = "🟢 ALTA PROB."
    elif total == 4:
        risk = "🟡 PROB. MEDIA"
    else:
        risk = "🔴 ALTO RIESGO"

    return t_dvp, t_vol, risk

def generate_roster_matrix(team: str, opp: str, frames: dict, out_players: list) -> pd.DataFrame:
    rdf = frames.get("roster", pd.DataFrame())
    if rdf.empty:
        return pd.DataFrame()
    
    tc = _col(rdf, ["team", "tm"])
    nc = _col(rdf, ["player", "name"])
    pc = _col(rdf, ["pos", "position"])
    
    if not (tc and nc):
        return pd.DataFrame()
    
    roster = rdf[rdf[tc].astype(str).str.lower().str.contains(team.lower(), na=False)]
    
    data = []
    for _, row in roster.iterrows():
        pname = str(row[nc])
        if pname in out_players:
            continue
        
        if pc:
            pos = str(row[pc])
        else:
            pos = "UNKN"
            
        m = _player_metrics(frames.get("players", pd.DataFrame()), frames.get("impact", pd.DataFrame()), pname)
        if np.isnan(m.get("pts", np.nan)):
            continue 
        
        dvp = _dvp_factor(frames.get("def", pd.DataFrame()), opp, pos)
        t_dvp, t_vol, risk = evaluate_risk(dvp, m.get("usg", np.nan), m.get("min", np.nan))
        
        data.append({
            "Jugador": pname,
            "Pos": pos,
            "PTS": round(m.get("pts", 0) * dvp, 1),
            "REB": round(m.get("reb", 0) * dvp, 1
