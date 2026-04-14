import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="NBA PROPS & HUNTER", page_icon="🏀", layout="wide")

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    c = df.columns.astype(str).str.strip().str.lower()
    c = c.str.replace(r"[\s\-/]", "_", regex=True)
    c = c.str.replace(r"[^a-z0-9_]", "", regex=True)
    df.columns = c
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def _col(df: pd.DataFrame, candidates: list) -> str | None:
    return next((c for c in candidates if c in df.columns), None)

@st.cache_data(show_spinner=False)
def load_data(files_bytes: dict) -> dict:
    cats = ["roster", "nbaleaguesumary", "def", "players", "impact", "trends", "shooting", "fouls", "frequency", "accuracy"]
    frames, temp_dict = {}, {k: [] for k in cats}
    keys_dict = {
        "roster": ["roster", "plantilla", "ref"],
        "nbaleaguesumary": ["summary", "league", "resumen"],
        "def": ["def", "defense", "dvp"],
        "players": ["players", "jugadores", "overview"],
        "impact": ["impact", "impacto", "onoff"],
        "trends": ["trends", "tendencias"],
        "shooting": ["shooting", "tiros"],
        "fouls": ["foul", "faltas"],
        "frequency": ["frequency", "frecuencia"],
        "accuracy": ["accuracy", "precision"]
    }
    for _, (name, raw) in files_bytes.items():
        fname = name.lower()
        match = next((k for k, aliases in keys_dict.items() if any(a in fname for a in aliases)), None)
        if match:
            try:
                if fname.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(io.BytesIO(raw))
                else:
                    df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
                temp_dict[match].append(_normalize_cols(df))
            except:
                pass
    for k, v in temp_dict.items():
        frames[k] = pd.concat(v, ignore_index=True) if v else pd.DataFrame()
    return frames

def calculate_absence_impact(team: str, out_pl: list, imp_df: pd.DataFrame) -> float:
    if imp_df.empty or not out_pl: return 0.0
    p_col = _col(imp_df, ["player", "name"])
    d_col = _col(imp_df, ["diff_pts_per_100_poss", "point_differential", "on_off_diff", "diff"])
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
    hm = _team_momentum(frames.get("nbaleaguesumary", pd.DataFrame()), home)
    am = _team_momentum(frames.get("nbaleaguesumary", pd.DataFrame()), away)
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
    tot, spread = h_pts + a_pts, a_pts - h_pts
    winner = home if (not np.isnan(h_pts) and not np.isnan(a_pts) and h_pts > a_pts) else away
    margin = abs(h_pts - a_pts) if (not np.isnan(h_pts) and not np.isnan(a_pts)) else 0.0
    return {"home_pts": h_pts, "away_pts": a_pts, "total": tot, "spread": spread, "winner": winner, "margin": margin}

def _player_metrics(pdf: pd.DataFrame, idf: pd.DataFrame, player: str) -> dict:
    res = {"pts": 0.0, "trb": 0.0, "ast": 0.0, "stl": 0.0, "blk": 0.0, "orb": 0.0, "drb": 0.0, "usg": np.nan, "mp": 0.0}
    if pdf.empty: return res
    nc = _col(pdf, ["player", "name"])
    if nc:
        p1 = pdf[pdf[nc].astype(str).str.lower().str.contains(player.lower(), na=False)].copy()
        for m in ["pts", "trb", "ast", "stl", "blk", "orb", "drb", "mp"]:
            if m in p1.columns:
                res[m] = pd.to_numeric(p1[m], errors="coerce").mean()
    if not idf.empty:
        nic = _col(idf, ["player", "name"])
        if nic:
            i1 = idf[idf[nic].astype(str).str.lower().str.contains(player.lower(), na=False)]
            c_usg = _col(i1, ["usage", "usg"])
            if c_usg:
                res["usg"] = pd.to_numeric(i1[c_usg], errors="coerce").mean()
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
    s_dvp = 3 if dvp > 1.05 else (1 if dvp < 0.95 else 2)
    t_dvp = "🟢 Favorable" if dvp > 1.05 else ("🔴 Difícil" if dvp < 0.95 else "🟡 Neutro")
    s_vol = 3 if (usg > 22 or mins > 28) else (1 if (usg < 15 or mins < 18) else 2)
    t_vol = "🟢 Foco" if s_vol == 3 else ("🔴 Bajo" if s_vol == 1 else "🟡 Rotación")
    tot = s_dvp + s_vol
    risk = "🟢 ALTÍSIMA" if tot == 6 else ("🟢 ALTA" if tot == 5 else ("🟡 MEDIA" if tot == 4 else "🔴 ALTO RIESGO"))
    return t_dvp, t_vol, risk

def generate_roster_matrix(team: str, opp: str, frames: dict, out_players: list) -> pd.DataFrame:
    rdf = frames.get("roster", pd.DataFrame())
    if rdf.empty: rdf = frames.get("players", pd.DataFrame())
    tc, nc, pc = _col(rdf, ["team", "tm"]), _col(rdf, ["player", "name"]), _col(rdf, ["pos", "position"])
    if rdf.empty or not nc: return pd.DataFrame()
    roster = rdf[rdf[tc].astype(str).str.lower().str.contains(team.lower(), na=False)] if tc else rdf
    data = []
    for _, row in roster.iterrows():
        pname = str(row[nc])
        if pname in out_players: continue
        m = _player_metrics(frames.get("players", pd.DataFrame()), frames.get("impact", pd.DataFrame()), pname)
        if m["pts"] == 0 and m["trb"] == 0: continue
        dvp = _dvp_factor(frames.get("def", pd.DataFrame()), opp, str(row[pc]) if pc else "UNKN")
        t_dvp, t_vol, risk = evaluate_risk(dvp, m["usg"], m["mp"])
        data.append({
            "Jugador": pname,
            "PTS": round(m["pts"]*dvp, 1),
            "TRB": round(m["trb"]*dvp, 1),
            "AST": round(m["ast"]*dvp, 1),
            "STL": round(m["stl"]*dvp, 1),
            "BLK": round(m["blk"]*dvp, 1),
            "ORB": round(m["orb"]*dvp, 1),
            "DRB": round(m["drb"]*dvp, 1),
            "PRA": round((m["pts"]+m["trb"]+m["ast"])*dvp, 1),
            "Matchup": t_dvp,
            "Riesgo": risk
        })
    return pd.DataFrame(data).sort_values(by="PTS", ascending=False) if data else pd.DataFrame()

def get_micro_metric(t_name: str, df_keys: list, cols: list, frames: dict, agg='mean'):
    for k in df_keys:
        df = frames.get(k, pd.DataFrame())
        if df.empty: continue
        tc = _col(df, ["team", "tm", "franchise"])
        c = _col(df, cols)
        if not c: continue
        tdf = df[df[tc].astype(str).str.lower().str.contains(t_name.lower(), na=False)] if tc else df
        if tdf.empty: continue
        if _col(tdf, ["player", "name"]):
            min_col = _col(tdf, ["min", "mpg", "mp"])
            if min_col: tdf = tdf.sort_values(by=min_col, ascending=False).head(8)
            v = pd.to_numeric(tdf[c], errors='coerce').dropna()
            if not v.empty: return v.sum() if agg == 'sum' else v.mean()
        else:
            return pd.to_numeric(tdf[c].iloc[0], errors='coerce')
    return np.nan

def analyze_micro_markets(t_off: str, t_def: str, frames: dict) -> pd.DataFrame:
    data = []
    o_sfld = get_micro_metric(t_off, ["fouls", "players"], ["sfld", "fta", "ft"], frames, agg='sum')
    if not np.isnan(o_sfld):
        sig = "🟢 OVER FALTAS" if o_sfld > 12 else "🔴 UNDER FALTAS"
        data.append({"Mercado": "Faltas Recibidas", "Ofensiva": round(o_sfld, 1), "Señal": sig})
    
    o_rim = get_micro_metric(t_off, ["frequency", "shooting"], ["rim", "paint"], frames)
    if not np.isnan(o_rim):
        sig = "🟢 ATAQUE ARO" if o_rim > 30 else "🔴 PERÍMETRO"
        data.append({"Mercado": "Rim Freq %", "Ofensiva": round(o_rim, 1), "Señal": sig})
        
    o_3p = get_micro_metric(t_off, ["nbaleaguesumary", "players", "shooting"], ["3pa", "3p_freq", "3p"], frames)
    if not np.isnan(o_3p):
        sig = "🟢 LÍNEA ALTA" if o_3p > 35 else "🔴 LÍNEA BAJA"
        data.append({"Mercado": "Volumen 3P", "Ofensiva": round(o_3p, 1), "Señal": sig})
        
    if not data:
        data.append({"Mercado": "Sin Datos", "Ofensiva": "-", "Señal": "-"})
    return pd.DataFrame(data)

with st.sidebar:
    st.title("NBA PROPS & HUNTER")
    uploaded = st.file_uploader("Sube archivos", accept_multiple_files=True, type=["csv", "xlsx"])
    if uploaded:
        frames = load_data({str(i): (f.name, f.read()) for i, f in enumerate(uploaded)})
        st.success("Fusionado")
    else: frames = {}

if not frames: st.stop()

teams_set = set()
for key in ["roster", "nbaleaguesumary", "def"]:
    df = frames.get(key, pd.DataFrame())
    tc = _col(df, ["team", "tm", "team_name", "franchise", "opp"])
    if tc: teams_set.update(df[tc].dropna().astype(str).unique().tolist())
teams = sorted([t for t in teams_set if len(t) > 1]) or ["UNK"]

c1, c2 = st.columns(2)
t_home = c1.selectbox("Local", teams)
t_away = c2.selectbox("Visitante", teams, index=1 if len(teams)>1 else 0)

def get_global_roster(frames):
    p = set()
    for k in ["roster", "players", "fouls", "frequency"]:
        df = frames.get(k, pd.DataFrame())
        col = _col(df, ["player", "name"])
        if col: p.update(df[col].dropna().astype(str).unique().tolist())
    return sorted(list(p))

st.sidebar.divider()
out_home = st.sidebar.multiselect(f"Bajas {t_home}", get_global_roster(frames))
out_away = st.sidebar.multiselect(f"Bajas {t_away}", get_global_roster(frames))

t1, t2, t3 = st.tabs(["🎯 Hunter", "💥 Props", "🎲 Micro"])
with t1:
    imp = frames.get("impact", pd.DataFrame())
    if st.button("Proyectar Partido"):
        g = project_game(t_home, t_away, frames, calculate_absence_impact(t_home, out_home, imp), calculate_absence_impact(t_away, out_away, imp))
        st.metric("Ganador", g["winner"].upper(), f"Margen: {g['margin']:.1f}")
        st.metric("Total (O/U)", f"{g['total']:.1f}")
with t2:
    if st.button("Generar Props"):
        st.dataframe(generate_roster_matrix(t_home, t_away, frames, out_home))
with t3:
    if st.button("Analizar Táctica"):
        st.dataframe(analyze_micro_markets(t_home, t_away, frames))
