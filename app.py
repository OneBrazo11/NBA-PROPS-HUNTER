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
    cats = ["roster", "nbaleaguesumary", "def", "players", "impact", "trends", "shooting", "fouls", "frequency"]
    frames, temp_dict = {}, {k: [] for k in cats}
    keys_dict = {
        "roster": ["roster", "plantilla", "ref"], 
        "nbaleaguesumary": ["summary", "league", "resumen"], 
        "def": ["def", "defense", "dvp"], 
        "players": ["players", "jugadores"], 
        "impact": ["impact", "impacto", "onoff"], 
        "trends": ["trends", "tendencias"],
        "shooting": ["shooting", "accuracy", "tiros", "overview"],
        "fouls": ["foul", "faltas"],
        "frequency": ["frequency", "frecuencia"]
    }
    
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
        c_pts, c_reb, c_ast = _col(p1, ["pts", "points", "pt"]), _col(p1, ["reb", "trb", "rb"]), _col(p1, ["ast", "assists", "as"])
        if c_pts: res["pts"] = pd.to_numeric(p1[c_pts], errors="coerce").mean()
        if c_reb: res["reb"] = pd.to_numeric(p1[c_reb], errors="coerce").mean()
        if c_ast: res["ast"] = pd.to_numeric(p1[c_ast], errors="coerce").mean()
    if not idf.empty:
        nic = _col(idf, ["player", "name"])
        if nic:
            i1 = idf[idf[nic].astype(str).str.lower().str.contains(player.lower(), na=False)]
            c_usg, c_min = _col(i1, ["usg", "usage"]), _col(i1, ["min", "minutes", "mp"])
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
    s_dvp = 3 if dvp > 1.05 else (1 if dvp < 0.95 else 2)
    t_dvp = "🟢 Favorable" if dvp > 1.05 else ("🔴 Difícil" if dvp < 0.95 else "🟡 Neutro")
    s_vol, t_vol = 2, "🟡 Rotación"
    if not np.isnan(usg):
        s_vol = 3 if usg > 22.0 else (1 if usg < 15.0 else 2)
        t_vol = "🟢 Foco Ofensivo" if usg > 22.0 else ("🔴 Bajo Uso" if usg < 15.0 else "🟡 Rotación")
    elif not np.isnan(mins):
        s_vol = 3 if mins > 28.0 else (1 if mins < 18.0 else 2)
        t_vol = "🟢 Titular Fijo" if mins > 28.0 else ("🔴 Pocos Minutos" if mins < 18.0 else "🟡 Rotación")
    tot = s_dvp + s_vol
    risk = "🟢 ALTÍSIMA PROB." if tot == 6 else ("🟢 ALTA PROB." if tot == 5 else ("🟡 PROB. MEDIA" if tot == 4 else "🔴 ALTO RIESGO"))
    return t_dvp, t_vol, risk

def generate_roster_matrix(team: str, opp: str, frames: dict, out_players: list) -> pd.DataFrame:
    rdf = frames.get("roster", pd.DataFrame())
    tc, nc, pc = _col(rdf, ["team", "tm"]), _col(rdf, ["player", "name"]), _col(rdf, ["pos", "position"])
    if rdf.empty or not (tc and nc): return pd.DataFrame()
    
    roster = rdf[rdf[tc].astype(str).str.lower().str.contains(team.lower(), na=False)]
    data = []
    for _, row in roster.iterrows():
        pname = str(row[nc])
        if pname in out_players: continue
        pos = str(row[pc]) if pc else "UNKN"
        m = _player_metrics(frames.get("players", pd.DataFrame()), frames.get("impact", pd.DataFrame()), pname)
        
        pts_val = m.get("pts", np.nan)
        if np.isnan(pts_val): continue 
            
        reb_val, ast_val = m.get("reb", 0.0), m.get("ast", 0.0)
        usg_val, min_val = m.get("usg", np.nan), m.get("min", np.nan)
        
        dvp = _dvp_factor(frames.get("def", pd.DataFrame()), opp, pos)
        t_dvp, t_vol, risk = evaluate_risk(dvp, usg_val, min_val)
        
        data.append({
            "Jugador": pname, "Pos": pos, "PTS": round(pts_val * dvp, 1),
            "REB": round(reb_val * dvp, 1), "AST": round(ast_val * dvp, 1),
            "PRA": round((pts_val + reb_val + ast_val) * dvp, 1),
            "1. Matchup (DVP)": t_dvp, "2. Volumen": t_vol, "3. RIESGO": risk
        })
    return pd.DataFrame(data).sort_values(by="PTS", ascending=False) if data else pd.DataFrame()

def analyze_micro_markets(team: str, opp: str, frames: dict) -> dict:
    res = {"status": "ok", "diagnostic": ""}
    freq_df = frames.get("frequency", pd.DataFrame())
    foul_df = frames.get("fouls", pd.DataFrame())
    
    if freq_df.empty and foul_df.empty:
        res["status"] = "missing"
        return res
        
    res["diagnostic"] = f"Columnas FREQ: {list(freq_df.columns) if not freq_df.empty else 'Vacio'} | Columnas FOULS: {list(foul_df.columns) if not foul_df.empty else 'Vacio'}"
    return res
    with st.sidebar:
    st.title("NBA PROPS & HUNTER")
    uploaded = st.file_uploader("Sube archivos CTG y NBA", accept_multiple_files=True, type=["csv", "xlsx"])
    if uploaded:
        frames = load_data({str(i): (f.name, f.read()) for i, f in enumerate(uploaded)})
        st.success("¡Base de Datos Fusionada!")
        st.write(f"📂 Archivos Subidos: {len(uploaded)}")
        for k, v in frames.items(): st.write(f"✅ {k.upper()} ({len(v)})" if not v.empty else f"❌ {k.upper()} (0)")
    else:
        frames = {}

st.title("🏀 NBA PROPS & HUNTER")
if not frames: st.stop()

teams_set = set()
for key in ["roster", "nbaleaguesumary", "def"]:
    df = frames.get(key, pd.DataFrame())
    tc = _col(df, ["team", "tm", "team_name", "franchise", "opp"])
    if not df.empty and tc: teams_set.update(df[tc].dropna().astype(str).unique().tolist())
            
teams = sorted([t for t in teams_set if len(t) > 1]) or ["UNK - Sin equipos"]
c1, c2 = st.columns(2)
with c1: t_home = st.selectbox("Local", teams)
with c2: t_away = st.selectbox("Visitante", teams, index=1 if len(teams)>1 else 0)

st.sidebar.divider()
st.sidebar.subheader("🚑 Bajas")

def get_team_roster(team: str) -> list:
    df = frames.get("roster", pd.DataFrame())
    c, p = _col(df, ["team", "tm"]), _col(df, ["player", "name"])
    return df[df[c] == team][p].dropna().unique().tolist() if not df.empty and c and p else []

out_home = st.sidebar.multiselect(f"Bajas {t_home}", get_team_roster(t_home))
out_away = st.sidebar.multiselect(f"Bajas {t_away}", get_team_roster(t_away))

tab1, tab2, tab3 = st.tabs(["🎯 Game Hunter", "💥 Prop Assassin", "🎲 Micro-Mercados"])

with tab1:
    impact_df = frames.get("impact", pd.DataFrame())
    h_impact = calculate_absence_impact(t_home, out_home, impact_df)
    a_impact = calculate_absence_impact(t_away, out_away, impact_df)

    if st.button("Proyectar Partido", type="primary"):
        g = project_game(t_home, t_away, frames, h_impact, a_impact)
        if np.isnan(g.get("total", np.nan)):
            st.error("Faltan datos en League Summary.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Ganador", g["winner"].upper(), f"Margin: {g['margin']:.1f}")
            c2.metric("Total (O/U)", f"{g['total']:.1f}")
            fav, hcap = (t_home, -abs(g["spread"])) if g["spread"] < 0 else (t_away, -abs(g["spread"]))
            c3.metric("Handicap", f"{fav} {hcap:.1f}")
            if out_home or out_away: st.caption(f"Ajuste: {t_home} ({h_impact:+.1f}) | {t_away} ({a_impact:+.1f})")

with tab2:
    t_target = st.radio("Analizar:", [t_home, t_away], horizontal=True, key="t_prop")
    t_opp, current_out = (t_away, out_home) if t_target == t_home else (t_home, out_away)
    
    if st.button(f"Generar Matriz {t_target}", type="primary"):
        df_props = generate_roster_matrix(t_target, t_opp, frames, current_out)
        if df_props.empty: 
            st.warning("Matriz vacía: Faltan promedios tradicionales (PTS, REB, AST).")
            st.info("💡 Diagnóstico de Columnas:")
            st.code(list(frames.get("players", pd.DataFrame()).columns))
            st.write("👉 Descarga archivos de Basketball-Reference y súbelos junto a CTG.")
        else: st.dataframe(df_props, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Análisis de Líneas Específicas (Triples, Faltas, Ritmo)")
    st.write("Cruza tendencias ofensivas vs. vulnerabilidades defensivas del rival.")
    
    t_target_micro = st.radio("Analizar Micro-Mercados para:", [t_home, t_away], horizontal=True, key="t_micro")
    
    if st.button(f"Analizar {t_target_micro}", type="primary"):
        m_data = analyze_micro_markets(t_target_micro, t_away if t_target_micro == t_home else t_home, frames)
        
        if m_data["status"] == "missing":
            st.info("👈 Sube los archivos de 'Frequency' y 'Foul Drawing' de CTG.")
        else:
            st.warning("Datos leídos. Envíame este diagnóstico para programar el algoritmo:")
            st.code(m_data["diagnostic"])
