import streamlit as st
import pandas as pd
import numpy as np
import io

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NBA PROPS & HUNTER", 
    page_icon="🏀", 
    layout="wide"
)

# ─────────────────────────────────────────────
# PROCESAMIENTO DE DATOS
# ─────────────────────────────────────────────
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[\s\-/]", "_", regex=True)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
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
    
    winner = away
    if not np.isnan(h_pts) and not np.isnan(a_pts):
        if h_pts > a_pts:
            winner = home
            
    win_margin = 0.0
    if not np.isnan(h_pts) and not np.isnan(a_pts):
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
        s_dvp = 3
        t_dvp = "🟢 Favorable"
    elif dvp < 0.95:
        s_dvp = 1
        t_dvp = "🔴 Difícil"
    else:
        s_dvp = 2
        t_dvp = "🟡 Neutro"

    s_vol = 2
    t_vol = "🟡 Rotación"
    
    if not np.isnan(usg):
        if usg > 22.0:
            s_vol = 3
            t_vol = "🟢 Foco Ofensivo"
        elif usg < 15.0:
            s_vol = 1
            t_vol = "🔴 Bajo Uso"
    elif not np.isnan(mins):
        if mins > 28.0:
            s_vol = 3
            t_vol = "🟢 Titular Fijo"
        elif mins < 18.0:
            s_vol = 1
            t_vol = "🔴 Pocos Minutos"

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
        
        pts_val = m.get("pts", np.nan)
        if np.isnan(pts_val):
            continue 
            
        reb_val = m.get("reb", 0.0)
        ast_val = m.get("ast", 0.0)
        usg_val = m.get("usg", np.nan)
        min_val = m.get("min", np.nan)
        
        dvp = _dvp_factor(frames.get("def", pd.DataFrame()), opp, pos)
        t_dvp, t_vol, risk = evaluate_risk(dvp, usg_val, min_val)
        
        pts_proj = round(pts_val * dvp, 1)
        reb_proj = round(reb_val * dvp, 1)
        ast_proj = round(ast_val * dvp, 1)
        pra_proj = round((pts_val + reb_val + ast_val) * dvp, 1)
        
        data.append({
            "Jugador": pname,
            "Pos": pos,
            "PTS": pts_proj,
            "REB": reb_proj,
            "AST": ast_proj,
            "PRA": pra_proj,
            "1. Matchup (DVP)": t_dvp,
            "2. Volumen": t_vol,
            "3. RIESGO": risk
        })
        
    if not data:
        return pd.DataFrame()
        
    df_res = pd.DataFrame(data)
    return df_res.sort_values(by="PTS", ascending=False)

# ─────────────────────────────────────────────
# INTERFAZ 
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("NBA PROPS & HUNTER")
    uploaded = st.file_uploader(
        "Sube archivos CTG", 
        accept_multiple_files=True, 
        type=["csv", "xlsx"]
    )
    
    fmap = {}
    if uploaded:
        # Reescrito para evitar líneas largas que causan error de sintaxis
        files_dict = {}
        for i, file_obj in enumerate(uploaded):
            files_dict[str(i)] = (file_obj.name, file_obj.read())
            
        frames = load_data(files_dict)
        
        st.success("¡Base de Datos Fusionada!")
        st.write(f"📂 Archivos Subidos: {len(uploaded)}")
        
        st.caption("Estado de las Tablas:")
        for k, v in frames.items():
            if not v.empty:
                st.write(f"✅ {k.upper()} ({len(v)} filas)")
            else:
                st.write(f"❌ {k.upper()} (Vacío)")
    else:
        frames = {}

st.title("🏀 NBA PROPS & HUNTER")

if not frames:
    st.info("👈 Sube los archivos en el panel lateral para iniciar.")
    st.stop()

# Selección de Equipos
teams_set = set()
for key in ["roster", "nbaleaguesumary", "def"]:
    df = frames.get(key, pd.DataFrame())
    if df.empty:
        continue
        
    tc = _col(df, ["team", "tm", "team_name", "franchise", "squad", "opp"])
    if tc:
        teams_set.update(df[tc].dropna().astype(str).unique().tolist())
            
teams = sorted([t for t in teams_set if len(t) > 1])
if not teams:
    teams = ["UNK - Sin equipos"]

col1, col2 = st.columns(2)
with col1:
    t_home = st.selectbox("Local (Home)", teams)
with col2:
    idx = 1 if len(teams) > 1 else 0
    t_away = st.selectbox("Visitante (Away)", teams, index=idx)

# Gestión de Lesiones
st.sidebar.divider()
st.sidebar.subheader("🚑 Bajas / Lesiones")

def get_team_roster(team: str) -> list:
    df = frames.get("roster", pd.DataFrame())
    if df.empty:
        return []
    c = _col(df, ["team", "tm"])
    p = _col(df, ["player", "name"])
    if c and p:
        return df[df[c] == team][p].dropna().unique().tolist()
    return []

out_home = st.sidebar.multiselect(f"Bajas {t_home}", get_team_roster(t_home))
out_away = st.sidebar.multiselect(f"Bajas {t_away}", get_team_roster(t_away))

tab1, tab2 = st.tabs(["🎯 Game Hunter", "💥 Prop Assassin"])

# ══════════════════════════════════════════════
# TAB 1: GAME HUNTER
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Proyección L10 + Ajuste de Bajas")
    
    impact_df = frames.get("impact", pd.DataFrame())
    h_impact = calculate_absence_impact(t_home, out_home, impact_df)
    a_impact = calculate_absence_impact(t_away, out_away, impact_df)

    if st.button("Proyectar Partido", type="primary"):
        g = project_game(t_home, t_away, frames, h_impact, a_impact)
        
        if np.isnan(g.get("total", np.nan)):
            st.error("Faltan datos en League Summary para estos equipos.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(
                    label="Ganador", 
                    value=g["winner"].upper(), 
                    delta=f"Margin: {g['margin']:.1f}"
                )
            with c2:
                st.metric(
                    label="Total (O/U)", 
                    value=f"{g['total']:.1f}"
                )
            with c3:
                fav = t_home if g["spread"] < 0 else t_away
                hcap = -abs(g["spread"])
                st.metric(
                    label="Handicap", 
                    value=f"{fav} {hcap:.1f}"
                )
            
            if out_home or out_away:
                st.caption(f"Ajuste Lesiones: {t_home} ({h_impact:+.1f}) | {t_away} ({a_impact:+.1f})")

# ══════════════════════════════════════════════
# TAB 2: PROP ASSASSIN
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Matriz de Comando Única")
    t_target = st.radio("Analizar:", [t_home, t_away], horizontal=True)
    t_opp = t_away if t_target == t_home else t_home
    current_out = out_home if t_target == t_home else out_away
    
    if st.button(f"Generar Matriz {t_target}", type="primary"):
        df_props = generate_roster_matrix(t_target, t_opp, frames, current_out)
        if df_props.empty:
            st.warning("No hay datos suficientes para proyectar a este equipo.")
        else:
            st.dataframe(
                df_props, 
                use_container_width=True, 
                hide_index=True
            )
