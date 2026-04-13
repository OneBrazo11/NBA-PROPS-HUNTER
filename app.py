import streamlit as st
import pandas as pd
import numpy as np
import io

# ─────────────────────────────────────────────
# PAGE CONFIG & CSS
# ─────────────────────────────────────────────
st.set_page_config(page_title="NBA Edge Analyzer", page_icon="🏀", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Mono', monospace; }
    .stApp { background: #0a0c10; color: #e8eaf0; }
    .metric-box { background: #12151f; border: 1px solid #1e2a3a; padding: 15px; border-radius: 8px; text-align: center; }
    .metric-title { font-size: 0.8rem; color: #8890a0; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #3a8ef6; }
    .metric-green { color: #30d96a !important; }
    .section-title { color: #3a8ef6; border-bottom: 1px solid #1e2a3a; padding-bottom: 5px; margin-top: 30px; margin-bottom: 15px; text-transform: uppercase; font-size: 1rem;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CORE HELPERS
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
# GAME HUNTER LOGIC
# ─────────────────────────────────────────────
def _team_momentum(summary_df: pd.DataFrame, team: str) -> dict:
    res = {"offrtg": np.nan, "defrtg": np.nan, "pace": np.nan}
    if summary_df.empty: return res
    tc = _col(summary_df, ["team", "team_name"])
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

def project_game(home: str, away: str, frames: dict) -> dict:
    sdf = frames.get("nbaleaguesumary", pd.DataFrame())
    hm = _team_momentum(sdf, home)
    am = _team_momentum(sdf, away)
    
    pace = np.nanmean([hm["pace"], am["pace"]]) if not np.isnan(np.nanmean([hm["pace"], am["pace"]])) else 98.5
    h_pts = np.nanmean([hm["offrtg"], am["defrtg"]]) * (pace / 100) if not np.isnan(hm["offrtg"]) else np.nan
    a_pts = np.nanmean([am["offrtg"], hm["defrtg"]]) * (pace / 100) if not np.isnan(am["offrtg"]) else np.nan
    
    # Home Court Advantage (approx 1.5 pts)
    if not np.isnan(h_pts): h_pts += 1.5
    if not np.isnan(a_pts): a_pts -= 1.5

    tot = h_pts + a_pts
    spread = a_pts - h_pts # Negative means Home is favorite
    
    winner = home if h_pts > a_pts else away
    win_margin = abs(h_pts - a_pts)

    return {"home_pts": h_pts, "away_pts": a_pts, "total": tot, "spread": spread, "winner": winner, "margin": win_margin}

# ─────────────────────────────────────────────
# PROP ASSASSIN LOGIC (MATRIX)
# ─────────────────────────────────────────────
def _player_metrics(pdf: pd.DataFrame, idf: pd.DataFrame, player: str) -> dict:
    res = {"pts": np.nan, "reb": np.nan, "ast": np.nan, "usg": np.nan, "min": np.nan}
    if pdf.empty: return res
    
    # Base Stats (Last 10)
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

    # Impact / Usage
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
    tc, pc = _col(def_df, ["team", "opp"]), _col(def_df, ["pos", "position"])
    if not tc: return 1.0
    t_df = def_df[def_df[tc].astype(str).str.lower().str.contains(opp.lower(), na=False)]
    if pc and not t_df.empty:
        p_df = t_df[t_df[pc].astype(str).str.lower().str.contains(str(pos).lower()[:2], na=False)]
        if not p_df.empty: t_df = p_df
    
    rc = _col(t_df, ["rel", "diff", "vs_avg"])
    if rc and not t_df.empty:
        val = pd.to_numeric(t_df[rc], errors="coerce").mean()
        if not np.isnan(val): return 1.0 + (val / 100.0)
    return 1.0

def evaluate_risk(dvp: float, usg: float, mins: float) -> tuple:
    # Score DVP
    if dvp > 1.05: s_dvp, t_dvp = 3, "🟢 Favorable"
    elif dvp < 0.95: s_dvp, t_dvp = 1, "🔴 Difícil"
    else: s_dvp, t_dvp = 2, "🟡 Neutro"

    # Score Volume (Usage prioritized over Minutes)
    s_vol, t_vol = 2, "🟡 Rotación"
    if not np.isnan(usg):
        if usg > 22.0: s_vol, t_vol = 3, "🟢 Foco Ofensivo"
        elif usg < 15.0: s_vol, t_vol = 1, "🔴 Bajo Uso"
    elif not np.isnan(mins):
        if mins > 28.0: s_vol, t_vol = 3, "🟢 Titular Fijo"
        elif mins < 18.0: s_vol, t_vol = 1, "🔴 Pocos Minutos"

    # Compound Score
    total = s_dvp + s_vol
    if total == 6: risk = "🟢 ALTÍSIMA PROB."
    elif total == 5: risk = "🟢 ALTA PROB."
    elif total == 4: risk = "🟡 PROB. MEDIA"
    else: risk = "🔴 ALTO RIESGO"

    return t_dvp, t_vol, risk

def generate_roster_matrix(team: str, opp: str, frames: dict) -> pd.DataFrame:
    rdf = frames.get("roster", pd.DataFrame())
    if rdf.empty: return pd.DataFrame()
    
    tc = _col(rdf, ["team"])
    nc = _col(rdf, ["player", "name"])
    pc = _col(rdf, ["pos", "position"])
    if not (tc and nc): return pd.DataFrame()
    
    roster = rdf[rdf[tc].astype(str).str.lower().str.contains(team.lower(), na=False)]
    
    data = []
    for _, row in roster.iterrows():
        pname = str(row[nc])
        pos = str(row[pc]) if pc else "UNKN"
        
        m = _player_metrics(frames.get("players", pd.DataFrame()), frames.get("impact", pd.DataFrame()), pname)
        if np.isnan(m["pts"]): continue # Skip players with no L10 data
        
        dvp = _dvp_factor(frames.get("def", pd.DataFrame()), opp, pos)
        t_dvp, t_vol, risk = evaluate_risk(dvp, m["usg"], m["min"])
        
        data.append({
            "Jugador": pname,
            "Pos": pos,
            "PTS": round(m["pts"] * dvp, 1),
            "REB": round(m["reb"] * dvp, 1),
            "AST": round(m["ast"] * dvp, 1),
            "PRA": round((m["pts"] + m["reb"] + m["ast"]) * dvp, 1),
            "1. Matchup (DVP)": t_dvp,
            "2. Volumen (Uso/Min)": t_vol,
            "3. RIESGO FINAL": risk
        })
        
    return pd.DataFrame(data).sort_values(by="PTS", ascending=False)

# ─────────────────────────────────────────────
# UI LAYOUT
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Data Engine")
    uploaded = st.file_uploader("Sube los 6 archivos CTG (CSV/Excel)", accept_multiple_files=True, type=["csv", "xlsx"])
    fmap = {}
    if uploaded:
        keys = ["roster", "nbaleaguesumary", "def", "players", "impact", "trends"]
        for f in uploaded:
            for k in keys:
                if k in f.name.lower() and k not in fmap:
                    fmap[k] = f
                    break
        frames = load_data({k: (v.name, v.read()) for k, v in fmap.items()})
    else:
        frames = {}

st.title("🏀 NBA Algorithmic Edge")

if not frames:
    st.info("👈 Sube los archivos en el panel lateral para iniciar el motor de proyecciones.")
    st.stop()

# Get Teams
teams = sorted(list(set(frames.get("roster", pd.DataFrame())[_col(frames.get("roster"), ["team"])].dropna().unique()))) if _col(frames.get("roster", pd.DataFrame()), ["team"]) else ["UNK"]

col1, col2 = st.columns(2)
with col1: t_home = st.selectbox("Equipo Local (Home)", teams)
with col2: t_away = st.selectbox("Equipo Visitante (Away)", teams, index=1 if len(teams)>1 else 0)

tab1, tab2 = st.tabs(["🎯 Game Hunter (Partidos)", "💥 Prop Assassin (Matriz de Jugadores)"])

# ══════════════════════════════════════════════
# TAB 1: GAME HUNTER
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Proyección Pura L10 (Sin sesgo de mercado)</div>', unsafe_allow_html=True)
    if st.button("Generar Proyección de Partido", type="primary"):
        g = project_game(t_home, t_away, frames)
        if np.isnan(g["total"]):
            st.error("Datos insuficientes para proyectar estos equipos.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(f'<div class="metric-box"><div class="metric-title">Ganador Probable</div><div class="metric-value metric-green">{g["winner"].upper()}</div><div style="font-size:0.8rem; color:#8890a0;">Ventaja: {g["margin"]:.1f} pts</div></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="metric-box"><div class="metric-title">Línea Total (O/U)</div><div class="metric-value">{g["total"]:.1f}</div><div style="font-size:0.8rem; color:#8890a0;">Proyección Matemática</div></div>', unsafe_allow_html=True)
            with c3:
                fav = t_home if g["spread"] < 0 else t_away
                hcap = -abs(g["spread"])
                st.markdown(f'<div class="metric-box"><div class="metric-title">Handicap Justo</div><div class="metric-value">{fav} {hcap:.1f}</div><div style="font-size:0.8rem; color:#8890a0;">Spread Algorítmico</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2: PROP ASSASSIN
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Matriz de Comando Única</div>', unsafe_allow_html=True)
    st.caption("Evalúa el Roster completo. Riesgo = Cruce entre Vulnerabilidad del Rival (DVP) y Volumen del Jugador (Usage/Minutos).")
    
    t_target = st.radio("Selecciona el equipo a analizar:", [t_home, t_away], horizontal=True)
    t_opp = t_away if t_target == t_home else t_home
    
    if st.button(f"Generar Matriz para {t_target}", type="primary"):
        with st.spinner("Procesando L10, cruzando DVP e Impacto..."):
            df_props = generate_roster_matrix(t_target, t_opp, frames)
            
        if df_props.empty:
            st.warning("No se pudo generar la matriz. Revisa el cruce de nombres entre Roster y Players.")
        else:
            def style_matrix(val):
                v = str(val)
                if "🟢" in v: return 'color: #30d96a; font-weight: bold;'
                if "🔴" in v: return 'color: #f04545;'
                if "🟡" in v: return 'color: #f0c945;'
                return ''
            
            st.dataframe(
                df_props.style.map(style_matrix, subset=["1. Matchup (DVP)", "2. Volumen (Uso/Min)", "3. RIESGO FINAL"]).format(precision=1),
                use_container_width=True,
                height=600,
                hide_index=True
            )
