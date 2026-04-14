import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="NBA PROPS & HUNTER", page_icon="🏀", layout="wide")

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    c = df.columns.astype(str).str.strip().str.lower()
    c = c.str.replace(r"[\s\-/]", "_", regex=True).str.replace(r"[^a-z0-9_]", "", regex=True)
    df.columns = c
    return df.loc[:, ~df.columns.duplicated()]

def _col(df: pd.DataFrame, candidates: list) -> str | None:
    return next((c for c in candidates if c in df.columns), None)

def _match_team(val, target):
    v, t = str(val).lower().strip(), str(target).lower().strip()
    return t in v or v in t

@st.cache_data(show_spinner=False)
def load_data(files_bytes: dict) -> dict:
    # Agregamos Trends y Ref al motor de búsqueda
    cats = ["nbaleaguesumary", "players", "impact", "def", "trends"]
    frames, temp_dict = {}, {k: [] for k in cats}
    keys = {
        "nbaleaguesumary": ["summary", "league"],
        "players": ["players", "overview", "ref", "roster"],
        "impact": ["impact", "onoff"],
        "def": ["defense", "dvp", "def"],
        "trends": ["trends", "tendencias"]
    }
    for _, (name, raw) in files_bytes.items():
        fname = name.lower()
        match = next((k for k, aliases in keys.items() if any(a in fname for a in aliases)), None)
        if match:
            try:
                df = pd.read_excel(io.BytesIO(raw)) if fname.endswith((".xlsx", ".xls")) else pd.read_csv(io.BytesIO(raw), encoding="latin-1")
                df = _normalize_cols(df)
                temp_dict[match].append(df)
            except: pass
    for k, v in temp_dict.items():
        if v:
            res = pd.concat(v, ignore_index=True)
            p_col = _col(res, ["player", "name"])
            if p_col: res = res.drop_duplicates(subset=[p_col])
            frames[k] = res
        else: frames[k] = pd.DataFrame()
    return frames

def calculate_absence_impact(out_pl, imp_df):
    if imp_df.empty or not out_pl: return 0.0
    p_col = _col(imp_df, ["player", "name"])
    d_col = _col(imp_df, ["diff_pts_per_100_poss", "on_off_diff", "diff"])
    if not (p_col and d_col): return 0.0
    impacto = imp_df[imp_df[p_col].astype(str).isin(out_pl)]
    return -pd.to_numeric(impacto[d_col], errors='coerce').sum() * 1.5

def _team_momentum(sum_df, trend_df, team):
    res = {"off": 110.0, "def": 110.0, "pace": 98.5}
    # Intentar sacar de Trends primero (más reciente)
    for df in [trend_df, sum_df]:
        if not df.empty:
            tc = _col(df, ["team", "tm", "team_name"])
            if tc:
                tdf = df[df[tc].apply(lambda x: _match_team(x, team))].copy()
                if not tdf.empty:
                    oc, dc, pc = _col(tdf, ["offrtg", "ortg", "offense"]), _col(tdf, ["defrtg", "drtg", "defense"]), _col(tdf, ["pace", "poss"])
                    if oc: res["off"] = pd.to_numeric(tdf[oc], errors="coerce").mean()
                    if dc: res["def"] = pd.to_numeric(tdf[dc], errors="coerce").mean()
                    if pc: res["pace"] = pd.to_numeric(tdf[pc], errors="coerce").mean()
                    return res
    return res

def project_game(home, away, frames, h_imp, a_imp):
    h = _team_momentum(frames.get("nbaleaguesumary", pd.DataFrame()), frames.get("trends", pd.DataFrame()), home)
    a = _team_momentum(frames.get("nbaleaguesumary", pd.DataFrame()), frames.get("trends", pd.DataFrame()), away)
    
    # El Pace es el promedio de ambos equipos
    pace = np.nanmean([h["pace"], a["pace"]])
    
    h_pts = (h["off"] + h_imp + a["def"]) / 2 * (pace/100) + 1.5
    a_pts = (a["off"] + a_imp + h["def"]) / 2 * (pace/100)
    
    return {"h_pts": h_pts, "a_pts": a_pts, "total": h_pts + a_pts, "winner": home if h_pts > a_pts else away, "spread": a_pts - h_pts}

def generate_roster(team, frames, out_p):
    df = frames.get("players", pd.DataFrame())
    tc, nc = _col(df, ["team", "tm"]), _col(df, ["player", "name"])
    if df.empty or not nc: return pd.DataFrame()
    tdf = df[df[tc].apply(lambda x: _match_team(x, team))].copy()
    
    # Factor de absorción si hay bajas importantes
    boost = 1.15 if len(out_p) > 0 else 1.0
    
    data = []
    for _, row in tdf.iterrows():
        name = str(row[nc])
        if name in out_p or name.lower() in ["nan", "none"]: continue
        pts = pd.to_numeric(row.get("pts", 0), errors="coerce") * boost
        reb = pd.to_numeric(row.get("trb", row.get("reb", 0)), errors="coerce") * boost
        ast = pd.to_numeric(row.get("ast", 0), errors="coerce") * boost
        if pts > 0:
            data.append({"Jugador": name, "Floor Safe": round(pts*0.7, 1), "PTS": round(pts,1), "REB": round(reb,1), "AST": round(ast,1), "PRA": round(pts+reb+ast,1)})
    return pd.DataFrame(data).sort_values(by="PTS", ascending=False) if data else pd.DataFrame()

with st.sidebar:
    st.title("🏀 NBA HUNTER PRO")
    up = st.file_uploader("Sube archivos", accept_multiple_files=True)
    if up:
        frames = load_data({str(i): (f.name, f.read()) for i, f in enumerate(up)})
        st.success("¡Archivos Detectados!")
        # Checklist visual de lo que la app ESTÁ leyendo
        for k, v in frames.items():
            if not v.empty:
                st.write(f"✅ {k.upper()} ({len(v)} filas)")
            else:
                st.write(f"❌ {k.upper()} (No detectado)")
    else: frames = {}

if not frames: st.stop()

# Selector de equipos
summary_df = frames.get("nbaleaguesumary", pd.DataFrame())
t_col = _col(summary_df, ["team", "tm"])
teams = sorted(list(summary_df[t_col].dropna().unique()))

c1, c2 = st.columns(2)
t_h = c1.selectbox("Local", teams)
t_a = c2.selectbox("Visitante", teams, index=1 if len(teams)>1 else 0)

# Jugadores
def get_players(f, t):
    p = set()
    df = f.get("players", pd.DataFrame())
    nc, tc = _col(df, ["player", "name"]), _col(df, ["team", "tm"])
    if not df.empty and nc:
        tdf = df[df[tc].apply(lambda x: _match_team(x, t))]
        p.update(tdf[nc].astype(str).tolist())
    return sorted([str(x) for x in p if str(x).lower() not in ["nan", "none", ""]])

out_h = st.sidebar.multiselect(f"Bajas {t_h}", get_players(frames, t_h))
out_a = st.sidebar.multiselect(f"Bajas {t_a}", get_players(frames, t_a))

tab1, tab2 = st.tabs(["🎯 Game Hunter", "💥 Prop Assassin"])

with tab1:
    if st.button("Ejecutar Proyección"):
        imp_h = calculate_absence_impact(out_h, frames.get("impact", pd.DataFrame()))
        imp_a = calculate_absence_impact(out_a, frames.get("impact", pd.DataFrame()))
        g = project_game(t_h, t_a, frames, imp_h, imp_a)
        
        c_1, c_2, c_3 = st.columns(3)
        c_1.metric("GANADOR", g["winner"])
        hcap = f"{t_h} {g['spread']:+.1f}" if g['spread'] < 0 else f"{t_a} {-g['spread']:+.1f}"
        c_1.metric("HÁNDICAP", hcap)
        
        c_2.metric("PROYECCIÓN TOTAL", f"{g['total']:.1f}")
        # Comparación contra promedio Vegas para O/U
        c_2.metric("TENDENCIA", "OVER" if g['total'] > 223 else "UNDER")
        
        c_3.metric(f"PTS {t_h}", f"{g['h_pts']:.1f}")
        c_3.metric(f"PTS {t_a}", f"{g['a_pts']:.1f}")

with tab2:
    if st.button("Generar Props"):
        st.subheader(f"Props Proyectados: {t_h}")
        st.dataframe(generate_roster(t_h, frames, out_h), use_container_width=True)
        st.subheader(f"Props Proyectados: {t_away if 't_away' in locals() else t_a}")
        st.dataframe(generate_roster(t_a, frames, out_a), use_container_width=True)
