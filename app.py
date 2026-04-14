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

def _match_team(val, target, fname=""):
    v, t, f = str(val).lower(), str(target).lower(), fname.lower()
    if v == t or v in t or t in v or t in f: return True
    d = {"sas":"san antonio", "den":"denver", "bos":"boston", "gsw":"golden", "lal":"lakers", "nyk":"york", "okc":"oklahoma"}
    return any(k in v or k in f for k, full in d.items() if full in t)

@st.cache_data(show_spinner=False)
def load_data(files_bytes: dict) -> dict:
    cats = ["nbaleaguesumary", "players", "impact", "def", "trends"]
    frames, temp_dict = {}, {k: [] for k in cats}
    keys = {"nbaleaguesumary": ["summary", "league"], "players": ["players", "overview", "ref", "roster"], "impact": ["impact", "onoff"], "def": ["defense", "dvp"], "trends": ["trends", "tendencias"]}
    for _, (name, raw) in files_bytes.items():
        fname = name.lower()
        match = next((k for k, aliases in keys.items() if any(a in fname for a in aliases)), None)
        if match:
            try:
                df = pd.read_excel(io.BytesIO(raw)) if fname.endswith((".xlsx", ".xls")) else pd.read_csv(io.BytesIO(raw), encoding="latin-1")
                df = _normalize_cols(df)
                df["_source"] = fname
                temp_dict[match].append(df)
            except: pass
    for k, v in temp_dict.items():
        if v:
            res = pd.concat(v, ignore_index=True)
            p_col = _col(res, ["player", "name"])
            if p_col: res = res.drop_duplicates(subset=[p_col, "_source"])
            frames[k] = res
        else: frames[k] = pd.DataFrame()
    return frames

def calculate_absence_impact(team: str, out_pl: list, imp_df: pd.DataFrame) -> float:
    if imp_df.empty or not out_pl: return 0.0
    p_col, d_col = _col(imp_df, ["player", "name"]), _col(imp_df, ["diff_pts_per_100_poss", "diff"])
    if not (p_col and d_col): return 0.0
    # Filtramos por jugadores que están OUT
    impacto = imp_df[imp_df[p_col].isin(out_pl)]
    return -pd.to_numeric(impacto[d_col], errors='coerce').sum()

def _team_momentum(sdf: pd.DataFrame, team: str) -> dict:
    res = {"off": 110.0, "def": 110.0, "pace": 98.5}
    tc = _col(sdf, ["team", "tm"])
    if not tc or sdf.empty: return res
    tdf = sdf[sdf.apply(lambda r: _match_team(r[tc], team, r["_source"]), axis=1)].copy()
    if tdf.empty: return res
    oc, dc, pc = _col(tdf, ["offrtg", "ortg"]), _col(tdf, ["defrtg", "drtg"]), _col(tdf, ["pace", "poss"])
    if oc: res["off"] = pd.to_numeric(tdf[oc], errors="coerce").mean()
    if dc: res["def"] = pd.to_numeric(tdf[dc], errors="coerce").mean()
    if pc: res["pace"] = pd.to_numeric(tdf[pc], errors="coerce").mean()
    return res

def project_game(home: str, away: str, frames: dict, h_imp: float, a_imp: float) -> dict:
    h = _team_momentum(frames.get("nbaleaguesumary", pd.DataFrame()), home)
    a = _team_momentum(frames.get("nbaleaguesumary", pd.DataFrame()), away)
    pace = np.nanmean([h["pace"], a["pace"]])
    # Proyección con impacto de bajas (Wemby effect)
    h_pts = (h["off"] + h_imp + a["def"]) / 2 * (pace/100) + 1.5
    a_pts = (a["off"] + a_imp + h["def"]) / 2 * (pace/100)
    return {"h_pts": h_pts, "a_pts": a_pts, "total": h_pts + a_pts, "winner": home if h_pts > a_pts else away, "spread": a_pts - h_pts}

def generate_roster(team: str, frames: dict, out_players: list) -> pd.DataFrame:
    df = frames.get("players", pd.DataFrame())
    tc, nc = _col(df, ["team", "tm"]), _col(df, ["player", "name"])
    if df.empty or not nc: return pd.DataFrame()
    # Filtro estricto por equipo para evitar mezcla Jokic/Wemby
    tdf = df[df.apply(lambda r: _match_team(r[tc] if tc else "", team, r["_source"]), axis=1)]
    data = []
    for _, row in tdf.iterrows():
        pname = str(row[nc])
        if pname in out_players or pname == "nan": continue
        pts = pd.to_numeric(row.get("pts", 0), errors="coerce")
        reb = pd.to_numeric(row.get("trb", row.get("reb", 0)), errors="coerce")
        ast = pd.to_numeric(row.get("ast", 0), errors="coerce")
        if pts > 0:
            data.append({"Jugador": pname, "Floor Safe": round(pts*0.7, 1), "PTS": pts, "REB": reb, "AST": ast, "PRA": pts+reb+ast})
    return pd.DataFrame(data).sort_values(by="PTS", ascending=False) if data else pd.DataFrame()

with st.sidebar:
    st.title("🏀 NBA HUNTER PRO")
    up = st.file_uploader("Sube archivos CTG/REF", accept_multiple_files=True)
    if up:
        frames = load_data({str(i): (f.name, f.read()) for i, f in enumerate(up)})
        st.success("Carga Exitosa")
        for k, v in frames.items():
            st.write(f"✅ {k.upper()} ({len(v)})" if not v.empty else f"❌ {k.upper()}")
    else: frames = {}

if not frames: st.stop()

teams = sorted(list(frames["nbaleaguesumary"][_col(frames["nbaleaguesumary"], ["team", "tm"])].dropna().unique()))
c1, c2 = st.columns(2)
t_h = c1.selectbox("Local", teams)
t_a = c2.selectbox("Visitante", teams, index=1 if len(teams)>1 else 0)

# Obtener lista de jugadores por equipo (Sin mezcla)
def get_players_by_team(f, t):
    p = set()
    df = f.get("players", pd.DataFrame())
    nc, tc = _col(df, ["player", "name"]), _col(df, ["team", "tm"])
    if not df.empty and nc:
        tdf = df[df.apply(lambda r: _match_team(r[tc] if tc else "", t, r["_source"]), axis=1)]
        p.update(tdf[nc].astype(str).tolist())
    return sorted([x for x in p if x != "nan"])

out_h = st.sidebar.multiselect(f"Bajas {t_h}", get_players_by_team(frames, t_h))
out_a = st.sidebar.multiselect(f"Bajas {t_a}", get_players_by_team(frames, t_a))

tab1, tab2 = st.tabs(["🎯 Game Hunter", "💥 Prop Assassin"])

with tab1:
    if st.button("Proyectar Game"):
        imp_h = calculate_absence_impact(t_h, out_h, frames.get("impact", pd.DataFrame()))
        imp_a = calculate_absence_impact(t_a, out_a, frames.get("impact", pd.DataFrame()))
        g = project_game(t_h, t_a, frames, imp_h, imp_a)
        res1, res2, res3 = st.columns(3)
        res1.metric("Ganador", g["winner"])
        res1.metric("Hándicap", f"{g['spread']:.1f}")
        res2.metric("Total O/U", f"{g['total']:.1f}")
        res3.metric(f"PTS {t_h}", f"{g['h_pts']:.1f}")
        res3.metric(f"PTS {t_a}", f"{g['a_pts']:.1f}")
        if imp_h != 0 or imp_a != 0:
            st.info(f"Ajuste por bajas: {t_h} ({imp_h:+.1f}) | {t_a} ({imp_a:+.1f})")

with tab2:
    if st.button("Generar Props"):
        st.subheader(f"Props {t_h}"); st.dataframe(generate_roster(t_h, frames, out_h), use_container_width=True)
        st.subheader(f"Props {t_a}"); st.dataframe(generate_roster(t_a, frames, out_a), use_container_width=True)
