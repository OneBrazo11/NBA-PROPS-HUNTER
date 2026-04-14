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

def _is_team(val, target, fname):
    v, t, f = str(val).lower(), str(target).lower(), str(fname).lower()
    if t in v or v in t or t in f: return True
    d = {"sas":"san antonio", "den":"denver", "gsw":"golden", "lal":"lakers", "nyk":"york", "okc":"oklahoma"}
    return any(k in v or k in f for k, full in d.items() if full in t)

@st.cache_data(show_spinner=False)
def load_data(files_bytes: dict) -> dict:
    cats = ["roster", "nbaleaguesumary", "def", "players", "impact", "fouls", "frequency"]
    frames, temp_dict = {}, {k: [] for k in cats}
    keys = {"roster": ["ref", "roster"], "nbaleaguesumary": ["summary", "league"], "def": ["defense", "dvp"], "players": ["players", "overview"], "impact": ["impact", "onoff"], "fouls": ["foul"], "frequency": ["frequency"]}
    for _, (name, raw) in files_bytes.items():
        fname = name.lower()
        match = next((k for k, aliases in keys.items() if any(a in fname for a in aliases)), None)
        if match:
            try:
                df = pd.read_excel(io.BytesIO(raw)) if fname.endswith((".xlsx", ".xls")) else pd.read_csv(io.BytesIO(raw), encoding="latin-1")
                df = _normalize_cols(df)
                df["_file"] = fname
                temp_dict[match].append(df)
            except: pass
    for k, v in temp_dict.items():
        frames[k] = pd.concat(v, ignore_index=True) if v else pd.DataFrame()
    return frames

def calculate_absence_impact(team, out_pl, imp_df):
    if imp_df.empty or not out_pl: return 0.0
    p_c, d_c = _col(imp_df, ["player", "name"]), _col(imp_df, ["diff_pts_per_100_poss", "diff"])
    if not (p_c and d_c): return 0.0
    v = imp_df[imp_df[p_c].isin(out_pl)][d_c]
    return -pd.to_numeric(v, errors='coerce').sum()

def _team_momentum(sdf, team):
    res = {"off": 110.0, "def": 110.0, "pace": 98.5}
    tc = _col(sdf, ["team", "tm"])
    if not tc or sdf.empty: return res
    tdf = sdf[sdf.apply(lambda r: _is_team(r[tc], team, r["_file"]), axis=1)]
    if tdf.empty: return res
    oc, dc, pc = _col(tdf, ["offrtg", "ortg"]), _col(tdf, ["defrtg", "drtg"]), _col(tdf, ["pace", "poss"])
    if oc: res["off"] = pd.to_numeric(tdf[oc], errors="coerce").mean()
    if dc: res["def"] = pd.to_numeric(tdf[dc], errors="coerce").mean()
    if pc: res["pace"] = pd.to_numeric(tdf[pc], errors="coerce").mean()
    return res

def generate_roster(team, frames, out_p):
    df = frames.get("players", pd.DataFrame())
    if df.empty: df = frames.get("roster", pd.DataFrame())
    nc, tc = _col(df, ["player", "name"]), _col(df, ["team", "tm"])
    if df.empty or not nc: return pd.DataFrame()
    tdf = df[df.apply(lambda r: _is_team(r[tc] if tc else "", team, r["_file"]), axis=1)]
    data = []
    for _, r in tdf.iterrows():
        name = str(r[nc])
        if name in out_p or name == "nan" or name == "None": continue
        pts = pd.to_numeric(r.get("pts", 0), errors='coerce')
        if pts > 0:
            data.append({"Jugador": name, "Floor": round(pts*0.75, 1), "Proj PTS": round(pts, 1), "REB": r.get("trb", 0), "AST": r.get("ast", 0)})
    return pd.DataFrame(data).sort_values("Proj PTS", ascending=False) if data else pd.DataFrame()

def analyze_tactics(team, frames):
    res = {"Faltas": "-", "Pintura": "-", "Triples": "-"}
    for cat, k, cols in [("fouls", "Faltas", ["sfld", "fta"]), ("frequency", "Pintura", ["rim", "paint"]), ("nbaleaguesumary", "Triples", ["3pa", "3p_freq"])]:
        df = frames.get(cat, pd.DataFrame())
        if not df.empty:
            tc, c = _col(df, ["team", "tm"]), _col(df, cols)
            tdf = df[df.apply(lambda r: _is_team(r[tc] if tc else "", team, r["_file"]), axis=1)]
            if not tdf.empty and c:
                val = pd.to_numeric(tdf[c], errors='coerce').mean()
                res[k] = f"{round(val, 1)}%" if "rim" in c or "paint" in c else round(val, 1)
    return res

with st.sidebar:
    st.title("NBA PROPS HUNTER")
    up = st.file_uploader("Subir CTG/REF", accept_multiple_files=True)
    frames = load_data({str(i): (f.name, f.read()) for i, f in enumerate(up)}) if up else {}
    if frames: st.success("Base de Datos Lista")

if not frames: st.stop()
sum_df = frames.get("nbaleaguesumary", pd.DataFrame())
t_col = _col(sum_df, ["team", "tm"])
teams = sorted(list(sum_df[t_col].astype(str).unique())) if not sum_df.empty else ["Carga Summary"]

c1, c2 = st.columns(2)
t_h, t_a = c1.selectbox("Local", teams, 0), c2.selectbox("Visita", teams, 1 if len(teams)>1 else 0)

def get_p_list(f, t):
    players = set()
    for k in ["players", "roster"]:
        df = f.get(k, pd.DataFrame())
        nc, tc = _col(df, ["player", "name"]), _col(df, ["team", "tm"])
        if not df.empty and nc:
            tdf = df[df.apply(lambda r: _is_team(r[tc] if tc else "", t, r["_file"]), axis=1)]
            # Convertimos explícitamente a string y filtramos basura
            vals = tdf[nc].dropna().astype(str).tolist()
            players.update(vals)
    # Filtro final antes de ordenar
    final = [p for p in players if p.lower() not in ["nan", "none", "player", "name", ""]]
    return sorted(final)

out_h = st.sidebar.multiselect(f"Bajas {t_h}", get_p_list(frames, t_h))
out_a = st.sidebar.multiselect(f"Bajas {t_a}", get_p_list(frames, t_a))

tab1, tab2, tab3 = st.tabs(["🎯 Hunter", "💥 Props", "🎲 Micro"])
with tab1:
    if st.button("Proyectar"):
        h, a = _team_momentum(sum_df, t_h), _team_momentum(sum_df, t_a)
        imp_h = calculate_absence_impact(t_h, out_h, frames.get("impact", pd.DataFrame()))
        imp_a = calculate_absence_impact(t_a, out_a, frames.get("impact", pd.DataFrame()))
        pace = np.nanmean([h["pace"], a["pace"]])
        p_h = (h["off"] + imp_h + a["def"]) / 2 * (pace/100)
        p_a = (a["off"] + imp_a + h["def"]) / 2 * (pace/100)
        st.metric(f"Ganador: {t_h if p_h > p_a else t_a}", f"{p_h:.1f} - {p_a:.1f}")
with tab2:
    if st.button("Generar Tablas"):
        st.subheader(t_h); st.dataframe(generate_roster(t_h, frames, out_h), use_container_width=True)
        st.subheader(t_a); st.dataframe(generate_roster(t_a, frames, out_a), use_container_width=True)
with tab3:
    if st.button("Analizar Táctica"):
        col1, col2 = st.columns(2)
        col1.table(pd.Series(analyze_tactics(t_h, frames), name=t_h))
        col2.table(pd.Series(analyze_tactics(t_a, frames), name=t_a))
