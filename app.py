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
    return df.loc[:, ~df.columns.duplicated()]

def _col(df: pd.DataFrame, candidates: list) -> str | None:
    return next((c for c in candidates if c in df.columns), None)

def _match_team(val, target, fname=""):
    v, t = str(val).lower().strip(), str(target).lower().strip()
    if v == t or v in t or t in v or t in fname.lower(): return True
    d = {"sas":"san antonio", "den":"denver", "bos":"boston", "gsw":"golden", "lal":"lakers", "lac":"clipper", "nyk":"new york", "okc":"oklahoma", "phx":"phoenix", "nop":"new orleans", "min":"minnesota", "sac":"sacramento", "dal":"dallas", "mia":"miami", "mem":"memphis", "orl":"orlando", "phi":"philadelphia", "tor":"toronto", "chi":"chicago", "cle":"cleveland", "det":"detroit", "ind":"indiana", "mil":"milwaukee", "atl":"atlanta", "cha":"charlotte", "was":"washington", "uta":"utah", "por":"portland", "hou":"houston", "bkn":"brooklyn"}
    if v in d and d[v] in t: return True
    if t in d and d[t] in v: return True
    return False

@st.cache_data(show_spinner=False)
def load_data(files_bytes: dict) -> dict:
    cats = ["nbaleaguesumary", "players", "impact"]
    frames, temp_dict = {}, {k: [] for k in cats}
    keys_dict = {
        "nbaleaguesumary": ["summary", "league", "resumen"],
        "players": ["players", "jugadores", "ref", "roster"],
        "impact": ["impact", "impacto", "onoff"]
    }
    for _, (name, raw) in files_bytes.items():
        fname = name.lower()
        match = next((k for k, aliases in keys_dict.items() if any(a in fname for a in aliases)), None)
        if match:
            try:
                df = pd.read_excel(io.BytesIO(raw)) if fname.endswith((".xlsx", ".xls")) else pd.read_csv(io.BytesIO(raw), encoding="latin-1")
                df = _normalize_cols(df)
                df["_source_file"] = fname
                temp_dict[match].append(df)
            except: pass
    for k, v in temp_dict.items():
        if v:
            res = pd.concat(v, ignore_index=True)
            p_col = _col(res, ["player", "name"])
            if p_col: res = res.drop_duplicates(subset=[p_col, "_source_file"])
            frames[k] = res
        else: frames[k] = pd.DataFrame()
    return frames

def calculate_absence_impact(team: str, out_pl: list, imp_df: pd.DataFrame) -> float:
    if imp_df.empty or not out_pl: return 0.0
    p_col, d_col = _col(imp_df, ["player", "name"]), _col(imp_df, ["diff_pts_per_100_poss", "on_off_diff", "diff"])
    if not (p_col and d_col): return 0.0
    tot = 0.0
    for p in out_pl:
        v = imp_df[imp_df[p_col].astype(str) == p][d_col]
        if not v.empty:
            val = pd.to_numeric(v.iloc[0], errors='coerce')
            if not np.isnan(val): tot -= val
    return tot

def _team_momentum(sdf: pd.DataFrame, team: str) -> dict:
    res = {"offrtg": np.nan, "defrtg": np.nan, "pace": np.nan}
    tc = _col(sdf, ["team", "tm", "team_name"])
    if not tc or sdf.empty: return res
    tdf = sdf[sdf.apply(lambda r: _match_team(r[tc], team, r.get("_source_file", "")), axis=1)].copy()
    if tdf.empty: return res
    l10 = _col(tdf, ["split", "last_n", "games"])
    if l10 and tdf[l10].astype(str).str.lower().str.contains("l10|last", regex=True, na=False).any():
        tdf = tdf[tdf[l10].astype(str).str.lower().str.contains("l10|last", regex=True, na=False)]
    oc, dc, pc = _col(tdf, ["offrtg", "ortg", "offense"]), _col(tdf, ["defrtg", "drtg", "defense"]), _col(tdf, ["pace", "poss"])
    if oc: res["offrtg"] = pd.to_numeric(tdf[oc], errors="coerce").mean()
    if dc: res["defrtg"] = pd.to_numeric(tdf[dc], errors="coerce").mean()
    if pc: res["pace"] = pd.to_numeric(tdf[pc], errors="coerce").mean()
    return res

def project_game(home: str, away: str, frames: dict, h_imp: float, a_imp: float) -> dict:
    hm = _team_momentum(frames.get("nbaleaguesumary", pd.DataFrame()), home)
    am = _team_momentum(frames.get("nbaleaguesumary", pd.DataFrame()), away)
    pace = np.nanmean([hm.get("pace", np.nan), am.get("pace", np.nan)])
    if np.isnan(pace): pace = 98.5
    ho, hd, ao, ad = hm.get("offrtg", np.nan), hm.get("defrtg", np.nan), am.get("offrtg", np.nan), am.get("defrtg", np.nan)
    ho_adj, ao_adj = ho + h_imp if not np.isnan(ho) else np.nan, ao + a_imp if not np.isnan(ao) else np.nan
    h_pts = np.nanmean([ho_adj, ad]) * (pace/100) if not np.isnan(ho_adj) else np.nan
    a_pts = np.nanmean([ao_adj, hd]) * (pace/100) if not np.isnan(ao_adj) else np.nan
    if not np.isnan(h_pts): h_pts += 1.5
    tot = (h_pts + a_pts) if (not np.isnan(h_pts) and not np.isnan(a_pts)) else np.nan
    winner = home if (h_pts > a_pts) else away
    spread = a_pts - h_pts
    return {"h_pts": h_pts, "a_pts": a_pts, "total": tot, "winner": winner, "spread": spread}

def generate_roster_matrix(team: str, frames: dict, out_players: list) -> pd.DataFrame:
    df = frames.get("players", pd.DataFrame())
    tc, nc = _col(df, ["team", "tm"]), _col(df, ["player", "name"])
    if df.empty or not nc: return pd.DataFrame()
    tdf = df[df.apply(lambda r: _match_team(r[tc] if tc else "", team, r.get("_source_file", "")), axis=1)]
    data = []
    for _, row in tdf.iterrows():
        pname = str(row[nc])
        if pname in out_players: continue
        pts = pd.to_numeric(row.get("pts", 0), errors="coerce")
        reb = pd.to_numeric(row.get("trb", row.get("reb", 0)), errors="coerce")
        ast = pd.to_numeric(row.get("ast", 0), errors="coerce")
        if pts > 0 or reb > 0:
            data.append({"Jugador": pname, "Puntos": pts, "Rebotes": reb, "Asistencias": ast, "PRA": pts+reb+ast})
    return pd.DataFrame(data).sort_values(by="Puntos", ascending=False) if data else pd.DataFrame()

with st.sidebar:
    st.title("NBA HUNTER SIMPLIFIED")
    uploaded = st.file_uploader("Sube archivos (CTG y REF)", accept_multiple_files=True)
    if uploaded:
        frames = load_data({str(i): (f.name, f.read()) for i, f in enumerate(uploaded)})
        st.success("Cargado")
        for k, v in frames.items(): st.write(f"✅ {k.upper()} ({len(v)})")
    else: frames = {}

if not frames: st.stop()

summary_df = frames.get("nbaleaguesumary", pd.DataFrame())
t_col = _col(summary_df, ["team", "tm"])
teams = sorted(list(summary_df[t_col].dropna().unique())) if t_col else ["Error en League Summary"]

c1, c2 = st.columns(2)
t_home = c1.selectbox("Local", teams)
t_away = c2.selectbox("Visitante", teams, index=1 if len(teams)>1 else 0)

all_p = sorted(list(set([p for k in ["players"] if not (df := frames.get(k, pd.DataFrame())).empty and (c := _col(df, ["player", "name"])) for p in df[c].dropna().unique()])))
out_h = st.sidebar.multiselect(f"Bajas {t_home}", all_p)
out_a = st.sidebar.multiselect(f"Bajas {t_away}", all_p)

t1, t2 = st.tabs(["🎯 Hunter (Game)", "💥 Props"])

with t1:
    if st.button("Proyectar Game"):
        imp = frames.get("impact", pd.DataFrame())
        g = project_game(t_home, t_away, frames, calculate_absence_impact(t_home, out_h, imp), calculate_absence_impact(t_away, out_a, imp))
        res_c1, res_c2, res_c3 = st.columns(3)
        res_c1.metric("Ganador Proyectado", g["winner"])
        res_c1.metric("Hándicap", f"{g['spread']:.1f}")
        res_c2.metric("Total O/U", f"{g['total']:.1f}")
        res_c3.metric(f"PTS {t_home}", f"{g['h_pts']:.1f}")
        res_c3.metric(f"PTS {t_away}", f"{g['a_pts']:.1f}")

with t2:
    if st.button("Generar Props"):
        st.subheader(f"Props: {t_home}")
        st.dataframe(generate_roster_matrix(t_home, frames, out_h), use_container_width=True)
        st.subheader(f"Props: {t_away}")
        st.dataframe(generate_roster_matrix(t_away, frames, out_a), use_container_width=True)
