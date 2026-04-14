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

def _match_team(val, target):
    v, t = str(val).lower().strip(), str(target).lower().strip()
    if v == t or v in t or t in v: return True
    d = {"sas":"san antonio", "den":"denver", "bos":"boston", "gsw":"golden", "lal":"lakers", "lac":"clipper", "nyk":"new york", "okc":"oklahoma", "phx":"phoenix", "nop":"new orleans", "min":"minnesota", "sac":"sacramento", "dal":"dallas", "mia":"miami", "mem":"memphis", "orl":"orlando", "phi":"philadelphia", "tor":"toronto", "chi":"chicago", "cle":"cleveland", "det":"detroit", "ind":"indiana", "mil":"milwaukee", "atl":"atlanta", "cha":"charlotte", "was":"washington", "uta":"utah", "por":"portland", "hou":"houston", "bkn":"brooklyn"}
    if v in d and d[v] in t: return True
    if t in d and d[t] in v: return True
    return False

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
                df = _normalize_cols(df)
                # Si el archivo es un REF y no tiene columna TEAM, la creamos basada en el nombre del archivo
                if "ref" in fname and "team" not in df.columns and "tm" not in df.columns:
                    df["team"] = fname.split("_")[0]
                temp_dict[match].append(df)
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
    tdf = sdf[sdf[tc].apply(lambda x: _match_team(x, team))].copy()
    if tdf.empty: return res
    l10 = _col(tdf, ["split", "last_n", "games"])
    if l10 and tdf[l10].astype(str).str.lower().str.contains("l10|last", regex=True, na=False).any():
        tdf = tdf[tdf[l10].astype(str).str.lower().str.contains("l10|last", regex=True, na=False)]
    oc = _col(tdf, ["last_2_weeks_offense", "offense", "offrtg", "ortg", "pts_per_100"])
    dc = _col(tdf, ["last_2_weeks_defense", "defense", "defrtg", "drtg", "opp_pts_per_100"])
    pc = _col(tdf, ["pace", "poss", "ritmo"])
    if oc: res["offrtg"] = pd.to_numeric(tdf[oc], errors="coerce").mean()
    if dc: res["defrtg"] = pd.to_numeric(tdf[dc], errors="coerce").mean()
    if pc: res["pace"] = pd.to_numeric(tdf[pc], errors="coerce").mean()
    return res

def project_game(home: str, away: str, frames: dict, h_imp: float, a_imp: float) -> dict:
    hm, am = _team_momentum(frames.get("nbaleaguesumary", pd.DataFrame()), home), _team_momentum(frames.get("nbaleaguesumary", pd.DataFrame()), away)
    pace = np.nanmean([hm.get("pace", np.nan), am.get("pace", np.nan)])
    if np.isnan(pace): pace = 98.5
    ho, hd, ao, ad = hm.get("offrtg", np.nan), hm.get("defrtg", np.nan), am.get("offrtg", np.nan), am.get("defrtg", np.nan)
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
            if m in p1.columns: res[m] = pd.to_numeric(p1[m], errors="coerce").mean()
    if not idf.empty:
        nic = _col(idf, ["player", "name"])
        if nic:
            i1 = idf[idf[nic].astype(str).str.lower().str.contains(player.lower(), na=False)]
            c_usg = _col(i1, ["usage", "usg"])
            if c_usg: res["usg"] = pd.to_numeric(i1[c_usg], errors="coerce").mean()
    return res

def generate_roster_matrix(team: str, opp: str, frames: dict, out_players: list) -> pd.DataFrame:
    rdf = frames.get("roster", pd.DataFrame())
    if rdf.empty: rdf = frames.get("players", pd.DataFrame())
    tc, nc, pc = _col(rdf, ["team", "tm"]), _col(rdf, ["player", "name"]), _col(rdf, ["pos", "position"])
    if rdf.empty or not nc: return pd.DataFrame()
    roster = rdf[rdf[tc].apply(lambda x: _match_team(x, team))] if tc else rdf
    data = []
    for _, row in roster.iterrows():
        pname = str(row[nc])
        if pname in out_players: continue
        m = _player_metrics(frames.get("players", pd.DataFrame()), frames.get("impact", pd.DataFrame()), pname)
        if m["pts"] == 0 and m["trb"] == 0: continue
        dvp = 1.0 # Simplificado para velocidad
        pts_p = m["pts"] * dvp
        data.append({
            "Jugador": pname,
            "Floor Safe": round(pts_p * 0.75, 1), # Línea mínima segura
            "Proj PTS": round(pts_p, 1),
            "TRB": round(m["trb"], 1), "AST": round(m["ast"], 1),
            "STL/BLK": f"{round(m['stl'],1)}/{round(m['blk'],1)}",
            "ORB": round(m["orb"], 1), "MP": round(m["mp"], 1)
        })
    return pd.DataFrame(data).sort_values(by="Proj PTS", ascending=False) if data else pd.DataFrame()

def analyze_team_tactics(team: str, frames: dict):
    data = []
    def _get(cat, cols):
        df = frames.get(cat, pd.DataFrame())
        if df.empty: return np.nan
        tc, c = _col(df, ["team", "tm", "franchise"]), _col(df, cols)
        if not (tc and c): return np.nan
        val = df[df[tc].apply(lambda x: _match_team(x, team))][c]
        return pd.to_numeric(val, errors='coerce').mean() if not val.empty else np.nan

    sfld = _get("fouls", ["sfld", "fta"])
    rim = _get("frequency", ["rim", "paint"])
    tp = _get("nbaleaguesumary", ["3pa", "3p_freq"])
    
    return {"SFLD (Faltas)": round(sfld,1), "RIM (Pintura)": f"{round(rim,1)}%", "3PA (Triples)": round(tp,1)}

with st.sidebar:
    st.title("NBA PROPS & HUNTER")
    uploaded = st.file_uploader("Sube archivos", accept_multiple_files=True)
    if uploaded:
        frames = load_data({str(i): (f.name, f.read()) for i, f in enumerate(uploaded)})
        st.success("Fusionado")
    else: frames = {}

if not frames: st.stop()

teams = sorted(list(set(frames.get("nbaleaguesumary", pd.DataFrame())[_col(frames.get("nbaleaguesumary"), ["team", "tm"])].dropna().unique())))
t_home = st.columns(2)[0].selectbox("Local", teams)
t_away = st.columns(2)[1].selectbox("Visitante", teams, index=1 if len(teams)>1 else 0)

# BAJAS SIN DUPLICADOS
def get_unique_players(frames):
    p = set()
    for k in ["roster", "players"]:
        df = frames.get(k, pd.DataFrame())
        c = _col(df, ["player", "name"])
        if c: p.update(df[c].dropna().unique().tolist())
    return sorted(list(p))

all_p = get_unique_players(frames)
out_h = st.sidebar.multiselect(f"Bajas {t_home}", all_p)
out_a = st.sidebar.multiselect(f"Bajas {t_away}", all_p)

t1, t2, t3 = st.tabs(["🎯 Hunter", "💥 Props", "🎲 Micro"])

with t1:
    if st.button("Proyectar"):
        imp = frames.get("impact", pd.DataFrame())
        g = project_game(t_home, t_away, frames, calculate_absence_impact(t_home, out_h, imp), calculate_absence_impact(t_away, out_a, imp))
        st.metric(f"Ganador: {g['winner']}", f"{g['home_pts']:.1f} - {g['away_pts']:.1f}", f"Total: {g['total']:.1f}")

with t2:
    st.subheader(f"Props Proyectados: {t_home}")
    st.dataframe(generate_roster_matrix(t_home, t_away, frames, out_h), use_container_width=True)
    st.subheader(f"Props Proyectados: {t_away}")
    st.dataframe(generate_roster_matrix(t_away, t_home, frames, out_a), use_container_width=True)

with t3:
    c1, c2 = st.columns(2)
    c1.write(f"📊 Táctica {t_home}")
    c1.json(analyze_team_tactics(t_home, frames))
    c2.write(f"📊 Táctica {t_away}")
    c2.json(analyze_team_tactics(t_away, frames))
