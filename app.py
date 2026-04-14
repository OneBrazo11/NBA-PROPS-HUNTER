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
    cats = ["nbaleaguesumary", "players", "impact", "def"]
    frames, temp_dict = {}, {k: [] for k in cats}
    keys = {"nbaleaguesumary": ["summary", "league"], "players": ["players", "overview", "ref", "roster"], "impact": ["impact", "onoff"], "def": ["defense", "dvp", "def"]}
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
    # Penalización agresiva: multiplicamos x1.2 el impacto negativo
    return -pd.to_numeric(impacto[d_col], errors='coerce').sum() * 1.2

def _team_momentum(sdf, team):
    res = {"off": 110.0, "def": 110.0, "pace": 98.5}
    tc = _col(sdf, ["team", "tm"])
    if not tc or sdf.empty: return res
    tdf = sdf[sdf[tc].apply(lambda x: _match_team(x, team))].copy()
    if tdf.empty: return res
    oc, dc, pc = _col(tdf, ["offrtg", "ortg"]), _col(tdf, ["defrtg", "drtg"]), _col(tdf, ["pace", "poss"])
    if oc: res["off"] = pd.to_numeric(tdf[oc], errors="coerce").mean()
    if dc: res["def"] = pd.to_numeric(tdf[dc], errors="coerce").mean()
    if pc: res["pace"] = pd.to_numeric(tdf[pc], errors="coerce").mean()
    return res

def project_game(home, away, frames, h_imp, a_imp):
    h = _team_momentum(frames.get("nbaleaguesumary", pd.DataFrame()), home)
    a = _team_momentum(frames.get("nbaleaguesumary", pd.DataFrame()), away)
    pace = np.nanmean([h["pace"], a["pace"]])
    h_pts = (h["off"] + h_imp + a["def"]) / 2 * (pace/100) + 1.5
    a_pts = (a["off"] + a_imp + h["def"]) / 2 * (pace/100)
    total = h_pts + a_pts
    winner = home if h_pts > a_pts else away
    spread = a_pts - h_pts
    return {"h_pts": h_pts, "a_pts": a_pts, "total": total, "winner": winner, "spread": spread}

def generate_roster(team, frames, out_p, has_major_out):
    df = frames.get("players", pd.DataFrame())
    tc, nc = _col(df, ["team", "tm"]), _col(df, ["player", "name"])
    if df.empty or not nc: return pd.DataFrame()
    tdf = df[df[tc].apply(lambda x: _match_team(x, team))].copy()
    data = []
    # Si hay bajas estrella (has_major_out), los demás absorben un 15% más de volumen
    boost = 1.15 if has_major_out else 1.0
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
        st.success("Cargado!")
        for k, v in frames.items():
            st.write(f"✅ {k.upper()} ({len(v)})" if not v.empty else f"❌ {k.upper()}")
    else: frames = {}

if not frames: st.stop()

teams = sorted(list(frames["nbaleaguesumary"][_col(frames["nbaleaguesumary"], ["team", "tm"])].dropna().unique()))
c1, c2 = st.columns(2)
t_h = c1.selectbox("Local", teams)
t_a = c2.selectbox("Visitante", teams, index=1 if len(teams)>1 else 0)

def get_players(f, t):
    p = set()
    df = f.get("players", pd.DataFrame())
    nc, tc = _col(df, ["player", "name"]), _col(df, ["team", "tm"])
    if not df.empty and nc:
        tdf = df[df[tc].apply(lambda x: _match_team(x, t))]
        p.update(tdf[nc].astype(str).tolist())
    clean = [str(x) for x in p if str(x).lower() not in ["nan", "none", ""]]
    return sorted(clean)

out_h = st.sidebar.multiselect(f"Bajas {t_h}", get_players(frames, t_h))
out_a = st.sidebar.multiselect(f"Bajas {t_a}", get_players(frames, t_a))

tab1, tab2 = st.tabs(["🎯 Game Hunter", "💥 Prop Assassin"])

with tab1:
    if st.button("Proyectar Partido"):
        imp_h = calculate_absence_impact(out_h, frames.get("impact", pd.DataFrame()))
        imp_a = calculate_absence_impact(out_a, frames.get("impact", pd.DataFrame()))
        g = project_game(t_h, t_a, frames, imp_h, imp_a)
        
        c_1, c_2, c_3 = st.columns(3)
        c_1.metric("GANADOR", g["winner"])
        hcap_txt = f"{t_h} {g['spread']:+.1f}" if g['spread'] < 0 else f"{t_a} {-g['spread']:+.1f}"
        c_1.metric("HÁNDICAP", hcap_txt)
        
        c_2.metric("PUNTOS TOTALES", f"{g['total']:.1f}")
        # Lógica de Over/Under (basada en promedio NBA ~222)
        ou_txt = "OVER" if g['total'] > 222 else "UNDER"
        c_2.metric("RECOMENDACIÓN O/U", ou_txt)
        
        c_3.metric(f"PTS {t_h}", f"{g['h_pts']:.1f}")
        c_3.metric(f"PTS {t_a}", f"{g['a_pts']:.1f}")

with tab2:
    if st.button("Generar Tablas"):
        st.subheader(f"Props {t_h}")
        st.dataframe(generate_roster(t_h, frames, out_h, len(out_h)>0), use_container_width=True)
        st.subheader(f"Props {t_a}")
        st.dataframe(generate_roster(t_a, frames, out_a, len(out_a)>0), use_container_width=True)
