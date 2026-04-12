import streamlit as st
import pandas as pd
import numpy as np
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Edge Analyzer",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
.stApp {
    background: #0a0c10;
    color: #e8eaf0;
}
[data-testid="stSidebar"] {
    background: #0f1117 !important;
    border-right: 1px solid #1e2330;
}
[data-testid="stSidebar"] * {
    color: #c8ccd8 !important;
}
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}
.metric-card {
    background: #12151f;
    border: 1px solid #1e2a3a;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.6rem;
}
.metric-card .label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #5a6070;
    margin-bottom: 4px;
}
.metric-card .value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #e8eaf0;
}
.pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 1px;
}
.pill-green  { background: #0d2e1a; color: #30d96a; border: 1px solid #30d96a44; }
.pill-red    { background: #2e0d0d; color: #f04545; border: 1px solid #f0454544; }
.pill-yellow { background: #2e2a0d; color: #f0c945; border: 1px solid #f0c94544; }
.pill-grey   { background: #1a1e2a; color: #8890a0; border: 1px solid #3a3e50; }
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: #3a8ef6;
    border-bottom: 1px solid #1e2330;
    padding-bottom: 6px;
    margin-bottom: 1rem;
    margin-top: 1.6rem;
}
.ev-positive { color: #30d96a !important; font-weight: 700; }
.ev-negative { color: #f04545 !important; }
.ev-neutral  { color: #8890a0 !important; }
stDataFrame { background: #0f1117 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS: SAFE FILE READERS
# ─────────────────────────────────────────────

def _read_file(uploaded) -> pd.DataFrame:
    """Read CSV or Excel from an UploadedFile object."""
    name = uploaded.name.lower()
    raw = uploaded.read()
    uploaded.seek(0)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))
    else:
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=enc)
            except Exception:
                continue
    return pd.DataFrame()


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[\s\-/]", "_", regex=True)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return df


def _col(df: pd.DataFrame, candidates: list) -> str | None:
    """Return first matching column name from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_float(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series([np.nan] * len(df))
    return pd.to_numeric(df[col], errors="coerce")


# ─────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data(files_bytes: dict) -> dict:
    """Parse all uploaded files into DataFrames."""
    frames = {}
    for key, (name, raw) in files_bytes.items():
        try:
            if name.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(io.BytesIO(raw))
            else:
                df = None
                for enc in ("utf-8", "latin-1", "cp1252"):
                    try:
                        df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                        break
                    except Exception:
                        continue
            if df is not None:
                frames[key] = _normalize_cols(df)
        except Exception as e:
            st.sidebar.warning(f"⚠️ {key}: {e}")
            frames[key] = pd.DataFrame()
    return frames


# ─────────────────────────────────────────────
# MODULE 1 HELPERS – GAME HUNTER
# ─────────────────────────────────────────────

def _team_momentum(summary_df: pd.DataFrame, team: str) -> dict:
    """
    Extract last-10-game OffRtg, DefRtg, Pace for a given team.
    Expects columns: team, offrtg/ortg, defrtg/drtg, pace, gp/game, date/g
    """
    result = {"offrtg": np.nan, "defrtg": np.nan, "pace": np.nan}
    if summary_df.empty:
        return result

    team_col = _col(summary_df, ["team", "team_name", "franchise", "squad"])
    if team_col is None:
        return result

    mask = summary_df[team_col].astype(str).str.lower().str.contains(team.lower(), na=False)
    team_df = summary_df[mask].copy()

    # Try to isolate last-10 rows (CTG may include split rows labeled L10)
    l10_col = _col(team_df, ["split", "segment", "last_n", "games"])
    if l10_col:
        l10_mask = team_df[l10_col].astype(str).str.lower().str.contains("l10|last.?10|last 10", regex=True, na=False)
        if l10_mask.any():
            team_df = team_df[l10_mask]

    off_col  = _col(team_df, ["offrtg", "ortg", "offensive_rating", "off_rtg", "o_rtg"])
    def_col  = _col(team_df, ["defrtg", "drtg", "defensive_rating", "def_rtg", "d_rtg"])
    pace_col = _col(team_df, ["pace", "pace_adj", "team_pace"])

    result["offrtg"] = pd.to_numeric(team_df[off_col],  errors="coerce").mean() if off_col else np.nan
    result["defrtg"] = pd.to_numeric(team_df[def_col],  errors="coerce").mean() if def_col else np.nan
    result["pace"]   = pd.to_numeric(team_df[pace_col], errors="coerce").mean() if pace_col else np.nan
    return result


def project_game(home: str, away: str, frames: dict) -> dict:
    """Return projected total, spread, and component ratings."""
    summary = frames.get("nbaleaguesumary", pd.DataFrame())

    home_m = _team_momentum(summary, home)
    away_m = _team_momentum(summary, away)

    # Expected points per 100 poss scaled to expected possessions
    avg_pace  = np.nanmean([home_m["pace"], away_m["pace"]])
    avg_pace  = avg_pace if not np.isnan(avg_pace) else 98.0

    # Each team's expected pts = their offrtg vs opponent defrtg, scaled by pace
    h_pts = np.nanmean([home_m["offrtg"], away_m["defrtg"]]) * (avg_pace / 100) if not np.isnan(home_m["offrtg"]) else np.nan
    a_pts = np.nanmean([away_m["offrtg"], home_m["defrtg"]]) * (avg_pace / 100) if not np.isnan(away_m["offrtg"]) else np.nan

    # Home court adjustment (+2.5 pts industry standard)
    h_adj = 1.5
    if not np.isnan(h_pts):
        h_pts += h_adj
    if not np.isnan(a_pts):
        a_pts -= h_adj * 0.5

    total  = (h_pts + a_pts) if (not np.isnan(h_pts) and not np.isnan(a_pts)) else np.nan
    spread = (h_pts - a_pts) if (not np.isnan(h_pts) and not np.isnan(a_pts)) else np.nan

    return {
        "home_pts":   round(h_pts,  1) if not np.isnan(h_pts)  else None,
        "away_pts":   round(a_pts,  1) if not np.isnan(a_pts)  else None,
        "total":      round(total,  1) if not np.isnan(total)  else None,
        "spread":     round(spread, 1) if not np.isnan(spread) else None,
        "home_offrtg": home_m["offrtg"],
        "home_defrtg": home_m["defrtg"],
        "away_offrtg": away_m["offrtg"],
        "away_defrtg": away_m["defrtg"],
        "avg_pace":    avg_pace,
    }


# ─────────────────────────────────────────────
# MODULE 2 HELPERS – PROP ASSASSIN
# ─────────────────────────────────────────────

def _player_momentum(players_df: pd.DataFrame, roster_df: pd.DataFrame, player: str) -> dict:
    """
    Average PTS, REB, AST from last-10 games for player.
    Cross-references roster for position.
    """
    result = {"pts": np.nan, "reb": np.nan, "ast": np.nan, "pos": "UNKNOWN"}

    # ── position from roster
    if not roster_df.empty:
        name_col = _col(roster_df, ["player", "player_name", "name", "full_name"])
        pos_col  = _col(roster_df, ["pos", "position", "primary_position"])
        if name_col and pos_col:
            mask = roster_df[name_col].astype(str).str.lower().str.contains(player.lower(), na=False)
            if mask.any():
                result["pos"] = str(roster_df.loc[mask, pos_col].iloc[0])

    if players_df.empty:
        return result

    name_col = _col(players_df, ["player", "player_name", "name", "full_name"])
    if name_col is None:
        return result

    mask = players_df[name_col].astype(str).str.lower().str.contains(player.lower(), na=False)
    p_df = players_df[mask].copy()

    if p_df.empty:
        return result

    # Try to get L10 split rows first
    split_col = _col(p_df, ["split", "segment", "last_n", "games"])
    if split_col:
        l10 = p_df[p_df[split_col].astype(str).str.lower().str.contains("l10|last.?10", regex=True, na=False)]
        if not l10.empty:
            p_df = l10

    pts_col = _col(p_df, ["pts", "points", "ppg", "pts_per_game"])
    reb_col = _col(p_df, ["reb", "rebounds", "rpg", "trb", "total_reb"])
    ast_col = _col(p_df, ["ast", "assists", "apg"])

    result["pts"] = pd.to_numeric(p_df[pts_col], errors="coerce").mean() if pts_col else np.nan
    result["reb"] = pd.to_numeric(p_df[reb_col], errors="coerce").mean() if reb_col else np.nan
    result["ast"] = pd.to_numeric(p_df[ast_col], errors="coerce").mean() if ast_col else np.nan
    return result


def _dvp_factor(def_df: pd.DataFrame, opp_team: str, position: str) -> float:
    """
    Defense vs Position multiplier.
    Returns a float near 1.0 (e.g., 1.08 = opp allows 8% more to this position).
    """
    if def_df.empty:
        return 1.0

    team_col = _col(def_df, ["team", "team_name", "opponent", "opp"])
    pos_col  = _col(def_df, ["pos", "position", "position_group"])

    if team_col is None:
        return 1.0

    mask = def_df[team_col].astype(str).str.lower().str.contains(opp_team.lower(), na=False)
    t_df = def_df[mask]

    if pos_col:
        p_mask = t_df[pos_col].astype(str).str.lower().str.contains(position.lower()[:2], na=False)
        if p_mask.any():
            t_df = t_df[p_mask]

    # Look for a "relative" or "allowed vs league avg" column
    rel_col = _col(t_df, ["rel", "relative", "diff", "vs_avg", "allowed_rel", "pts_allowed_rel", "pct_diff"])
    if rel_col and not t_df.empty:
        val = pd.to_numeric(t_df[rel_col], errors="coerce").mean()
        if not np.isnan(val):
            # If expressed as percentage points (e.g., +3.2 means 3.2% more)
            return 1.0 + (val / 100.0)

    # Fallback: absolute pts allowed vs position
    abs_col = _col(t_df, ["pts", "points", "pts_allowed", "ppg_allowed"])
    if abs_col and not t_df.empty:
        val = pd.to_numeric(t_df[abs_col], errors="coerce").mean()
        league_avg = 25.0  # rough positional avg
        if not np.isnan(val) and league_avg > 0:
            return val / league_avg

    return 1.0


def project_prop(player: str, opp_team: str, frames: dict) -> dict:
    """Full prop projection with DVP weighting."""
    players_df = frames.get("players", pd.DataFrame())
    roster_df  = frames.get("roster",  pd.DataFrame())
    def_df     = frames.get("def",     pd.DataFrame())

    mom = _player_momentum(players_df, roster_df, player)
    dvp = _dvp_factor(def_df, opp_team, mom["pos"])

    pts_proj = round(mom["pts"] * dvp, 1) if not np.isnan(mom["pts"]) else None
    reb_proj = round(mom["reb"] * dvp, 1) if not np.isnan(mom["reb"]) else None
    ast_proj = round(mom["ast"] * dvp, 1) if not np.isnan(mom["ast"]) else None
    pra_proj = None
    if pts_proj and reb_proj and ast_proj:
        pra_proj = round(pts_proj + reb_proj + ast_proj, 1)

    return {
        "player":    player,
        "position":  mom["pos"],
        "dvp_factor": round(dvp, 4),
        "pts_proj":  pts_proj,
        "reb_proj":  reb_proj,
        "ast_proj":  ast_proj,
        "pra_proj":  pra_proj,
    }


# ─────────────────────────────────────────────
# EV CALCULATION
# ─────────────────────────────────────────────

def ev_pct(projection, market_line, juice: float = -110) -> float | None:
    """
    Kelly-style EV%.  juice in American odds.
    Returns EV% of betting OVER (projection > market) or UNDER.
    """
    if projection is None or market_line is None:
        return None
    try:
        # Convert American odds to implied prob
        if juice < 0:
            implied = abs(juice) / (abs(juice) + 100)
        else:
            implied = 100 / (juice + 100)
        # Simple model: if projection > line → bet Over
        edge = (projection - market_line) / max(abs(market_line), 1)
        # EV = edge adjusted by implied prob
        ev = edge - (implied - 0.5)
        return round(ev * 100, 2)
    except Exception:
        return None


# ─────────────────────────────────────────────
# RESULT STYLER
# ─────────────────────────────────────────────

def style_ev(df: pd.DataFrame, ev_col: str) -> pd.io.formats.style.Styler:
    def row_color(row):
        val = row[ev_col]
        try:
            val = float(val)
        except Exception:
            return [""] * len(row)
        if val > 3:
            return [f"background-color: #0d2e1a; color: #30d96a"] * len(row)
        elif val < -3:
            return [f"background-color: #2e0d0d; color: #f04545"] * len(row)
        return ["background-color: #12151f"] * len(row)

    return (
        df.style
        .apply(row_color, axis=1)
        .format(precision=2, na_rep="—")
        .set_properties(**{
            "font-family": "Space Mono, monospace",
            "font-size": "0.82rem",
        })
    )


# ─────────────────────────────────────────────
# TEAM / PLAYER EXTRACTORS
# ─────────────────────────────────────────────

def get_teams(frames: dict) -> list[str]:
    teams = set()
    for key in ("nbaleaguesumary", "roster", "def"):
        df = frames.get(key, pd.DataFrame())
        if df.empty:
            continue
        tc = _col(df, ["team", "team_name", "franchise", "squad"])
        if tc:
            teams.update(df[tc].dropna().astype(str).unique().tolist())
    result = sorted([t for t in teams if len(t) > 1])
    return result if result else ["— upload data first —"]


def get_players(frames: dict) -> list[str]:
    players = set()
    for key in ("players", "roster", "impact"):
        df = frames.get(key, pd.DataFrame())
        if df.empty:
            continue
        nc = _col(df, ["player", "player_name", "name", "full_name"])
        if nc:
            players.update(df[nc].dropna().astype(str).unique().tolist())
    result = sorted([p for p in players if len(p) > 2])
    return result if result else ["— upload data first —"]


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏀 NBA Edge Analyzer")
    st.markdown('<div class="section-header">Data Upload</div>', unsafe_allow_html=True)
    st.caption("Upload your 6 CTG files (CSV or Excel)")

    file_keys = {
        "roster":          "1 · Roster (master)",
        "nbaleaguesumary": "2 · NBA League Summary",
        "def":             "3 · Defense vs Position",
        "players":         "4 · Players",
        "impact":          "5 · Impact",
        "trends":          "6 · Trends",
    }

    uploaded_raw = st.file_uploader(
        "Drop files here",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Map uploaded files to keys by filename matching
    file_map: dict = {}
    if uploaded_raw:
        for f in uploaded_raw:
            low = f.name.lower()
            matched = False
            for key in file_keys:
                if key in low or low.startswith(key):
                    file_map[key] = f
                    matched = True
                    break
            if not matched:
                # fallback: assign by upload order
                for key in file_keys:
                    if key not in file_map:
                        file_map[key] = f
                        break

        # Status pills
        for key, label in file_keys.items():
            loaded = key in file_map
            pill = f'<span class="pill {"pill-green" if loaded else "pill-grey"}">{"✓" if loaded else "·"}</span>'
            st.markdown(f"{pill} {label}", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Match Selector</div>', unsafe_allow_html=True)

    frames: dict = {}
    if file_map:
        bytes_map = {k: (v.name, v.read()) for k, v in file_map.items()}
        for k, v in file_map.items():
            v.seek(0)
        frames = load_data(bytes_map)

    teams   = get_teams(frames)
    players_list = get_players(frames)

    home_team = st.selectbox("Home Team", teams, key="home")
    away_team = st.selectbox("Away Team", teams, index=min(1, len(teams)-1), key="away")

    st.markdown('<div class="section-header">Prop Selector</div>', unsafe_allow_html=True)
    selected_player = st.selectbox("Player", players_list)
    prop_opponent   = st.selectbox("Opponent (DVP)", teams, key="prop_opp")


# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────

st.markdown("# NBA Edge Analyzer")
st.markdown(
    '<span class="pill pill-yellow">QUANTITATIVE BETTING INTELLIGENCE</span>',
    unsafe_allow_html=True,
)
st.divider()

tab1, tab2 = st.tabs(["🎯  Game Hunter", "💥  Prop Assassin"])

# ══════════════════════════════════════════════
# TAB 1 – GAME HUNTER
# ══════════════════════════════════════════════
with tab1:

    st.markdown("### Game Hunter — Total & Spread Projector")
    st.caption("Model uses last-10-game OffRtg / DefRtg / Pace from your League Summary file.")

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        mkt_total  = st.number_input("Market O/U Line", min_value=180.0, max_value=280.0,
                                     value=220.5, step=0.5, format="%.1f")
    with col_b:
        mkt_spread = st.number_input("Market Spread (Home)", min_value=-30.0, max_value=30.0,
                                     value=-3.5, step=0.5, format="%.1f")
    with col_c:
        juice      = st.number_input("Juice (American odds)", min_value=-130, max_value=-100,
                                     value=-110, step=1)

    run_game = st.button("🔍  Run Game Model", use_container_width=True, type="primary")

    if run_game:
        if not frames:
            st.error("Upload your CTG files first.")
        else:
            with st.spinner("Crunching last-10 momentum..."):
                gp = project_game(home_team, away_team, frames)

            st.markdown('<div class="section-header">Team Ratings (L10)</div>', unsafe_allow_html=True)

            r1, r2, r3, r4, r5 = st.columns(5)
            def mc(col, label, val, fmt=".1f"):
                with col:
                    disp = f"{val:{fmt}}" if val is not None else "—"
                    st.markdown(
                        f'<div class="metric-card"><div class="label">{label}</div>'
                        f'<div class="value">{disp}</div></div>',
                        unsafe_allow_html=True,
                    )

            mc(r1, f"{home_team} OffRtg", gp["home_offrtg"])
            mc(r2, f"{home_team} DefRtg", gp["home_defrtg"])
            mc(r3, "Avg Pace",            gp["avg_pace"])
            mc(r4, f"{away_team} OffRtg", gp["away_offrtg"])
            mc(r5, f"{away_team} DefRtg", gp["away_defrtg"])

            st.markdown('<div class="section-header">Projections vs Market</div>', unsafe_allow_html=True)

            ev_total  = ev_pct(gp["total"],  mkt_total,  juice)
            ev_spread = ev_pct(gp["spread"] if gp["spread"] else None,
                               abs(mkt_spread), juice)

            results = []
            results.append({
                "Metric":        "Total (O/U)",
                "Model Proj":    gp["total"]  if gp["total"]  else "—",
                "Market Line":   mkt_total,
                "Diff":          round(gp["total"] - mkt_total, 1) if gp["total"] else "—",
                "EV %":          ev_total if ev_total else "—",
                "Signal":        ("OVER ✅" if (gp["total"] or 0) > mkt_total else "UNDER ⬇️") if gp["total"] else "—",
            })
            results.append({
                "Metric":        "Spread (Home)",
                "Model Proj":    gp["spread"] if gp["spread"] else "—",
                "Market Line":   mkt_spread,
                "Diff":          round(gp["spread"] - mkt_spread, 1) if gp["spread"] else "—",
                "EV %":          ev_spread if ev_spread else "—",
                "Signal":        ("COVER ✅" if (gp["spread"] or 0) > mkt_spread else "NO COVER ⬇️") if gp["spread"] else "—",
            })

            df_res = pd.DataFrame(results)

            # Numeric EV col for styling
            df_style = df_res.copy()
            df_style["EV %"] = pd.to_numeric(df_style["EV %"], errors="coerce")

            st.dataframe(
                style_ev(df_style, "EV %"),
                use_container_width=True,
                hide_index=True,
            )

            # Score breakdown
            st.markdown('<div class="section-header">Score Projection</div>', unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            mc(s1, f"{home_team}",          gp["home_pts"])
            mc(s2, f"{away_team}",          gp["away_pts"])
            mc(s3, "Projected Total",       gp["total"])


# ══════════════════════════════════════════════
# TAB 2 – PROP ASSASSIN
# ══════════════════════════════════════════════
with tab2:

    st.markdown("### Prop Assassin — PRA Analyzer")
    st.caption("L10 momentum × DVP weighting from your Defense vs Position file.")

    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        mkt_pts  = st.number_input("Market PTS Line", min_value=0.0, max_value=60.0,
                                    value=20.5, step=0.5, format="%.1f")
    with col_p2:
        mkt_reb  = st.number_input("Market REB Line", min_value=0.0, max_value=30.0,
                                    value=5.5,  step=0.5, format="%.1f")
    with col_p3:
        mkt_ast  = st.number_input("Market AST Line", min_value=0.0, max_value=20.0,
                                    value=4.5,  step=0.5, format="%.1f")
    with col_p4:
        mkt_pra  = st.number_input("Market PRA Line", min_value=0.0, max_value=100.0,
                                    value=30.5, step=0.5, format="%.1f")
    juice_prop = st.number_input("Juice (props)", min_value=-130, max_value=-100,
                                  value=-115, step=1, key="juice_prop")

    run_prop = st.button("💥  Run Prop Model", use_container_width=True, type="primary")

    if run_prop:
        if not frames:
            st.error("Upload your CTG files first.")
        else:
            with st.spinner("Calculating DVP-weighted projections..."):
                pp = project_prop(selected_player, prop_opponent, frames)

            st.markdown(
                f'<div class="section-header">{selected_player} '
                f'— {pp["position"]} · DVP factor: {pp["dvp_factor"]:.4f}</div>',
                unsafe_allow_html=True,
            )

            ev_pts = ev_pct(pp["pts_proj"], mkt_pts, juice_prop)
            ev_reb = ev_pct(pp["reb_proj"], mkt_reb, juice_prop)
            ev_ast = ev_pct(pp["ast_proj"], mkt_ast, juice_prop)
            ev_pra = ev_pct(pp["pra_proj"], mkt_pra, juice_prop)

            def sig(proj, mkt):
                if proj is None:
                    return "—"
                return "OVER ✅" if proj > mkt else "UNDER ⬇️"

            prop_rows = [
                {
                    "Prop":        "Points",
                    "Model Proj":  pp["pts_proj"] or "—",
                    "Market Line": mkt_pts,
                    "Diff":        round(pp["pts_proj"] - mkt_pts, 1) if pp["pts_proj"] else "—",
                    "EV %":        ev_pts or "—",
                    "Signal":      sig(pp["pts_proj"], mkt_pts),
                },
                {
                    "Prop":        "Rebounds",
                    "Model Proj":  pp["reb_proj"] or "—",
                    "Market Line": mkt_reb,
                    "Diff":        round(pp["reb_proj"] - mkt_reb, 1) if pp["reb_proj"] else "—",
                    "EV %":        ev_reb or "—",
                    "Signal":      sig(pp["reb_proj"], mkt_reb),
                },
                {
                    "Prop":        "Assists",
                    "Model Proj":  pp["ast_proj"] or "—",
                    "Market Line": mkt_ast,
                    "Diff":        round(pp["ast_proj"] - mkt_ast, 1) if pp["ast_proj"] else "—",
                    "EV %":        ev_ast or "—",
                    "Signal":      sig(pp["ast_proj"], mkt_ast),
                },
                {
                    "Prop":        "PRA (Combined)",
                    "Model Proj":  pp["pra_proj"] or "—",
                    "Market Line": mkt_pra,
                    "Diff":        round(pp["pra_proj"] - mkt_pra, 1) if pp["pra_proj"] else "—",
                    "EV %":        ev_pra or "—",
                    "Signal":      sig(pp["pra_proj"], mkt_pra),
                },
            ]

            df_prop = pd.DataFrame(prop_rows)
            df_prop_s = df_prop.copy()
            df_prop_s["EV %"] = pd.to_numeric(df_prop_s["EV %"], errors="coerce")

            st.dataframe(
                style_ev(df_prop_s, "EV %"),
                use_container_width=True,
                hide_index=True,
            )

            # DVP explainer
            st.markdown('<div class="section-header">DVP Breakdown</div>', unsafe_allow_html=True)
            dvp_val = pp["dvp_factor"]
            if dvp_val > 1.05:
                dvp_label = f'<span class="pill pill-green">FAVORABLE MATCHUP +{(dvp_val-1)*100:.1f}%</span>'
            elif dvp_val < 0.95:
                dvp_label = f'<span class="pill pill-red">TOUGH MATCHUP {(dvp_val-1)*100:.1f}%</span>'
            else:
                dvp_label = f'<span class="pill pill-grey">NEUTRAL MATCHUP {(dvp_val-1)*100:+.1f}%</span>'

            st.markdown(
                f"{dvp_label}  vs **{prop_opponent}** for a **{pp['position']}**",
                unsafe_allow_html=True,
            )
            st.caption(
                "DVP factor multiplies the player's L10 averages. "
                ">1.05 = opponent allows more than average; <0.95 = tough defense."
            )


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    '<p style="font-family:Space Mono,monospace;font-size:0.65rem;'
    'color:#3a3e50;text-align:center;">NBA Edge Analyzer · Powered by CTG Data · '
    'For informational purposes only</p>',
    unsafe_allow_html=True,
)
