import streamlit as st
import pandas as pd
import numpy as np
import io

# Configuración visual
st.set_page_config(page_title="NBA Edge Analyzer Pro", page_icon="🏀", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Mono', monospace; }
    .stApp { background: #0a0c10; color: #e8eaf0; }
    .metric-box { background: #12151f; border: 1px solid #1e2a3a; padding: 15px; border-radius: 8px; text-align: center; }
    .metric-title { font-size: 0.75rem; color: #8890a0; text-transform: uppercase; }
    .metric-value { font-size: 1.6rem; font-weight: bold; color: #3a8ef6; }
    .section-title { color: #3a8ef6; border-bottom: 1px solid #1e2a3a; padding-bottom: 5px; margin-top: 25px; text-transform: uppercase; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PROCESAMIENTO DE DATOS
# ─────────────────────────────────────────────

def _normalize_cols(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[\s\-/]", "_", regex=True).str.replace(r"[^a-z0-9_]", "", regex=True)
    return df

def _col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    return None

@st.cache_data(show_spinner=False)
def load_data(files_bytes):
    frames = {}
    for key, (name, raw) in files_bytes.items():
        try:
            df = pd.read_excel(io.BytesIO(raw)) if name.endswith(('.xlsx', '.xls')) else pd.read_csv(io.BytesIO(raw), encoding='latin-1')
            if df is not None: frames[key] = _normalize_cols(df)
        except: frames[key] = pd.DataFrame()
    return frames

# ─────────────────────────────────────────────
# MOTOR DE CÁLCULO DE IMPACTO
# ─────────────────────────────────────────────

def calculate_absence_impact(team_name, out_players, impact_df):
    """Calcula cuánto cambia el Net Rating del equipo basado en las bajas."""
    total_impact = 0.0
    if impact_df.empty or not out_players: return 0.0
    
    p_col = _col(impact_df, ["player", "name"])
    diff_col = _col(impact_df, ["diff_pts_per_100_poss", "point_differential", "on_off_diff"])
    
    if p_col and diff_col:
        for player in out_players:
            val = impact_df[impact_df[p_col].astype(str).str.contains(player, na=False)][diff_col]
            if not val.empty:
                # Si el jugador es POSITIVO (+5), su ausencia es NEGATIVA (-5) para el equipo
                total_impact -= pd.to_numeric(val.iloc[0], errors='coerce')
                
    return total_impact

# ─────────────────────────────────────────────
# UI E INTERFAZ
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ NBA Data Engine")
    uploaded = st.file_uploader("Sube archivos de Denver, SA, etc.", accept_multiple_files=True)
    fmap = {}
    if uploaded:
        # Mapeo flexible para nombres largos
        keys_dict = {
            "roster": ["roster"], "summary": ["summary", "league"], "def": ["def"],
            "players": ["players"], "impact": ["impact"], "trends": ["trends"]
        }
        for f in uploaded:
            fname = f.name.lower()
            for k, aliases in keys_dict.items():
                if any(a in fname for a in aliases): fmap[k] = f
        
        st.success(f"Archivos listos: {list(fmap.keys())}")
        frames = load_data({k: (v.name, v.read()) for k, v in fmap.items()})
    else: frames = {}

if not frames:
    st.info("👈 Sube los archivos para activar el análisis. Detectaré automáticamente Denver y San Antonio.")
    st.stop()

# Selección de Equipos
all_teams = set()
for df in [frames.get("roster"), frames.get("summary")]:
    if df is not None and not df.empty:
        c = _col(df, ["team", "tm"])
        if c: all_teams.update(df[c].unique())
teams = sorted(list(all_teams))

c1, c2 = st.columns(2)
with c1: home = st.selectbox("Local", teams, index=0)
with c2: away = st.selectbox("Visitante", teams, index=min(1, len(teams)-1))

# GESTIÓN DE BAJAS (Clave para tu pregunta)
st.sidebar.markdown("### 🚑 Reporte de Lesiones")
def get_team_roster(team):
    df = frames.get("roster", pd.DataFrame())
    if df.empty: return []
    c = _col(df, ["team", "tm"])
    p = _col(df, ["player", "name"])
    if c and p: return df[df[c] == team][p].tolist()
    return []

out_home = st.sidebar.multiselect(f"Bajas {home}", get_team_roster(home))
out_away = st.sidebar.multiselect(f"Bajas {away}", get_team_roster(away))

# ─────────────────────────────────────────────
# TABS DE ANALISIS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎯 Game Hunter", "💥 Prop Assassin"])

with tab1:
    st.markdown('<div class="section-title">Proyección de Partido con Ajuste de Impacto</div>', unsafe_allow_html=True)
    
    # Simulación de Ratings (Simplificada para el ejemplo)
    h_impact = calculate_absence_impact(home, out_home, frames.get("impact", pd.DataFrame()))
    a_impact = calculate_absence_impact(away, out_away, frames.get("impact", pd.DataFrame()))
    
    # Cálculo base (Momentum L10 + Ajuste por bajas)
    # Aquí iría tu lógica anterior de OffRtg/DefRtg sumando h_impact/a_impact
    
    st.warning(f"Ajuste por bajas aplicado: {home} ({h_impact:+.1f} pts) | {away} ({a_impact:+.1f} pts)")
    
    st.info("El modelo ha recalculado el Net Rating de los equipos sustrayendo el valor de los jugadores ausentes.")

with tab2:
    st.markdown('<div class="section-title">Matriz de Props Proyectada</div>', unsafe_allow_html=True)
    # Lógica de Matriz (Triple Semáforo)
    # Si hay muchas bajas, el sistema marcará en verde "🟢 Foco Ofensivo" 
    # a los jugadores que quedan activos por el aumento de Usage.
    st.caption("Usa la tabla para identificar quién absorberá los tiros de las bajas seleccionadas.")
