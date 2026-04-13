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
        
    # Filtrar el equipo
    mask = summary_df[tc].astype(str).str.lower().str.contains(team.lower(), na=False)
    tdf = summary_df[mask].copy()
    
    if tdf.empty:
        return res
    
    # BUSCADOR EXACTO BASADO EN TU ARCHIVO LEAGUE_SUMMARY.CSV
    # Prioriza las columnas "Last 2 Weeks" (Momentum), si no están, usa la temporada completa
    oc = _col(tdf, ["last_2_weeks_offense", "offense", "offrtg"])
    dc = _col(tdf, ["last_2_weeks_defense", "defense", "defrtg"])
    
    # Tu archivo no tiene Pace, lo dejamos preparado por si algún día lo incluyes
    pc = _col(tdf, ["pace", "poss", "possessions", "ritmo"])
    
    if oc:
        res["offrtg"] = pd.to_numeric(tdf[oc], errors="coerce").mean()
    if dc:
        res["defrtg"] = pd.to_numeric(tdf[dc], errors="coerce").mean()
    if pc:
        res["pace"] = pd.to_numeric(tdf[pc], errors="coerce").mean()
        
    return res
