"""
Dashboard de Campañas — CMR Puntos
Página única, flujo top-down:
  1. KPIs generales + tabla de todas las campañas (con semáforo)
  2. Detalle de campaña: email | quién canjeó | GM vs GC
  3. [Expander] Base enviada + descarga CSV

Requiere haber ejecutado: python extract.py --fecha-inicio YYYY-MM-DD
"""
import os, warnings
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

FUNNEL_LABELS = {
    "NO_CANJEADOR_HISTORICO": "Sin historial de canje",
    "FUGADOS":                "Fugados",
    "NUEVO":                  "Nuevo canjeador",
    "RECUPERADO":             "Recuperado",
    "RECURRENTE":             "Recurrente",
}
FUNNEL_ORDER = ["NO_CANJEADOR_HISTORICO", "FUGADOS", "NUEVO", "RECUPERADO", "RECURRENTE"]
FUNNEL_COLORS = {
    "NO_CANJEADOR_HISTORICO": "#9E9E9E",
    "FUGADOS":                "#EF5350",
    "NUEVO":                  "#66BB6A",
    "RECUPERADO":             "#42A5F5",
    "RECURRENTE":             "#26C6DA",
}

st.set_page_config(page_title="Campañas CMR Puntos", layout="wide")
st.markdown("""
<style>
    div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; }
    div[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #555; }
    .section-title { font-size: 1.1rem; font-weight: 600; color: #1a1a2e;
                     border-left: 4px solid #1565C0; padding-left: 10px; margin: 20px 0 10px; }
    .camp-label { font-size: 1.3rem; font-weight: 600; color: #1565C0; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def spct(a, b, dec=1):
    return round(a / b * 100, dec) if b and b > 0 else 0.0


def detect(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def norm_camp(s):
    return str(s).strip().upper().replace(" ", "")


# ── Carga de datos ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando datos...")
def load_data():
    out = {}
    for key, fname in [("r", "redemption_metrics.parquet"),
                       ("omm", "omm_metrics.parquet"),
                       ("hist", "historical_campaigns.parquet")]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df.columns = [c.upper() for c in df.columns]
            out[key] = df
        else:
            out[key] = pd.DataFrame()
    return out


@st.cache_data(show_spinner="Cargando datos RFM...")
def load_rfm():
    path = os.path.join(DATA_DIR, "rfm_all_clients.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Cargando RFM × Campañas...")
def load_rfm_campaigns():
    path = os.path.join(DATA_DIR, "rfm_campaigns.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_rfm_campaign_individual(camp_name: str) -> pd.DataFrame:
    """Carga datos individuales (cliente × propensity × canje) para una campaña específica.
    Lee solo la partición de la campaña seleccionada del dataset particionado por campaña.
    """
    path = os.path.join(DATA_DIR, "rfm_campaigns_individual")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(path, filters=[("NOMBRE_CAMPANHA", "=", camp_name)])
        # Limpiar metadatos pandas (evita error con tipos BigQuery como dbdate)
        table = table.replace_schema_metadata({})
        return table.to_pandas()
    except Exception:
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_campaign_model():
    """Carga el modelo de canje en 5 días (campaign_model.pkl)."""
    path = os.path.join(DATA_DIR, "campaign_model.pkl")
    if not os.path.exists(path):
        return None
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_rfm_features_pred():
    """Carga columnas necesarias de rfm_all_clients para predicción del modelo 5d."""
    path = os.path.join(DATA_DIR, "rfm_all_clients.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    import pyarrow.parquet as pq_f
    schema_cols = set(pq_f.read_schema(path).names)
    needed = [
        "cust_id", "tier", "gender", "city", "rfm_segment",
        "cust_active_deb_flg", "cust_active_cmr_flg",
        "frequency_monthly_avg", "monetary_monthly_avg", "redeem_rate",
        "retailer_entropy", "pct_redeem_digital", "earn_velocity_90",
        "days_since_last_activity", "points_pressure", "stock_points_at_t0",
        "redeem_count_pre", "frequency_total", "monetary_total", "tenure_months",
        "age", "stock_points", "exp_points_current", "exp_points_next",
        "recency_days", "r_score", "f_score", "m_score", "propensity_score",
    ]
    cols = [c for c in needed if c in schema_cols]
    df = pd.read_parquet(path, columns=cols)
    # Derivar features faltantes (igual que en train)
    if "frequency_monthly_avg" not in df.columns and "frequency_total" in df.columns:
        df["frequency_monthly_avg"] = df["frequency_total"] / 12.0
    if "monetary_monthly_avg" not in df.columns and "monetary_total" in df.columns:
        df["monetary_monthly_avg"] = df["monetary_total"] / 12.0
    if "redeem_rate" not in df.columns and "frequency_total" in df.columns:
        df["redeem_rate"] = df.get("redeem_count_pre", 0) / df["frequency_total"].replace(0, 1)
    if "stock_points_at_t0" not in df.columns and "stock_points" in df.columns:
        df["stock_points_at_t0"] = df["stock_points"]
    return df


def predict_5d(bundle, df_ind, df_rfm_feat, camp_fecha):
    """Predice probabilidad de canje en 5 días para clientes de la campaña.
    Retorna df_ind con columna p_5dias (probabilidad calibrada).
    """
    import numpy as np
    model       = bundle["model"]
    enc         = bundle["ordinal_encoder"]
    features    = bundle["features"]
    cat_feats   = bundle["cat_features"]
    bool_feats  = bundle["bool_features"]
    prior_ratio = bundle["prior_ratio"]

    df = df_ind[["cust_id", "canjeo", "GMGC"]].merge(df_rfm_feat, on="cust_id", how="left")

    # Guardar rfm_segment original (string) antes del encoding para el dashboard
    rfm_seg_orig = df["rfm_segment"].copy() if "rfm_segment" in df.columns else None

    fecha = pd.to_datetime(camp_fecha)
    df["mes_envio"]        = float(fecha.month)
    df["dia_semana_envio"] = float(fecha.dayofweek)
    df["es_gm"]            = (df["GMGC"] == "GM").astype(float)
    df["es_cyber_month"]   = float(1 if fecha.month == 11 else 0)
    df["es_diciembre"]     = float(1 if fecha.month == 12 else 0)

    for col in features:
        if col not in df.columns:
            df[col] = 0.0

    if enc is not None and cat_feats:
        df[cat_feats] = enc.transform(df[cat_feats].astype(str).fillna("__nan__"))

    for col in bool_feats:
        if col in df.columns:
            df[col] = df[col].map(lambda v: 1 if str(v).upper() in ("1", "TRUE", "SI", "YES", "Y") else 0)

    X     = df[features].fillna(0).astype(float)
    p_bal = model.predict_proba(X)[:, 1]

    # Calibración: escalar al base rate real de la campaña (cuando está disponible)
    # Preserva el ranking relativo entre segmentos y ajusta la media al valor real.
    # Fallback: corrección de prior si la campaña no tiene datos de canje aún.
    import numpy as np
    known_rate = float(df["canjeo"].mean()) if "canjeo" in df.columns else 0.0
    mean_p_bal = float(p_bal.mean())

    if known_rate > 0.001 and mean_p_bal > 0:
        # Campaña histórica: escalar scores al base rate real de la campaña
        p_cal = np.clip(p_bal * (known_rate / mean_p_bal), 0.0, 1.0)
    else:
        # Campaña futura: corrección de prior (muestra balanceada → población)
        odds  = p_bal / (1 - p_bal + 1e-9)
        p_cal = (odds * prior_ratio) / (1 + odds * prior_ratio)

    df["p_5dias"] = p_cal

    # Restaurar rfm_segment con nombre legible para el groupby del dashboard
    if rfm_seg_orig is not None:
        df["rfm_segment"] = rfm_seg_orig

    return df


# ── Construir tabla resumen de campañas ───────────────────────────────────────
def build_camp_table(df_r, df_omm, df_rfm_camp=None):
    """Una fila por campaña con todas las métricas consolidadas."""
    if df_r.empty:
        return pd.DataFrame()

    col_camp  = detect(df_r, ["NOMBRE_CAMPANHA", "CAMPANHA"])
    col_fecha = detect(df_r, ["FECHA_ENVIO"])
    col_gm    = detect(df_r, ["GMGC"])
    col_flag  = detect(df_r, ["FLAG_CANJE"])
    col_cli   = detect(df_r, ["CLIENTES"])
    col_post  = detect(df_r, ["GRUPO_CANJE_POST"])

    # Pre-indexar rfm_campaigns por campaña para lookup rápido
    rfm_idx = {}
    if df_rfm_camp is not None and not df_rfm_camp.empty:
        for camp_k, grp_r in df_rfm_camp.groupby("NOMBRE_CAMPANHA"):
            rfm_idx[camp_k] = grp_r

    rows = []
    for camp, grp in df_r.groupby(col_camp):
        fecha = pd.to_datetime(grp[col_fecha]).max().strftime("%Y-%m-%d") if col_fecha else "—"

        gm = grp[grp[col_gm] == "GM"] if col_gm else grp
        gc = grp[grp[col_gm] == "GC"] if col_gm else pd.DataFrame()

        def canje_rate(g):
            if g.empty or col_flag is None: return 0.0, 0
            flags = g[col_flag].astype(str).str.upper()
            mask  = flags.isin(["CANJEO", "1", "TRUE", "SI", "S"])
            cnt   = int(g[col_cli].sum()) if col_cli else len(g)
            canjeadores = int(g.loc[mask, col_cli].sum()) if col_cli else int(mask.sum())
            return spct(canjeadores, cnt), canjeadores

        pct_gm, canj_gm = canje_rate(gm)
        pct_gc, _       = canje_rate(gc)
        lift = max(0.0, round(pct_gm - pct_gc, 2))

        total_gm = int(gm[col_cli].sum()) if col_cli and not gm.empty else len(gm)
        col_pre_bt = detect(df_r, ["GRUPO_CANJE_PRE"])
        nuevos_gm = 0
        if col_pre_bt and col_flag and col_cli and not gm.empty:
            mask_nuevo = (gm[col_pre_bt] == "NO_CANJEADOR_HISTORICO") & (gm[col_flag] == "CANJEO")
            nuevos_gm = int(gm.loc[mask_nuevo, col_cli].sum())
        elif col_pre_bt and col_flag and not gm.empty:
            nuevos_gm = int(((gm[col_pre_bt] == "NO_CANJEADOR_HISTORICO") & (gm[col_flag] == "CANJEO")).sum())

        # ── Columnas RFM ──────────────────────────────────────────────────────
        seg_dom   = "—"
        pct_dom   = 0.0
        if camp in rfm_idx:
            rfm_gm = rfm_idx[camp][rfm_idx[camp]["GMGC"] == "GM"]
            if not rfm_gm.empty:
                totals = rfm_gm.groupby("rfm_segment")["CLIENTES"].sum()
                seg_dom = totals.idxmax()
                sub_dom = rfm_gm[rfm_gm["rfm_segment"] == seg_dom]
                tot_dom = int(sub_dom["CLIENTES"].sum())
                can_dom = int(sub_dom.loc[sub_dom["FLAG_CANJE"] == "CANJEO", "CLIENTES"].sum())
                pct_dom = spct(can_dom, tot_dom)

        rows.append({
            "_camp_key":       norm_camp(camp),
            "Campaña":         camp,
            "Fecha":           fecha,
            "Canjeadores":     canj_gm,
            "% Canje GM":      pct_gm,
            "% Canje GC":      pct_gc,
            "Lift (pp)":       lift,
            "Nuevos":          nuevos_gm,
            "% Conv.":         spct(nuevos_gm, total_gm),
            "Seg. RFM dom.":   seg_dom,
            "% Canje top seg.": pct_dom,
        })

    df_out = pd.DataFrame(rows)

    # Join con OMM
    # Estrategia: primero match exacto por nombre normalizado;
    # para los que no matchean, intentar match por nombre-base (sin fecha al final: _DDMMYYYY)
    if not df_omm.empty:
        omm_camp_col = detect(df_omm, ["CAMPAIGN_NAME", "NOMBRE_CAMPANHA"])
        if omm_camp_col:
            agg_cols = {c: "sum" for c in ["CASOS","TOTAL_MAILS","TOTAL_OPENS","TOTAL_CLICKS",
                                            "TOTAL_BOUNCES","TOTAL_UNSUBS"] if c in df_omm.columns}
            omm_agg = df_omm.groupby(omm_camp_col).agg(agg_cols).reset_index()
            omm_agg["_camp_key"] = omm_agg[omm_camp_col].apply(norm_camp)

            # Clave base: nombre sin los últimos 8 dígitos de fecha (_DDMMYYYY)
            import re
            def base_key(k):
                return re.sub(r'_\d{8}$', '', k)

            omm_agg["_base_key"] = omm_agg["_camp_key"].apply(base_key)
            df_out["_base_key"]  = df_out["_camp_key"].apply(base_key)

            if "TOTAL_MAILS" in omm_agg.columns:
                for col_n, label in [("TOTAL_OPENS","Aperturas %"),
                                     ("TOTAL_CLICKS","Clicks %"),
                                     ("TOTAL_BOUNCES","Bounces %")]:
                    if col_n in omm_agg.columns:
                        omm_agg[label] = (omm_agg[col_n] / omm_agg["TOTAL_MAILS"] * 100).round(1)
            if "CASOS" in omm_agg.columns and "TOTAL_MAILS" in omm_agg.columns:
                omm_agg["Enviados"] = omm_agg["TOTAL_MAILS"].astype(int)

            keep = ["_camp_key","_base_key","Enviados","Aperturas %","Clicks %","Bounces %"]
            omm_agg = omm_agg[[c for c in keep if c in omm_agg.columns]]

            # 1er intento: match exacto
            df_out = df_out.merge(omm_agg.drop(columns=["_base_key"]),
                                  on="_camp_key", how="left")
            # 2do intento: para los sin match, intentar por base_key
            no_match = df_out["Enviados"].isna() if "Enviados" in df_out.columns else pd.Series([True]*len(df_out))
            if no_match.any():
                omm_base = omm_agg.drop_duplicates("_base_key")[
                    ["_base_key"] + [c for c in ["Enviados","Aperturas %","Clicks %","Bounces %"]
                                     if c in omm_agg.columns]
                ]
                df_fallback = df_out[no_match][["_camp_key","_base_key"]].merge(
                    omm_base, on="_base_key", how="left"
                )
                for col in ["Enviados","Aperturas %","Clicks %","Bounces %"]:
                    if col in df_fallback.columns:
                        df_out.loc[no_match, col] = df_fallback[col].values

    df_out = df_out.drop(columns=["_camp_key","_base_key"], errors="ignore")
    # Reordenar columnas
    cols_order = ["Campaña","Fecha","Enviados","Aperturas %","Clicks %",
                  "Bounces %","Canjeadores","% Canje GM","% Canje GC","Lift (pp)",
                  "Nuevos","% Conv.","Seg. RFM dom.","% Canje top seg."]
    df_out = df_out[[c for c in cols_order if c in df_out.columns]]
    return df_out.sort_values("Fecha", ascending=False) if "Fecha" in df_out.columns else df_out


def style_table(df):
    """Aplica semáforo a % Canje GM y Lift."""
    def color_pct(val, avg):
        if pd.isna(val): return ""
        if val > avg * 1.1:   return "background-color: #C8E6C9; color: #1B5E20"
        if val < avg * 0.9:   return "background-color: #FFCDD2; color: #B71C1C"
        return "background-color: #FFF9C4; color: #F57F17"

    def color_lift(val):
        if pd.isna(val): return ""
        if val > 1:    return "background-color: #C8E6C9; color: #1B5E20; font-weight:bold"
        if val < -1:   return "background-color: #FFCDD2; color: #B71C1C"
        return ""

    def fmt_int(x):
        try:
            return f"{int(x):,}" if pd.notna(x) and x == x else "—"
        except Exception:
            return "—"

    def fmt_pct(x, sign=False):
        try:
            if pd.isna(x) or x != x: return "—"
            return f"{x:+.1f}pp" if sign else f"{x:.1f}%"
        except Exception:
            return "—"

    fmt_map = {
        "Enviados":          fmt_int,
        "Canjeadores":       fmt_int,
        "Nuevos":            fmt_int,
        "% Canje GM":        fmt_pct,
        "% Canje GC":        fmt_pct,
        "% Conv.":           fmt_pct,
        "Lift (pp)":         lambda x: fmt_pct(x, sign=True),
        "Aperturas %":       fmt_pct,
        "Clicks %":          fmt_pct,
        "Bounces %":         fmt_pct,
        "% Canje top seg.":  fmt_pct,
    }
    styler = df.style.format({k: v for k, v in fmt_map.items() if k in df.columns}, na_rep="—")

    if "% Canje GM" in df.columns:
        avg = df["% Canje GM"].mean()
        styler = styler.map(lambda v: color_pct(v, avg), subset=["% Canje GM"])
    if "Lift (pp)" in df.columns:
        styler = styler.map(color_lift, subset=["Lift (pp)"])
    if "% Conv." in df.columns:
        avg_conv = df["% Conv."].mean()
        styler = styler.map(lambda v: color_pct(v, avg_conv), subset=["% Conv."])
    if "% Canje top seg." in df.columns:
        avg_top = df["% Canje top seg."].mean()
        styler = styler.map(lambda v: color_pct(v, avg_top), subset=["% Canje top seg."])

    return styler


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
def build_sidebar(df_r):
    st.sidebar.title("Filtros")

    col_fecha = detect(df_r, ["FECHA_ENVIO"])
    if col_fecha and not df_r.empty:
        df_r[col_fecha] = pd.to_datetime(df_r[col_fecha], errors="coerce")
        min_d = df_r[col_fecha].min().date()
        max_d = df_r[col_fecha].max().date()
    else:
        min_d = pd.Timestamp("2026-01-01").date()
        max_d = pd.Timestamp.today().date()

    rango = st.sidebar.date_input("Período", value=(min_d, max_d),
                                  min_value=min_d, max_value=max_d)
    f_ini, f_fin = (rango[0], rango[1]) if len(rango) == 2 else (min_d, max_d)

    # Fix 6: Filtros tipo cliente y categoría/tier
    st.sidebar.markdown("---")
    tipo_cli_opts = st.sidebar.multiselect(
        "Tipo de cliente", ["BANCO", "OMP"],
        default=["BANCO", "OMP"], key="sb_tipo_cli",
    )
    tier_opts = st.sidebar.multiselect(
        "Categoría / Tier", ["ELITE", "PREMIUM", "NORMAL", "FAN"],
        default=["ELITE", "PREMIUM", "NORMAL", "FAN"], key="sb_tier",
    )

    # Fix 9: Valor por punto en sidebar
    st.sidebar.markdown("---")
    valor_punto = st.sidebar.number_input(
        "Valor por punto ($)", min_value=0.0001, max_value=0.10,
        value=0.01, format="%.4f", key="sidebar_vpp",
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("Recargar datos"):
        st.cache_data.clear()
        st.rerun()

    return f_ini, f_fin, tipo_cli_opts, tier_opts, valor_punto


# ── SECCIÓN 1: KPIs + Tabla ───────────────────────────────────────────────────
def seccion_resumen(df_camp):
    """Renderiza KPIs + tabla clickable. Retorna el nombre de campaña seleccionada (o None)."""
    st.markdown('<p class="section-title">Resumen General de Campañas</p>', unsafe_allow_html=True)

    n_camps   = len(df_camp)
    enviados  = int(df_camp["Enviados"].sum())    if "Enviados"    in df_camp.columns else 0
    canjead   = int(df_camp["Canjeadores"].sum())  if "Canjeadores" in df_camp.columns else 0
    nuevos    = int(df_camp["Nuevos"].sum())        if "Nuevos"      in df_camp.columns else 0
    avg_canje = df_camp["% Canje GM"].mean()        if "% Canje GM"  in df_camp.columns else 0.0
    avg_lift  = df_camp["Lift (pp)"].mean()          if "Lift (pp)"   in df_camp.columns else 0.0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Campañas", f"{n_camps}")
    c2.metric("Emails enviados", f"{enviados:,}")
    c3.metric("Canjeadores GM", f"{canjead:,}")
    c4.metric("Nuevos canjeadores", f"{nuevos:,}")
    c5.metric("% Canje promedio GM", f"{avg_canje:.1f}%")
    lift_label = f"{avg_lift:+.1f}pp" if avg_lift != 0 else "—"
    c6.metric("Lift promedio vs GC", lift_label,
              delta_color="normal" if avg_lift >= 0 else "inverse")

    st.markdown("---")
    st.markdown('<p class="section-title">Detalle por Campaña</p>', unsafe_allow_html=True)
    st.caption("Haz clic en una fila para ver el detalle. Verde = sobre promedio | Amarillo = promedio | Rojo = bajo promedio")

    event = st.dataframe(
        style_table(df_camp),
        use_container_width=True,
        hide_index=True,
        height=min(420, 60 + 38 * len(df_camp)),
        selection_mode="single-row",
        on_select="rerun",
        key="camp_table",
    )

    selected_rows = event.selection.rows if hasattr(event, "selection") else []
    sel_camp = df_camp.iloc[selected_rows[0]]["Campaña"] if selected_rows else None

    # ── Expander: pivot % Canje GM por campaña × segmento RFM ────────────────
    df_rc_res = load_rfm_campaigns()
    if not df_rc_res.empty:
        with st.expander("Breakdown RFM por campaña — % Canje GM por segmento"):
            RFM_ORDER_P = ["Champions", "Loyal", "Nuevos", "En Riesgo", "Perdidos", "Otros"]
            gm_rc = df_rc_res[df_rc_res["GMGC"] == "GM"]

            pivot_rows = []
            for camp_n in df_camp["Campaña"]:
                sub = gm_rc[gm_rc["NOMBRE_CAMPANHA"] == camp_n]
                row = {"Campaña": camp_n}
                for seg in RFM_ORDER_P:
                    s = sub[sub["rfm_segment"] == seg]
                    tot = int(s["CLIENTES"].sum())
                    can = int(s.loc[s["FLAG_CANJE"] == "CANJEO", "CLIENTES"].sum())
                    row[seg] = spct(can, tot) if tot > 0 else None
                pivot_rows.append(row)

            pivot_df = pd.DataFrame(pivot_rows).set_index("Campaña")
            seg_cols = [c for c in RFM_ORDER_P if c in pivot_df.columns]

            fmt_pivot = {c: "{:.1f}%" for c in seg_cols}

            def color_green(v):
                try:
                    val = float(str(v).replace("%", ""))
                    if val >= 10: return "background-color:#1B5E20;color:white;font-weight:bold"
                    if val >= 6:  return "background-color:#388E3C;color:white"
                    if val >= 3:  return "background-color:#81C784"
                    if val > 0:   return "background-color:#C8E6C9"
                except Exception: pass
                return ""

            styled_pivot = (
                pivot_df[seg_cols]
                .style
                .format(fmt_pivot, na_rep="—")
                .map(color_green)
            )
            st.dataframe(styled_pivot, use_container_width=True)
            st.caption("% Canje GM por segmento RFM. Verde más intenso = mayor conversión.")

    return sel_camp


# ── SECCIÓN 2: Detalle de campaña ─────────────────────────────────────────────
def seccion_detalle(camp_name, df_r, df_omm, df_hist, valor_punto: float = 0.01):
    st.markdown("---")
    st.markdown(f'<p class="camp-label">Detalle: {camp_name}</p>', unsafe_allow_html=True)

    col_camp  = detect(df_r, ["NOMBRE_CAMPANHA"])
    col_gm    = detect(df_r, ["GMGC"])
    col_flag  = detect(df_r, ["FLAG_CANJE"])
    col_pre   = detect(df_r, ["GRUPO_CANJE_PRE","TIPO_CANJEADOR_PRE","GRUPO_CANJE"])
    col_post  = detect(df_r, ["GRUPO_CANJE_POST"])
    col_cli   = detect(df_r, ["CLIENTES"])
    col_tipo  = detect(df_r, ["TIPO_CLIENTE"])

    df_sel = df_r[df_r[col_camp] == camp_name].copy() if col_camp else df_r.copy()
    gm = df_sel[df_sel[col_gm] == "GM"] if col_gm else df_sel

    # ── Bloque de conversión ──────────────────────────────────────────────────
    st.markdown('<p class="section-title">Conversión generada por la campaña</p>', unsafe_allow_html=True)

    col_flag_det = detect(df_sel, ["FLAG_CANJE"])
    gc = df_sel[df_sel[col_gm] == "GC"] if col_gm else pd.DataFrame()

    if col_pre and col_cli and col_flag_det and not gm.empty:
        canjeo_gm = gm[col_flag_det] == "CANJEO"
        total_gm    = int(gm[col_cli].sum())
        nuevos      = int(gm.loc[(gm[col_pre] == "NO_CANJEADOR_HISTORICO") & canjeo_gm, col_cli].sum())
        recuperados = int(gm.loc[(gm[col_pre] == "FUGADOS")                & canjeo_gm, col_cli].sum())
        recurrentes = int(gm.loc[(gm[col_pre] == "RECURRENTE")             & canjeo_gm, col_cli].sum())
        activados   = nuevos + recuperados

        # Totales de canjeadores GM y GC para TRI
        can_gm_total = int(gm.loc[canjeo_gm, col_cli].sum())
        total_gc     = int(gc[col_cli].sum()) if col_gm and not gc.empty else 0
        canjeo_gc    = gc[col_flag_det] == "CANJEO" if col_gm and not gc.empty else None
        can_gc_total = int(gc.loc[canjeo_gc, col_cli].sum()) if canjeo_gc is not None else 0

        pct_gm_main  = spct(can_gm_total, total_gm)
        pct_gc_main  = spct(can_gc_total, total_gc)
        tri_incr     = max(0, int((pct_gm_main / 100 - pct_gc_main / 100) * total_gm))

        # Puntos TRI (Fix 10)
        col_puntos_k = detect(df_sel, ["PUNTOS_CANJEADOS","PUNTOS"])
        avg_puntos_k = 0.0
        if col_puntos_k:
            df_canj_k = gm.loc[canjeo_gm]
            total_pts_k = df_canj_k[col_puntos_k].fillna(0).sum()
            avg_puntos_k = total_pts_k / max(can_gm_total, 1)
        puntos_tri = int(tri_incr * avg_puntos_k)

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Total GM", f"{total_gm:,}")
        mc2.metric("Nuevos canjeadores", f"{nuevos:,}",
                   help="PRE=Sin historial de canje → canjearon por 1ª vez")
        mc3.metric("Recuperados", f"{recuperados:,}",
                   help="PRE=Fugados → volvieron a canjear")
        mc4.metric("Total activados", f"{activados:,}",
                   delta=f"{spct(activados, total_gm):.1f}% del GM",
                   delta_color="normal")
        mc5.metric("Recurrentes canjearon", f"{recurrentes:,}",
                   help="PRE=Recurrente → se mantuvieron activos")

        # Segunda fila de KPIs: nominales + TRI
        mk1, mk2, mk3, mk4, mk5 = st.columns(5)
        mk1.metric("Canjeadores GM", f"{can_gm_total:,}",
                   delta=f"{pct_gm_main:.1f}% de GM")
        mk2.metric("Canjeadores GC", f"{can_gc_total:,}",
                   delta=f"{pct_gc_main:.1f}% de GC")
        mk3.metric("TRI (incrementales)", f"{tri_incr:,}",
                   help="Canjeadores atribuibles a la campaña = (% GM − % GC) × total GM",
                   delta=f"{spct(tri_incr, total_gm):.1f}% del GM")
        if puntos_tri > 0:
            mk4.metric("Puntos TRI", f"{puntos_tri:,}",
                       help=f"{tri_incr:,} increm. × {avg_puntos_k:,.0f} pts/canje")
        else:
            mk4.metric("Puntos TRI", "—", help="Sin dato de puntos canjeados")
        mk5.metric("Lift GM vs GC", f"{max(0.0, round(pct_gm_main - pct_gc_main, 2)):+.1f}pp")

        # ── Gráfico GM vs GC por segmento PRE ────────────────────────────────
        pre_rows = []
        for estado in FUNNEL_ORDER:
            label = FUNNEL_LABELS.get(estado, estado)
            color = FUNNEL_COLORS.get(estado, "#BDBDBD")

            sub_gm = gm[gm[col_pre] == estado]
            tot_gm = int(sub_gm[col_cli].sum()) if not sub_gm.empty else 0
            can_gm = int(sub_gm.loc[sub_gm[col_flag_det] == "CANJEO", col_cli].sum()) if not sub_gm.empty else 0
            pct_gm_s = spct(can_gm, tot_gm)

            sub_gc = gc[gc[col_pre] == estado] if not gc.empty else pd.DataFrame()
            tot_gc = int(sub_gc[col_cli].sum()) if not sub_gc.empty else 0
            can_gc = int(sub_gc.loc[sub_gc[col_flag_det] == "CANJEO", col_cli].sum()) if not sub_gc.empty else 0
            pct_gc_s = spct(can_gc, tot_gc)

            lift_s = max(0.0, round(pct_gm_s - pct_gc_s, 2))
            pre_rows.append({
                "estado": estado, "label": label, "color": color,
                "GM Clientes": tot_gm, "GM Canjean": can_gm, "GM %": pct_gm_s,
                "GC Clientes": tot_gc, "GC Canjean": can_gc, "GC %": pct_gc_s,
                "Lift (pp)": lift_s,
            })

        df_pre = pd.DataFrame(pre_rows)
        df_pre_plot = df_pre[df_pre["GM Clientes"] > 0]

        if not df_pre_plot.empty:
            # ── Toggle nominal / % ────────────────────────────────────────────
            vista = st.radio(
                "Vista del gráfico",
                ["% Canje", "N Canjeadores"],
                horizontal=True,
                key=f"vista_gmgc_{camp_name}",
            )
            usar_pct = vista == "% Canje"

            fig_cmp = go.Figure()
            if usar_pct:
                y_gm   = df_pre_plot["GM %"]
                y_gc   = df_pre_plot["GC %"]
                txt_gm = df_pre_plot["GM %"].apply(lambda v: f"{v:.1f}%")
                txt_gc = df_pre_plot["GC %"].apply(lambda v: f"{v:.1f}%")
                y_lbl  = "% canje"
            else:
                y_gm   = df_pre_plot["GM Canjean"]
                y_gc   = df_pre_plot["GC Canjean"]
                txt_gm = df_pre_plot["GM Canjean"].apply(lambda v: f"{int(v):,}")
                txt_gc = df_pre_plot["GC Canjean"].apply(lambda v: f"{int(v):,}")
                y_lbl  = "Canjeadores"

            fig_cmp.add_trace(go.Bar(
                x=df_pre_plot["label"], y=y_gm,
                name="GM (recibió campaña)", marker_color="#1565C0",
                text=txt_gm, textposition="outside",
            ))
            fig_cmp.add_trace(go.Bar(
                x=df_pre_plot["label"], y=y_gc,
                name="GC (control)", marker_color="#9E9E9E",
                text=txt_gc, textposition="outside",
            ))

            # Anotaciones de Lift (solo en vista %)
            if usar_pct:
                for _, r in df_pre_plot.iterrows():
                    color_lift = "#2E7D32" if r["Lift (pp)"] > 0 else "#777"  # lift ≥ 0 guaranteed
                    fig_cmp.add_annotation(
                        x=r["label"],
                        y=max(r["GM %"], r["GC %"]) * 1.18 + 0.3,
                        text=f"Lift: {r['Lift (pp)']:+.1f}pp",
                        showarrow=False,
                        font=dict(size=11, color=color_lift),
                    )

            fig_cmp.update_layout(
                barmode="group",
                yaxis_title=y_lbl,
                height=300,
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

            # ── Tabla con ambas vistas siempre ───────────────────────────────
            def fmt_lift(v):
                if pd.isna(v): return "—"
                return f"{v:+.1f}pp"
            def color_lift_cell(v):
                try:
                    val = float(str(v).replace("pp",""))
                    if val > 0: return "background-color:#C8E6C9;color:#1B5E20;font-weight:bold"
                    # val ≤ 0 → neutro (lift flooreado en 0)
                except Exception: pass
                return ""

            tbl_cols = ["label","GM Clientes","GM Canjean","GM %","GC Clientes","GC Canjean","GC %","Lift (pp)"]
            df_tbl = df_pre[tbl_cols].copy()
            df_tbl.rename(columns={"label": "Estado PRE"}, inplace=True)
            styler = df_tbl.style.format({
                "GM Clientes": lambda x: f"{int(x):,}",
                "GM Canjean":  lambda x: f"{int(x):,}",
                "GM %":        lambda x: f"{x:.1f}%",
                "GC Clientes": lambda x: f"{int(x):,}",
                "GC Canjean":  lambda x: f"{int(x):,}",
                "GC %":        lambda x: f"{x:.1f}%",
                "Lift (pp)":   fmt_lift,
            }).map(color_lift_cell, subset=["Lift (pp)"])
            st.dataframe(styler, hide_index=True, use_container_width=True)
    else:
        st.info("Sin datos de funnel PRE para esta campaña.")

    st.markdown("---")

    # ── Bloque RFM: conversión por segmento RFM ───────────────────────────────
    st.markdown('<p class="section-title">Conversión por segmento RFM</p>', unsafe_allow_html=True)

    df_rc_all = load_rfm_campaigns()
    if df_rc_all.empty:
        st.info("Sin datos RFM × Campañas. Ejecuta `python extract.py --fecha-inicio 2025-01-01`.")
    else:
        RFM_COLORS_DET = {
            "Champions": "#1565C0", "Loyal": "#1E88E5", "En Riesgo": "#FB8C00",
            "Nuevos": "#43A047",    "Perdidos": "#E53935", "Otros": "#9E9E9E",
            "Sin score RFM": "#BDBDBD",
        }
        RFM_ORDER_DET = ["Champions", "Loyal", "Nuevos", "En Riesgo", "Perdidos", "Otros", "Sin score RFM"]

        df_rc_camp = df_rc_all[df_rc_all["NOMBRE_CAMPANHA"] == camp_name].copy()

        if df_rc_camp.empty:
            st.info("Esta campaña no tiene datos en rfm_campaigns. Es posible que la fecha de extract no la cubra.")
        else:
            # Agregar por rfm_segment × GMGC × FLAG_CANJE
            agg_rfm = (
                df_rc_camp
                .groupby(["rfm_segment", "GMGC", "FLAG_CANJE"])["CLIENTES"]
                .sum()
                .reset_index()
            )

            # ── Predicciones del modelo 5d por segmento RFM ──────────────────────
            seg_pred_pct = {}
            df_ind_camp  = load_rfm_campaign_individual(camp_name)
            _camp_bundle = load_campaign_model()
            _rfm_feat    = load_rfm_features_pred()
            if _camp_bundle and not _rfm_feat.empty and not df_ind_camp.empty:
                try:
                    _col_f = detect(df_r, ["FECHA_ENVIO"])
                    _rows  = df_r[df_r[col_camp] == camp_name] if col_camp else df_r
                    _fecha = pd.to_datetime(_rows[_col_f], errors="coerce").max() if _col_f and not _rows.empty else pd.Timestamp.today()
                    _dp    = predict_5d(_camp_bundle, df_ind_camp, _rfm_feat, _fecha)
                    seg_pred_pct = (
                        _dp[_dp["GMGC"] == "GM"]
                        .groupby("rfm_segment", observed=True)["p_5dias"]
                        .mean().mul(100).round(1).to_dict()
                    )
                except Exception:
                    seg_pred_pct = {}

            rfm_rows = []
            for seg in RFM_ORDER_DET:
                sub_gm = agg_rfm[(agg_rfm["rfm_segment"] == seg) & (agg_rfm["GMGC"] == "GM")]
                sub_gc = agg_rfm[(agg_rfm["rfm_segment"] == seg) & (agg_rfm["GMGC"] == "GC")]

                tot_gm = int(sub_gm["CLIENTES"].sum())
                can_gm = int(sub_gm.loc[sub_gm["FLAG_CANJE"] == "CANJEO", "CLIENTES"].sum())
                pct_gm = spct(can_gm, tot_gm)

                tot_gc = int(sub_gc["CLIENTES"].sum())
                can_gc = int(sub_gc.loc[sub_gc["FLAG_CANJE"] == "CANJEO", "CLIENTES"].sum())
                pct_gc = spct(can_gc, tot_gc)

                if tot_gm == 0 and tot_gc == 0:
                    continue

                rfm_rows.append({
                    "seg": seg, "tot_gm": tot_gm, "can_gm": can_gm, "pct_gm": pct_gm,
                    "tot_gc": tot_gc, "can_gc": can_gc, "pct_gc": pct_gc,
                    "lift": max(0.0, round(pct_gm - pct_gc, 1)),
                    "pred_pct": seg_pred_pct.get(seg, None),
                })

            if rfm_rows:
                rfm_df = pd.DataFrame(rfm_rows)

                # Toggle nominal / %
                vista_rfm = st.radio(
                    "Vista", ["% Canje", "N Canjeadores"], horizontal=True,
                    key=f"rfm_det_vista_{camp_name}"
                )
                usar_pct_rfm = vista_rfm == "% Canje"

                fig_rfm = go.Figure()
                fig_rfm.add_trace(go.Bar(
                    name="GM",
                    x=rfm_df["seg"].tolist(),
                    y=rfm_df["pct_gm"].tolist() if usar_pct_rfm else rfm_df["can_gm"].tolist(),
                    marker_color=[RFM_COLORS_DET.get(s, "#9E9E9E") for s in rfm_df["seg"]],
                    text=[f"{v:.1f}%" if usar_pct_rfm else f"{v:,}" for v in
                          (rfm_df["pct_gm"] if usar_pct_rfm else rfm_df["can_gm"])],
                    textposition="auto",
                    hovertemplate="%{x}<br>GM: %{text}<extra></extra>",
                ))
                fig_rfm.add_trace(go.Bar(
                    name="GC",
                    x=rfm_df["seg"].tolist(),
                    y=rfm_df["pct_gc"].tolist() if usar_pct_rfm else rfm_df["can_gc"].tolist(),
                    marker_color=["#BDBDBD"] * len(rfm_df),
                    text=[f"{v:.1f}%" if usar_pct_rfm else f"{v:,}" for v in
                          (rfm_df["pct_gc"] if usar_pct_rfm else rfm_df["can_gc"])],
                    textposition="auto",
                    hovertemplate="%{x}<br>GC: %{text}<extra></extra>",
                ))

                # Anotaciones de Lift solo en modo %
                if usar_pct_rfm:
                    for _, row in rfm_df.iterrows():
                        lift_txt = f"+{row['lift']:.1f}pp" if row["lift"] >= 0 else f"{row['lift']:.1f}pp"
                        lift_col = "green" if row["lift"] >= 0 else "red"
                        fig_rfm.add_annotation(
                            x=row["seg"], y=max(row["pct_gm"], row["pct_gc"]) + 1.5,
                            text=f"<b>{lift_txt}</b>", showarrow=False,
                            font=dict(color=lift_col, size=11),
                        )

                fig_rfm.update_layout(
                    barmode="group", height=300,
                    margin=dict(l=0, r=0, t=40, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis_title="Segmento RFM",
                    yaxis_title="% canje" if usar_pct_rfm else "Canjeadores",
                    title="Conversión GM vs GC por segmento RFM",
                )
                st.plotly_chart(fig_rfm, use_container_width=True)

                # Tabla detalle
                def color_lift_rfm(val):
                    try:
                        v = float(val)
                        if v > 0: return "color: #2E7D32; font-weight: bold"
                        if v < 0: return "color: #C62828; font-weight: bold"
                    except Exception:
                        pass
                    return ""

                has_pred = rfm_df["pred_pct"].notna().any()
                base_cols = ["Segmento RFM", "GM Clientes", "GM Canjean", "GM %",
                             "GC Clientes", "GC Canjean", "GC %", "Lift (pp)"]
                rename_map = {
                    "seg": "Segmento RFM", "tot_gm": "GM Clientes", "can_gm": "GM Canjean",
                    "pct_gm": "GM %", "tot_gc": "GC Clientes", "can_gc": "GC Canjean",
                    "pct_gc": "GC %", "lift": "Lift (pp)", "pred_pct": "Pred % (5d)",
                }
                tbl_cols = base_cols + (["Pred % (5d)"] if has_pred else [])
                tbl = rfm_df.rename(columns=rename_map)[tbl_cols]

                fmt_rfm = {
                    "GM Clientes": "{:,}", "GM Canjean": "{:,}",
                    "GC Clientes": "{:,}", "GC Canjean": "{:,}",
                    "GM %": "{:.1f}%", "GC %": "{:.1f}%", "Lift (pp)": "{:.1f}",
                }
                if has_pred:
                    fmt_rfm["Pred % (5d)"] = lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
                st.dataframe(
                    tbl.style.format(fmt_rfm, na_rep="—").map(color_lift_rfm, subset=["Lift (pp)"]),
                    hide_index=True, use_container_width=True,
                )
                if has_pred:
                    st.caption("**Pred % (5d)**: probabilidad media predicha por el modelo de canje en 5 días.")

                # Fix 7: Vista combinada engagement × conversión × RFM
                with st.expander("Vista combinada: Engagement × Conversión × RFM"):
                    # Obtener email metrics para esta campaña (misma fila para todos los segmentos)
                    open_rate_pct, click_rate_pct = None, None
                    if not df_omm.empty:
                        omm_camp_col7 = detect(df_omm, ["CAMPAIGN_NAME","NOMBRE_CAMPANHA"])
                        if omm_camp_col7:
                            match7 = df_omm[df_omm[omm_camp_col7].apply(norm_camp) == norm_camp(camp_name)]
                            if not match7.empty:
                                row7   = match7.agg({c: "sum" for c in match7.select_dtypes("number").columns})
                                mails7 = float(row7.get("TOTAL_MAILS", 0))
                                opens7 = float(row7.get("TOTAL_OPENS", 0))
                                clicks7= float(row7.get("TOTAL_CLICKS", 0))
                                if mails7 > 0:
                                    open_rate_pct  = round(opens7 / mails7 * 100, 1)
                                    click_rate_pct = round(clicks7 / mails7 * 100, 1)

                    combined_rows = []
                    for rr in rfm_rows:
                        combined_rows.append({
                            "Segmento RFM":  rr["seg"],
                            "Open rate":     f"{open_rate_pct:.1f}%" if open_rate_pct is not None else "—",
                            "Click rate":    f"{click_rate_pct:.1f}%" if click_rate_pct is not None else "—",
                            "% Canje GM":    f"{rr['pct_gm']:.1f}%",
                            "% Canje GC":    f"{rr['pct_gc']:.1f}%",
                            "Lift (pp)":     rr["lift"],
                            "Pred % (5d)":   f"{rr['pred_pct']:.1f}%" if rr["pred_pct"] is not None else "—",
                        })
                    if combined_rows:
                        def color_lift_comb(v):
                            try:
                                val = float(v)
                                if val > 0: return "color: #2E7D32; font-weight: bold"
                            except Exception:
                                pass
                            return ""
                        df_comb = pd.DataFrame(combined_rows)
                        st.dataframe(
                            df_comb.style.map(color_lift_comb, subset=["Lift (pp)"]),
                            hide_index=True, use_container_width=True,
                        )
                        st.caption("Open/Click rate a nivel de campaña (igual para todos los segmentos).")

    st.markdown("---")

    # ── ¿Quién canjeó? (estado PRE campaña) — ancho completo ─────────────────
    # [Email (Salesforce) movido a pestaña Planificación > Historial de envíos]
    st.markdown("**¿Quién canjeó? (estado PRE campaña)**")
    if col_pre and col_flag and not df_sel.empty:
        from plotly.subplots import make_subplots

        flags = df_sel[col_flag].astype(str).str.upper()
        df_sel["_canjeo"] = flags.isin(["CANJEO","1","TRUE","SI","S"])

        # Construir resumen por estado PRE — respetando FUNNEL_ORDER
        pre_rows = []
        for estado in FUNNEL_ORDER:
            sub = df_sel[df_sel[col_pre] == estado]
            if sub.empty:
                continue
            total = int(sub[col_cli].sum()) if col_cli else len(sub)
            canjeadores = int(sub.loc[sub["_canjeo"], col_cli].sum()) if col_cli else int(sub["_canjeo"].sum())
            pre_rows.append({
                "estado":      estado,
                "label":       FUNNEL_LABELS.get(estado, estado),
                "color":       FUNNEL_COLORS.get(estado, "#BDBDBD"),
                "total":       total,
                "canjeadores": canjeadores,
                "pct":         spct(canjeadores, total),
            })

        if pre_rows:
            # Fix 4: orden por FUNNEL_ORDER (ya preservado por iteración arriba)
            df_pre = pd.DataFrame(pre_rows)
            # Para el gráfico, invertir para que el funnel quede de abajo→arriba
            df_pre_plot = df_pre.iloc[::-1].reset_index(drop=True)

            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=["Tasa de conversión por segmento", "Tamaño del segmento (GM)"],
                vertical_spacing=0.18,
            )

            # Panel superior: % canje por segmento
            fig.add_trace(go.Bar(
                y=df_pre_plot["label"],
                x=df_pre_plot["pct"],
                orientation="h",
                marker_color=df_pre_plot["color"].tolist(),
                text=[f"{p:.1f}%  ({c:,})" for p, c in zip(df_pre_plot["pct"], df_pre_plot["canjeadores"])],
                textposition="outside",
                cliponaxis=False,
                showlegend=False,
            ), row=1, col=1)

            # Panel inferior: total de clientes por segmento
            fig.add_trace(go.Bar(
                y=df_pre_plot["label"],
                x=df_pre_plot["total"],
                orientation="h",
                marker_color=df_pre_plot["color"].tolist(),
                opacity=0.6,
                text=df_pre_plot["total"].apply(lambda v: f"{v:,}"),
                textposition="outside",
                cliponaxis=False,
                showlegend=False,
            ), row=2, col=1)

            fig.update_xaxes(title_text="% que canjeó", row=1, col=1)
            fig.update_xaxes(title_text="Total clientes", row=2, col=1)
            fig.update_layout(
                height=520,
                margin=dict(l=10, r=90, t=35, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Toggle para mostrar tabla con todos los estados
            show_all_states = st.checkbox("Mostrar tabla detallada por estado funnel", key="show_all_states")
            if show_all_states:
                # Fix 3: tabla con todos los estados (incluye 0-clientes si se quiere)
                tbl_pre_rows = []
                for estado in FUNNEL_ORDER:
                    sub = df_sel[df_sel[col_pre] == estado]
                    total = int(sub[col_cli].sum()) if col_cli else len(sub)
                    canjeadores = int(sub.loc[sub["_canjeo"], col_cli].sum()) if col_cli else int(sub["_canjeo"].sum())
                    tbl_pre_rows.append({
                        "Estado":       FUNNEL_LABELS.get(estado, estado),
                        "GM Clientes":  total,
                        "Canjeadores":  canjeadores,
                        "% Canje":      f"{spct(canjeadores, total):.1f}%",
                    })
                st.dataframe(pd.DataFrame(tbl_pre_rows), hide_index=True, use_container_width=True)
    else:
        st.info("Sin datos de funnel PRE para esta campaña.")

    # ── Fila B: GM vs GC ─────────────────────────────────────────────────────
    st.markdown("**Efecto incremental — GM (recibió campaña) vs GC (control)**")

    if col_gm and col_flag and not df_sel.empty:
        def rate(g):
            if g.empty: return 0.0
            flags = g[col_flag].astype(str).str.upper()
            mask  = flags.isin(["CANJEO","1","TRUE","SI","S"])
            cnt   = int(g[col_cli].sum()) if col_cli else len(g)
            canjeadores = int(g.loc[mask, col_cli].sum()) if col_cli else int(mask.sum())
            return spct(canjeadores, cnt)

        pct_gm = rate(df_sel[df_sel[col_gm] == "GM"])
        pct_gc = rate(df_sel[df_sel[col_gm] == "GC"])
        lift   = max(0.0, round(pct_gm - pct_gc, 2))

        lift_txt = (f"La campaña generó **{lift:+.1f}pp** de canje adicional vs grupo control"
                    if lift > 0 else "Sin diferencia incremental entre GM y GC")
        st.markdown(lift_txt)

        fig2 = go.Figure(data=[
            go.Bar(x=["GM (recibió campaña)", "GC (control)"],
                   y=[pct_gm, pct_gc],
                   marker_color=["#1565C0", "#BDBDBD"],
                   text=[f"{pct_gm:.1f}%", f"{pct_gc:.1f}%"],
                   textposition="outside",
                   width=0.4),
        ])
        fig2.add_annotation(
            x=0.5, y=max(pct_gm, pct_gc) * 1.15,
            text=f"Lift: {lift:+.1f}pp",
            showarrow=False,
            font=dict(size=16, color="#1565C0" if lift > 0 else "#B71C1C"),
            xref="paper",
        )
        fig2.update_layout(
            yaxis_title="% Canje",
            yaxis_range=[0, max(pct_gm, pct_gc) * 1.4 + 1],
            height=280,
            showlegend=False,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Sin datos de grupo GM/GC para esta campaña.")

    # ── Expander: Base enviada ────────────────────────────────────────────────
    with st.expander("Ver base enviada (historical_campaigns)"):
        if not df_hist.empty:
            hist_camp = detect(df_hist, ["NOMBRE_CAMPANHA","CAMPANHA"])
            hist_fecha= detect(df_hist, ["FECHA_ENVIO"])

            df_h = df_hist.copy()
            if hist_camp:
                df_h = df_h[df_h[hist_camp] == camp_name]

            if not df_h.empty:
                grp_cols = [c for c in ["TIPO_CLIENTE","GMGC","CANAL","TIPO_MEDICION"] if c in df_h.columns]
                if grp_cols:
                    resumen_h = df_h.groupby(grp_cols).size().reset_index(name="RUTs")
                    st.dataframe(resumen_h, hide_index=True, use_container_width=True)

                # Fix 5: canal de canje real desde redemption_metrics
                if not df_r.empty:
                    col_canal_r = detect(df_r, ["CANAL_CANJE"])
                    col_flag_r  = detect(df_r, ["FLAG_CANJE"])
                    col_camp_r  = detect(df_r, ["NOMBRE_CAMPANHA","CAMPANHA"])
                    col_cli_r   = detect(df_r, ["CLIENTES","N_CLIENTES","TOTAL"])
                    if col_canal_r and col_flag_r and col_camp_r:
                        canal_df = df_r[
                            (df_r[col_camp_r] == camp_name) &
                            (df_r[col_flag_r].astype(str).str.upper() == "CANJEO")
                        ]
                        if not canal_df.empty:
                            st.markdown("**Canjes por canal (real):**")
                            if col_cli_r:
                                grp_canal = canal_df.groupby(col_canal_r)[col_cli_r].sum().reset_index()
                                grp_canal.columns = ["Canal", "Canjeadores"]
                            else:
                                grp_canal = canal_df.groupby(col_canal_r).size().reset_index(name="Canjeadores")
                            st.dataframe(grp_canal, hide_index=True, use_container_width=True)

                csv = df_h.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Descargar CSV",
                    data=csv,
                    file_name=f"base_{camp_name.replace(' ','_')}.csv",
                    mime="text/csv",
                )
            else:
                st.info("No hay datos de historical_campaigns para esta campaña.")
        else:
            st.info("No se cargaron datos de historical_campaigns.")


# ── PIVOT TABLE ──────────────────────────────────────────────────────────────
def build_pivot_table(df_sel, valor_punto: float = 0.01):
    """
    Construye la tabla pivote con MultiIndex:
        Columnas: TIPO_CLIENTE × TIPO × GMGC
        Filas:    métricas (Clientes totales, Canjeadores, No canj. hist., Nuevos,
                            % Canje, Puntos canjeados, Valor canjeado ($))
    """
    df = df_sel[df_sel["GMGC"].isin(["GM", "GC"])].copy()

    TIPO_CLI_ORDER = ["BANCO", "OMP"]
    TIPO_ORDER     = ["CANJEADOR HISTORICO", "NO CANJEADOR HISTORICO"]
    GMGC_ORDER     = ["GM", "GC"]
    METRICS        = ["Clientes totales", "Canjeadores", "No canj. hist.", "Nuevos",
                      "% Canje", "Puntos canjeados", "Valor canjeado ($)"]

    col_tuples, cell_data = [], {}
    for tc in TIPO_CLI_ORDER:
        for tp in TIPO_ORDER:
            for gm in GMGC_ORDER:
                key = (tc, tp, gm)
                col_tuples.append(key)
                grp = df[(df["TIPO_CLIENTE"] == tc) &
                         (df["TIPO"]         == tp) &
                         (df["GMGC"]         == gm)]
                if grp.empty:
                    cell_data[key] = dict.fromkeys(METRICS, 0)
                    cell_data[key]["% Canje"] = "0.0%"
                    cell_data[key]["Valor canjeado ($)"] = "$0"
                else:
                    total    = int(grp["CLIENTES"].sum())
                    canj_mask = grp["FLAG_CANJE"] == "CANJEO"
                    canjeadores = int(grp.loc[canj_mask, "CLIENTES"].sum())
                    no_hist  = int(grp.loc[grp["TIPO"] == "NO CANJEADOR HISTORICO", "CLIENTES"].sum())
                    nuevos   = int(grp.loc[grp["GRUPO_CANJE_POST"] == "NUEVO", "CLIENTES"].sum())
                    pct      = spct(canjeadores, total)
                    puntos   = int(grp["PUNTOS_CANJEADOS"].fillna(0).sum()) if "PUNTOS_CANJEADOS" in grp.columns else 0
                    valor    = puntos * valor_punto
                    cell_data[key] = {
                        "Clientes totales": total,
                        "Canjeadores":      canjeadores,
                        "No canj. hist.":   no_hist,
                        "Nuevos":           nuevos,
                        "% Canje":          f"{pct:.1f}%",
                        "Puntos canjeados": puntos,
                        "Valor canjeado ($)": f"${valor:,.0f}",
                    }

    # Etiquetas legibles para el MultiIndex
    TIPO_LABELS = {
        "CANJEADOR HISTORICO":    "Canj. histórico",
        "NO CANJEADOR HISTORICO": "No canj. histórico",
    }
    mi = pd.MultiIndex.from_tuples(
        [(tc, TIPO_LABELS.get(tp, tp), gm) for (tc, tp, gm) in col_tuples],
        names=["Tipo cliente", "Historial canje", "Grupo"],
    )
    rows = {m: [cell_data[k][m] for k in col_tuples] for m in METRICS}
    df_pivot = pd.DataFrame(rows, index=mi).T
    df_pivot.index.name = "Métrica"
    return df_pivot


def style_pivot(df_pivot):
    """Formato numérico y semáforo en % Canje para la tabla pivote."""
    def fmt(v):
        if isinstance(v, str): return v          # ya formateado (e.g. "5.2%")
        try: return f"{int(v):,}"
        except Exception: return v

    def color_pct(v):
        if not isinstance(v, str) or "%" not in v: return ""
        try:
            val = float(v.replace("%", ""))
            if val >= 10:  return "background-color:#C8E6C9;color:#1B5E20;font-weight:bold"
            if val >= 5:   return "background-color:#FFF9C4;color:#F57F17"
            if val > 0:    return "background-color:#FFCDD2;color:#B71C1C"
        except Exception: pass
        return ""

    return df_pivot.style.format(fmt).map(color_pct)


# ── VISTA GRANULAR ────────────────────────────────────────────────────────────
def seccion_granular(df_r, valor_punto: float = 0.01):
    if df_r.empty:
        st.warning("Sin datos para el período seleccionado.")
        return

    col_camp  = detect(df_r, ["NOMBRE_CAMPANHA"])
    col_fecha = detect(df_r, ["FECHA_ENVIO"])
    col_pre   = detect(df_r, ["GRUPO_CANJE_PRE"])
    col_post  = detect(df_r, ["GRUPO_CANJE_POST"])
    col_cli   = detect(df_r, ["CLIENTES"])
    col_canal = detect(df_r, ["CANAL_CANJE"])

    # ── Selectores fecha + campaña ────────────────────────────────────────────
    st.markdown('<p class="section-title">Seleccionar campaña</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    if col_fecha:
        df_r[col_fecha] = pd.to_datetime(df_r[col_fecha], errors="coerce")
        fechas = sorted(df_r[col_fecha].dt.date.unique(), reverse=True)
        sel_fecha = c1.selectbox("Fecha de envío", fechas, key="gran_fecha")
        df_fecha = df_r[df_r[col_fecha].dt.date == sel_fecha]
    else:
        df_fecha = df_r
        sel_fecha = None

    if col_camp:
        camps_disp = sorted(df_fecha[col_camp].dropna().unique().tolist())
        sel_camp = c2.selectbox("Campaña", camps_disp, key="gran_camp")
        df_sel = df_fecha[df_fecha[col_camp] == sel_camp].copy()
    else:
        df_sel = df_fecha.copy()
        sel_camp = "—"

    st.markdown(f"**{sel_camp}** — {sel_fecha}")
    st.markdown("---")

    # ── Tabla pivote principal ────────────────────────────────────────────────
    st.markdown('<p class="section-title">Resultados por segmento</p>', unsafe_allow_html=True)
    st.caption("Banco / OMP  ×  Canjeador histórico / No canjeador histórico  ×  GM (recibió) / GC (control)")

    df_pivot = build_pivot_table(df_sel, valor_punto=valor_punto)
    # Renderizar via HTML para evitar error Arrow con columnas de tipo mixto (int + "x.x%")
    st.markdown(
        style_pivot(df_pivot).to_html(border=0),
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # [Canjes por canal eliminado — ver detalle en pestaña Planificación > Historial]

    # Fix 11: segunda tabla por estado funnel (GRUPO_CANJE_PRE)
    with st.expander("Ver detalle por estado funnel (PRE campaña)"):
        if col_pre and col_cli and "FLAG_CANJE" in df_sel.columns:
            FUNNEL_TIPO_ORDER = ["RECURRENTE", "RECUPERADO", "NUEVO", "FUGADOS", "NO_CANJEADOR_HISTORICO"]
            funnel_rows = []
            for estado in FUNNEL_TIPO_ORDER:
                for grupo in ["GM", "GC"]:
                    sub = df_sel[(df_sel[col_pre] == estado) & (df_sel["GMGC"] == grupo)]
                    if sub.empty:
                        continue
                    total_f = int(sub[col_cli].sum())
                    canj_f  = int(sub.loc[sub["FLAG_CANJE"] == "CANJEO", col_cli].sum())
                    funnel_rows.append({
                        "Estado PRE":  FUNNEL_LABELS.get(estado, estado),
                        "Grupo":       grupo,
                        "Clientes":    total_f,
                        "Canjeadores": canj_f,
                        "% Canje":     f"{spct(canj_f, total_f):.1f}%",
                    })
            if funnel_rows:
                st.dataframe(pd.DataFrame(funnel_rows), hide_index=True, use_container_width=True)
            else:
                st.info("Sin datos de GRUPO_CANJE_PRE para esta selección.")
        else:
            st.info("Sin datos de estado funnel disponibles.")

    # ── Movimiento PRE → POST (solo GM) ──────────────────────────────────────
    st.markdown('<p class="section-title">Movimiento de funnel PRE → POST (grupo GM)</p>', unsafe_allow_html=True)
    st.caption("Muestra cómo cambió el estado de los clientes después de la campaña")

    if col_pre and col_post and col_cli:
        df_gm = df_sel[df_sel["GMGC"] == "GM"].copy()
        if not df_gm.empty:
            flujo = df_gm.groupby([col_pre, col_post])[col_cli].sum().reset_index()
            flujo.columns = ["PRE", "POST", "Clientes"]

            # Fix 8: radio selector para vista canjeadores activos vs completa
            vista_trans = st.radio(
                "Vista transición",
                ["Solo canjeadores activos", "Vista completa"],
                horizontal=True,
                key=f"vista_trans_{sel_camp}",
            )

            # Pivot con FUNNEL_ORDER
            all_states = FUNNEL_ORDER + [s for s in flujo["PRE"].unique() if s not in FUNNEL_ORDER]
            flujo_pivot = flujo.pivot_table(index="PRE", columns="POST", values="Clientes",
                                            fill_value=0, aggfunc="sum")

            if vista_trans == "Solo canjeadores activos":
                excluir = ["NO_CANJEADOR_HISTORICO"]
                row_ord = [r for r in all_states if r in flujo_pivot.index and r not in excluir]
            else:
                row_ord = [r for r in all_states if r in flujo_pivot.index]

            col_ord = [c for c in all_states if c in flujo_pivot.columns]
            flujo_pivot = flujo_pivot.reindex(index=row_ord, columns=col_ord, fill_value=0)

            # Renombrar con labels legibles
            flujo_pivot.index   = [FUNNEL_LABELS.get(r, r) for r in flujo_pivot.index]
            flujo_pivot.columns = [FUNNEL_LABELS.get(c, c) for c in flujo_pivot.columns]
            flujo_pivot.index.name   = "PRE campaña ↓"
            flujo_pivot.columns.name = "POST campaña →"

            # Estilo: diagonal azul (sin cambio), resto proporcional
            def highlight_diag(df):
                styles = pd.DataFrame("", index=df.index, columns=df.columns)
                for i, row in enumerate(df.index):
                    for j, col in enumerate(df.columns):
                        if row == col:
                            styles.iloc[i, j] = "background-color:#E3F2FD;font-weight:bold"
                return styles

            styled = flujo_pivot.style\
                .apply(highlight_diag, axis=None)\
                .format(lambda x: f"{int(x):,}" if x > 0 else "—")
            st.dataframe(styled, use_container_width=True)
            st.caption("Diagonal = sin cambio de estado. Fuera de diagonal = el cliente cambió su perfil.")
        else:
            st.info("Sin datos de GM para esta campaña.")
    st.markdown("---")

    # ── Descarga ──────────────────────────────────────────────────────────────
    csv = df_sel.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar datos completos (CSV)",
        data=csv,
        file_name=f"granular_{sel_camp}_{sel_fecha}.csv",
        mime="text/csv",
    )


# ── Sección RFM ───────────────────────────────────────────────────────────────
def seccion_rfm(df_rfm, df_r):
    """Pestaña de Segmentación RFM: perfiles de segmento y conexión con campañas."""

    RFM_COLORS = {
        "Champions": "#1565C0",
        "Loyal":     "#1E88E5",
        "En Riesgo": "#FB8C00",
        "Nuevos":    "#43A047",
        "Perdidos":  "#E53935",
        "Otros":     "#9E9E9E",
    }
    RFM_ORDER = ["Champions", "Loyal", "Nuevos", "En Riesgo", "Perdidos", "Otros"]

    if df_rfm.empty:
        st.warning(
            "No se encontró `data/rfm_all_clients.parquet`. "
            "Ejecuta primero:\n```bash\npython extract.py --fecha-inicio 2025-01-01\n```"
        )
        return

    subtab_a, subtab_b, subtab_c = st.tabs(["Perfiles de Segmento", "RFM × Campañas", "Calibración del modelo"])

    # ── Sub-tab A: Perfiles de Segmento ───────────────────────────────────────
    with subtab_a:
        t0_val = df_rfm["t0"].max() if "t0" in df_rfm.columns else "—"
        st.markdown(
            f'<p class="section-title">Segmentos RFM — Base completa (partición: {t0_val})</p>',
            unsafe_allow_html=True,
        )

        total_clientes = len(df_rfm)
        n_segmentos    = df_rfm["rfm_segment"].nunique() if "rfm_segment" in df_rfm.columns else 0
        recencia_med   = df_rfm["recency_days"].median() if "recency_days" in df_rfm.columns else 0

        k1, k2, k3 = st.columns(3)
        k1.metric("Total clientes", f"{total_clientes:,}")
        k2.metric("Segmentos RFM", str(n_segmentos))
        k3.metric("Recencia mediana", f"{int(recencia_med)} días")

        if "rfm_segment" not in df_rfm.columns:
            st.info("La columna `rfm_segment` no está disponible.")
            return

        # Agrupar por segmento
        agg_dict = {"cust_id": "count"}
        for col in ["r_score", "f_score", "m_score", "recency_days",
                    "frequency_total", "monetary_total", "stock_points"]:
            if col in df_rfm.columns:
                agg_dict[col] = "mean"
        if "prioridad" in df_rfm.columns:
            agg_dict["prioridad"] = lambda x: x.value_counts().index[0]

        seg_agg = df_rfm.groupby("rfm_segment").agg(agg_dict).reset_index()
        seg_agg = seg_agg.rename(columns={"cust_id": "Clientes"})

        # Ordenar por RFM_ORDER
        seg_agg["_order"] = seg_agg["rfm_segment"].map(
            {s: i for i, s in enumerate(RFM_ORDER)}
        ).fillna(99)
        seg_agg = seg_agg.sort_values("_order").drop(columns="_order").reset_index(drop=True)

        # Gráfico barras horizontales
        color_list = [RFM_COLORS.get(s, "#9E9E9E") for s in seg_agg["rfm_segment"]]
        pct_list   = [spct(v, total_clientes) for v in seg_agg["Clientes"]]

        fig_seg = go.Figure(go.Bar(
            x=seg_agg["Clientes"],
            y=seg_agg["rfm_segment"],
            orientation="h",
            marker_color=color_list,
            text=[f"{v:,}  ({p:.1f}%)" for v, p in zip(seg_agg["Clientes"], pct_list)],
            textposition="auto",
            hovertemplate="%{y}: %{x:,} clientes<extra></extra>",
        ))
        fig_seg.update_layout(
            height=max(260, 55 * len(seg_agg)),
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis=dict(autorange="reversed"),
            xaxis_title="N° clientes",
            title="Distribución por segmento RFM",
        )
        st.plotly_chart(fig_seg, use_container_width=True)

        # Tabla detalle por segmento
        rename_map = {
            "rfm_segment":     "Segmento",
            "r_score":         "R score",
            "f_score":         "F score",
            "m_score":         "M score",
            "recency_days":    "Recencia (días)",
            "frequency_total": "Frecuencia",
            "monetary_total":  "Gasto total (CLP)",
            "stock_points":    "Puntos en stock",
            "prioridad":       "Prioridad",
        }
        display_cols = ["rfm_segment", "Clientes"] + [
            c for c in ["r_score", "f_score", "m_score", "recency_days",
                        "frequency_total", "monetary_total", "stock_points", "prioridad"]
            if c in seg_agg.columns
        ]
        df_display = seg_agg[display_cols].rename(columns=rename_map)
        df_display.insert(2, "% Base", [f"{p:.1f}%" for p in pct_list])

        fmt = {"Clientes": "{:,}"}
        for c in ["R score", "F score", "M score"]:
            if c in df_display.columns: fmt[c] = "{:.1f}"
        for c in ["Recencia (días)", "Frecuencia"]:
            if c in df_display.columns: fmt[c] = "{:.0f}"
        for c in ["Gasto total (CLP)", "Puntos en stock"]:
            if c in df_display.columns: fmt[c] = "{:,.0f}"

        st.dataframe(df_display.style.format(fmt), use_container_width=True, hide_index=True)

        # Breakdown por tier si está disponible
        if "tier" in df_rfm.columns:
            st.markdown('<p class="section-title">Distribución RFM por tier</p>',
                        unsafe_allow_html=True)
            pivot_tier = (
                df_rfm.groupby(["tier", "rfm_segment"])
                .size()
                .reset_index(name="n")
                .pivot(index="tier", columns="rfm_segment", values="n")
                .fillna(0)
            )
            # Mantener solo columnas presentes en RFM_ORDER
            cols_order = [c for c in RFM_ORDER if c in pivot_tier.columns]
            pivot_tier = pivot_tier[cols_order]

            fig_tier = go.Figure()
            for seg in cols_order:
                fig_tier.add_trace(go.Bar(
                    name=seg,
                    x=pivot_tier.index.tolist(),
                    y=pivot_tier[seg].tolist(),
                    marker_color=RFM_COLORS.get(seg, "#9E9E9E"),
                ))
            fig_tier.update_layout(
                barmode="stack",
                height=320,
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis_title="Tier",
                yaxis_title="N° clientes",
                title="Segmentos RFM por tier de cliente",
            )
            st.plotly_chart(fig_tier, use_container_width=True)

    # ── Sub-tab B: RFM × Campañas ─────────────────────────────────────────────
    with subtab_b:
        st.markdown('<p class="section-title">RFM × Campañas — Conversión real por segmento RFM</p>',
                    unsafe_allow_html=True)

        df_rc = load_rfm_campaigns()

        if df_rc.empty:
            st.warning(
                "No se encontró `data/rfm_campaigns.parquet`. "
                "Ejecuta `python extract.py --fecha-inicio 2025-01-01` para generarlo."
            )
            return

        # Filtro opcional por campaña
        camps_disponibles = sorted(df_rc["NOMBRE_CAMPANHA"].dropna().unique().tolist())
        sel_camps = st.multiselect(
            "Filtrar por campaña (vacío = todas)",
            options=camps_disponibles,
            default=[],
            key="rfm_camp_filter",
        )
        df_rc_f = df_rc[df_rc["NOMBRE_CAMPANHA"].isin(sel_camps)] if sel_camps else df_rc

        # Agregar por rfm_segment × GMGC × FLAG_CANJE
        agg = (
            df_rc_f
            .groupby(["rfm_segment", "GMGC", "FLAG_CANJE"])["CLIENTES"]
            .sum()
            .reset_index()
        )

        def conv_rate_df(tipo):
            sub = agg[agg["GMGC"] == tipo]
            total   = sub.groupby("rfm_segment")["CLIENTES"].sum()
            canjean = sub[sub["FLAG_CANJE"] == "CANJEO"].groupby("rfm_segment")["CLIENTES"].sum()
            df = pd.DataFrame({"clientes": total, "canjean": canjean}).fillna(0)
            df["conv_rate"] = (df["canjean"] / df["clientes"] * 100).round(1)
            return df

        gm_df = conv_rate_df("GM")
        gc_df = conv_rate_df("GC")

        # Ordenar segmentos por conv_rate GM descendente
        segs = [s for s in RFM_ORDER if s in gm_df.index or s in gc_df.index]
        segs_sorted = sorted(segs, key=lambda s: gm_df.loc[s, "conv_rate"] if s in gm_df.index else 0, reverse=True)

        # Gráfico barras agrupadas: GM vs GC por segmento RFM
        fig_rc = go.Figure()
        fig_rc.add_trace(go.Bar(
            name="GM % canje",
            x=segs_sorted,
            y=[gm_df.loc[s, "conv_rate"] if s in gm_df.index else 0 for s in segs_sorted],
            marker_color="#1565C0",
            text=[f"{gm_df.loc[s,'conv_rate']:.1f}%" if s in gm_df.index else "—" for s in segs_sorted],
            textposition="auto",
            hovertemplate="%{x}<br>GM: %{y:.1f}%<extra></extra>",
        ))
        fig_rc.add_trace(go.Bar(
            name="GC % canje",
            x=segs_sorted,
            y=[gc_df.loc[s, "conv_rate"] if s in gc_df.index else 0 for s in segs_sorted],
            marker_color="#9E9E9E",
            text=[f"{gc_df.loc[s,'conv_rate']:.1f}%" if s in gc_df.index else "—" for s in segs_sorted],
            textposition="auto",
            hovertemplate="%{x}<br>GC: %{y:.1f}%<extra></extra>",
        ))
        fig_rc.update_layout(
            barmode="group",
            height=340,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Segmento RFM",
            yaxis_title="% canje",
            title="% Canje por segmento RFM — GM vs GC",
        )
        st.plotly_chart(fig_rc, use_container_width=True)

        # Tabla detalle por segmento
        def color_lift(val):
            try:
                v = float(val)
                if v > 0: return "color: #2E7D32; font-weight: bold"
                if v < 0: return "color: #C62828; font-weight: bold"
            except Exception:
                pass
            return ""

        tbl_rows = []
        total_base = len(df_rfm) if not df_rfm.empty else None
        for seg in segs_sorted:
            gm_r = gm_df.loc[seg] if seg in gm_df.index else None
            gc_r = gc_df.loc[seg] if seg in gc_df.index else None
            lift  = max(0.0, round(
                (gm_r["conv_rate"] if gm_r is not None else 0) -
                (gc_r["conv_rate"] if gc_r is not None else 0), 1
            ))
            # % del total de clientes en base RFM
            n_base = (
                int((df_rfm["rfm_segment"] == seg).sum())
                if not df_rfm.empty and "rfm_segment" in df_rfm.columns else None
            )
            row = {
                "Segmento RFM":   seg,
                "% Base total":   f"{spct(n_base, total_base):.1f}%" if n_base and total_base else "—",
                "GM Clientes":    int(gm_r["clientes"]) if gm_r is not None else 0,
                "GM Canjean":     int(gm_r["canjean"])  if gm_r is not None else 0,
                "GM %":           round(gm_r["conv_rate"], 1) if gm_r is not None else 0.0,
                "GC Clientes":    int(gc_r["clientes"]) if gc_r is not None else 0,
                "GC Canjean":     int(gc_r["canjean"])  if gc_r is not None else 0,
                "GC %":           round(gc_r["conv_rate"], 1) if gc_r is not None else 0.0,
                "Lift (pp)":      lift,
            }
            tbl_rows.append(row)

        df_tbl = pd.DataFrame(tbl_rows)
        fmt_tbl = {
            "GM Clientes": "{:,}", "GM Canjean": "{:,}",
            "GC Clientes": "{:,}", "GC Canjean": "{:,}",
            "GM %": "{:.1f}%", "GC %": "{:.1f}%", "Lift (pp)": "{:.1f}",
        }
        styled = df_tbl.style.format(fmt_tbl).map(color_lift, subset=["Lift (pp)"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.caption(
            "**Lift (pp)** = GM% − GC%. "
            "**% Base total** = participación del segmento en los 11.8M clientes activos. "
            "Join cliente a cliente: `historical_campaigns.rut` ↔ `rfm_all_clients.cust_id`."
        )

    # ── Sub-tab C: Calibración del modelo ─────────────────────────────────────
    with subtab_c:
        st.markdown('<p class="section-title">Calibración del modelo de propensión</p>',
                    unsafe_allow_html=True)
        st.caption(
            "Compara la propensión predicha (promedio por segmento) contra la conversión real "
            "observada en campañas. Permite detectar si el modelo sobreestima o subestima."
        )

        has_prop = not df_rfm.empty and "propensity_score" in df_rfm.columns and df_rfm["propensity_score"].notna().any()
        df_rc_cal = load_rfm_campaigns()

        if not has_prop:
            st.warning("propensity_score no disponible. Ejecuta `python extract.py` para generarlo.")
        elif df_rc_cal.empty:
            st.info("Sin datos rfm_campaigns para calcular conversión real.")
        else:
            # Propensión predicha promedio por rfm_segment
            prop_seg = (
                df_rfm.dropna(subset=["propensity_score"])
                .groupby("rfm_segment")["propensity_score"]
                .mean() * 100
            ).round(2)

            # Conversión real GM por rfm_segment (todas las campañas)
            gm_cal = df_rc_cal[df_rc_cal["GMGC"] == "GM"]
            gc_cal = df_rc_cal[df_rc_cal["GMGC"] == "GC"]

            RFM_CAL_ORDER = ["Champions", "Loyal", "Nuevos", "En Riesgo", "Perdidos", "Otros"]
            cal_rows = []
            for seg in RFM_CAL_ORDER:
                if seg not in prop_seg.index:
                    continue
                s_gm = gm_cal[gm_cal["rfm_segment"] == seg]
                tot_gm = int(s_gm["CLIENTES"].sum())
                can_gm = int(s_gm.loc[s_gm["FLAG_CANJE"] == "CANJEO", "CLIENTES"].sum())
                pct_gm = spct(can_gm, tot_gm) if tot_gm > 0 else None

                s_gc = gc_cal[gc_cal["rfm_segment"] == seg]
                tot_gc = int(s_gc["CLIENTES"].sum())
                can_gc = int(s_gc.loc[s_gc["FLAG_CANJE"] == "CANJEO", "CLIENTES"].sum())
                pct_gc = spct(can_gc, tot_gc) if tot_gc > 0 else None

                pred = float(prop_seg[seg])
                gap  = round(pred - pct_gm, 1) if pct_gm is not None else None
                cal_rows.append({
                    "seg": seg, "predicho": pred,
                    "real_gm": pct_gm, "real_gc": pct_gc,
                    "gap": gap, "tot_gm": tot_gm,
                })

            if cal_rows:
                cal_df = pd.DataFrame(cal_rows).dropna(subset=["real_gm"])

                # KPIs calibración
                mae = round(cal_df["gap"].abs().mean(), 1) if "gap" in cal_df else None
                k1, k2, k3 = st.columns(3)
                k1.metric("Segmentos evaluados", str(len(cal_df)))
                if mae is not None:
                    k2.metric("Error absoluto medio (MAE)", f"{mae:.1f}pp",
                              help="Diferencia promedio entre propensión predicha y canje real GM")
                bias = round(cal_df["gap"].mean(), 1) if "gap" in cal_df else None
                if bias is not None:
                    k3.metric("Sesgo promedio", f"{bias:+.1f}pp",
                              help="Positivo = modelo sobreestima | Negativo = subestima")

                # Gráfico calibración: predicho vs real (scatter + línea perfecta)
                fig_cal = go.Figure()
                # Línea perfecta y=x
                max_val = max(
                    cal_df["predicho"].max(),
                    cal_df["real_gm"].max()
                ) * 1.1
                fig_cal.add_trace(go.Scatter(
                    x=[0, max_val], y=[0, max_val],
                    mode="lines", name="Calibración perfecta",
                    line=dict(color="#BDBDBD", dash="dash"), showlegend=True,
                ))
                # Puntos por segmento
                for _, row in cal_df.iterrows():
                    fig_cal.add_trace(go.Scatter(
                        x=[row["predicho"]], y=[row["real_gm"]],
                        mode="markers+text",
                        name=row["seg"],
                        text=[row["seg"]],
                        textposition="top center",
                        marker=dict(
                            size=max(8, min(30, row["tot_gm"] / 500000)),
                            color={"Champions": "#1565C0", "Loyal": "#1E88E5",
                                   "Nuevos": "#43A047", "En Riesgo": "#FB8C00",
                                   "Perdidos": "#E53935", "Otros": "#9E9E9E"}.get(row["seg"], "#9E9E9E"),
                            line=dict(width=1, color="white"),
                        ),
                        hovertemplate=(
                            f"<b>{row['seg']}</b><br>"
                            f"Predicho: {row['predicho']:.1f}%<br>"
                            f"Real GM: {row['real_gm']:.1f}%<br>"
                            f"Gap: {row['gap']:+.1f}pp<br>"
                            f"Clientes GM: {row['tot_gm']:,}<extra></extra>"
                        ),
                    ))
                fig_cal.update_layout(
                    height=380,
                    margin=dict(l=0, r=0, t=40, b=0),
                    xaxis_title="Propensión predicha por modelo (%)",
                    yaxis_title="Conversión real GM (%)",
                    title="Calibración: predicho vs real (tamaño = volumen GM en campañas)",
                    showlegend=False,
                )
                st.plotly_chart(fig_cal, use_container_width=True)

                # Tabla de calibración
                def color_gap_cal(val):
                    try:
                        v = float(val)
                        if abs(v) <= 5:  return "color: #2E7D32; font-weight: bold"
                        if abs(v) <= 10: return "color: #F57F17"
                        return "color: #C62828; font-weight: bold"
                    except Exception:
                        return ""

                tbl_cal = cal_df.rename(columns={
                    "seg": "Segmento RFM", "predicho": "Predicho %",
                    "real_gm": "Real GM %", "real_gc": "Real GC %",
                    "gap": "Gap (pp)", "tot_gm": "GM Clientes campaña",
                })[["Segmento RFM", "GM Clientes campaña", "Predicho %",
                    "Real GM %", "Real GC %", "Gap (pp)"]]

                fmt_cal = {
                    "GM Clientes campaña": "{:,}",
                    "Predicho %": "{:.1f}%", "Real GM %": "{:.1f}%",
                    "Real GC %": "{:.1f}%", "Gap (pp)": "{:+.1f}",
                }
                st.dataframe(
                    tbl_cal.style.format(fmt_cal, na_rep="—")
                    .map(color_gap_cal, subset=["Gap (pp)"]),
                    hide_index=True, use_container_width=True,
                )
                st.caption(
                    "**Gap** = Predicho − Real GM. 🟢 ≤5pp bien calibrado | 🟡 ≤10pp | 🔴 >10pp. "
                    "El tamaño del punto en el scatter es proporcional al volumen GM en campañas."
                )


# ── PESTAÑA 4: PLANIFICACIÓN ──────────────────────────────────────────────────
_RFM_SEGS_ALL = ["Champions", "Loyal", "Nuevos", "En Riesgo", "Perdidos", "Otros"]


def seccion_planificador(df_rfm_feat, bundle):
    """Subtab A: estimación de canjeadores y costo para una campaña futura."""
    st.markdown('<p class="section-title">Planificador de audiencia</p>', unsafe_allow_html=True)

    if bundle is None:
        st.warning("Modelo no disponible. Ejecuta `python train_campaign_model.py` primero.")
        return
    if df_rfm_feat.empty:
        st.warning("Sin datos RFM. Verifica `rfm_all_clients.parquet`.")
        return

    # ── Filtros de audiencia ────────────────────────────────────────────────
    f1, f2, f3 = st.columns(3)
    with f1:
        seg_sel = st.multiselect(
            "Segmentos RFM", _RFM_SEGS_ALL, default=_RFM_SEGS_ALL, key="plan_segs"
        )
        tipo_cli_plan = st.multiselect(
            "Tipo de cliente", ["BANCO", "OMP"], default=["BANCO", "OMP"], key="plan_tipo"
        )
    with f2:
        tier_plan = st.multiselect(
            "Categoría / Tier", ["ELITE", "PREMIUM", "NORMAL", "FAN"],
            default=["ELITE", "PREMIUM", "NORMAL", "FAN"], key="plan_tier",
        )
        fecha_envio = st.date_input(
            "Fecha estimada de envío", value=pd.Timestamp.today().date(), key="plan_fecha"
        )
    with f3:
        vpp_plan = st.number_input(
            "Valor por punto ($)", min_value=0.0001, max_value=0.10,
            value=0.01, format="%.4f", key="vpp_plan",
        )
        mes_envio_plan = int(pd.Timestamp(fecha_envio).month)
        st.info(f"Mes de envío detectado: **{fecha_envio.strftime('%B %Y')}**")

    # ── Filtrar df_rfm_feat ─────────────────────────────────────────────────
    df_base = df_rfm_feat.copy()

    # Filtrar por segmento RFM
    if "rfm_segment" in df_base.columns and seg_sel:
        df_base = df_base[df_base["rfm_segment"].isin(seg_sel)]

    # Filtrar por tier
    col_tier_p = detect(df_base, ["tier", "TIER", "categoria", "CATEGORIA"])
    if col_tier_p and tier_plan:
        df_base = df_base[df_base[col_tier_p].str.upper().isin([t.upper() for t in tier_plan])]

    # Filtrar por tipo_cliente (si existe)
    col_tipo_p = detect(df_base, ["tipo_cliente", "TIPO_CLIENTE"])
    if col_tipo_p and tipo_cli_plan:
        df_base = df_base[df_base[col_tipo_p].isin(tipo_cli_plan)]

    total_audiencia = len(df_base)

    if total_audiencia == 0:
        st.warning("Sin clientes con los filtros seleccionados.")
        return

    # ── Predicción ──────────────────────────────────────────────────────────
    btn_predict = st.button("Estimar canjeadores esperados", type="primary", key="plan_predict")
    if btn_predict or st.session_state.get("plan_resultado") is not None:
        if btn_predict:
            with st.spinner(f"Prediciendo para {total_audiencia:,} clientes..."):
                df_ind_syn = pd.DataFrame({
                    "cust_id": df_base["cust_id"].values,
                    "canjeo":  0,
                    "GMGC":    "GM",
                })
                df_pred = predict_5d(bundle, df_ind_syn, df_rfm_feat, pd.Timestamp(fecha_envio))
            st.session_state["plan_resultado"] = df_pred
        else:
            df_pred = st.session_state["plan_resultado"]

        # KPIs estimados
        exp_canjead = float(df_pred["p_5dias"].sum())
        pct_conv    = exp_canjead / max(total_audiencia, 1) * 100

        # Avg puntos por canje (desde rfm o parámetro fijo 5000)
        col_avg_pts = detect(df_rfm_feat, ["avg_redeem_points", "monetary_monthly_avg"])
        avg_pts = float(df_rfm_feat[col_avg_pts].dropna().median()) if col_avg_pts else 5000.0
        puntos_esp = exp_canjead * avg_pts
        costo_est  = puntos_esp * vpp_plan

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Audiencia filtrada", f"{total_audiencia:,}")
        k2.metric("Canjeadores esperados (5d)", f"{exp_canjead:,.0f}")
        k3.metric("% Conversión esperada", f"{pct_conv:.1f}%")
        k4.metric("Puntos esperados", f"{puntos_esp:,.0f}")
        k5.metric("Costo estimado ($)", f"${costo_est:,.0f}")

        # Gráfico por segmento RFM
        if "rfm_segment" in df_pred.columns:
            seg_agg = df_pred.groupby("rfm_segment").agg(
                Audiencia=("cust_id", "count"),
                Canjeadores_esp=("p_5dias", "sum"),
            ).reset_index()
            seg_agg["% Conv"] = seg_agg["Canjeadores_esp"] / seg_agg["Audiencia"] * 100
            seg_agg = seg_agg.sort_values("% Conv", ascending=False)

            fig_plan = go.Figure()
            fig_plan.add_trace(go.Bar(
                x=seg_agg["rfm_segment"], y=seg_agg["Canjeadores_esp"],
                name="Canjeadores esperados", marker_color="#1565C0",
                text=seg_agg["% Conv"].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
            ))
            fig_plan.add_trace(go.Bar(
                x=seg_agg["rfm_segment"], y=seg_agg["Audiencia"],
                name="Audiencia total", marker_color="#BDBDBD",
                opacity=0.5,
            ))
            fig_plan.update_layout(
                barmode="group", height=320,
                margin=dict(l=0, r=0, t=30, b=0),
                yaxis_title="Clientes",
                title="Canjeadores esperados vs Audiencia por segmento RFM",
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
            )
            st.plotly_chart(fig_plan, use_container_width=True)

            # Tabla de detalle
            tbl_plan = seg_agg.copy()
            tbl_plan["Puntos est."] = tbl_plan["Canjeadores_esp"] * avg_pts
            tbl_plan["Costo est. ($)"] = tbl_plan["Puntos est."] * vpp_plan
            tbl_plan.columns = ["Segmento", "Audiencia", "Canjeadores esp.", "% Conv",
                                 "Puntos est.", "Costo est. ($)"]
            st.dataframe(
                tbl_plan.style.format({
                    "Audiencia": "{:,.0f}", "Canjeadores esp.": "{:,.0f}",
                    "% Conv": "{:.1f}%", "Puntos est.": "{:,.0f}", "Costo est. ($)": "${:,.0f}",
                }),
                hide_index=True, use_container_width=True,
            )


def seccion_historial_envios(df_hist, df_omm, df_r):
    """Subtab B: tabla histórica de campañas enviadas con métricas de email y canje."""
    st.markdown('<p class="section-title">Historial de envíos</p>', unsafe_allow_html=True)

    if df_hist.empty and df_r.empty:
        st.info("Sin datos de histórico. Ejecuta `python extract.py` primero.")
        return

    # ── Construir tabla consolidada ─────────────────────────────────────────
    col_camp_h  = detect(df_hist, ["NOMBRE_CAMPANHA", "CAMPANHA"]) if not df_hist.empty else None
    col_fecha_h = detect(df_hist, ["FECHA_ENVIO"]) if not df_hist.empty else None
    col_camp_r  = detect(df_r, ["NOMBRE_CAMPANHA", "CAMPANHA"]) if not df_r.empty else None
    col_flag_r  = detect(df_r, ["FLAG_CANJE"]) if not df_r.empty else None
    col_cli_r   = detect(df_r, ["CLIENTES"]) if not df_r.empty else None
    col_gm_r    = detect(df_r, ["GMGC"]) if not df_r.empty else None
    col_fecha_r = detect(df_r, ["FECHA_ENVIO"]) if not df_r.empty else None

    # Obtener lista de campañas de redemption_metrics
    rows_hist = []
    if not df_r.empty and col_camp_r:
        for camp_k, grp_k in df_r.groupby(col_camp_r):
            fecha_k = None
            if col_fecha_r and col_fecha_r in grp_k.columns:
                fecha_k = pd.to_datetime(grp_k[col_fecha_r], errors="coerce").min()

            # GM stats
            gm_k = grp_k[grp_k[col_gm_r] == "GM"] if col_gm_r else grp_k
            gc_k = grp_k[grp_k[col_gm_r] == "GC"] if col_gm_r else pd.DataFrame()
            tot_gm_k  = int(gm_k[col_cli_r].sum()) if col_cli_r and not gm_k.empty else 0
            canj_gm_k = int(gm_k.loc[gm_k[col_flag_r] == "CANJEO", col_cli_r].sum()) if col_cli_r and col_flag_r and not gm_k.empty else 0
            tot_gc_k  = int(gc_k[col_cli_r].sum()) if col_cli_r and not gc_k.empty else 0
            canj_gc_k = int(gc_k.loc[gc_k[col_flag_r] == "CANJEO", col_cli_r].sum()) if col_cli_r and col_flag_r and not gc_k.empty else 0
            pct_gm_k  = spct(canj_gm_k, tot_gm_k)
            pct_gc_k  = spct(canj_gc_k, tot_gc_k)
            lift_k    = max(0.0, round(pct_gm_k - pct_gc_k, 1))

            # Email stats desde OMM
            open_r_k, click_r_k = None, None
            if not df_omm.empty:
                omm_cc = detect(df_omm, ["CAMPAIGN_NAME", "NOMBRE_CAMPANHA"])
                if omm_cc:
                    m_k = df_omm[df_omm[omm_cc].apply(norm_camp) == norm_camp(camp_k)]
                    if not m_k.empty:
                        row_k = m_k.agg({c: "sum" for c in m_k.select_dtypes("number").columns})
                        mails_k = float(row_k.get("TOTAL_MAILS", 0))
                        if mails_k > 0:
                            open_r_k  = round(float(row_k.get("TOTAL_OPENS", 0)) / mails_k * 100, 1)
                            click_r_k = round(float(row_k.get("TOTAL_CLICKS", 0)) / mails_k * 100, 1)

            rows_hist.append({
                "Campaña":         camp_k,
                "Fecha":           fecha_k.date() if fecha_k and not pd.isna(fecha_k) else None,
                "GM Total":        tot_gm_k,
                "GM Canjean":      canj_gm_k,
                "% Canje GM":      f"{pct_gm_k:.1f}%",
                "% Canje GC":      f"{pct_gc_k:.1f}%",
                "Lift (pp)":       lift_k,
                "Open rate":       f"{open_r_k:.1f}%" if open_r_k is not None else "—",
                "Click rate":      f"{click_r_k:.1f}%" if click_r_k is not None else "—",
            })

    if not rows_hist:
        st.info("Sin datos de campañas disponibles.")
        return

    df_thist = pd.DataFrame(rows_hist).sort_values("Fecha", ascending=False, na_position="last")

    # ── Filtro por nombre ───────────────────────────────────────────────────
    buscar = st.text_input("Buscar campaña por nombre", key="hist_buscar")
    if buscar:
        df_thist = df_thist[df_thist["Campaña"].str.contains(buscar, case=False, na=False)]

    st.dataframe(
        df_thist.style.format({"GM Total": "{:,}", "GM Canjean": "{:,}", "Lift (pp)": "{:+.1f}"}),
        hide_index=True, use_container_width=True,
    )

    # ── Gráfico de tendencia mensual ───────────────────────────────────────
    st.markdown("**Tendencia mensual**")
    df_tmon = df_thist.copy()
    df_tmon["Fecha"] = pd.to_datetime(df_tmon["Fecha"], errors="coerce")
    df_tmon["Mes"]   = df_tmon["Fecha"].dt.to_period("M").astype(str)
    df_tmon["% Canje GM num"] = df_tmon["% Canje GM"].str.replace("%", "").astype(float)

    if df_tmon["Mes"].notna().any():
        mon_agg = df_tmon.groupby("Mes").agg(
            Campañas=("Campaña", "count"),
            Canje_GM_avg=("% Canje GM num", "mean"),
        ).reset_index().sort_values("Mes")

        from plotly.subplots import make_subplots
        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        fig_trend.add_trace(go.Bar(
            x=mon_agg["Mes"], y=mon_agg["Campañas"],
            name="# Campañas", marker_color="#CFD8DC", opacity=0.7,
        ), secondary_y=False)
        fig_trend.add_trace(go.Scatter(
            x=mon_agg["Mes"], y=mon_agg["Canje_GM_avg"],
            name="% Canje GM promedio", line=dict(color="#1565C0", width=2),
            mode="lines+markers",
        ), secondary_y=True)
        fig_trend.update_layout(
            height=280, margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        )
        fig_trend.update_yaxes(title_text="# Campañas", secondary_y=False)
        fig_trend.update_yaxes(title_text="% Canje GM", secondary_y=True)
        st.plotly_chart(fig_trend, use_container_width=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    st.title("Dashboard de Campañas — CMR Puntos")

    dfs = load_data()
    df_r, df_omm, df_hist = dfs["r"], dfs["omm"], dfs["hist"]

    if all(v.empty for v in dfs.values()):
        st.error(
            "No hay datos en `data/`. Ejecuta primero:\n"
            "```bash\ncd campanhas_estefi\npython extract.py --fecha-inicio 2026-02-01\n```"
        )
        st.stop()

    # Filtrar por rango de fechas
    col_fecha = detect(df_r, ["FECHA_ENVIO"])
    if col_fecha and not df_r.empty:
        df_r[col_fecha] = pd.to_datetime(df_r[col_fecha], errors="coerce")

    df_rfm_camp_main = load_rfm_campaigns()
    df_camp = build_camp_table(df_r, df_omm, df_rfm_camp_main)

    f_ini, f_fin, tipo_cli_opts, tier_opts, valor_punto = build_sidebar(df_r)

    # Aplicar filtro de fechas a df_r
    if col_fecha and not df_r.empty:
        df_r_f = df_r[
            (df_r[col_fecha].dt.date >= f_ini) &
            (df_r[col_fecha].dt.date <= f_fin)
        ].copy()
    else:
        df_r_f = df_r.copy()

    # Fix 6: Filtrar por tipo_cliente si la columna existe
    col_tipo_r = detect(df_r_f, ["TIPO_CLIENTE"])
    if col_tipo_r and tipo_cli_opts:
        df_r_f = df_r_f[df_r_f[col_tipo_r].isin(tipo_cli_opts)]

    # Recalcular tabla con fechas filtradas (rfm_campaigns no se filtra por fecha — es la base completa)
    df_camp_f = build_camp_table(df_r_f, df_omm, df_rfm_camp_main)

    # Filtrar OMM
    if not df_omm.empty:
        omm_f = detect(df_omm, ["INICIO_VIGENCIA","FECHA_ENVIO"])
        if omm_f:
            df_omm[omm_f] = pd.to_datetime(df_omm[omm_f], errors="coerce")
            df_omm_f = df_omm[
                (df_omm[omm_f].dt.date >= f_ini) &
                (df_omm[omm_f].dt.date <= f_fin)
            ]
        else:
            df_omm_f = df_omm
    else:
        df_omm_f = df_omm

    tab1, tab2, tab3, tab4 = st.tabs([
        "Resumen de Campañas", "Vista Granular", "Segmentación RFM", "🚀 Planificación"
    ])

    with tab1:
        # ── Sección 1: resumen general ────────────────────────────────────────
        if df_camp_f.empty:
            st.warning("Sin campañas en el período seleccionado.")
            sel_from_table = None
        else:
            sel_from_table = seccion_resumen(df_camp_f)

        # ── Sección 2: detalle de campaña (click en tabla) ────────────────────
        active_camp = sel_from_table
        if active_camp:
            seccion_detalle(active_camp, df_r_f, df_omm_f, df_hist, valor_punto=valor_punto)
        else:
            st.info("Haz clic en una fila de la tabla para ver el detalle de la campaña.")

    with tab2:
        seccion_granular(df_r_f, valor_punto=valor_punto)

    with tab3:
        df_rfm_tab = load_rfm()
        # Fix 6: filtrar RFM por tier seleccionado
        if not df_rfm_tab.empty and tier_opts:
            col_tier_rfm = detect(df_rfm_tab, ["tier","TIER","categoria","CATEGORIA"])
            if col_tier_rfm:
                df_rfm_tab = df_rfm_tab[
                    df_rfm_tab[col_tier_rfm].str.upper().isin([t.upper() for t in tier_opts])
                ]
        seccion_rfm(df_rfm_tab, df_r)

    with tab4:
        bundle_plan = load_campaign_model()
        df_rfm_feat_plan = load_rfm_features_pred()
        plan_sub1, plan_sub2 = st.tabs(["Planificador de audiencia", "Historial de envíos"])
        with plan_sub1:
            seccion_planificador(df_rfm_feat_plan, bundle_plan)
        with plan_sub2:
            seccion_historial_envios(df_hist, df_omm_f, df_r_f)


if __name__ == "__main__":
    main()
