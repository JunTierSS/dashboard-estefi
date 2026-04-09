"""
Extracción de datos de campañas desde BigQuery.
Guarda parquet en data/ para que el dashboard cargue sin costo de BQ en cada refresh.

Uso:
    python extract.py --fecha-inicio 2026-02-01
    python extract.py --fecha-inicio 2026-01-01 --fecha-fin 2026-03-31
"""
import sys, os, time, argparse, warnings
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
warnings.filterwarnings('ignore')

from google.cloud import bigquery
import pandas as pd

# ── Proyectos ────────────────────────────────────────────────────────────────
PROJECT_CONSUMPTION = "fif-loy-cl-consumption"
PROJECT_DISCOVERY   = "fif-loy-cl-cg-discovery"
OUTPUT_DIR          = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Clientes BQ ──────────────────────────────────────────────────────────────
client_cons  = bigquery.Client(project=PROJECT_CONSUMPTION)
client_disc  = bigquery.Client(project=PROJECT_DISCOVERY)


def q(client, sql, label=""):
    t0 = time.time()
    print(f"  [{label}] ejecutando...", flush=True)
    df = client.query(sql).to_dataframe(create_bqstorage_client=True)
    elapsed = time.time() - t0
    print(f"  [{label}] {len(df):,} filas en {elapsed:.1f}s", flush=True)
    return df


def normalize_types(df):
    """Convierte tipos BigQuery (dbdate, dbbool, etc.) a tipos pandas estándar."""
    for col in df.columns:
        col_dtype = str(df[col].dtype)
        if "date" in col_dtype.lower() or col_dtype == "dbdate":
            df[col] = pd.to_datetime(df[col], errors="coerce")
        elif col_dtype == "dbbool":
            df[col] = df[col].astype(bool)
        elif col_dtype == "object":
            pass  # ya es string-compatible
    return df


def save(df, name):
    df = normalize_types(df)
    path = os.path.join(OUTPUT_DIR, f"{name}.parquet")
    df.to_parquet(path, index=False)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  -> Guardado: {path} ({size_mb:.1f} MB)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fecha-inicio", default="2026-02-01",
                        help="Fecha inicio en formato YYYY-MM-DD (default: 2026-02-01)")
    parser.add_argument("--fecha-fin",    default=None,
                        help="Fecha fin en formato YYYY-MM-DD (default: hoy)")
    args = parser.parse_args()

    fecha_inicio = args.fecha_inicio
    fecha_fin    = args.fecha_fin or pd.Timestamp.today().strftime("%Y-%m-%d")

    print("=" * 70)
    print("EXTRACCION CAMPAÑAS")
    print(f"  Rango: {fecha_inicio} → {fecha_fin}")
    print("=" * 70)

    # ── 1. TABLA PRINCIPAL: funnel pre/post + flag canje ─────────────────────
    # Tabla custom en dataset temporal de discovery. Se hace SELECT * porque
    # el DDL no está en el repo. Si los nombres de columna difieren, ajustar aquí:
    #   TIPO_CANJEADOR_PRE  → estado funnel antes de la campaña
    #   TIPO_CANJEADOR_POST → estado funnel después de la campaña
    #   FLAG_CANJE          → si canjeó durante la campaña
    #   FLAG_CANJEADOR_HISTORICO → si ya era canjeador antes
    print("\n[1] Tabla principal: campaigns_redemption_metrics (discovery.temporal)")
    sql_redemption = f"""
        SELECT *
        FROM `{PROJECT_DISCOVERY}.temporal.campaigns_redemption_metrics`
        WHERE FECHA_ENVIO >= '{fecha_inicio}'
          AND FECHA_ENVIO <= '{fecha_fin}'
    """
    try:
        df_redemption = q(client_disc, sql_redemption, "redemption_metrics")
        save(df_redemption, "redemption_metrics")
        print(f"  Columnas disponibles: {list(df_redemption.columns)}", flush=True)
    except Exception as e:
        print(f"  ERROR en tabla principal: {e}", flush=True)
        print("  Creando DataFrame vacío de placeholder...", flush=True)
        df_redemption = pd.DataFrame()
        df_redemption.to_parquet(os.path.join(OUTPUT_DIR, "redemption_metrics.parquet"), index=False)

    # ── 2. MÉTRICAS OMM (Salesforce: open/click/bounce/unsub) ───────────────
    print("\n[2] OMM Salesforce: evaluador_resultados_omm")
    sql_omm = f"""
        SELECT
            id_jobenvio,
            campaign_name,
            casos,
            inicio_vigencia,
            fin_vigencia,
            flag_vigencia,
            desc_emailname,
            desc_asunto,
            total_mails,
            total_opens,
            total_clicks,
            total_bounces,
            total_unsubs,
            porc_envio
        FROM `{PROJECT_CONSUMPTION}.evaluador_campanhas.evaluador_resultados_omm`
        WHERE inicio_vigencia >= '{fecha_inicio}'
          AND inicio_vigencia <= '{fecha_fin}'
    """
    try:
        df_omm = q(client_cons, sql_omm, "omm_metrics")
        save(df_omm, "omm_metrics")
    except Exception as e:
        print(f"  ERROR en OMM: {e}", flush=True)
        df_omm = pd.DataFrame()
        df_omm.to_parquet(os.path.join(OUTPUT_DIR, "omm_metrics.parquet"), index=False)

    # ── 3. GRANULAR: historical_campaigns (base enviada) ────────────────────
    print("\n[3] Granular: historical_campaigns")
    sql_hist = f"""
        SELECT
            FECHA_ENVIO,
            NOMBRE_CAMPANHA,
            TIPO_CLIENTE,
            CANAL,
            TIPO_MEDICION,
            tipo_campanha,
            GMGC,
            INICIO_VIGENCIA,
            FIN_VIGENCIA
        FROM `{PROJECT_CONSUMPTION}.loy_campaigns.historical_campaigns`
        WHERE FECHA_ENVIO >= '{fecha_inicio}'
          AND FECHA_ENVIO <= '{fecha_fin}'
        LIMIT 1000000
    """
    try:
        df_hist = q(client_cons, sql_hist, "historical")
        save(df_hist, "historical_campaigns")
    except Exception as e:
        print(f"  ERROR en historical_campaigns: {e}", flush=True)
        df_hist = pd.DataFrame()
        df_hist.to_parquet(os.path.join(OUTPUT_DIR, "historical_campaigns.parquet"), index=False)

    # ── 4. RFM: todos los clientes de la partición actual ───────────────────
    print("\n[4] RFM — todos los clientes (partición actual de clients_entity)")
    try:
        # Paso 1: calcular la partición más reciente (último fin de mes) en Python
        # clients_entity tiene require_partition_filter, no permite MAX() sin filtro.
        hoy = pd.Timestamp.today()
        t0_dt      = (hoy.replace(day=1) - pd.Timedelta(days=1)).date()
        t0_literal = str(t0_dt)
        t0_minus12 = str(
            (pd.Timestamp(t0_literal) - pd.DateOffset(months=12)).date()
        )
        print(f"  Partición actual (último fin de mes): {t0_literal}", flush=True)
        print(f"  Ventana transacciones: {t0_minus12} → {t0_literal}", flush=True)

        # Paso 2: query principal con la fecha como literal (permite partition pruning)
        # Incluye features adicionales necesarias para el modelo de propensión (51 features)
        t0_minus90 = str(
            (pd.Timestamp(t0_literal) - pd.DateOffset(days=90)).date()
        )
        t0_minus30 = str(
            (pd.Timestamp(t0_literal) - pd.DateOffset(days=30)).date()
        )
        sql_rfm = f"""
        WITH clientes AS (
            SELECT
                c.cust_id,
                UPPER(c.cat_cust_name)        AS tier,
                c.cust_gender_desc            AS gender,
                c.cust_age_num                AS age,
                c.city_name                   AS city,
                c.cust_active_cmr_flg,
                c.cust_active_deb_flg,
                c.contact_email_flg,
                c.contact_phone_flg,
                c.contact_push_flg,
                c.cust_active_omp_flg,
                c.cust_stock_point_amt        AS stock_points,
                c.exp_point_current_month_amt AS exp_points_current,
                c.exp_point_next_month_amt    AS exp_points_next,
                c.cust_enroll_date            AS enrollment_date,
                c.cat_cust_name               AS tier_raw,
                DATE '{t0_literal}'           AS t0
            FROM `{PROJECT_CONSUMPTION}.svw_fif_loy_cl_datalake_prod__trf_loy_cl_prd.svw_clients_entity` c
            WHERE c.partition_date = DATE '{t0_literal}'
        ),
        trx AS (
            SELECT
                t.cust_id,
                MAX(t.tran_date)            AS last_tran_date,
                COUNT(DISTINCT t.tran_id)   AS frequency_total,
                SUM(t.tran_amt)             AS monetary_total,
                -- Velocidades por ventana
                COUNT(DISTINCT CASE WHEN t.tran_date >= DATE '{t0_minus30}' THEN t.tran_id END) AS earn_velocity_30,
                COUNT(DISTINCT CASE WHEN t.tran_date >= DATE '{t0_minus90}' THEN t.tran_id END) AS earn_velocity_90,
                -- Gasto por retailer (12 meses)
                SUM(CASE WHEN UPPER(t.channel_name) = 'FALABELLA' THEN t.tran_amt ELSE 0 END) AS spend_falabella,
                SUM(CASE WHEN UPPER(t.channel_name) = 'SODIMAC'   THEN t.tran_amt ELSE 0 END) AS spend_sodimac,
                SUM(CASE WHEN UPPER(t.channel_name) = 'TOTTUS'    THEN t.tran_amt ELSE 0 END) AS spend_tottus,
                SUM(CASE WHEN UPPER(t.channel_name) = 'FCOM'      THEN t.tran_amt ELSE 0 END) AS spend_fcom,
                SUM(CASE WHEN UPPER(t.channel_name) = 'IKEA'      THEN t.tran_amt ELSE 0 END) AS spend_ikea,
                -- Métodos de pago
                SAFE_DIVIDE(
                    COUNTIF(UPPER(t.payment_method_name) LIKE '%CMR%'),
                    COUNT(DISTINCT t.tran_id)
                ) AS pct_cmr_payments,
                SAFE_DIVIDE(
                    COUNTIF(UPPER(t.payment_method_name) LIKE '%DEB%'),
                    COUNT(DISTINCT t.tran_id)
                ) AS pct_debit_payments
            FROM `{PROJECT_CONSUMPTION}.control_de_gestion.frozen_transaction_entity` t
            WHERE t.tran_date >= DATE '{t0_minus12}'
              AND t.tran_date  < DATE '{t0_literal}'
              AND t.tran_type  = 'COMPRA'
              AND t.tran_amt   > 0
              AND t.tran_valid_flg = 1
            GROUP BY t.cust_id
        ),
        rdm AS (
            SELECT
                r.cust_id,
                COUNT(DISTINCT r.redemption_id_unico) AS redeem_count_pre,
                MAX(r.redemption_date)                AS last_redeem_date,
                SAFE_DIVIDE(
                    COUNTIF(LOWER(r.bu_channel_name) LIKE '%digital%' OR
                            LOWER(r.bu_channel_name) LIKE '%online%'  OR
                            LOWER(r.bu_channel_name) LIKE '%app%'),
                    COUNT(*)
                )                                     AS pct_redeem_digital,
                AVG(r.redemption_points_amt)          AS avg_redeem_points,
                SAFE_DIVIDE(
                    COUNTIF(LOWER(r.price_type_desc) LIKE '%catalogo%'),
                    COUNT(*)
                )                                     AS pct_redeem_catalogo,
                SAFE_DIVIDE(
                    COUNTIF(LOWER(r.price_type_desc) LIKE '%giftcard%'),
                    COUNT(*)
                )                                     AS pct_redeem_giftcard
            FROM `{PROJECT_CONSUMPTION}.control_de_gestion.frozen_redemption_entity` r
            WHERE r.redemption_date >= DATE '{t0_minus12}'
              AND r.redemption_date  < DATE '{t0_literal}'
              AND r.return_flag IS FALSE
            GROUP BY r.cust_id
        )
        SELECT
            c.cust_id,
            c.tier,
            c.gender,
            c.age,
            c.city,
            -- Flags de actividad
            c.cust_active_cmr_flg,
            c.cust_active_deb_flg,
            c.contact_email_flg,
            c.contact_phone_flg,
            c.contact_push_flg,
            c.cust_active_omp_flg,
            -- Puntos
            c.stock_points,
            c.exp_points_current,
            c.exp_points_next,
            c.enrollment_date,
            c.t0,
            -- RFM base
            COALESCE(DATE_DIFF(c.t0, t.last_tran_date, DAY), 999) AS recency_days,
            COALESCE(t.frequency_total, 0)                         AS frequency_total,
            COALESCE(t.monetary_total,  0)                         AS monetary_total,
            -- Velocidades
            COALESCE(t.earn_velocity_30, 0)                        AS earn_velocity_30,
            COALESCE(t.earn_velocity_90, 0)                        AS earn_velocity_90,
            -- Gasto por retailer
            COALESCE(t.spend_falabella, 0)  AS spend_falabella,
            COALESCE(t.spend_sodimac,   0)  AS spend_sodimac,
            COALESCE(t.spend_tottus,    0)  AS spend_tottus,
            COALESCE(t.spend_fcom,      0)  AS spend_fcom,
            COALESCE(t.spend_ikea,      0)  AS spend_ikea,
            -- Métodos de pago
            COALESCE(t.pct_cmr_payments,   0) AS pct_cmr_payments,
            COALESCE(t.pct_debit_payments,  0) AS pct_debit_payments,
            -- Redemptions
            COALESCE(rdm.redeem_count_pre,    0) AS redeem_count_pre,
            COALESCE(rdm.pct_redeem_digital,  0) AS pct_redeem_digital,
            COALESCE(rdm.avg_redeem_points,   0) AS avg_redeem_points,
            COALESCE(rdm.pct_redeem_catalogo, 0) AS pct_redeem_catalogo,
            COALESCE(rdm.pct_redeem_giftcard, 0) AS pct_redeem_giftcard,
            COALESCE(DATE_DIFF(c.t0, rdm.last_redeem_date, DAY), 999) AS days_since_last_redeem,
            -- Actividad general
            COALESCE(
                DATE_DIFF(c.t0, GREATEST(
                    COALESCE(t.last_tran_date, DATE '2000-01-01'),
                    COALESCE(rdm.last_redeem_date, DATE '2000-01-01')
                ), DAY), 999
            )                                                      AS days_since_last_activity,
            SAFE_DIVIDE(c.exp_points_current, NULLIF(c.stock_points, 0)) AS points_pressure,
            DATE_DIFF(c.t0, c.enrollment_date, MONTH)              AS tenure_months,
            -- Retailer dominante (spend máximo)
            CASE
                WHEN GREATEST(
                    COALESCE(t.spend_falabella,0), COALESCE(t.spend_sodimac,0),
                    COALESCE(t.spend_tottus,0),    COALESCE(t.spend_fcom,0),
                    COALESCE(t.spend_ikea,0)
                ) = 0 THEN 'NINGUNO'
                WHEN COALESCE(t.spend_falabella,0) >= COALESCE(t.spend_sodimac,0)
                 AND COALESCE(t.spend_falabella,0) >= COALESCE(t.spend_tottus,0)
                 AND COALESCE(t.spend_falabella,0) >= COALESCE(t.spend_fcom,0)
                 AND COALESCE(t.spend_falabella,0) >= COALESCE(t.spend_ikea,0) THEN 'FALABELLA'
                WHEN COALESCE(t.spend_sodimac,0) >= COALESCE(t.spend_tottus,0)
                 AND COALESCE(t.spend_sodimac,0) >= COALESCE(t.spend_fcom,0)
                 AND COALESCE(t.spend_sodimac,0) >= COALESCE(t.spend_ikea,0) THEN 'SODIMAC'
                WHEN COALESCE(t.spend_tottus,0) >= COALESCE(t.spend_fcom,0)
                 AND COALESCE(t.spend_tottus,0) >= COALESCE(t.spend_ikea,0) THEN 'TOTTUS'
                WHEN COALESCE(t.spend_fcom,0) >= COALESCE(t.spend_ikea,0) THEN 'FCOM'
                ELSE 'IKEA'
            END AS dominant_retailer,
            -- Funnel state (lógica fase3)
            CASE
                WHEN COALESCE(rdm.redeem_count_pre, 0) >= 1
                 AND DATE_DIFF(c.t0, COALESCE(rdm.last_redeem_date, DATE '2000-01-01'), DAY) >
                     CASE WHEN UPPER(c.tier_raw) IN ('ELITE','PREMIUM') THEN 730 ELSE 365 END
                     THEN 'FUGADO'
                WHEN COALESCE(rdm.redeem_count_pre, 0) >= 2 THEN 'RECURRENTE'
                WHEN COALESCE(rdm.redeem_count_pre, 0) = 1  THEN 'CANJEADOR'
                WHEN COALESCE(t.frequency_total, 0) >= 1
                 AND COALESCE(rdm.redeem_count_pre, 0) = 0
                 AND c.stock_points >= 1000 THEN 'POSIBILIDAD_CANJE'
                WHEN COALESCE(t.frequency_total, 0) >= 1
                 AND COALESCE(rdm.redeem_count_pre, 0) = 0 THEN 'PARTICIPANTE'
                ELSE 'INSCRITO'
            END AS funnel_state_at_t0
        FROM clientes c
        LEFT JOIN trx t ON c.cust_id = t.cust_id
        LEFT JOIN rdm   ON c.cust_id = rdm.cust_id
        """
        df_rfm = q(client_cons, sql_rfm, "rfm_all")

        # ── Retailer entropy (Shannon) sobre gasto por retailer ──────────────────
        import numpy as np
        print("  Calculando retailer_entropy (Shannon sobre gasto)...", flush=True)
        spends = ["spend_falabella", "spend_sodimac", "spend_tottus", "spend_fcom", "spend_ikea"]
        spends_ok = [s for s in spends if s in df_rfm.columns]
        if spends_ok:
            total_sp = df_rfm[spends_ok].sum(axis=1).replace(0, 1)
            entropy  = pd.Series(0.0, index=df_rfm.index)
            for col in spends_ok:
                p = (df_rfm[col] / total_sp).clip(lower=0)
                entropy -= (p * np.log(p + 1e-9)).where(p > 0, 0)
            df_rfm["retailer_entropy"] = entropy.round(4)

        # ── Scoring RFM en Python (quintiles 1-5) ────────────────────────────────
        print("  Calculando quintiles R/F/M...", flush=True)
        n = len(df_rfm)

        for col, score_col, ascending in [
            ("recency_days",    "r_score", False),   # menor días = mejor
            ("frequency_total", "f_score", True),
            ("monetary_total",  "m_score", True),
        ]:
            if col in df_rfm.columns and n > 0:
                labels = [5, 4, 3, 2, 1] if not ascending else [1, 2, 3, 4, 5]
                try:
                    df_rfm[score_col] = pd.qcut(
                        df_rfm[col].rank(method="first"), 5, labels=labels
                    ).astype(int)
                except Exception:
                    df_rfm[score_col] = 3

        def rfm_segment(row):
            R = row.get("r_score", 3)
            F = row.get("f_score", 3)
            M = row.get("m_score", 3)
            if R >= 4 and F >= 4 and M >= 4: return "Champions"
            if F >= 3 and M >= 3:            return "Loyal"
            if R >= 4 and F <= 2:            return "Nuevos"
            if R <= 2 and F >= 3 and M >= 3: return "En Riesgo"
            if R <= 2 and F <= 2 and M <= 2: return "Perdidos"
            return "Otros"

        df_rfm["rfm_segment"] = df_rfm.apply(rfm_segment, axis=1)

        PRIORIDAD = {
            "Champions": "Alta",
            "Loyal":     "Alta",
            "En Riesgo": "Media",
            "Nuevos":    "Media",
            "Perdidos":  "Baja",
            "Otros":     "Baja",
        }
        df_rfm["prioridad"] = df_rfm["rfm_segment"].map(PRIORIDAD)

        seg_counts = df_rfm["rfm_segment"].value_counts().to_dict()
        print(f"  Segmentos: {seg_counts}", flush=True)

        # ── Aplicar modelo de propensión ──────────────────────────────────────
        MODEL_PATH = os.path.join(os.path.dirname(__file__), "../fase3/models/models.pkl")
        PSM_FEATURES = [
            "frequency_monthly_avg", "monetary_monthly_avg", "redeem_rate",
            "retailer_entropy",      "pct_redeem_digital",   "earn_velocity_90",
            "days_since_last_activity", "points_pressure",   "stock_points_at_t0",
            "redeem_count_pre",      "frequency_total",      "monetary_total",
            "tenure_months",
        ]
        if os.path.exists(MODEL_PATH):
            print("  Aplicando modelo de propensión...", flush=True)
            import pickle
            # Derivar features calculables
            df_rfm["frequency_monthly_avg"] = df_rfm["frequency_total"] / 12.0
            df_rfm["monetary_monthly_avg"]  = df_rfm["monetary_total"]  / 12.0
            df_rfm["redeem_rate"]           = df_rfm["redeem_count_pre"] / df_rfm["frequency_total"].replace(0, 1)
            df_rfm["stock_points_at_t0"]    = df_rfm["stock_points"]
            # earn_velocity_90 ya viene de BQ como freq_90; renombrar si necesario
            if "earn_velocity_90" not in df_rfm.columns and "freq_90" in df_rfm.columns:
                df_rfm["earn_velocity_90"] = df_rfm["freq_90"]

            try:
                with open(MODEL_PATH, "rb") as f:
                    models = pickle.load(f)
                feats_ok = [c for c in PSM_FEATURES if c in df_rfm.columns]
                feats_miss = [c for c in PSM_FEATURES if c not in df_rfm.columns]
                if feats_miss:
                    print(f"  AVISO: features faltantes (→ fillna 0): {feats_miss}", flush=True)
                    for fm in feats_miss:
                        df_rfm[fm] = 0
                X_psm = df_rfm[PSM_FEATURES].fillna(0)
                df_rfm["propensity_score"] = models["propensity_model"].predict_proba(X_psm)[:, 1]
                avg_prop = df_rfm["propensity_score"].mean()
                print(f"  propensity_score calculado — promedio: {avg_prop:.3f}", flush=True)
            except Exception as e_model:
                print(f"  ERROR aplicando modelo: {e_model}", flush=True)
                df_rfm["propensity_score"] = float("nan")
        else:
            print(f"  AVISO: modelo no encontrado en {MODEL_PATH} — propensity_score omitido", flush=True)
            df_rfm["propensity_score"] = float("nan")

        save(df_rfm, "rfm_all_clients")

    except Exception as e:
        print(f"  ERROR en RFM all clients: {e}", flush=True)
        import traceback; traceback.print_exc()
        pd.DataFrame().to_parquet(
            os.path.join(OUTPUT_DIR, "rfm_all_clients.parquet"), index=False
        )

    # ── 5. RFM × CAMPAÑAS: join individual cust_id ──────────────────────────
    print("\n[5] RFM × Campañas — join a nivel de cliente")
    sql_rfm_camp = f"""
        WITH campanas AS (
            SELECT DISTINCT
                A.FECHA_ENVIO,
                A.NOMBRE_CAMPANHA,
                A.GMGC,
                A.RUT        AS cust_id,
                A.INICIO_VIGENCIA,
                A.FIN_VIGENCIA
            FROM `{PROJECT_CONSUMPTION}.loy_campaigns.historical_campaigns` A
            WHERE A.FECHA_ENVIO >= '{fecha_inicio}'
              AND A.FECHA_ENVIO <= '{fecha_fin}'
        ),
        canjes AS (
            SELECT DISTINCT CUST_ID, REDEMPTION_DATE
            FROM `{PROJECT_CONSUMPTION}.control_de_gestion.frozen_redemption_entity`
            WHERE RETURN_FLAG IS FALSE
              AND mes >= '{fecha_inicio}'
        )
        SELECT
            c.FECHA_ENVIO,
            c.NOMBRE_CAMPANHA,
            c.GMGC,
            c.cust_id,
            MAX(CASE WHEN j.CUST_ID IS NOT NULL THEN "CANJEO" ELSE "NO CANJEO" END) AS FLAG_CANJE
        FROM campanas c
        LEFT JOIN canjes j
          ON c.cust_id = j.CUST_ID
         AND j.REDEMPTION_DATE BETWEEN c.INICIO_VIGENCIA AND c.FIN_VIGENCIA
        GROUP BY c.FECHA_ENVIO, c.NOMBRE_CAMPANHA, c.GMGC, c.cust_id
    """
    try:
        df_camp_ind = q(client_cons, sql_rfm_camp, "rfm_camp_individual")
        print(f"  Clientes únicos: {df_camp_ind['cust_id'].nunique():,}", flush=True)

        # Cargar rfm_all_clients para el join (incluir propensity_score)
        rfm_path = os.path.join(OUTPUT_DIR, "rfm_all_clients.parquet")
        if os.path.exists(rfm_path):
            rfm_cols = ["cust_id", "rfm_segment", "r_score", "f_score", "m_score"]
            try:
                import pyarrow.parquet as pq_check
                avail = pq_check.read_schema(rfm_path).names
                if "propensity_score" in avail:
                    rfm_cols.append("propensity_score")
            except Exception:
                pass
            df_rfm_all = pd.read_parquet(rfm_path, columns=rfm_cols)
            df_camp_ind = df_camp_ind.merge(df_rfm_all, on="cust_id", how="left")
            df_camp_ind["rfm_segment"] = df_camp_ind["rfm_segment"].fillna("Sin score RFM")
            print(f"  Con score RFM: {df_camp_ind['rfm_segment'].ne('Sin score RFM').sum():,} / {len(df_camp_ind):,}", flush=True)
        else:
            df_camp_ind["rfm_segment"] = "Sin score RFM"
            print("  AVISO: rfm_all_clients.parquet no encontrado — ejecutar paso [4] primero", flush=True)

        # ── Guardar datos individuales particionados por campaña ──────────────
        # Permite al dashboard leer solo la partición de la campaña seleccionada
        if "propensity_score" in df_camp_ind.columns:
            print("  Guardando rfm_campaigns_individual/ (particionado por campaña)...", flush=True)
            cols_indiv = ["NOMBRE_CAMPANHA", "GMGC", "cust_id",
                          "rfm_segment", "FLAG_CANJE", "propensity_score"]
            df_indiv = df_camp_ind[[c for c in cols_indiv if c in df_camp_ind.columns]].copy()
            df_indiv["canjeo"]           = (df_indiv["FLAG_CANJE"] == "CANJEO")
            df_indiv["propensity_score"] = df_indiv["propensity_score"].astype("float32")
            df_indiv["GMGC"]             = df_indiv["GMGC"].astype("category")
            df_indiv["rfm_segment"]      = df_indiv["rfm_segment"].astype("category")
            import pyarrow as pa_ind, pyarrow.parquet as pq_ind
            tbl_ind = pa_ind.Table.from_pandas(
                df_indiv.drop(columns=["FLAG_CANJE"]), preserve_index=False
            )
            # Limpiar metadatos pandas para evitar tipos BigQuery (ej. dbdate) al leer
            tbl_ind = tbl_ind.replace_schema_metadata({})
            out_ind = os.path.join(OUTPUT_DIR, "rfm_campaigns_individual")
            # Limpiar particiones previas si existen
            import shutil
            if os.path.exists(out_ind):
                shutil.rmtree(out_ind)
            pq_ind.write_to_dataset(
                tbl_ind,
                root_path=out_ind,
                partition_cols=["NOMBRE_CAMPANHA"],
                compression="snappy",
            )
            size_mb = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fns in os.walk(out_ind) for f in fns
            ) / 1e6
            print(f"  -> rfm_campaigns_individual/ ({len(df_indiv):,} filas, {size_mb:.0f} MB)", flush=True)
        else:
            print("  AVISO: propensity_score no disponible — rfm_campaigns_individual/ no generado", flush=True)

        # Agrupar por campaña × segmento RFM × GMGC → clientes + canjeadores
        df_rfm_camp = (
            df_camp_ind
            .groupby(["NOMBRE_CAMPANHA", "FECHA_ENVIO", "GMGC", "rfm_segment", "FLAG_CANJE"])
            .size()
            .reset_index(name="CLIENTES")
        )
        seg_dist = df_rfm_camp.groupby("rfm_segment")["CLIENTES"].sum().sort_values(ascending=False).to_dict()
        print(f"  Distribución RFM en campañas: {seg_dist}", flush=True)
        save(df_rfm_camp, "rfm_campaigns")

    except Exception as e:
        print(f"  ERROR en RFM × Campañas: {e}", flush=True)
        import traceback; traceback.print_exc()
        pd.DataFrame().to_parquet(
            os.path.join(OUTPUT_DIR, "rfm_campaigns.parquet"), index=False
        )

    print("\n" + "=" * 70)
    print("EXTRACCION COMPLETADA")
    print(f"  Archivos en: {OUTPUT_DIR}/")
    print("  Ejecuta: streamlit run dashboard.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
