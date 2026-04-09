"""
Entrena un modelo de conversión de campaña en 5 días.

Usa los mismos 13 PSM features de fase3 + features adicionales disponibles
en rfm_all_clients (tier, gender, age, flags de actividad, scores RFM).

El cascade de fase3 usa ~52 features de customer_snapshot (incluye canales de
contacto, dominant_retailer, funnel_state). Con los ~25 disponibles capturamos
la mayoría de la señal. Para acercarse a 52 habría que extender extract.py.

Target: canjeo en los primeros 5 días del envío de la campaña.

Uso:
    python train_campaign_model.py                          # últimos 3 meses
    python train_campaign_model.py --fecha-inicio 2025-10-01   # 6 meses
"""
import os, sys, time, argparse, warnings, pickle
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import pyarrow.parquet as pq_r
from google.cloud import bigquery
from sklearn.preprocessing import OrdinalEncoder

PROJECT_CONSUMPTION = "fif-loy-cl-consumption"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── Features PSM (mismos 13 de fase3) ────────────────────────────────────────
PSM_FEATURES = [
    "frequency_monthly_avg", "monetary_monthly_avg", "redeem_rate",
    "retailer_entropy",      "pct_redeem_digital",   "earn_velocity_90",
    "days_since_last_activity", "points_pressure",   "stock_points_at_t0",
    "redeem_count_pre",      "frequency_total",      "monetary_total",
    "tenure_months",
]

# ── Features adicionales de rfm_all_clients ───────────────────────────────────
# Categóricas (como en cascade de fase3)
CATEGORICAL_FEATURES = [
    "tier", "gender", "city", "rfm_segment",
    "dominant_retailer", "funnel_state_at_t0",
]

# Booleanas / flags
BOOLEAN_FEATURES = [
    "cust_active_deb_flg", "cust_active_cmr_flg",
    "contact_email_flg", "contact_phone_flg", "contact_push_flg", "cust_active_omp_flg",
]

# Numéricas adicionales (no PSM)
EXTRA_NUMERIC = [
    "age", "stock_points", "exp_points_current", "exp_points_next",
    "recency_days", "r_score", "f_score", "m_score",
    "propensity_score",       # modelo anterior como feature (stacking)
    "earn_velocity_30",
    "spend_falabella", "spend_sodimac", "spend_tottus", "spend_fcom", "spend_ikea",
    "pct_cmr_payments", "pct_debit_payments",
    "avg_redeem_points", "pct_redeem_catalogo", "pct_redeem_giftcard",
    "days_since_last_redeem",
]

# Features de contexto de campaña
CAMP_FEATURES = [
    "mes_envio",          # estacionalidad (1-12)
    "dia_semana_envio",   # lunes=0 … domingo=6
    "es_gm",              # 1=GM recibió campaña, 0=GC control
    "es_cyber_month",     # noviembre = cyber month (alta conversión)
    "es_diciembre",       # diciembre = navidad
]

# Orden final de features (categ. primero, igual que cascade)
ALL_FEATURES = (
    CATEGORICAL_FEATURES
    + BOOLEAN_FEATURES
    + PSM_FEATURES
    + EXTRA_NUMERIC
    + CAMP_FEATURES
)


def run_bq_5d(fecha_inicio, fecha_fin):
    client = bigquery.Client(project=PROJECT_CONSUMPTION)
    sql = f"""
    WITH campanas AS (
        SELECT DISTINCT
            A.FECHA_ENVIO,
            A.NOMBRE_CAMPANHA,
            A.GMGC,
            A.RUT                                      AS cust_id,
            DATE_ADD(A.FECHA_ENVIO, INTERVAL 5 DAY)    AS cutoff_5d
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
        MAX(CASE WHEN j.CUST_ID IS NOT NULL THEN TRUE ELSE FALSE END) AS canjeo_5d
    FROM campanas c
    LEFT JOIN canjes j
      ON c.cust_id   = j.CUST_ID
     AND j.REDEMPTION_DATE BETWEEN c.FECHA_ENVIO AND c.cutoff_5d
    GROUP BY c.FECHA_ENVIO, c.NOMBRE_CAMPANHA, c.GMGC, c.cust_id
    """
    t0 = time.time()
    print("  Ejecutando query BQ (ventana 5 días)...", flush=True)
    df = client.query(sql).to_dataframe(create_bqstorage_client=True)
    print(f"  {len(df):,} filas en {time.time()-t0:.1f}s", flush=True)
    df["FECHA_ENVIO"] = pd.to_datetime(df["FECHA_ENVIO"], errors="coerce")
    df["canjeo_5d"]   = df["canjeo_5d"].astype(bool)
    return df


def load_client_features():
    rfm_path = os.path.join(OUTPUT_DIR, "rfm_all_clients.parquet")
    if not os.path.exists(rfm_path):
        print("ERROR: rfm_all_clients.parquet no encontrado.", flush=True)
        sys.exit(1)

    schema_cols = set(pq_r.read_schema(rfm_path).names)
    needed = (
        ["cust_id"]
        + [c for c in CATEGORICAL_FEATURES if c in schema_cols]
        + [c for c in BOOLEAN_FEATURES     if c in schema_cols]
        + [c for c in PSM_FEATURES         if c in schema_cols]
        + [c for c in EXTRA_NUMERIC        if c in schema_cols]
        # columnas base para derivar features faltantes
        + [c for c in ["frequency_total","monetary_total","stock_points","redeem_count_pre"]
           if c in schema_cols]
    )
    needed = list(dict.fromkeys(needed))  # deduplicar preservando orden
    df = pd.read_parquet(rfm_path, columns=needed)

    # Derivar PSM features si faltan (igual que scoring_pipeline.py)
    if "frequency_monthly_avg" not in df.columns and "frequency_total" in df.columns:
        df["frequency_monthly_avg"] = df["frequency_total"] / 12.0
    if "monetary_monthly_avg" not in df.columns and "monetary_total" in df.columns:
        df["monetary_monthly_avg"]  = df["monetary_total"]  / 12.0
    if "redeem_rate" not in df.columns:
        df["redeem_rate"] = df.get("redeem_count_pre", 0) / df["frequency_total"].replace(0, 1)
    if "stock_points_at_t0" not in df.columns and "stock_points" in df.columns:
        df["stock_points_at_t0"] = df["stock_points"]

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fecha-inicio", default="2026-01-01",
        help="Inicio período de entrenamiento. Default: 2026-01-01 (últimos 3 meses)"
    )
    args = parser.parse_args()

    fecha_inicio = args.fecha_inicio
    fecha_fin    = pd.Timestamp.today().date().isoformat()

    print("=" * 70)
    print("ENTRENAMIENTO: Modelo de Canje en 5 Días")
    print(f"  Período:  {fecha_inicio} → {fecha_fin}")
    print(f"  Features: {len(PSM_FEATURES)} PSM + {len(CATEGORICAL_FEATURES)} cat + "
          f"{len(BOOLEAN_FEATURES)} bool + {len(EXTRA_NUMERIC)} num + {len(CAMP_FEATURES)} campaña "
          f"= {len(ALL_FEATURES)} total")
    print("=" * 70)

    # ── [1] Query BQ ─────────────────────────────────────────────────────────
    print("\n[1] Query BigQuery — canje en primeros 5 días", flush=True)
    df_camp = run_bq_5d(fecha_inicio, fecha_fin)

    n_total = len(df_camp)
    n_pos   = int(df_camp["canjeo_5d"].sum())
    n_camps = df_camp["NOMBRE_CAMPANHA"].nunique()
    rate_5d = n_pos / n_total * 100
    print(f"  Campañas: {n_camps} | Clientes: {df_camp['cust_id'].nunique():,}", flush=True)
    print(f"  Tasa canje 5d: {n_pos:,}/{n_total:,} = {rate_5d:.2f}%", flush=True)

    # ── [2] Cargar features de cliente ────────────────────────────────────────
    print("\n[2] Cargando features de rfm_all_clients.parquet...", flush=True)
    df_rfm = load_client_features()
    avail_feats = [c for c in ALL_FEATURES if c in df_rfm.columns]
    print(f"  {len(df_rfm):,} clientes | {len(avail_feats)}/{len(ALL_FEATURES)} features disponibles", flush=True)
    missing_f = [c for c in ALL_FEATURES if c not in df_rfm.columns and c not in CAMP_FEATURES]
    if missing_f:
        print(f"  Faltantes en rfm_all_clients (→ 0): {missing_f}", flush=True)

    # ── [3] Muestreo ANTES del merge (evitar OOM con 55M filas) ─────────────
    print("\n[3] Muestreo previo al merge (todos canjeadores + 5x no-canjeadores)...", flush=True)
    _pos_camp = df_camp[df_camp["canjeo_5d"] == True]
    _neg_camp = df_camp[df_camp["canjeo_5d"] == False]
    _n_neg_s  = min(len(_pos_camp) * 5, len(_neg_camp))
    df_camp_s = pd.concat([
        _pos_camp,
        _neg_camp.sample(n=_n_neg_s, random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Muestra campañas: {len(df_camp_s):,} filas | pos: {df_camp_s['canjeo_5d'].sum():,} ({df_camp_s['canjeo_5d'].mean()*100:.1f}%)", flush=True)
    base_rate_real = n_pos / n_total   # tasa real ANTES del muestreo (para calibración)
    del _pos_camp, _neg_camp, df_camp   # liberar memoria

    # ── [4] Merge (muestra reducida × features) ───────────────────────────────
    print("\n[4] Merge campañas × features...", flush=True)
    df = df_camp_s.merge(df_rfm, on="cust_id", how="inner")
    del df_camp_s, df_rfm               # liberar memoria
    print(f"  {len(df):,} filas ({df['canjeo_5d'].mean()*100:.2f}% tasa 5d)", flush=True)

    # Features de campaña
    df["mes_envio"]        = df["FECHA_ENVIO"].dt.month.astype(float)
    df["dia_semana_envio"] = df["FECHA_ENVIO"].dt.dayofweek.astype(float)
    df["es_gm"]            = (df["GMGC"] == "GM").astype(float)
    df["es_cyber_month"]   = (df["FECHA_ENVIO"].dt.month == 11).astype(float)
    df["es_diciembre"]     = (df["FECHA_ENVIO"].dt.month == 12).astype(float)

    # Rellenar features que no existen en df
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    # ── [5] Encodear categóricas (OrdinalEncoder igual que cascade) ───────────
    cat_in_df  = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    bool_in_df = [c for c in BOOLEAN_FEATURES     if c in df.columns]

    if cat_in_df:
        df[cat_in_df] = df[cat_in_df].astype(str).fillna("__nan__")
    if bool_in_df:
        # Convertir flags a 0/1
        for col in bool_in_df:
            df[col] = df[col].map(lambda v: 1 if str(v).upper() in ("1", "TRUE", "SI", "YES", "Y") else 0)

    ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if cat_in_df:
        df[cat_in_df] = ord_enc.fit_transform(df[cat_in_df].astype(str))

    # Cast a float32 DESPUÉS del encoding (cuando todo es numérico)
    num_cols = [c for c in ALL_FEATURES if c in df.columns]
    df[num_cols] = df[num_cols].fillna(0).astype("float32")

    df_s = df
    del df
    print(f"  {len(df_s):,} filas | positivos: {df_s['canjeo_5d'].sum():,} ({df_s['canjeo_5d'].mean()*100:.1f}%)", flush=True)

    # ── [6] Split temporal ────────────────────────────────────────────────────
    print("\n[5] Split temporal (80% train / 20% test por fecha campaña)...", flush=True)
    camp_dates  = df_s.groupby("NOMBRE_CAMPANHA")["FECHA_ENVIO"].max().sort_values()
    cutoff_date = camp_dates.iloc[int(len(camp_dates) * 0.8)]
    print(f"  Cutoff: {cutoff_date.date()} (campañas posteriores → test)", flush=True)

    mask_tr = df_s["FECHA_ENVIO"] <= cutoff_date
    mask_te = df_s["FECHA_ENVIO"] >  cutoff_date

    X_all = df_s[ALL_FEATURES].fillna(0).astype(float)
    y_all = df_s["canjeo_5d"].astype(int)

    X_tr, X_te = X_all[mask_tr], X_all[mask_te]
    y_tr, y_te = y_all[mask_tr], y_all[mask_te]
    print(f"  Train: {len(X_tr):,} ({y_tr.mean()*100:.1f}% pos) | Test: {len(X_te):,} ({y_te.mean()*100:.1f}% pos)", flush=True)

    if len(X_te) == 0 or y_te.sum() == 0:
        print("ERROR: test set vacío — ampliar --fecha-inicio", flush=True)
        sys.exit(1)

    # ── [7] Entrenar XGBoost binario (hiperparáms del cascade de fase3) ───────
    print("\n[6] Entrenando XGBoost (objective=binary, hiperparáms fase3)...", flush=True)
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    # scale_pos_weight eliminado: el muestreo 10x ya maneja el desbalance.
    # Usar scale_pos_weight sobre muestra balanceada genera doble ponderación
    # que hace que el modelo sobreestime masivamente las probabilidades.
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="auc",
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.6,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=0.01,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )

    t0 = time.time()
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    print(f"  Entrenado en {time.time()-t0:.1f}s", flush=True)

    # ── [8] Evaluar ───────────────────────────────────────────────────────────
    print("\n[7] Evaluación...", flush=True)
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    y_pred = model.predict_proba(X_te)[:, 1]

    # ── Calibración isotónica (mapea raw scores → probs calibradas en test set) ─
    from sklearn.isotonic import IsotonicRegression
    iso_cal = IsotonicRegression(out_of_bounds="clip")
    iso_cal.fit(y_pred, y_te)
    y_pred_iso = iso_cal.predict(y_pred)
    print(f"  Isotonic: mean raw={y_pred.mean():.3f} → calibrado={y_pred_iso.mean():.3f} (real={y_te.mean():.3f})", flush=True)

    # Corrección de prior: test_rate (balanceada) → real_rate (población)
    sample_rate    = float(y_pred_iso.mean())   # media calibrada isotónica
    odds           = y_pred_iso / (1 - y_pred_iso + 1e-9)
    prior_ratio    = (base_rate_real / (1 - base_rate_real + 1e-9)) / (sample_rate / (1 - sample_rate + 1e-9))
    y_pred_cal     = (odds * prior_ratio) / (1 + odds * prior_ratio)

    auc_new   = roc_auc_score(y_te, y_pred)
    ap_new    = average_precision_score(y_te, y_pred)
    brier_cal = brier_score_loss(y_te, y_pred_cal)

    # ── Diagnóstico de overfitting: AUC en train ──────────────────────────────
    y_pred_tr   = model.predict_proba(X_tr)[:, 1]
    auc_train   = roc_auc_score(y_tr, y_pred_tr)
    overfit_gap = auc_train - auc_new
    flag_ov     = "⚠️  posible overfitting" if overfit_gap > 0.05 else "✓ OK"
    print(f"  AUC Train: {auc_train:.4f} | AUC Test: {auc_new:.4f} | Gap: {overfit_gap:+.4f}  {flag_ov}", flush=True)

    print(f"\n  {'Métrica':<32} {'Valor':>8}", flush=True)
    print(f"  {'-'*42}", flush=True)
    print(f"  {'AUC-ROC  (↑ mejor)':<32} {auc_new:>8.4f}", flush=True)
    print(f"  {'Avg Precision  (↑ mejor)':<32} {ap_new:>8.4f}", flush=True)
    print(f"  {'Brier Score calibrado  (↓ mejor)':<32} {brier_cal:>8.6f}", flush=True)
    print(f"  {'Base rate real':<32} {base_rate_real:>8.4%}", flush=True)

    # Lift por decil
    df_eval = pd.DataFrame({"y": y_te.values, "score": y_pred, "cal": y_pred_cal})
    df_eval["decil"] = pd.qcut(df_eval["score"], q=10, labels=False, duplicates="drop")
    base = y_te.mean()
    lifts = df_eval.groupby("decil")["y"].mean() / base
    lift_top = lifts.iloc[-1]
    print(f"\n  Lift top decil: {lift_top:.1f}x  (top 10% clientes convierten {lift_top:.1f}x más)", flush=True)

    # Tabla de calibración
    print(f"\n  Calibración por decil:", flush=True)
    print(f"  {'D':>3} {'N':>8} {'Predicho%':>10} {'Real%':>8} {'Gap(pp)':>9}", flush=True)
    print(f"  {'-'*42}", flush=True)
    cal_tbl = df_eval.groupby("decil").agg(cal_avg=("cal","mean"), real=("y","mean"), n=("y","count")).reset_index()
    for _, r in cal_tbl.iterrows():
        gap = (r["cal_avg"] - r["real"]) * 100
        flag = "✓" if abs(gap) <= 5 else ("~" if abs(gap) <= 10 else "✗")
        print(f"  {int(r['decil'])+1:>3} {int(r['n']):>8,} {r['cal_avg']*100:>9.1f}% {r['real']*100:>7.2f}% {gap:>+8.1f} {flag}", flush=True)

    # Feature importance
    feat_imp = pd.Series(model.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
    print(f"\n  Top 15 features:", flush=True)
    print(f"  {'Feature':<30} {'Importancia':>12}", flush=True)
    print(f"  {'-'*44}", flush=True)
    for feat, imp in feat_imp.head(15).items():
        bar = "█" * max(1, int(imp * 400))
        print(f"  {feat:<30} {imp:>12.4f}  {bar}", flush=True)

    # ── [9] Guardar modelo ────────────────────────────────────────────────────
    model_path = os.path.join(OUTPUT_DIR, "campaign_model.pkl")
    model_data = {
        "model":              model,
        "iso_calibrator":     iso_cal,
        "ordinal_encoder":    ord_enc if cat_in_df else None,
        "cat_features":       cat_in_df,
        "bool_features":      bool_in_df,
        "features":           ALL_FEATURES,
        "psm_features":       PSM_FEATURES,
        "camp_features":      CAMP_FEATURES,
        "feature_importance": feat_imp.to_dict(),
        "base_rate_real":     float(base_rate_real),
        "prior_ratio":        float(prior_ratio),
        "metrics": {
            "auc":            float(auc_new),
            "auc_train":      float(auc_train),
            "overfit_gap":    float(overfit_gap),
            "avg_precision":  float(ap_new),
            "brier_cal":      float(brier_cal),
            "lift_top_decil": float(lift_top),
        },
        "train_config": {
            "fecha_inicio": fecha_inicio,
            "fecha_fin":    fecha_fin,
            "n_campanas":   int(n_camps),
            "n_train":      int(len(X_tr)),
            "n_test":       int(len(X_te)),
            "n_features":   len(ALL_FEATURES),
            "cutoff_date":  cutoff_date.isoformat(),
        },
        "trained_at": pd.Timestamp.now().isoformat(),
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    size_kb = os.path.getsize(model_path) / 1024
    print(f"\n{'='*70}", flush=True)
    print(f"MODELO GUARDADO: {model_path} ({size_kb:.0f} KB)", flush=True)
    print(f"  AUC: {auc_new:.4f} | Lift top decil: {lift_top:.1f}x | Features: {len(ALL_FEATURES)}", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
