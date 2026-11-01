from __future__ import annotations

from pathlib import Path
from datetime import date, timedelta
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# Demand Intelligence — retail demand forecasting dashboard
# Uses real project files. Inventory recommendations are simulations.
# ============================================================

st.set_page_config(
    page_title="Demand Intelligence | Retail AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
POSSIBLE_BASES = [
    APP_DIR,
    Path("/content/drive/MyDrive/AI_Demand_Forecasting_RELEASE"),
    Path("/content/drive/MyDrive/AI_Demand_Forecasting_Project"),
]
BASE_DIR = next(
    (p for p in POSSIBLE_BASES if (p / "data" / "sales_clean.parquet").exists()),
    APP_DIR,
)
DATA_PATHS = [
    BASE_DIR / "data" / "sales_clean.parquet",
    BASE_DIR / "dataset" / "processed" / "sales_clean.parquet",
    BASE_DIR / "dataset" / "processed" / "sales_clean.csv",
    BASE_DIR / "data" / "sales_clean.csv",
]
MODEL_DIRS = [BASE_DIR / "models", BASE_DIR.parent / "AI_Demand_Forecasting_Project" / "models"]
REPORT_DIRS = [BASE_DIR / "reports", BASE_DIR.parent / "AI_Demand_Forecasting_RESEARCH_V2" / "reports"]
FORECAST_DIR = BASE_DIR / "generated" / "forecasts"
SETTINGS_DIR = BASE_DIR / "generated" / "settings"
EXPORT_DIR = BASE_DIR / "generated" / "reports"

FEATURES = [
    "store_nbr", "family", "year", "month", "day", "day_of_week",
    "day_of_year", "week_of_year", "is_weekend", "onpromotion",
    "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "sales_rolling_mean_7", "sales_rolling_mean_28", "sales_rolling_std_7",
]
C = {
    "teal": "#176B67", "dark": "#104C49", "beige": "#F4EBDD",
    "cream": "#FCF9F3", "ink": "#263735", "muted": "#667773",
    "border": "#D8D2C5", "blue": "#0875D1", "red": "#B7473E",
    "orange": "#C77D32", "green": "#39785B",
}
RISK_ORDER = ["Critical", "High", "Medium", "Low"]

st.markdown(
    f"""
    <style>
    :root {{ --primary-color: {C['teal']}; }}
    .stApp {{ background: {C['beige']}; color: {C['ink']}; }}
    [data-testid="stHeader"] {{ background: rgba(244,235,221,.94); }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {C['dark']} 0%, {C['teal']} 55%, #205E59 100%);
        border-right: 1px solid #0c4541;
    }}
    [data-testid="stSidebar"] * {{ color: #F8F5ED; }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: #E2F0EC;
    }}
    [data-testid="stMetric"] {{
        background: {C['cream']}; border: 1px solid {C['border']};
        border-radius: 14px; padding: 14px 17px;
    }}
    [data-testid="stMetricLabel"] {{ color: {C['muted']}; }}
    .brand-card {{
        background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.18);
        border-radius: 16px; padding: 17px 15px; margin: 4px 0 14px 0;
    }}
    .brand-title {{ font-size: 1.12rem; font-weight: 750; color: white; }}
    .brand-sub {{ font-size: .69rem; letter-spacing: .12em; color: #D7E9E4; }}
    .hero {{
        background: linear-gradient(115deg, {C['dark']}, {C['teal']});
        color: white; padding: 25px 29px; border-radius: 17px; margin-bottom: 18px;
    }}
    .hero h1 {{ color: white; margin: 0 0 6px 0; font-size: 1.9rem; }}
    .hero p {{ color: #E7F3EF; margin: 0; }}
    .section-note {{
        background: #DDEFEA; border-left: 4px solid {C['teal']};
        border-radius: 7px; padding: 10px 13px; margin: 8px 0 16px 0;
    }}
    .muted {{ color: {C['muted']}; font-size: .88rem; }}
    .stButton > button, .stDownloadButton > button {{
        border-radius: 9px; font-weight: 600;
    }}
    h1, h2, h3 {{ color: {C['dark']}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Paths / data loading ----------

def find_data_path():
    for p in DATA_PATHS:
        if p.exists():
            return p
    return None

def find_report(name: str):
    for root in REPORT_DIRS:
        candidate = root / name
        if candidate.exists():
            return candidate
        # Search one level down for commonly nested research reports.
        if root.exists():
            matches = list(root.glob(f"**/{name}"))
            if matches:
                return matches[0]
    return None

def find_model_files():
    found = {}
    for root in MODEL_DIRS:
        if not root.exists():
            continue
        for p in root.glob("*.joblib"):
            key = p.stem.lower()
            # Prefer explicitly selected best model if duplicate filenames exist.
            if key not in found or "best" in p.stem:
                found[key] = p
    return found

@st.cache_data(show_spinner="Loading and validating historical sales...")
def load_sales(path_str: str, mtime: float):
    path = Path(path_str)
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    needed = {"date", "store_nbr", "family", "sales"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["store_nbr"] = pd.to_numeric(df["store_nbr"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df["family"] = df["family"].astype("string").str.strip()
    if "onpromotion" not in df:
        df["onpromotion"] = 0
    df["onpromotion"] = pd.to_numeric(df["onpromotion"], errors="coerce").fillna(0)
    df = df.dropna(subset=["date", "store_nbr", "family", "sales"])
    df["store_nbr"] = df["store_nbr"].astype(int)
    df["family"] = df["family"].astype(str)
    df["sales"] = df["sales"].clip(lower=0)
    df = (
        df.groupby(["date", "store_nbr", "family"], as_index=False, observed=True)
        .agg(sales=("sales", "sum"), onpromotion=("onpromotion", "sum"))
        .sort_values(["date", "store_nbr", "family"])
        .reset_index(drop=True)
    )
    return df

@st.cache_resource(show_spinner="Loading trained model pipeline...")
def load_model(path_str: str, mtime: float):
    model = joblib.load(path_str)
    if not hasattr(model, "predict"):
        raise TypeError("Saved model does not provide predict().")
    return model

def read_csv_report(filename):
    p = find_report(filename)
    if p is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def read_json_report(filename):
    p = find_report(filename)
    if p is None:
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

DATA_PATH = find_data_path()
MODEL_FILES = find_model_files()
if DATA_PATH is None:
    st.error("Historical sales data was not found. Expected sales_clean.parquet or sales_clean.csv in the app data folder.")
    st.info("For deployment, place the cleaned dataset at data/sales_clean.parquet relative to app.py.")
    st.stop()

sales_all = load_sales(str(DATA_PATH), DATA_PATH.stat().st_mtime)
if sales_all.empty:
    st.error("The historical sales dataset contains no usable rows.")
    st.stop()

last_history_date = pd.Timestamp(sales_all["date"].max()).normalize()
first_history_date = pd.Timestamp(sales_all["date"].min()).normalize()
stores_all = sorted(sales_all["store_nbr"].unique().tolist())
families_all = sorted(sales_all["family"].unique().tolist())

# Model selection only lists files that actually exist. No fake alternatives.
if not MODEL_FILES:
    st.error("No trained .joblib model was found. Add the verified model to a models/ folder.")
    st.stop()

def model_label(key, path):
    k = key.lower()
    if "xgboost" in k or k == "best_demand_model":
        return f"XGBoost — {path.name}"
    if "random_forest" in k:
        return f"Random Forest — {path.name}"
    if "lightgbm" in k:
        return f"LightGBM — {path.name}"
    if "catboost" in k:
        return f"CatBoost — {path.name}"
    if "extra_trees" in k:
        return f"Extra Trees — {path.name}"
    return path.stem.replace("_", " ").title()

model_options = {model_label(k, p): p for k, p in sorted(MODEL_FILES.items())}
preferred_label = next((label for label, path in model_options.items() if "best_demand_model" in path.name.lower()), None)
if preferred_label is None:
    preferred_label = next((label for label, path in model_options.items() if "xgboost" in path.name.lower()), next(iter(model_options)))

# Sidebar brand and navigation
with st.sidebar:
    st.markdown(
        """
        <div class="brand-card">
          <div style="font-size:1.55rem">◈ &nbsp;<span class="brand-title">Demand Intelligence</span></div>
          <div class="brand-sub">RETAIL AI · FORECASTING</div>
          <hr style="border-color:#7AA7A0">
          <div style="font-size:.83rem">Demand planning, inventory signals and decision support in one place.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("↻ Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.markdown("### WORKSPACE")
    pages = [
        "Executive Overview", "Demand Forecasting", "Inventory Optimizer",
        "Smart Alerts", "Sales Explorer", "What-if Simulator",
        "Model Evaluation", "Store & Product Analytics",
        "Business Intelligence", "Data Quality & Explorer",
    ]
    page = st.radio("Navigate", pages, label_visibility="collapsed")
    st.markdown("---")
    st.caption("PROJECT STATUS")
    st.write("**Model:** Verified saved pipeline")
    st.write(f"**Stores:** {len(stores_all)}")
    st.write(f"**Product families:** {len(families_all)}")
    st.write(f"**Historical end:** {last_history_date:%d %b %Y}")
    st.caption("Research prototype · Inventory outputs are simulated")

# Global filter values live in the Executive Overview, as requested.
# Persist them in session state for other pages.
if "global_stores" not in st.session_state:
    st.session_state.global_stores = stores_all
if "global_families" not in st.session_state:
    st.session_state.global_families = families_all
if "global_start" not in st.session_state:
    st.session_state.global_start = first_history_date.date()
if "global_end" not in st.session_state:
    st.session_state.global_end = last_history_date.date()

selected_stores = [s for s in st.session_state.global_stores if s in stores_all]
selected_families = [f for f in st.session_state.global_families if f in families_all]
if not selected_stores:
    selected_stores = stores_all
if not selected_families:
    selected_families = families_all

start_date = pd.Timestamp(st.session_state.global_start)
end_date = pd.Timestamp(st.session_state.global_end)
if start_date > end_date:
    start_date, end_date = first_history_date, last_history_date

filtered = sales_all[
    sales_all["store_nbr"].isin(selected_stores)
    & sales_all["family"].isin(selected_families)
    & sales_all["date"].between(start_date, end_date)
].copy()

# ---------- Forecasting engine ----------

def make_features(store, family, forecast_date, history, promotion=0.0):
    def lag(n):
        d = forecast_date - pd.Timedelta(days=n)
        return float(history.get(d, 0.0))
    last7 = [float(history.get(forecast_date - pd.Timedelta(days=i), 0.0)) for i in range(1, 8)]
    last28 = [float(history.get(forecast_date - pd.Timedelta(days=i), 0.0)) for i in range(1, 29)]
    dow = int(forecast_date.dayofweek)
    return {
        "store_nbr": int(store),
        "family": str(family),
        "year": int(forecast_date.year),
        "month": int(forecast_date.month),
        "day": int(forecast_date.day),
        "day_of_week": dow,
        "day_of_year": int(forecast_date.dayofyear),
        "week_of_year": int(forecast_date.isocalendar().week),
        "is_weekend": int(dow >= 5),
        "onpromotion": float(promotion),
        "sales_lag_7": lag(7),
        "sales_lag_14": lag(14),
        "sales_lag_28": lag(28),
        "sales_rolling_mean_7": float(np.mean(last7)),
        "sales_rolling_mean_28": float(np.mean(last28)),
        "sales_rolling_std_7": float(np.std(last7, ddof=1)) if len(last7) > 1 else 0.0,
    }

@st.cache_data(show_spinner="Generating forecasts (batched model prediction)...", ttl=3600, max_entries=12)
def generate_forecast(
    data_fingerprint: str,
    _data: pd.DataFrame,
    model_path: str,
    model_mtime: float,
    horizon: int,
    stores: tuple,
    families: tuple,
    promotion_mode: str,
):
    """
    Generate recursive forecasts efficiently.

    The previous implementation called model.predict once for every
    store-family pair on every forecast day. With 54 stores and 33 families,
    that can mean 12,000+ tiny model calls for a 7-day forecast. This version
    makes one batch prediction per forecast day while preserving recursive
    lag updates.
    """
    model = load_model(model_path, model_mtime)
    sub = _data[
        _data["store_nbr"].isin(stores) & _data["family"].isin(families)
    ].copy()

    daily = (
        sub.groupby(["date", "store_nbr", "family"], as_index=False, observed=True)
        .agg(sales=("sales", "sum"), onpromotion=("onpromotion", "sum"))
    )
    all_dates = pd.date_range(first_history_date, last_history_date, freq="D")
    indexed = daily.set_index(["store_nbr", "family", "date"]).sort_index()

    # Prepare the history and recent promotion series for each requested pair.
    pairs = []
    for store in stores:
        for family in families:
            try:
                hist = indexed.loc[(int(store), str(family))]
                history_series = hist["sales"]
                promo_series = hist["onpromotion"]
            except KeyError:
                history_series = pd.Series(dtype=float)
                promo_series = pd.Series(dtype=float)

            history_series.index = pd.to_datetime(history_series.index)
            history_series = history_series.groupby(level=0).sum().reindex(all_dates, fill_value=0.0)
            promo_series.index = pd.to_datetime(promo_series.index)
            promo_series = promo_series.groupby(level=0).sum().reindex(all_dates, fill_value=0.0)

            pairs.append({
                "store": int(store),
                "family": str(family),
                "history": {pd.Timestamp(d): float(v) for d, v in history_series.items()},
                "promo_recent_7": float(promo_series.tail(7).max()) if len(promo_series) else 0.0,
                "promo_recent_28": float(promo_series.tail(28).mean()) if len(promo_series) else 0.0,
            })

    rows = []
    forecast_start = last_history_date + pd.Timedelta(days=1)

    for step in range(int(horizon)):
        forecast_date = forecast_start + pd.Timedelta(days=step)
        feature_rows = []
        for pair in pairs:
            if promotion_mode == "Use recent average":
                promo = pair["promo_recent_28"]
            elif promotion_mode == "Zero promotion":
                promo = 0.0
            else:
                promo = pair["promo_recent_7"]

            feature_rows.append(
                make_features(
                    pair["store"], pair["family"], forecast_date,
                    pair["history"], promo
                )
            )

        if not feature_rows:
            return pd.DataFrame()

        X = pd.DataFrame(feature_rows, columns=FEATURES)
        predictions = np.asarray(model.predict(X), dtype=float).reshape(-1)
        if len(predictions) != len(pairs):
            raise ValueError(
                f"Model returned {len(predictions)} predictions for {len(pairs)} store-family pairs."
            )

        for pair, raw_pred in zip(pairs, predictions):
            pred = max(0.0, float(raw_pred)) if np.isfinite(raw_pred) else 0.0
            rows.append({
                "forecast_date": forecast_date,
                "store_nbr": pair["store"],
                "family": pair["family"],
                "forecast_demand": pred,
                "raw_prediction": float(raw_pred) if np.isfinite(raw_pred) else np.nan,
                "onpromotion_assumption": (
                    pair["promo_recent_28"] if promotion_mode == "Use recent average"
                    else 0.0 if promotion_mode == "Zero promotion"
                    else pair["promo_recent_7"]
                ),
                "horizon_day": step + 1,
                "model": Path(model_path).stem,
            })
            # Recursive update: tomorrow's lag features use today's prediction.
            pair["history"][forecast_date] = pred

    result = pd.DataFrame(rows)
    if not result.empty:
        result["forecast_date"] = pd.to_datetime(result["forecast_date"])
    return result

def current_forecast(horizon=None, stores=None, families=None, promotion_mode="Use recent average"):
    horizon = int(horizon or st.session_state.get("forecast_horizon", 7))
    stores = tuple(stores or selected_stores)
    families = tuple(families or selected_families)
    model_path = st.session_state.get("selected_model_path", str(model_options[preferred_label]))
    mtime = Path(model_path).stat().st_mtime
    fingerprint = f"{DATA_PATH}:{DATA_PATH.stat().st_mtime}:{len(sales_all)}"
    return generate_forecast(
        fingerprint, sales_all, model_path, mtime, horizon, stores, families, promotion_mode
    )

def metric_wape(actual, predicted):
    denom = float(np.abs(np.asarray(actual, dtype=float)).sum())
    return float(np.abs(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)).sum() / denom * 100) if denom else np.nan

def metrics_from_backtest(df):
    if df.empty or not {"actual_sales", "predicted_sales"}.issubset(df.columns):
        return None
    a = pd.to_numeric(df["actual_sales"], errors="coerce").to_numpy()
    p = pd.to_numeric(df["predicted_sales"], errors="coerce").to_numpy()
    ok = np.isfinite(a) & np.isfinite(p)
    a, p = a[ok], p[ok]
    if not len(a):
        return None
    return {
        "MAE": float(np.mean(np.abs(a-p))),
        "RMSE": float(np.sqrt(np.mean((a-p)**2))),
        "WAPE": metric_wape(a, p),
        "Bias (predicted − actual)": float(np.mean(p-a)),
        "Rows": int(len(a)),
    }

def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path

def money(v):
    return f"{float(v):,.0f}"

def show_hero(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )

def show_note(text):
    st.markdown(f'<div class="section-note">{text}</div>', unsafe_allow_html=True)

def chart_layout(fig, height=390):
    fig.update_layout(
        height=height, paper_bgcolor=C["beige"], plot_bgcolor=C["beige"],
        font=dict(color=C["ink"]), margin=dict(l=18, r=18, t=45, b=20),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#E4DCCF", zeroline=False)
    st.plotly_chart(fig, use_container_width=True)

# ---------- Shared report data ----------
model_comparison = read_csv_report("model_comparison.csv")
if model_comparison.empty:
    model_comparison = read_csv_report("validated_model_metrics.csv")
backtest = read_csv_report("stage7_recursive_predictions.csv")
cross_origin = read_csv_report("stage15_cross_origin_overall_comparison.csv")
horizon_backtest = read_csv_report("stage15_cross_origin_by_horizon.csv")
readiness_summary = read_json_report("stage15_final_readiness_summary.json")

# ---------- Pages ----------

if page == "Executive Overview":
    show_hero("Demand Intelligence", "AI-powered demand forecasting and inventory planning for supermarket retail.")
    st.markdown("### Global filters")
    st.caption("Choose the historical scope used by the dashboard. Filters are shared across pages.")

    with st.container(border=True):
        f1, f2 = st.columns(2)
        with f1:
            new_stores = st.multiselect(
                "Stores", stores_all, default=selected_stores, key="overview_store_filter"
            )
        with f2:
            new_families = st.multiselect(
                "Product families", families_all, default=selected_families, key="overview_family_filter"
            )
        f3, f4, f5 = st.columns([1, 1, 1])
        with f3:
            new_start = st.date_input(
                "Historical start date", value=start_date.date(),
                min_value=first_history_date.date(), max_value=last_history_date.date(),
                key="overview_start_filter",
            )
        with f4:
            new_end = st.date_input(
                "Historical end date", value=end_date.date(),
                min_value=first_history_date.date(), max_value=last_history_date.date(),
                key="overview_end_filter",
            )
        with f5:
            st.selectbox("Forecast model", list(model_options.keys()), index=list(model_options.keys()).index(preferred_label), key="overview_model")
        if st.button("Apply dashboard filters", type="primary", use_container_width=True):
            st.session_state.global_stores = new_stores or stores_all
            st.session_state.global_families = new_families or families_all
            st.session_state.global_start = new_start
            st.session_state.global_end = new_end
            chosen_label = st.session_state.overview_model
            st.session_state.selected_model_path = str(model_options[chosen_label])
            st.rerun()

    # Model selector can be changed dynamically without needing a separate page.
    if "overview_model" in st.session_state:
        st.session_state.selected_model_path = str(model_options[st.session_state.overview_model])
    else:
        st.session_state.selected_model_path = str(model_options[preferred_label])

    forecast_horizon = st.selectbox("Forecast horizon", [7, 14, 30, 60], index=0, format_func=lambda x: f"{x} days", key="overview_horizon")
    st.session_state.forecast_horizon = forecast_horizon
    st.caption(
        f"Forecast scope: {len(selected_stores)} stores × {len(selected_families)} product families. "
        "Forecasts are generated in batches and cached for repeated dashboard navigation."
    )

    actual_total = float(filtered["sales"].sum()) if not filtered.empty else 0.0
    daily_avg = float(filtered.groupby("date")["sales"].sum().mean()) if not filtered.empty else 0.0
    try:
        fc = current_forecast(forecast_horizon)
        fc_total = float(fc["forecast_demand"].sum()) if not fc.empty else 0.0
        model = load_model(
            st.session_state.selected_model_path,
            Path(st.session_state.selected_model_path).stat().st_mtime
        )
    except Exception as exc:
        st.error("Forecast generation or model loading failed.")
        st.exception(exc)
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Historical sales in scope", money(actual_total))
    m2.metric("Average daily sales", money(daily_avg))
    m3.metric(f"Forecast demand · {forecast_horizon} days", money(fc_total))
    m4.metric("Stores / families", f"{len(selected_stores)} / {len(selected_families)}")

    st.caption(
        f"Historical scope: {start_date:%d %b %Y} – {end_date:%d %b %Y}. "
        f"Forecast starts {last_history_date + pd.Timedelta(days=1):%d %b %Y}; "
        "future sales are model estimates, not observed results."
    )

    st.markdown("### Forecast demand timeline")
    if not fc.empty:
        daily_fc = fc.groupby("forecast_date", as_index=False)["forecast_demand"].sum()
        fig = px.area(daily_fc, x="forecast_date", y="forecast_demand", markers=True)
        fig.update_traces(line_color=C["blue"], fillcolor="rgba(8,117,209,0.48)")
        fig.update_yaxes(title="Predicted sales")
        fig.update_xaxes(title="Forecast date")
        chart_layout(fig, 380)
        family_fc = fc.groupby("family", as_index=False)["forecast_demand"].sum().sort_values("forecast_demand", ascending=False).head(15)
        fig2 = px.bar(family_fc.sort_values("forecast_demand"), x="forecast_demand", y="family", orientation="h", title="Forecast contribution by product family")
        fig2.update_traces(marker_color=C["blue"])
        fig2.update_xaxes(title="Predicted sales")
        fig2.update_yaxes(title="")
        chart_layout(fig2, 500)
        st.download_button(
            "Download current forecast CSV",
            fc.to_csv(index=False).encode("utf-8"),
            file_name=f"demand_forecast_{forecast_horizon}d.csv",
            mime="text/csv",
        )

elif page == "Demand Forecasting":
    show_hero("Demand Forecasting", "Generate recursive daily forecasts from the saved model pipeline.")
    col1, col2, col3 = st.columns(3)
    with col1:
        horizon = st.selectbox("Forecast horizon", [7, 14, 30, 60], key="demand_horizon")
    with col2:
        promo_mode = st.selectbox("Promotion assumption", ["Use recent average", "Zero promotion", "Recent maximum"], key="demand_promo")
    with col3:
        st.metric("Forecast model", Path(st.session_state.get("selected_model_path", str(model_options[preferred_label]))).stem)
    fc = current_forecast(horizon, promotion_mode=promo_mode)
    if fc.empty:
        st.warning("No forecast rows were generated for the current filters.")
    else:
        daily = fc.groupby("forecast_date", as_index=False)["forecast_demand"].sum()
        a, b, c = st.columns(3)
        a.metric("Forecast total", money(fc["forecast_demand"].sum()))
        b.metric("Daily average", money(daily["forecast_demand"].mean()))
        c.metric("Forecast rows", f"{len(fc):,}")
        fig = px.line(daily, x="forecast_date", y="forecast_demand", markers=True, title="Daily predicted demand")
        fig.update_traces(line_color=C["blue"])
        chart_layout(fig, 400)
        fig = px.bar(fc.groupby("store_nbr", as_index=False)["forecast_demand"].sum().sort_values("forecast_demand", ascending=False), x="store_nbr", y="forecast_demand", title="Forecast demand by store")
        fig.update_traces(marker_color=C["blue"])
        chart_layout(fig, 400)
        st.dataframe(fc.sort_values("forecast_demand", ascending=False), use_container_width=True, hide_index=True)
        st.download_button("Download forecast", fc.to_csv(index=False).encode(), f"forecast_{horizon}d.csv", "text/csv")
        if st.button("Save this forecast to project storage", type="primary"):
            path = save_csv(fc, FORECAST_DIR / f"forecast_{horizon}d_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv")
            st.success(f"Forecast saved to {path}")

elif page == "Inventory Optimizer":
    show_hero("Inventory Optimizer", "Forecast-driven replenishment planning with explicit, configurable assumptions.")
    show_note("SIMULATED INVENTORY: the dataset does not contain live stock-on-hand, supplier lead times, purchase orders, or SKU-level records. Recommendations are planning estimates, not live inventory facts.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        lead_days = st.number_input("Assumed supplier lead time (days)", min_value=1, max_value=60, value=7)
    with c2:
        review_days = st.number_input("Review cycle (days)", min_value=1, max_value=30, value=7)
    with c3:
        service_level = st.selectbox("Target service level", [0.90, 0.95, 0.975, 0.99], index=1, format_func=lambda x: f"{x:.1%}")
    with c4:
        assumed_stock_days = st.number_input("Assumed stock cover (days)", min_value=0, max_value=120, value=7)
    z_lookup = {0.90: 1.282, 0.95: 1.645, 0.975: 1.960, 0.99: 2.326}
    horizon = min(60, max(14, int(lead_days + review_days + 7)))
    fc = current_forecast(horizon)
    if not fc.empty:
        grp = fc.groupby(["store_nbr", "family"]).agg(
            forecast_total=("forecast_demand", "sum"),
            daily_mean=("forecast_demand", "mean"),
            daily_std=("forecast_demand", "std"),
        ).reset_index()
        grp["daily_std"] = grp["daily_std"].fillna(0)
        grp["safety_stock_simulated"] = z_lookup[service_level] * grp["daily_std"] * np.sqrt(lead_days)
        grp["reorder_point_simulated"] = grp["daily_mean"] * lead_days + grp["safety_stock_simulated"]
        grp["assumed_stock_simulated"] = grp["daily_mean"] * assumed_stock_days
        grp["suggested_order_simulated"] = np.maximum(
            0, grp["reorder_point_simulated"] + grp["daily_mean"] * review_days - grp["assumed_stock_simulated"]
        )
        grp["risk"] = np.select(
            [
                grp["assumed_stock_simulated"] <= grp["daily_mean"] * lead_days,
                grp["assumed_stock_simulated"] < grp["reorder_point_simulated"],
            ],
            ["High", "Medium"],
            default="Low",
        )
        x1, x2, x3 = st.columns(3)
        x1.metric("High simulated-risk pairs", int((grp["risk"] == "High").sum()))
        x2.metric("Pairs below reorder point", int((grp["assumed_stock_simulated"] < grp["reorder_point_simulated"]).sum()))
        x3.metric("Suggested simulated order units", money(grp["suggested_order_simulated"].sum()))
        fig = px.bar(grp.groupby("risk", as_index=False).size(), x="risk", y="size", title="Simulated risk distribution", category_orders={"risk": RISK_ORDER}, color="risk", color_discrete_map={"High": C["red"], "Medium": C["orange"], "Low": C["green"]})
        chart_layout(fig, 330)
        st.dataframe(grp.sort_values("suggested_order_simulated", ascending=False), use_container_width=True, hide_index=True)
        st.download_button("Download simulated inventory plan", grp.to_csv(index=False).encode(), "simulated_inventory_plan.csv", "text/csv")
        if st.button("Save inventory assumptions and plan"):
            SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
            settings = {
                "lead_time_days": lead_days, "review_cycle_days": review_days,
                "service_level": service_level, "assumed_stock_cover_days": assumed_stock_days,
                "inventory_status": "SIMULATED — not live stock data",
                "saved_at": pd.Timestamp.now().isoformat(),
            }
            save_json(settings, SETTINGS_DIR / "inventory_assumptions.json")
            save_csv(grp, EXPORT_DIR / f"simulated_inventory_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv")
            st.success("Assumptions and simulated plan saved to generated/.")

elif page == "Smart Alerts":
    show_hero("Smart Alerts", "Risk signals recalculated from the selected forecast and your planning thresholds.")
    a, b, c = st.columns(3)
    with a:
        horizon = st.selectbox("Forecast horizon", [7, 14, 30, 60], key="alerts_horizon")
    with b:
        high_threshold = st.number_input("High-risk daily demand threshold", min_value=0.0, value=1000.0, step=100.0)
    with c:
        low_threshold = st.number_input("Low-demand alert threshold", min_value=0.0, value=10.0, step=1.0)
    fc = current_forecast(horizon)
    if not fc.empty:
        by_pair = fc.groupby(["store_nbr", "family"]).agg(
            total_forecast=("forecast_demand", "sum"),
            avg_daily=("forecast_demand", "mean"),
            peak_daily=("forecast_demand", "max"),
        ).reset_index()
        by_pair["alert"] = np.select(
            [by_pair["peak_daily"] >= high_threshold, by_pair["avg_daily"] <= low_threshold],
            ["High forecast demand", "Low forecast demand"],
            default="Monitor",
        )
        by_pair["priority"] = np.select(
            [by_pair["alert"] == "High forecast demand", by_pair["alert"] == "Low forecast demand"],
            ["High", "Medium"], default="Low"
        )
        alerts = by_pair[by_pair["alert"] != "Monitor"].sort_values(["priority", "total_forecast"], ascending=[True, False])
        m1, m2, m3 = st.columns(3)
        m1.metric("Triggered alerts", len(alerts))
        m2.metric("High demand alerts", int((alerts["alert"] == "High forecast demand").sum()))
        m3.metric("Low demand alerts", int((alerts["alert"] == "Low forecast demand").sum()))
        if alerts.empty:
            st.success("No alert conditions were triggered under the current thresholds.")
        else:
            st.dataframe(alerts, use_container_width=True, hide_index=True)
            st.download_button("Download action list", alerts.to_csv(index=False).encode(), "smart_alert_action_list.csv", "text/csv")
            if st.button("Save current alert list"):
                path = save_csv(alerts, EXPORT_DIR / f"smart_alerts_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv")
                st.success(f"Saved to {path}")

elif page == "Sales Explorer":
    show_hero("Sales Explorer", "Explore observed historical sales using the current global scope.")
    if filtered.empty:
        st.warning("No historical rows match the current global filters.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Historical sales", money(filtered["sales"].sum()))
        c2.metric("Daily average", money(filtered.groupby("date")["sales"].sum().mean()))
        c3.metric("Records in view", f"{len(filtered):,}")
        daily = filtered.groupby("date", as_index=False)["sales"].sum()
        fig = px.line(daily, x="date", y="sales", title="Observed daily sales")
        fig.update_traces(line_color=C["blue"])
        chart_layout(fig, 390)
        family = filtered.groupby("family", as_index=False)["sales"].sum().sort_values("sales", ascending=False).head(20)
        fig = px.bar(family.sort_values("sales"), x="sales", y="family", orientation="h", title="Top product families by observed sales")
        fig.update_traces(marker_color=C["blue"])
        chart_layout(fig, 500)
        store = filtered.groupby("store_nbr", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
        fig = px.bar(store, x="store_nbr", y="sales", title="Observed sales by store")
        fig.update_traces(marker_color=C["blue"])
        chart_layout(fig, 380)
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.download_button("Download filtered historical sales", filtered.to_csv(index=False).encode(), "filtered_historical_sales.csv", "text/csv")

elif page == "What-if Simulator":
    show_hero("What-if Simulator", "Estimate demand changes under transparent promotion and demand-shock assumptions.")
    show_note("Scenario outputs are simulated. They are not causal estimates; promotion response is not learned from a dedicated causal model.")
    c1, c2, c3 = st.columns(3)
    with c1:
        horizon = st.selectbox("Forecast horizon", [7, 14, 30, 60], key="whatif_horizon")
    with c2:
        demand_change = st.slider("Demand shock", -50, 100, 0, step=5, format="%d%%")
    with c3:
        promotion_change = st.slider("Promotion proxy adjustment", 0, 100, 0, step=5, format="%d%%")
    fc = current_forecast(horizon)
    if not fc.empty:
        baseline = float(fc["forecast_demand"].sum())
        # Transparent scenario arithmetic, not a model retraining or causal claim.
        scenario_factor = max(0, 1 + demand_change / 100) * (1 + promotion_change / 100)
        scenario_total = baseline * scenario_factor
        a, b, c = st.columns(3)
        a.metric("Baseline model forecast", money(baseline))
        b.metric("Scenario forecast", money(scenario_total), delta=f"{scenario_total-baseline:,.0f}")
        c.metric("Scenario change", f"{(scenario_factor-1)*100:.1f}%")
        scenario = fc[["forecast_date", "store_nbr", "family", "forecast_demand"]].copy()
        scenario["baseline_demand"] = scenario["forecast_demand"]
        scenario["scenario_demand_simulated"] = scenario["baseline_demand"] * scenario_factor
        daily = scenario.groupby("forecast_date", as_index=False)[["baseline_demand", "scenario_demand_simulated"]].sum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["forecast_date"], y=daily["baseline_demand"], mode="lines+markers", name="Baseline"))
        fig.add_trace(go.Scatter(x=daily["forecast_date"], y=daily["scenario_demand_simulated"], mode="lines+markers", name="What-if scenario"))
        chart_layout(fig, 390)
        st.dataframe(scenario, use_container_width=True, hide_index=True)
        st.download_button("Download scenario results", scenario.to_csv(index=False).encode(), "what_if_scenario.csv", "text/csv")
        if st.button("Save what-if scenario"):
            save_csv(scenario, EXPORT_DIR / f"what_if_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv")
            save_json({
                "horizon_days": horizon, "demand_shock_percent": demand_change,
                "promotion_proxy_adjustment_percent": promotion_change,
                "scenario_factor": scenario_factor,
                "label": "SIMULATED WHAT-IF; not a causal estimate",
                "saved_at": pd.Timestamp.now().isoformat(),
            }, SETTINGS_DIR / "last_what_if_scenario.json")
            st.success("Scenario and assumptions saved.")

elif page == "Model Evaluation":
    show_hero("Model Evaluation", "Use saved validation and chronological backtesting evidence; do not confuse one-step and recursive metrics.")
    if not model_comparison.empty:
        st.markdown("### Saved model comparison")
        st.caption("Loaded from project evaluation reports. Values are not recalculated here unless raw predictions are available.")
        st.dataframe(model_comparison, use_container_width=True, hide_index=True)
        metric_col = next((x for x in model_comparison.columns if x.lower().replace("_", "").startswith("wape")), None)
        model_col = next((x for x in model_comparison.columns if x.lower() in {"model", "model_name", "name"}), model_comparison.columns[0])
        if metric_col:
            fig = px.bar(model_comparison.sort_values(metric_col, ascending=False), x=metric_col, y=model_col, orientation="h", title="Model comparison — lower WAPE is better")
            fig.update_traces(marker_color=C["blue"])
            chart_layout(fig, 390)
    else:
        st.warning("Model comparison report was not found. No comparison values are fabricated.")
    if not backtest.empty:
        metrics = metrics_from_backtest(backtest)
        if metrics:
            st.markdown("### Recursive chronological backtesting")
            st.caption("Metrics calculated from the saved recursive prediction rows.")
            cols = st.columns(5)
            for col, key in zip(cols, ["MAE", "RMSE", "WAPE", "Bias (predicted − actual)", "Rows"]):
                val = metrics[key]
                col.metric(key, f"{val:.3f}%" if key == "WAPE" else (f"{val:,.3f}" if key != "Rows" else f"{val:,}"))
            eval_df = backtest.copy()
            if "horizon_step" in eval_df:
                eval_df["horizon_step"] = pd.to_numeric(eval_df["horizon_step"], errors="coerce")
                by_h = eval_df.groupby("horizon_step").apply(
                    lambda x: pd.Series({
                        "MAE": np.mean(np.abs(x["actual_sales"] - x["predicted_sales"])),
                        "RMSE": np.sqrt(np.mean((x["actual_sales"] - x["predicted_sales"])**2)),
                        "WAPE_percent": metric_wape(x["actual_sales"], x["predicted_sales"]),
                    }),
                    include_groups=False,
                ).reset_index()
                fig = px.line(by_h, x="horizon_step", y="WAPE_percent", markers=True, title="Recursive backtest WAPE by forecast horizon")
                fig.update_traces(line_color=C["blue"])
                chart_layout(fig, 350)
            if {"actual_sales", "predicted_sales"}.issubset(eval_df.columns):
                sample = eval_df.sample(min(2000, len(eval_df)), random_state=42)
                fig = px.scatter(sample, x="actual_sales", y="predicted_sales", opacity=.45, title="Actual vs predicted (sample of saved backtest rows)")
                maxv = float(max(sample["actual_sales"].max(), sample["predicted_sales"].max()))
                fig.add_trace(go.Scatter(x=[0, maxv], y=[0, maxv], mode="lines", name="Ideal prediction", line=dict(dash="dash")))
                chart_layout(fig, 420)
            st.download_button("Download recursive backtest rows", backtest.to_csv(index=False).encode(), "recursive_backtest_rows.csv", "text/csv")
        else:
            st.warning("Backtest file exists but does not contain the expected actual/predicted columns.")
    else:
        st.warning("Recursive backtesting report was not found. Add the Stage 7 prediction CSV to reports/stage7_backtesting/.")

    if not cross_origin.empty:
        st.markdown("### Stage 15 cross-origin comparison")
        st.dataframe(cross_origin, use_container_width=True, hide_index=True)
    if readiness_summary:
        with st.expander("Final readiness audit summary"):
            st.json(readiness_summary)

elif page == "Store & Product Analytics":
    show_hero("Store & Product Analytics", "Compare demand concentration, replenishment needs, and risk across retail segments.")
    tab1, tab2 = st.tabs(["📊 Store performance", "🔒 Product-family performance"])
    fc = current_forecast(7)
    if not fc.empty:
        with tab1:
            store_fc = fc.groupby("store_nbr", as_index=False)["forecast_demand"].sum().sort_values("store_nbr")
            fig = px.bar(store_fc, x="store_nbr", y="forecast_demand", title="7-day demand by store")
            fig.update_traces(marker_color=C["blue"])
            chart_layout(fig, 390)
            store_hist = filtered.groupby("store_nbr", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
            st.markdown("#### Historical sales by store")
            fig = px.bar(store_hist, x="store_nbr", y="sales")
            fig.update_traces(marker_color=C["teal"])
            chart_layout(fig, 350)
            st.dataframe(store_fc.sort_values("forecast_demand", ascending=False), use_container_width=True, hide_index=True)
        with tab2:
            family_fc = fc.groupby("family", as_index=False)["forecast_demand"].sum().sort_values("forecast_demand", ascending=False)
            fig = px.bar(family_fc.head(20).sort_values("forecast_demand"), x="forecast_demand", y="family", orientation="h", title="7-day forecast by product family")
            fig.update_traces(marker_color=C["blue"])
            chart_layout(fig, 500)
            family_hist = filtered.groupby("family", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
            st.markdown("#### Historical sales by product family")
            st.dataframe(family_hist, use_container_width=True, hide_index=True)
    else:
        st.warning("No forecasts available for the current selection.")

elif page == "Business Intelligence":
    show_hero("Business Intelligence", "Business summaries and demand concentration derived from the selected historical scope.")
    if filtered.empty:
        st.warning("No data matches the current global filters.")
    else:
        daily = filtered.groupby("date", as_index=False)["sales"].sum()
        family = filtered.groupby("family", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
        store = filtered.groupby("store_nbr", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
        top_share = float(family.head(5)["sales"].sum() / family["sales"].sum() * 100) if family["sales"].sum() else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total observed sales", money(filtered["sales"].sum()))
        c2.metric("Peak sales day", f"{daily.loc[daily.sales.idxmax(), 'date']:%d %b %Y}")
        c3.metric("Top 5 family share", f"{top_share:.1f}%")
        c4.metric("Highest-sales store", int(store.iloc[0]["store_nbr"]) if not store.empty else "—")
        fig = px.line(daily, x="date", y="sales", title="Historical sales trend")
        fig.update_traces(line_color=C["blue"])
        chart_layout(fig, 370)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(family.head(10), values="sales", names="family", title="Top product-family contribution")
            chart_layout(fig, 400)
        with col2:
            fig = px.bar(store.head(15).sort_values("sales"), x="sales", y="store_nbr", orientation="h", title="Top 15 stores by historical sales")
            fig.update_traces(marker_color=C["blue"])
            chart_layout(fig, 400)
        st.markdown("### Key findings")
        st.write(f"- Top five product families account for **{top_share:.1f}%** of observed sales in the selected scope.")
        st.write(f"- Highest-sales store in the current scope: **Store {int(store.iloc[0]['store_nbr'])}**.")
        st.write(f"- Historical peak daily sales: **{money(daily['sales'].max())}**.")
        st.download_button("Download BI summary — daily sales", daily.to_csv(index=False).encode(), "business_intelligence_daily_sales.csv", "text/csv")

elif page == "Data Quality & Explorer":
    show_hero("Data Quality & Explorer", "Inspect the source dataset, missing values, date coverage, and current filtered records.")
    # Data is aggregated by date/store/family during loading. These are post-cleaning checks.
    raw = sales_all
    dup_after_aggregation = int(raw.duplicated(["date", "store_nbr", "family"]).sum())
    missing_by_col = raw.isna().sum().rename("missing_values").reset_index().rename(columns={"index": "column"})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Usable rows", f"{len(raw):,}")
    c2.metric("Stores", raw["store_nbr"].nunique())
    c3.metric("Product families", raw["family"].nunique())
    c4.metric("Duplicate keys after aggregation", dup_after_aggregation)
    st.caption("Duplicate-key count is measured after the app's defensive aggregation; it is not a count of duplicates in the untouched raw source.")
    st.markdown("### Date coverage")
    st.write(f"**Start:** {raw['date'].min():%Y-%m-%d}  |  **End:** {raw['date'].max():%Y-%m-%d}  |  **Distinct dates:** {raw['date'].nunique():,}")
    st.markdown("### Missing values in loaded data")
    st.dataframe(missing_by_col, use_container_width=True, hide_index=True)
    st.markdown("### Explore records")
    q1, q2 = st.columns(2)
    with q1:
        explore_store = st.selectbox("Store", ["All"] + stores_all, key="explore_store")
    with q2:
        explore_family = st.selectbox("Product family", ["All"] + families_all, key="explore_family")
    explore = raw.copy()
    if explore_store != "All":
        explore = explore[explore["store_nbr"] == explore_store]
    if explore_family != "All":
        explore = explore[explore["family"] == explore_family]
    st.dataframe(explore.head(5000), use_container_width=True, hide_index=True)
    st.caption(f"Showing up to 5,000 of {len(explore):,} matching rows.")
    st.download_button("Download explored data", explore.to_csv(index=False).encode(), "explored_sales_data.csv", "text/csv")

# ---------- Footer / reproducibility ----------

st.markdown("---")
st.caption(
    f"Demand Intelligence · Data: {DATA_PATH.name} · Historical dates: "
    f"{first_history_date:%Y-%m-%d} to {last_history_date:%Y-%m-%d}. "
    "Forecasts are estimates. Inventory and what-if outputs are simulated."
)

