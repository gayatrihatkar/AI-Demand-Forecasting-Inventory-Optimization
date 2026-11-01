from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)

# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sales_clean.parquet"
MODEL_PATH = BASE_DIR / "models" / "best_demand_model.joblib"
REPORTS_DIR = BASE_DIR / "reports"
STAGE7_DIR = REPORTS_DIR / "stage7_backtesting"
STAGE15_DIR = REPORTS_DIR / "stage15_final_readiness_audit"

FEATURES = [
    "store_nbr", "family", "year", "month", "day",
    "day_of_week", "day_of_year", "week_of_year", "is_weekend",
    "onpromotion", "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "sales_rolling_mean_7", "sales_rolling_mean_28",
    "sales_rolling_std_7",
]

COLORS = {
    "teal": "#176B67",
    "dark_teal": "#104C49",
    "light_teal": "#DDEFEA",
    "beige": "#F4EBDD",
    "cream": "#FCF9F3",
    "ink": "#263735",
    "muted": "#667773",
    "border": "#D8D2C5",
    "orange": "#C77D32",
    "red": "#B7473E",
    "green": "#39785B",
}

st.set_page_config(
    page_title="Demand Intelligence | Retail AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    :root {{
        --primary-color: {COLORS['teal']};
    }}
    .stApp {{
        background: {COLORS['cream']};
        color: {COLORS['ink']};
    }}
    [data-testid="stHeader"] {{
        background: rgba(252,249,243,0.94);
    }}
    [data-testid="stSidebar"] {{
        background: {COLORS['beige']};
        border-right: 1px solid {COLORS['border']};
    }}
    [data-testid="stMetric"] {{
        background: white;
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 14px 16px;
    }}
    [data-testid="stMetricLabel"] {{
        color: {COLORS['muted']};
    }}
    .hero {{
        background: linear-gradient(115deg, {COLORS['dark_teal']}, {COLORS['teal']});
        color: white;
        padding: 26px 30px;
        border-radius: 16px;
        margin: 4px 0 20px 0;
    }}
    .hero h1 {{ color: white; margin: 0 0 8px 0; }}
    .hero p {{ color: #E7F3EF; margin: 0; }}
    .section-note {{
        background: {COLORS['light_teal']};
        border-left: 4px solid {COLORS['teal']};
        border-radius: 7px;
        padding: 11px 14px;
        margin: 8px 0 16px 0;
    }}
    .small-muted {{ color: {COLORS['muted']}; font-size: 0.88rem; }}
    div[data-testid="stDownloadButton"] button {{
        border-radius: 8px;
    }}
    h1, h2, h3 {{ color: {COLORS['dark_teal']}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ================================================================
# DATA AND MODEL LOADING
# ================================================================

@st.cache_data(show_spinner=False)
def load_sales(path_string, modified_time):
    path = Path(path_string)
    if not path.exists():
        raise FileNotFoundError(f"Historical dataset not found: {path}")

    df = pd.read_parquet(path)
    required = {"date", "store_nbr", "family", "sales"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["store_nbr"] = pd.to_numeric(
        df["store_nbr"], errors="coerce"
    )
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df["family"] = df["family"].astype("string").str.strip()

    if "onpromotion" not in df.columns:
        df["onpromotion"] = 0.0

    df["onpromotion"] = pd.to_numeric(
        df["onpromotion"], errors="coerce"
    ).fillna(0.0)

    df = df.dropna(subset=["date", "store_nbr", "family", "sales"])
    df["store_nbr"] = df["store_nbr"].astype(int)
    df["family"] = df["family"].astype(str)
    df["sales"] = df["sales"].clip(lower=0)

    # Aggregate defensively if duplicate date/store/family rows exist.
    df = (
        df.groupby(
            ["date", "store_nbr", "family"], as_index=False,
            observed=True
        )
        .agg(
            sales=("sales", "sum"),
            onpromotion=("onpromotion", "sum"),
        )
        .sort_values(["date", "store_nbr", "family"])
        .reset_index(drop=True)
    )
    return df


@st.cache_resource(show_spinner="Loading trained demand model...")
def load_model(path_string, modified_time):
    path = Path(path_string)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    model = joblib.load(path)
    if not hasattr(model, "predict"):
        raise TypeError("The saved model does not expose predict().")

    if hasattr(model, "n_features_in_"):
        if int(model.n_features_in_) != len(FEATURES):
            raise ValueError(
                "Model input feature count does not match the expected "
                f"{len(FEATURES)} features."
            )
    return model


def read_csv_if_exists(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_json_if_exists(path):
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def safe_float(value, default=0.0):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def fmt_number(value, decimals=0):
    if pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f}"


def apply_filters(df, filters):
    out = df.copy()
    if out.empty:
        return out
    if filters.get("stores"):
        out = out[out["store_nbr"].isin(filters["stores"])]
    if filters.get("families"):
        out = out[out["family"].isin(filters["families"])]
    if filters.get("start") is not None:
        out = out[out["date"] >= pd.Timestamp(filters["start"])]
    if filters.get("end") is not None:
        out = out[out["date"] <= pd.Timestamp(filters["end"])]
    return out


def add_csv_download(df, filename, label="Download CSV"):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=False,
    )


def metric_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)


def section_header(title, subtitle=None):
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


def make_chart_theme(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["ink"]),
        margin=dict(l=12, r=12, t=35, b=12),
        legend_title_text="",
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E5E0D6", zeroline=False)
    return fig


def daily_history(df):
    return (
        df.groupby("date", as_index=False)
        .agg(sales=("sales", "sum"))
        .sort_values("date")
    )


# ================================================================
# RECURSIVE FORECASTING USING THE ORIGINAL 16-FEATURE PIPELINE
# ================================================================

def date_features(date_value):
    iso_week = int(pd.Timestamp(date_value).isocalendar().week)
    return {
        "year": int(date_value.year),
        "month": int(date_value.month),
        "day": int(date_value.day),
        "day_of_week": int(date_value.dayofweek),
        "day_of_year": int(date_value.dayofyear),
        "week_of_year": iso_week,
        "is_weekend": int(date_value.dayofweek >= 5),
    }


@st.cache_data(show_spinner=False, max_entries=6)
def generate_recursive_forecast(
    horizon, sales_path, sales_mtime, model_path, model_mtime
):
    """
    Recursive forecasts begin on the day after the last date in the
    historical dataset. Future promotions are assumed to be zero.

    Missing historical dates are filled with zero sales for each
    store-family series. Predictions are clipped at zero.
    """
    history_df = load_sales(sales_path, sales_mtime)
    model = load_model(model_path, model_mtime)

    last_date = pd.Timestamp(history_df["date"].max()).normalize()
    first_date = pd.Timestamp(history_df["date"].min()).normalize()

    stores = sorted(history_df["store_nbr"].dropna().unique().astype(int))
    families = sorted(history_df["family"].dropna().unique().tolist())

    if not stores or not families:
        raise ValueError("No stores or product families are available.")

    all_dates = pd.date_range(first_date, last_date, freq="D")
    keys = [
        (int(store), str(family))
        for store in stores
        for family in families
    ]

    # A complete daily series per store/family makes lag offsets explicit.
    series_map = {}
    indexed = history_df.set_index(
        ["store_nbr", "family", "date"]
    )["sales"]

    for store, family in keys:
        try:
            values = indexed.loc[(store, family)]
            values = values.reindex(all_dates, fill_value=0.0)
            series_map[(store, family)] = (
                pd.to_numeric(values, errors="coerce")
                .fillna(0.0)
                .clip(lower=0)
                .astype(float)
                .tolist()
            )
        except KeyError:
            series_map[(store, family)] = [0.0] * len(all_dates)

    output = []
    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1),
        periods=int(horizon),
        freq="D",
    )

    for step, forecast_date in enumerate(future_dates, start=1):
        calendar = date_features(forecast_date)
        feature_rows = []

        for store, family in keys:
            values = series_map[(store, family)]
            if len(values) < 28:
                padded = [0.0] * (28 - len(values)) + values
            else:
                padded = values

            last7 = np.asarray(padded[-7:], dtype=float)
            last28 = np.asarray(padded[-28:], dtype=float)

            row = {
                "store_nbr": int(store),
                "family": str(family),
                **calendar,
                "onpromotion": 0.0,
                "sales_lag_7": float(padded[-7]),
                "sales_lag_14": float(padded[-14]),
                "sales_lag_28": float(padded[-28]),
                "sales_rolling_mean_7": float(np.mean(last7)),
                "sales_rolling_mean_28": float(np.mean(last28)),
                "sales_rolling_std_7": (
                    float(np.std(last7, ddof=1))
                    if len(last7) > 1 else 0.0
                ),
            }
            feature_rows.append(row)

        feature_df = pd.DataFrame(feature_rows, columns=FEATURES)
        predictions = np.asarray(model.predict(feature_df), dtype=float)
        predictions = np.nan_to_num(
            predictions, nan=0.0, posinf=0.0, neginf=0.0
        )
        predictions = np.clip(predictions, 0, None)

        for (store, family), prediction in zip(keys, predictions):
            series_map[(store, family)].append(float(prediction))
            output.append({
                "date": forecast_date,
                "horizon_step": step,
                "store_nbr": store,
                "family": family,
                "predicted_sales": float(prediction),
                "forecast_assumption": "Future onpromotion = 0",
            })

    forecast_df = pd.DataFrame(output)
    return forecast_df, last_date



# ================================================================
# ON-DEMAND SIX-METHOD MODEL EVALUATION
# ================================================================

@st.cache_data(show_spinner=False, max_entries=2)
def build_evaluation_frame(_sales_df, cutoff_days=28):
    """Build chronological lag features from observed sales; no future leakage."""
    d = _sales_df[["date", "store_nbr", "family", "sales", "onpromotion"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    grp = d.groupby(["store_nbr", "family"], sort=False)["sales"]
    for lag in (7, 14, 28):
        d[f"sales_lag_{lag}"] = grp.shift(lag)
    shifted = d.groupby(["store_nbr", "family"], sort=False)["sales"].shift(1)
    d["sales_rolling_mean_7"] = shifted.groupby(
        [d["store_nbr"], d["family"]], sort=False
    ).rolling(7, min_periods=7).mean().reset_index(level=[0, 1], drop=True)
    d["sales_rolling_mean_28"] = shifted.groupby(
        [d["store_nbr"], d["family"]], sort=False
    ).rolling(28, min_periods=28).mean().reset_index(level=[0, 1], drop=True)
    d["sales_rolling_std_7"] = shifted.groupby(
        [d["store_nbr"], d["family"]], sort=False
    ).rolling(7, min_periods=7).std().reset_index(level=[0, 1], drop=True)
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    d["day"] = d["date"].dt.day
    d["day_of_week"] = d["date"].dt.dayofweek
    d["day_of_year"] = d["date"].dt.dayofyear
    d["week_of_year"] = d["date"].dt.isocalendar().week.astype(int)
    d["is_weekend"] = (d["day_of_week"] >= 5).astype(int)
    d["onpromotion"] = pd.to_numeric(d["onpromotion"], errors="coerce").fillna(0)
    d = d.dropna(subset=FEATURES).copy()
    cutoff = d["date"].max() - pd.Timedelta(days=int(cutoff_days))
    train = d[d["date"] <= cutoff].copy()
    test = d[d["date"] > cutoff].copy()
    # Cap training rows for Streamlit Cloud resource limits; preserve chronological coverage.
    if len(train) > 60000:
        train = train.sample(n=60000, random_state=42).sort_values("date")
    return train, test


@st.cache_data(show_spinner=False, max_entries=1)
def run_six_method_evaluation(_sales_df, _saved_model, cutoff_days=28):
    """Compare the deployed XGBoost pipeline with five baseline/ML methods on one holdout."""
    from sklearn.base import clone
    train, test = build_evaluation_frame(_sales_df, cutoff_days)
    if train.empty or test.empty:
        raise ValueError("Not enough lagged history to build train and holdout sets.")

    numeric_features = [c for c in FEATURES if c != "family"]
    # One-hot encode product family for the sklearn estimators.
    x_train = pd.get_dummies(train[FEATURES], columns=["family"], dtype=float)
    x_test = pd.get_dummies(test[FEATURES], columns=["family"], dtype=float)
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)
    y_train = train["sales"].to_numpy(dtype=float)
    y_test = test["sales"].to_numpy(dtype=float)

    rows = []
    predictions_by_model = {}

    def add_metrics(name, pred, note):
        pred = np.nan_to_num(np.asarray(pred, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        pred = np.clip(pred, 0, None)
        mae = float(mean_absolute_error(y_test, pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        denominator = float(np.abs(y_test).sum())
        wape = float(100 * np.abs(y_test - pred).sum() / denominator) if denominator else np.nan
        bias = float((pred - y_test).mean())
        rows.append({"Model": name, "MAE": mae, "RMSE": rmse, "WAPE_percent": wape,
                     "Mean_Bias": bias, "Holdout_Days": int(cutoff_days),
                     "Training_Rows": len(train), "Test_Rows": len(test), "Method": note})
        predictions_by_model[name] = pred

    # Existing saved pipeline: evaluated directly with the original feature schema.
    try:
        add_metrics("XGBoost (saved model)", _saved_model.predict(test[FEATURES]),
                    "Existing saved model; chronological holdout, observed lag features")
    except Exception as exc:
        rows.append({"Model": "XGBoost (saved model)", "MAE": np.nan, "RMSE": np.nan,
                     "WAPE_percent": np.nan, "Mean_Bias": np.nan,
                     "Holdout_Days": int(cutoff_days), "Training_Rows": len(train),
                     "Test_Rows": len(test), "Method": f"Saved model prediction failed: {exc}"})

    models = [
        ("Random Forest", RandomForestRegressor(n_estimators=60, min_samples_leaf=2,
                                                max_features=0.8, n_jobs=-1, random_state=42)),
        ("Extra Trees", ExtraTreesRegressor(n_estimators=60, min_samples_leaf=2,
                                            max_features=0.8, n_jobs=-1, random_state=42)),
        ("HistGradientBoosting", HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=20,
                                                               learning_rate=0.08, random_state=42)),
        ("Decision Tree", DecisionTreeRegressor(max_depth=18, min_samples_leaf=5, random_state=42)),
    ]
    for name, estimator in models:
        try:
            estimator.fit(x_train, y_train)
            add_metrics(name, estimator.predict(x_test),
                        "Trained at evaluation time on capped historical training sample")
        except Exception as exc:
            rows.append({"Model": name, "MAE": np.nan, "RMSE": np.nan,
                         "WAPE_percent": np.nan, "Mean_Bias": np.nan,
                         "Holdout_Days": int(cutoff_days), "Training_Rows": len(train),
                         "Test_Rows": len(test), "Method": f"Evaluation failed: {exc}"})

    # Seasonal naive is a transparent baseline: use observed sales from 7 days earlier.
    try:
        add_metrics("Seasonal Naive (lag 7)", test["sales_lag_7"].to_numpy(),
                    "Baseline; predicts the value from seven days earlier")
    except Exception as exc:
        rows.append({"Model": "Seasonal Naive (lag 7)", "MAE": np.nan, "RMSE": np.nan,
                     "WAPE_percent": np.nan, "Mean_Bias": np.nan,
                     "Holdout_Days": int(cutoff_days), "Training_Rows": len(train),
                     "Test_Rows": len(test), "Method": f"Evaluation failed: {exc}"})
    return pd.DataFrame(rows), predictions_by_model, train, test


# ================================================================
# INITIALIZE APP
# ================================================================

st.markdown(
    """
    <div class="hero">
      <h1>Demand Intelligence</h1>
      <p>AI-powered demand forecasting and inventory planning for
      supermarket retail</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not DATA_PATH.exists() or not MODEL_PATH.exists():
    st.error(
        "Required release artifacts are missing. Check that the "
        "data/ and models/ folders are present beside app.py."
    )
    st.stop()

try:
    sales = load_sales(
        str(DATA_PATH), DATA_PATH.stat().st_mtime
    )
    model = load_model(
        str(MODEL_PATH), MODEL_PATH.stat().st_mtime
    )
except Exception as exc:
    st.error(f"Could not initialize the project: {exc}")
    st.stop()

model_comparison = read_csv_if_exists(
    REPORTS_DIR / "model_comparison.csv"
)
backtest = read_csv_if_exists(
    STAGE7_DIR / "stage7_recursive_predictions.csv"
)
stage15_overall = read_csv_if_exists(
    STAGE15_DIR / "stage15_cross_origin_overall_comparison.csv"
)
stage15_horizon = read_csv_if_exists(
    STAGE15_DIR / "stage15_cross_origin_by_horizon.csv"
)
stage15_store = read_csv_if_exists(
    STAGE15_DIR / "stage15_cross_origin_store_comparison.csv"
)
stage15_family = read_csv_if_exists(
    STAGE15_DIR / "stage15_cross_origin_family_comparison.csv"
)
stage15_summary = read_json_if_exists(
    STAGE15_DIR / "stage15_final_readiness_summary.json"
)

last_history_date = pd.Timestamp(sales["date"].max()).normalize()
first_history_date = pd.Timestamp(sales["date"].min()).normalize()
stores_all = sorted(sales["store_nbr"].unique().astype(int).tolist())
families_all = sorted(sales["family"].unique().tolist())

# Shared filter state: configured in Executive Overview, then used elsewhere.
if "global_filters" not in st.session_state:
    st.session_state["global_filters"] = {
        "stores": stores_all,
        "families": families_all,
        "start": first_history_date.date(),
        "end": last_history_date.date(),
    }

# ================================================================
# NAVIGATION
# ================================================================

pages = [
    "Executive Overview",
    "Demand Forecasting",
    "Inventory Optimization",
    "Smart Alerts",
    "Sales Explorer",
    "What-if Simulator",
    "Model Evaluation",
    "Store & Product Analytics",
    "Business Intelligence",
    "Data Quality & Explorer",
]

with st.sidebar:
    st.markdown("## 📦 Demand Intelligence")
    st.caption("Retail forecasting and planning")
    page = st.radio("Navigate", pages, label_visibility="collapsed")
    st.divider()
    st.caption("PROJECT STATUS")
    st.write("**Model:** Original XGBoost pipeline")
    st.write(f"**Stores:** {len(stores_all)}")
    st.write(f"**Product families:** {len(families_all)}")
    st.write(f"**Historical end:** {last_history_date:%d %b %Y}")
    st.caption("Research prototype · Not a live inventory feed")

filters = st.session_state["global_filters"]
filtered_sales = apply_filters(sales, filters)

# ================================================================
# EXECUTIVE OVERVIEW
# ================================================================

if page == "Executive Overview":
    section_header(
        "Executive Overview",
        "Configure the shared filters here. Applied filters are reused "
        "across the dashboard."
    )

    with st.form("global_filter_form"):
        c1, c2 = st.columns(2)
        selected_stores = c1.multiselect(
            "Stores",
            options=stores_all,
            default=filters["stores"],
            help="Select one or more stores.",
        )
        selected_families = c2.multiselect(
            "Product families",
            options=families_all,
            default=filters["families"],
        )

        c3, c4 = st.columns(2)
        start_date = c3.date_input(
            "Historical start date",
            value=max(
                first_history_date.date(),
                filters["start"]
            ),
            min_value=first_history_date.date(),
            max_value=last_history_date.date(),
        )
        end_date = c4.date_input(
            "Historical end date",
            value=min(
                last_history_date.date(),
                filters["end"]
            ),
            min_value=first_history_date.date(),
            max_value=last_history_date.date(),
        )

        apply_button = st.form_submit_button(
            "Apply dashboard filters",
            type="primary",
        )

    if apply_button:
        if start_date > end_date:
            st.error("The start date must be on or before the end date.")
        elif not selected_stores or not selected_families:
            st.error("Select at least one store and one product family.")
        else:
            st.session_state["global_filters"] = {
                "stores": selected_stores,
                "families": selected_families,
                "start": start_date,
                "end": end_date,
            }
            st.rerun()

    filters = st.session_state["global_filters"]
    filtered_sales = apply_filters(sales, filters)

    st.markdown(
        f'<div class="section-note">Showing '
        f'{len(filters["stores"]):,} selected stores and '
        f'{len(filters["families"]):,} selected product families '
        f'from {filters["start"]} to {filters["end"]}. '
        f'Historical data ends on {last_history_date:%Y-%m-%d}.</div>',
        unsafe_allow_html=True,
    )

    total_sales = filtered_sales["sales"].sum()
    avg_daily = (
        filtered_sales.groupby("date")["sales"].sum().mean()
        if not filtered_sales.empty else 0
    )
    number_days = filtered_sales["date"].nunique()
    number_stores = filtered_sales["store_nbr"].nunique()
    number_families = filtered_sales["family"].nunique()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total historical sales", fmt_number(total_sales))
    k2.metric("Average daily sales", fmt_number(avg_daily, 1))
    k3.metric("Stores selected", fmt_number(number_stores))
    k4.metric("Families selected", fmt_number(number_families))
    k5.metric("Days represented", fmt_number(number_days))

    if filtered_sales.empty:
        st.warning("No historical rows match these filters.")
    else:
        daily = daily_history(filtered_sales)
        c1, c2 = st.columns([1.65, 1])
        with c1:
            section_header("Historical demand trend")
            fig = px.line(
                daily, x="date", y="sales",
                labels={"date": "Date", "sales": "Sales"},
            )
            fig.update_traces(line=dict(color=COLORS["teal"], width=2))
            st.plotly_chart(make_chart_theme(fig), use_container_width=True)

        with c2:
            section_header("Sales by product family")
            fam = (
                filtered_sales.groupby("family", as_index=False)["sales"]
                .sum().sort_values("sales", ascending=True).tail(12)
            )
            fig = px.bar(
                fam, x="sales", y="family", orientation="h",
                labels={"sales": "Sales", "family": "Product family"},
            )
            fig.update_traces(marker_color=COLORS["teal"])
            st.plotly_chart(make_chart_theme(fig), use_container_width=True)

    st.divider()
    section_header("Forecast preview")
    horizon = st.selectbox(
        "Future forecast horizon",
        [7, 14, 30, 60],
        index=0,
        key="overview_horizon",
    )
    if st.button("Generate forecast preview", type="primary"):
        try:
            with st.spinner(
                f"Generating recursive {horizon}-day forecasts..."
            ):
                forecast_all, forecast_origin = generate_recursive_forecast(
                    horizon,
                    str(DATA_PATH), DATA_PATH.stat().st_mtime,
                    str(MODEL_PATH), MODEL_PATH.stat().st_mtime,
                )
            forecast_filtered = forecast_all[
                forecast_all["store_nbr"].isin(filters["stores"])
                & forecast_all["family"].isin(filters["families"])
            ]
            st.session_state["forecast_cache"] = forecast_all
            st.session_state["forecast_origin"] = forecast_origin
            st.session_state["forecast_horizon"] = horizon
            st.session_state["forecast_filter_preview"] = forecast_filtered
            st.success(
                f"Forecast generated from {forecast_origin:%Y-%m-%d} "
                f"for {horizon} future days."
            )
        except Exception as exc:
            st.error(f"Forecast generation failed: {exc}")

    preview = st.session_state.get("forecast_filter_preview")
    if isinstance(preview, pd.DataFrame) and not preview.empty:
        st.metric(
            "Forecasted demand in selected scope",
            fmt_number(preview["predicted_sales"].sum()),
        )
        preview_daily = (
            preview.groupby("date", as_index=False)["predicted_sales"]
            .sum()
        )
        fig = px.line(
            preview_daily, x="date", y="predicted_sales",
            labels={"date": "Forecast date", "predicted_sales": "Predicted sales"},
        )
        fig.update_traces(line=dict(color=COLORS["orange"], width=2))
        st.plotly_chart(make_chart_theme(fig), use_container_width=True)
        add_csv_download(preview, "filtered_future_demand_forecast.csv")

# ================================================================
# DEMAND FORECASTING
# ================================================================

elif page == "Demand Forecasting":
    section_header(
        "Demand Forecasting",
        "Future forecasts are generated recursively from the last date "
        "in the historical dataset—not from today's date."
    )
    st.markdown(
        f'<div class="section-note">Last observed historical date: '
        f'<b>{last_history_date:%Y-%m-%d}</b>. Future promotions are '
        f'assumed to be zero. Negative predictions are clipped to zero. '
        f'Missing historical dates within each store-family series are '
        f'filled with zero sales.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    horizon = c1.selectbox(
        "Forecast horizon (days)", [7, 14, 30, 60], index=0
    )
    store_options = filters["stores"]
    family_options = filters["families"]
    forecast_store = c2.selectbox("Store", store_options)
    forecast_family = c3.selectbox("Product family", family_options)

    if st.button("Run recursive forecast", type="primary"):
        try:
            with st.spinner("Generating recursive forecasts..."):
                future, origin = generate_recursive_forecast(
                    horizon,
                    str(DATA_PATH), DATA_PATH.stat().st_mtime,
                    str(MODEL_PATH), MODEL_PATH.stat().st_mtime,
                )
            st.session_state["forecast_cache"] = future
            st.session_state["forecast_origin"] = origin
            st.session_state["forecast_horizon"] = horizon
            st.success(f"Forecast generated. Origin date: {origin:%Y-%m-%d}")
        except Exception as exc:
            st.error(f"Forecast generation failed: {exc}")

    future = st.session_state.get("forecast_cache")
    if isinstance(future, pd.DataFrame) and not future.empty:
        selected = future[
            (future["store_nbr"] == forecast_store)
            & (future["family"] == forecast_family)
        ].copy()

        if selected.empty:
            st.warning("No forecasts found for this selection.")
        else:
            a, b, c = st.columns(3)
            a.metric("Forecast total", fmt_number(selected["predicted_sales"].sum()))
            b.metric("Average per day", fmt_number(selected["predicted_sales"].mean(), 2))
            c.metric("Peak forecast day", fmt_number(selected["predicted_sales"].max(), 2))

            fig = px.line(
                selected, x="date", y="predicted_sales",
                markers=True,
                labels={"date": "Forecast date", "predicted_sales": "Predicted sales"},
            )
            fig.update_traces(line=dict(color=COLORS["teal"], width=3))
            st.plotly_chart(make_chart_theme(fig), use_container_width=True)
            add_csv_download(selected, "demand_forecast.csv")

            st.dataframe(
                selected[[
                    "date", "horizon_step", "store_nbr", "family",
                    "predicted_sales", "forecast_assumption"
                ]],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("Choose a horizon and click **Run recursive forecast**.")

# ================================================================
# INVENTORY OPTIMIZATION
# ================================================================

elif page == "Inventory Optimization":
    section_header(
        "Inventory Optimization",
        "Planning estimates only. This dataset does not contain verified "
        "on-hand inventory, purchase orders, or supplier lead times."
    )

    c1, c2, c3 = st.columns(3)
    inv_store = c1.selectbox("Store", filters["stores"], key="inv_store")
    inv_family = c2.selectbox(
        "Product family", filters["families"], key="inv_family"
    )
    lead_time = c3.number_input(
        "Assumed supplier lead time (days)", 1, 90, 7
    )

    c4, c5, c6 = st.columns(3)
    review_period = c4.number_input(
        "Review period (days)", 1, 60, 7
    )
    service_level = c5.selectbox(
        "Target service level",
        [0.90, 0.95, 0.975, 0.99],
        index=1,
        format_func=lambda x: f"{x:.1%}",
    )
    stock_on_hand = c6.number_input(
        "Assumed stock on hand (units)", min_value=0.0,
        value=0.0, step=10.0,
        help="Enter an assumed value; it is not read from the dataset."
    )

    z_values = {0.90: 1.282, 0.95: 1.645, 0.975: 1.960, 0.99: 2.326}
    hist_series = (
        sales[
            (sales["store_nbr"] == inv_store)
            & (sales["family"] == inv_family)
        ]
        .sort_values("date")
        .set_index("date")["sales"]
        .asfreq("D", fill_value=0)
    )

    if hist_series.empty:
        st.warning("No historical series is available for this selection.")
    else:
        recent = hist_series.tail(28)
        mean_daily = float(recent.mean())
        std_daily = float(recent.std(ddof=1)) if len(recent) > 1 else 0.0
        safety_stock = z_values[service_level] * std_daily * np.sqrt(lead_time)
        reorder_point = mean_daily * lead_time + safety_stock
        protection_demand = mean_daily * (lead_time + review_period)
        suggested_order = max(
            0.0, mean_daily * (lead_time + review_period)
            + safety_stock - stock_on_hand
        )

        a, b, c, d = st.columns(4)
        a.metric("Recent average daily demand", fmt_number(mean_daily, 2))
        b.metric("Estimated safety stock", fmt_number(safety_stock, 2))
        c.metric("Estimated reorder point", fmt_number(reorder_point, 2))
        d.metric("Suggested order estimate", fmt_number(suggested_order, 2))

        st.markdown(
            f'<div class="section-note">Formula assumptions: safety stock = '
            f'z × recent daily-demand standard deviation × √lead time; '
            f'reorder point = average daily demand × lead time + safety stock. '
            f'The suggested order estimate covers lead time plus review period. '
            f'It is not a purchase order recommendation.</div>',
            unsafe_allow_html=True,
        )

        inventory_table = pd.DataFrame([{
            "store_nbr": inv_store,
            "family": inv_family,
            "recent_28_day_average_daily_demand": mean_daily,
            "recent_28_day_daily_std_dev": std_daily,
            "assumed_lead_time_days": lead_time,
            "assumed_review_period_days": review_period,
            "service_level": service_level,
            "estimated_safety_stock": safety_stock,
            "estimated_reorder_point": reorder_point,
            "assumed_stock_on_hand": stock_on_hand,
            "estimated_suggested_order": suggested_order,
        }])
        st.dataframe(inventory_table, use_container_width=True, hide_index=True)
        add_csv_download(inventory_table, "inventory_planning_estimate.csv")

        st.caption(
            "Historical daily sales are used as a demand proxy. "
            "Sales are not necessarily equivalent to units sold for every "
            "product-family interpretation, and stock availability is unknown."
        )

# ================================================================
# SMART ALERTS
# ================================================================

elif page == "Smart Alerts":
    section_header(
        "Smart Alerts",
        "Alerts are calculated from historical demand and forecast results. "
        "They are not live stockout notifications."
    )

    alert_horizon = st.selectbox(
        "Alert forecast horizon (days)", [7, 14, 30], index=0
    )
    if st.button("Calculate smart alerts", type="primary"):
        try:
            future, origin = generate_recursive_forecast(
                alert_horizon,
                str(DATA_PATH), DATA_PATH.stat().st_mtime,
                str(MODEL_PATH), MODEL_PATH.stat().st_mtime,
            )
            scope = future[
                future["store_nbr"].isin(filters["stores"])
                & future["family"].isin(filters["families"])
            ]
            totals = (
                scope.groupby(["store_nbr", "family"], as_index=False)
                .agg(
                    forecast_total=("predicted_sales", "sum"),
                    forecast_daily_average=("predicted_sales", "mean"),
                    peak_daily_forecast=("predicted_sales", "max"),
                )
            )
            historical = sales[
                sales["store_nbr"].isin(filters["stores"])
                & sales["family"].isin(filters["families"])
            ].copy()

            recent_date = historical["date"].max()
            last7 = historical[
                historical["date"] > recent_date - pd.Timedelta(days=7)
            ].groupby(["store_nbr", "family"])["sales"].mean()
            previous7 = historical[
                (historical["date"] <= recent_date - pd.Timedelta(days=7))
                & (historical["date"] > recent_date - pd.Timedelta(days=14))
            ].groupby(["store_nbr", "family"])["sales"].mean()

            totals["recent_7_day_average"] = [
                safe_float(last7.get((r.store_nbr, r.family), 0))
                for r in totals.itertuples()
            ]
            totals["previous_7_day_average"] = [
                safe_float(previous7.get((r.store_nbr, r.family), 0))
                for r in totals.itertuples()
            ]
            totals["change_vs_previous_week_percent"] = np.where(
                totals["previous_7_day_average"] > 0,
                100 * (
                    totals["recent_7_day_average"]
                    - totals["previous_7_day_average"]
                ) / totals["previous_7_day_average"],
                np.nan,
            )

            totals["alert"] = np.select(
                [
                    totals["change_vs_previous_week_percent"] >= 25,
                    totals["forecast_total"] >= totals["forecast_total"].quantile(0.90),
                    totals["forecast_total"] <= totals["forecast_total"].quantile(0.10),
                ],
                [
                    "Demand rising sharply",
                    "High forecast demand",
                    "Low forecast demand",
                ],
                default="Monitor",
            )
            totals["priority"] = np.select(
                [
                    totals["alert"] == "Demand rising sharply",
                    totals["alert"] == "High forecast demand",
                    totals["alert"] == "Low forecast demand",
                ],
                ["High", "Medium", "Review"],
                default="Normal",
            )
            totals = totals.sort_values(
                ["priority", "forecast_total"],
                ascending=[True, False],
            )

            st.session_state["smart_alerts"] = totals
            st.success(f"Calculated {len(totals):,} store-family alert records.")
        except Exception as exc:
            st.error(f"Could not calculate alerts: {exc}")

    alerts = st.session_state.get("smart_alerts")
    if isinstance(alerts, pd.DataFrame) and not alerts.empty:
        a, b, c = st.columns(3)
        a.metric("High-priority alerts", int((alerts["priority"] == "High").sum()))
        b.metric("High forecast-demand records", int((alerts["alert"] == "High forecast demand").sum()))
        c.metric("Demand-rise alerts", int((alerts["alert"] == "Demand rising sharply").sum()))

        priority_choice = st.multiselect(
            "Priority filter",
            ["High", "Medium", "Review", "Normal"],
            default=["High", "Medium", "Review"],
        )
        shown = alerts[alerts["priority"].isin(priority_choice)]
        st.dataframe(shown, use_container_width=True, hide_index=True)
        add_csv_download(shown, "smart_alerts_action_list.csv")
        st.caption(
            "A high-priority alert indicates a sharp rise in observed "
            "recent demand, not a confirmed stockout. Stockout status "
            "requires real inventory records."
        )
    else:
        st.info("Click **Calculate smart alerts** to generate the current alert list.")

# ================================================================
# SALES EXPLORER
# ================================================================

elif page == "Sales Explorer":
    section_header(
        "Sales Explorer",
        "Explore the underlying historical sales data using interactive filters."
    )

    c1, c2, c3 = st.columns(3)
    explorer_stores = c1.multiselect(
        "Stores", filters["stores"], default=filters["stores"]
    )
    explorer_families = c2.multiselect(
        "Product families", filters["families"], default=filters["families"]
    )
    grain = c3.selectbox("Time aggregation", ["Daily", "Weekly", "Monthly"])

    data = sales[
        sales["store_nbr"].isin(explorer_stores)
        & sales["family"].isin(explorer_families)
        & sales["date"].between(
            pd.Timestamp(filters["start"]),
            pd.Timestamp(filters["end"]),
        )
    ].copy()

    if data.empty:
        st.warning("No records match the current selections.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Sales in selection", fmt_number(data["sales"].sum()))
        c2.metric("Rows", fmt_number(len(data)))
        c3.metric("Mean row sales", fmt_number(data["sales"].mean(), 2))

        freq = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[grain]
        trend = (
            data.set_index("date")
            .groupby("family")["sales"]
            .resample(freq)
            .sum()
            .reset_index()
        )
        fig = px.line(
            trend, x="date", y="sales", color="family",
            labels={"date": "Date", "sales": "Sales", "family": "Family"},
        )
        st.plotly_chart(make_chart_theme(fig), use_container_width=True)

        family_rank = (
            data.groupby("family", as_index=False)["sales"]
            .sum().sort_values("sales", ascending=False)
        )
        store_rank = (
            data.groupby("store_nbr", as_index=False)["sales"]
            .sum().sort_values("sales", ascending=False)
        )
        c1, c2 = st.columns(2)
        with c1:
            section_header("Product-family ranking")
            st.dataframe(family_rank, use_container_width=True, hide_index=True)
        with c2:
            section_header("Store ranking")
            st.dataframe(store_rank, use_container_width=True, hide_index=True)

        add_csv_download(data, "sales_explorer_filtered_data.csv")

# ================================================================
# WHAT-IF SIMULATOR
# ================================================================

elif page == "What-if Simulator":
    section_header(
        "What-if Simulator",
        "Change planning assumptions and recalculate inventory requirements. "
        "These values are simulations, not live stock data."
    )

    c1, c2 = st.columns(2)
    sim_store = c1.selectbox("Store", filters["stores"], key="sim_store")
    sim_family = c2.selectbox(
        "Product family", filters["families"], key="sim_family"
    )

    c1, c2, c3 = st.columns(3)
    demand_change = c1.slider("Demand change", -50, 100, 10, step=5)
    sim_lead_time = c2.slider("Lead time (days)", 1, 60, 7)
    sim_review = c3.slider("Review period (days)", 1, 30, 7)

    c1, c2, c3 = st.columns(3)
    sim_service = c1.selectbox(
        "Service level", [0.90, 0.95, 0.975, 0.99],
        index=1, format_func=lambda x: f"{x:.1%}"
    )
    sim_stock = c2.number_input(
        "Assumed available stock", min_value=0.0, value=100.0, step=10.0
    )
    sim_promotion = c3.selectbox(
        "Promotion scenario",
        ["No additional promotion adjustment",
         "Increase demand by an extra 10%",
         "Increase demand by an extra 20%"],
    )

    series = (
        sales[
            (sales["store_nbr"] == sim_store)
            & (sales["family"] == sim_family)
        ]
        .sort_values("date")
        .set_index("date")["sales"]
        .asfreq("D", fill_value=0)
    )

    if series.empty:
        st.warning("No historical series is available.")
    else:
        avg = float(series.tail(28).mean())
        std = float(series.tail(28).std(ddof=1)) if len(series.tail(28)) > 1 else 0.0

        promo_factor = {
            "No additional promotion adjustment": 1.0,
            "Increase demand by an extra 10%": 1.10,
            "Increase demand by an extra 20%": 1.20,
        }[sim_promotion]
        adjusted_avg = avg * (1 + demand_change / 100) * promo_factor
        adjusted_std = std * (1 + max(demand_change, 0) / 100) * promo_factor

        z = {0.90: 1.282, 0.95: 1.645, 0.975: 1.960, 0.99: 2.326}[sim_service]
        safety = z * adjusted_std * np.sqrt(sim_lead_time)
        reorder = adjusted_avg * sim_lead_time + safety
        target = adjusted_avg * (sim_lead_time + sim_review) + safety
        order_qty = max(0, target - sim_stock)

        baseline_safety = 1.645 * std * np.sqrt(7)
        baseline_reorder = avg * 7 + baseline_safety
        baseline_order = max(0, avg * 14 + baseline_safety - sim_stock)

        a, b, c = st.columns(3)
        a.metric(
            "Scenario reorder point", fmt_number(reorder, 2),
            delta=fmt_number(reorder - baseline_reorder, 2) + " vs baseline",
        )
        b.metric(
            "Scenario safety stock", fmt_number(safety, 2),
            delta=fmt_number(safety - baseline_safety, 2) + " vs baseline",
        )
        c.metric(
            "Estimated order quantity", fmt_number(order_qty, 2),
            delta=fmt_number(order_qty - baseline_order, 2) + " vs baseline",
        )

        comparison = pd.DataFrame([
            {
                "scenario": "Baseline assumptions",
                "average_daily_demand": avg,
                "safety_stock": baseline_safety,
                "reorder_point": baseline_reorder,
                "estimated_order_quantity": baseline_order,
            },
            {
                "scenario": "Selected what-if scenario",
                "average_daily_demand": adjusted_avg,
                "safety_stock": safety,
                "reorder_point": reorder,
                "estimated_order_quantity": order_qty,
            },
        ])
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        fig = px.bar(
            comparison, x="scenario",
            y=["safety_stock", "reorder_point", "estimated_order_quantity"],
            barmode="group",
            labels={"value": "Estimated units", "variable": "Measure"},
        )
        st.plotly_chart(make_chart_theme(fig), use_container_width=True)
        add_csv_download(comparison, "what_if_simulation.csv")

        st.caption(
            "Promotion adjustments are user-defined scenarios. They are not "
            "learned promotion effects from the trained model."
        )

# ================================================================
# MODEL EVALUATION
# ================================================================

elif page == "Model Evaluation":
    section_header(
        "Model Evaluation",
        "Use the saved evaluation reports. One-step validation and recursive "
        "multi-day backtesting are different evaluation settings."
    )

    if not stage15_overall.empty:
        section_header("Cross-origin recursive evaluation")
        display_cols = [
            c for c in [
                "method", "observations", "MAE", "RMSE",
                "WAPE_percent", "bias_predicted_minus_actual",
                "overforecast_rate_percent",
            ] if c in stage15_overall.columns
        ]
        st.dataframe(
            stage15_overall[display_cols],
            use_container_width=True,
            hide_index=True,
        )

        if "method" in stage15_overall.columns and "WAPE_percent" in stage15_overall.columns:
            fig = px.bar(
                stage15_overall, x="method", y="WAPE_percent",
                labels={"method": "Evaluation method", "WAPE_percent": "WAPE (%)"},
                text="WAPE_percent",
            )
            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            st.plotly_chart(make_chart_theme(fig), use_container_width=True)

    if not stage15_horizon.empty:
        section_header("Forecast error by horizon")
        horizon_cols = [
            c for c in [
                "horizon_step", "observations",
                "original_MAE", "corrected_MAE",
                "original_RMSE", "corrected_RMSE",
                "original_WAPE_percent", "corrected_WAPE_percent",
            ] if c in stage15_horizon.columns
        ]
        st.dataframe(
            stage15_horizon[horizon_cols],
            use_container_width=True,
            hide_index=True,
        )
        if {"horizon_step", "original_WAPE_percent", "corrected_WAPE_percent"}.issubset(stage15_horizon.columns):
            long = stage15_horizon.melt(
                id_vars="horizon_step",
                value_vars=["original_WAPE_percent", "corrected_WAPE_percent"],
                var_name="method",
                value_name="WAPE_percent",
            )
            fig = px.line(
                long, x="horizon_step", y="WAPE_percent",
                color="method", markers=True,
                labels={"horizon_step": "Forecast day", "WAPE_percent": "WAPE (%)"},
            )
            st.plotly_chart(make_chart_theme(fig), use_container_width=True)

    if not model_comparison.empty:
        section_header("Saved model comparison")
        st.dataframe(model_comparison, use_container_width=True, hide_index=True)
        error_col = next(
            (c for c in ["WAPE_percent", "MAE", "RMSE"] if c in model_comparison.columns),
            None,
        )
        model_name_col = next(
            (c for c in ["Model", "model", "method"] if c in model_comparison.columns),
            None,
        )
        if error_col and model_name_col:
            fig = px.bar(
                model_comparison, x=model_name_col, y=error_col,
                labels={model_name_col: "Model", error_col: error_col},
            )
            st.plotly_chart(make_chart_theme(fig), use_container_width=True)

    if not backtest.empty:
        section_header("Backtesting prediction explorer")
        required_cols = {
            "actual_sales", "predicted_sales", "horizon_step"
        }
        if required_cols.issubset(backtest.columns):
            c1, c2 = st.columns(2)
            chosen_step = c1.selectbox(
                "Forecast day", sorted(backtest["horizon_step"].dropna().unique())
            )
            chosen_store = c2.selectbox(
                "Store", sorted(backtest["store_nbr"].dropna().unique())
                if "store_nbr" in backtest.columns else [None]
            )
            sample = backtest[backtest["horizon_step"] == chosen_step].copy()
            if chosen_store is not None and "store_nbr" in sample.columns:
                sample = sample[sample["store_nbr"] == chosen_store]
            if "family" in sample.columns:
                sample = sample[sample["family"].isin(filters["families"])]
            if len(sample) > 1500:
                sample = sample.sample(1500, random_state=42)

            fig = px.scatter(
                sample, x="actual_sales", y="predicted_sales",
                color="family" if "family" in sample.columns else None,
                hover_data=[c for c in ["date", "store_nbr", "family"] if c in sample.columns],
                labels={"actual_sales": "Actual sales", "predicted_sales": "Predicted sales"},
            )
            fig.add_shape(
                type="line", x0=0, y0=0,
                x1=max(1, safe_float(sample["actual_sales"].max())),
                y1=max(1, safe_float(sample["actual_sales"].max())),
                line=dict(dash="dash", color=COLORS["red"]),
            )
            st.plotly_chart(make_chart_theme(fig), use_container_width=True)
            add_csv_download(sample, "backtesting_prediction_sample.csv")

    st.markdown(
        '<div class="section-note"><b>Interpretation:</b> Stage 15 '
        'evaluated two forecast origins. Results are useful research evidence, '
        'but they do not prove robustness across every season or future market '
        'condition. WAPE is an error metric, not a generic accuracy score.</div>',
        unsafe_allow_html=True,
    )

    section_header("Six-Model Comparison", "Run six methods on the same chronological holdout and identify the lowest-error model.")
    st.markdown(
        '<div class="section-note"><b>Transparency:</b> The existing saved reports and backtesting explorer above are preserved. '
        'This additional comparison uses the saved XGBoost model plus five methods trained/evaluated on demand.</div>',
        unsafe_allow_html=True,
    )
    eval_days = st.selectbox("Six-model holdout length (days)", [14, 28, 42], index=1, key="six_eval_days")
    if st.button("Run six-model evaluation", type="primary", key="run_six_model_evaluation"):
        try:
            with st.spinner("Building time-series features and evaluating six methods. This may take a few minutes..."):
                result, predictions, eval_train, eval_test = run_six_method_evaluation(
                    sales, model, int(eval_days)
                )
            st.session_state["six_model_eval_results"] = result
            st.session_state["six_model_eval_predictions"] = predictions
            st.session_state["six_model_eval_train_rows"] = len(eval_train)
            st.session_state["six_model_eval_test_rows"] = len(eval_test)
            st.session_state["six_model_eval_days"] = int(eval_days)
            st.success("Six-method evaluation completed using the selected chronological holdout.")
        except Exception as exc:
            st.error(f"Six-model evaluation failed: {exc}")

    results = st.session_state.get("six_model_eval_results")
    if isinstance(results, pd.DataFrame) and not results.empty:
        valid = results[pd.to_numeric(results["WAPE_percent"], errors="coerce").notna()].copy()
        st.caption(
            f"Holdout: {st.session_state.get('six_model_eval_days', eval_days)} days | "
            f"Training rows: {st.session_state.get('six_model_eval_train_rows', 0):,} "
            f"(maximum 60,000 for resource limits) | "
            f"Holdout rows: {st.session_state.get('six_model_eval_test_rows', 0):,}. "
            "This is one-step evaluation with observed historical lag values, not recursive multi-day forecast accuracy."
        )
        st.dataframe(results, use_container_width=True, hide_index=True)
        if not valid.empty:
            best = valid.sort_values("WAPE_percent", ascending=True).iloc[0]
            a, b, c = st.columns(3)
            a.metric("Best model by WAPE", str(best["Model"]))
            b.metric("Lowest WAPE", f"{best['WAPE_percent']:.2f}%")
            c.metric("Best MAE", f"{valid['MAE'].min():.3f}")
            chart = valid.sort_values("WAPE_percent", ascending=True)
            fig = px.bar(chart, x="Model", y="WAPE_percent", text="WAPE_percent",
                         labels={"WAPE_percent": "WAPE (%)"})
            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            st.plotly_chart(make_chart_theme(fig, height=450), use_container_width=True)
        add_csv_download(results, "six_model_evaluation_results.csv")
        st.markdown("### How to interpret the results")
        st.write("- **MAE:** average absolute prediction error; lower is better.")
        st.write("- **RMSE:** penalizes large errors more heavily; lower is better.")
        st.write("- **WAPE:** total absolute error divided by total actual sales; lower is better.")
        st.write("- **Mean bias:** average prediction minus actual sales; positive means overprediction.")
        st.info("The model with the lowest WAPE is marked best for this holdout only. That does not prove it will be best across every season or forecast horizon.")
    else:
        st.info("Click **Run six-model evaluation** to calculate results. No metrics are pre-filled or fabricated.")

# ================================================================
# STORE & PRODUCT ANALYTICS
# ================================================================

elif page == "Store & Product Analytics":
    section_header(
        "Store & Product Analytics",
        "Rankings and patterns are calculated from the selected historical data."
    )

    data = filtered_sales.copy()
    if data.empty:
        st.warning("No historical rows match the current dashboard filters.")
    else:
        store_stats = (
            data.groupby("store_nbr", as_index=False)
            .agg(
                total_sales=("sales", "sum"),
                average_row_sales=("sales", "mean"),
                active_days=("date", "nunique"),
                product_families=("family", "nunique"),
            )
            .sort_values("total_sales", ascending=False)
        )
        family_stats = (
            data.groupby("family", as_index=False)
            .agg(
                total_sales=("sales", "sum"),
                average_row_sales=("sales", "mean"),
                stores_covered=("store_nbr", "nunique"),
                active_days=("date", "nunique"),
            )
            .sort_values("total_sales", ascending=False)
        )

        c1, c2 = st.columns(2)
        with c1:
            section_header("Store performance")
            fig = px.bar(
                store_stats.sort_values("total_sales").tail(20),
                x="total_sales", y="store_nbr", orientation="h",
                labels={"total_sales": "Total sales", "store_nbr": "Store"},
            )
            st.plotly_chart(make_chart_theme(fig), use_container_width=True)
            st.dataframe(store_stats, use_container_width=True, hide_index=True)
            add_csv_download(store_stats, "store_performance.csv")

        with c2:
            section_header("Product-family performance")
            fig = px.bar(
                family_stats.sort_values("total_sales").tail(20),
                x="total_sales", y="family", orientation="h",
                labels={"total_sales": "Total sales", "family": "Family"},
            )
            st.plotly_chart(make_chart_theme(fig), use_container_width=True)
            st.dataframe(family_stats, use_container_width=True, hide_index=True)
            add_csv_download(family_stats, "product_family_performance.csv")

        section_header("Store × product-family matrix")
        pivot = data.pivot_table(
            index="family", columns="store_nbr", values="sales",
            aggfunc="sum", fill_value=0,
        )
        fig = px.imshow(
            pivot,
            aspect="auto",
            labels=dict(x="Store", y="Product family", color="Sales"),
            color_continuous_scale="Teal",
        )
        st.plotly_chart(make_chart_theme(fig, height=520), use_container_width=True)

# ================================================================
# BUSINESS INTELLIGENCE
# ================================================================

elif page == "Business Intelligence":
    section_header(
        "Business Intelligence",
        "Evidence-based descriptive insights from the historical dataset "
        "and saved forecast evaluation."
    )

    data = filtered_sales.copy()
    if data.empty:
        st.warning("No data is available for the selected filters.")
    else:
        family_totals = (
            data.groupby("family")["sales"].sum().sort_values(ascending=False)
        )
        store_totals = (
            data.groupby("store_nbr")["sales"].sum().sort_values(ascending=False)
        )
        daily = daily_history(data)
        peak_day = daily.loc[daily["sales"].idxmax()] if not daily.empty else None
        low_day = daily.loc[daily["sales"].idxmin()] if not daily.empty else None

        total = float(data["sales"].sum())
        top_family = family_totals.index[0] if len(family_totals) else "N/A"
        top_family_share = (
            100 * float(family_totals.iloc[0]) / total if total > 0 else 0
        )
        top_store = int(store_totals.index[0]) if len(store_totals) else None

        a, b, c = st.columns(3)
        a.metric("Highest-sales product family", str(top_family))
        b.metric("Top family share of sales", f"{top_family_share:.1f}%")
        c.metric("Highest-sales store", str(top_store) if top_store is not None else "—")

        st.markdown("### Key findings")
        st.write(
            f"- **Product concentration:** {top_family} is the highest-sales "
            f"family in the selected data, contributing {top_family_share:.1f}% "
            f"of total recorded sales."
        )
        st.write(
            f"- **Store performance:** Store {top_store} has the largest "
            f"recorded sales total in the selected scope."
        )
        if peak_day is not None:
            st.write(
                f"- **Peak observed day:** {pd.Timestamp(peak_day['date']):%Y-%m-%d}, "
                f"with total daily sales of {peak_day['sales']:,.2f}."
            )
        if low_day is not None:
            st.write(
                f"- **Lowest observed day:** {pd.Timestamp(low_day['date']):%Y-%m-%d}, "
                f"with total daily sales of {low_day['sales']:,.2f}."
            )

        if not stage15_overall.empty:
            st.markdown("### Forecast reliability evidence")
            st.dataframe(stage15_overall, use_container_width=True, hide_index=True)
            st.caption(
                "The saved cross-origin evaluation used two forecast origins. "
                "Use the results as limited validation evidence, not a guarantee "
                "of future performance."
            )

        st.markdown("### Business implications")
        st.write(
            "- High-volume families may deserve more frequent monitoring, "
            "but sales volume alone does not determine profitability."
        )
        st.write(
            "- Store-level comparisons can help prioritize further analysis, "
            "but differences may reflect store size, location, or assortment."
        )
        st.write(
            "- Forecast-based inventory planning should be reviewed alongside "
            "real stock, supplier lead time, shelf life, costs, and purchase orders."
        )

        insight_export = pd.DataFrame([
            {"insight": "Top product family", "value": top_family},
            {"insight": "Top family share percent", "value": top_family_share},
            {"insight": "Top store", "value": top_store},
            {"insight": "Selected historical sales total", "value": total},
        ])
        add_csv_download(insight_export, "business_intelligence_summary.csv")

# ================================================================
# DATA QUALITY & EXPLORER
# ================================================================

elif page == "Data Quality & Explorer":
    section_header(
        "Data Quality & Explorer",
        "Inspect the actual loaded dataset. Data-quality checks below are "
        "calculated from the release dataset at runtime."
    )

    data = sales.copy()
    a, b, c, d = st.columns(4)
    a.metric("Rows loaded", fmt_number(len(data)))
    b.metric("Date coverage", f"{data['date'].min():%Y-%m-%d} → {data['date'].max():%Y-%m-%d}")
    c.metric("Stores", fmt_number(data["store_nbr"].nunique()))
    d.metric("Product families", fmt_number(data["family"].nunique()))

    quality = pd.DataFrame({
        "column": data.columns,
        "dtype": [str(data[c].dtype) for c in data.columns],
        "missing_values": [int(data[c].isna().sum()) for c in data.columns],
        "missing_percent": [
            100 * data[c].isna().mean() for c in data.columns
        ],
        "unique_values": [int(data[c].nunique(dropna=True)) for c in data.columns],
    })
    st.markdown("### Column-level quality")
    st.dataframe(quality, use_container_width=True, hide_index=True)
    add_csv_download(quality, "data_quality_report.csv")

    c1, c2, c3 = st.columns(3)
    c1.metric("Duplicate date/store/family keys", fmt_number(
        int(data.duplicated(["date", "store_nbr", "family"]).sum())
    ))
    c2.metric("Negative sales rows", fmt_number(int((data["sales"] < 0).sum())))
    c3.metric("Missing sales rows", fmt_number(int(data["sales"].isna().sum())))

    st.markdown("### Explore records")
    c1, c2, c3 = st.columns(3)
    quality_stores = c1.multiselect(
        "Store filter", stores_all, default=filters["stores"]
    )
    quality_families = c2.multiselect(
        "Family filter", families_all, default=filters["families"]
    )
    max_rows = c3.selectbox("Maximum rows to display", [100, 500, 1000, 5000])

    data_view = data[
        data["store_nbr"].isin(quality_stores)
        & data["family"].isin(quality_families)
        & data["date"].between(
            pd.Timestamp(filters["start"]),
            pd.Timestamp(filters["end"]),
        )
    ].sort_values("date", ascending=False)

    st.caption(
        f"{len(data_view):,} rows match the selected filters. "
        f"The table displays at most {max_rows:,} rows."
    )
    st.dataframe(
        data_view.head(max_rows),
        use_container_width=True,
        hide_index=True,
    )
    add_csv_download(data_view, "data_explorer_filtered_records.csv")

# ================================================================
# FOOTER
# ================================================================

st.divider()
st.markdown(
    """
    <p class="small-muted">
    AI-Powered Demand Forecasting and Inventory Optimization System for
    Supermarket Retail | Research prototype | Streamlit + Plotly +
    original trained machine-learning pipeline
    </p>
    """,
    unsafe_allow_html=True,
)
