import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Cost Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "used_car_canada_clean.csv"

# ─── Load & clean real dataset ────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Cast numerics
    for col in ["price", "miles", "year", "engine_size"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Basic quality filters
    df = df[
        (df["price"] >= 500) &
        (df["price"] <= 200_000) &
        (df["miles"] >= 0) &
        (df["miles"] <= 500_000) &
        (df["year"] >= 2000) &
        (df["engine_size"] > 0)
    ].copy()

    # Normalise text columns
    for col in ["make", "body_type", "fuel_type", "drivetrain",
                "transmission", "engine_block", "state"]:
        df[col] = df[col].astype(str).str.strip().str.title()

    # Consolidate rare fuel labels
    df["fuel_type"] = df["fuel_type"].replace({"Hyrid": "Hybrid", "Biodiesel": "Other"})

    df = df.dropna(subset=["price", "miles", "year", "make",
                            "body_type", "fuel_type", "drivetrain",
                            "transmission", "engine_size", "engine_block"])
    df["year"] = df["year"].astype(int)
    df["miles"] = df["miles"].astype(int)
    return df.reset_index(drop=True)


# ─── Train model ─────────────────────────────────────────────────────────────
CAT_COLS = ["make", "body_type", "fuel_type", "drivetrain",
            "transmission", "engine_block", "state"]
FEATURES  = ["make", "year", "miles", "body_type", "drivetrain",
             "transmission", "fuel_type", "engine_size", "engine_block", "state"]

@st.cache_resource
def train_model(df: pd.DataFrame):
    encoders = {}
    df_enc = df.copy()
    for col in CAT_COLS:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    X = df_enc[FEATURES]
    y = df_enc["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.08, max_depth=5,
        min_samples_leaf=15, subsample=0.8, random_state=42,
        n_iter_no_change=20, validation_fraction=0.1
    )
    model.fit(X_train, y_train)

    preds    = model.predict(X_test)
    mae      = mean_absolute_error(y_test, preds)
    r2       = r2_score(y_test, preds)
    feat_imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

    return model, encoders, mae, r2, feat_imp, X_test, y_test, preds


# ─── Bootstrap ───────────────────────────────────────────────────────────────
with st.spinner("Loading real listings & training model…"):
    df = load_data()
    model, encoders, mae, r2, feat_imp, X_test, y_test, preds = train_model(df)

MAKES         = sorted(df["make"].unique())
BODY_TYPES    = sorted(df["body_type"].unique())
FUEL_TYPES    = sorted(df["fuel_type"].unique())
DRIVETRAINS   = sorted(df["drivetrain"].unique())
TRANSMISSIONS = sorted(df["transmission"].unique())
ENG_BLOCKS    = sorted(df["engine_block"].unique())
STATES        = sorted(df["state"].unique())


# ─── Predict helper ───────────────────────────────────────────────────────────
def predict_price(make, year, miles, body_type, drivetrain, transmission,
                  fuel_type, engine_size, engine_block, state):
    def enc(col, val):
        le = encoders[col]
        val_title = str(val).strip().title()
        if val_title in le.classes_:
            return le.transform([val_title])[0]
        return le.transform([le.classes_[0]])[0]   # fallback

    row = {
        "make": enc("make", make),
        "year": year,
        "miles": miles,
        "body_type": enc("body_type", body_type),
        "drivetrain": enc("drivetrain", drivetrain),
        "transmission": enc("transmission", transmission),
        "fuel_type": enc("fuel_type", fuel_type),
        "engine_size": engine_size,
        "engine_block": enc("engine_block", engine_block),
        "state": enc("state", state),
    }
    X_in = pd.DataFrame([row])[FEATURES]
    return float(model.predict(X_in)[0])


# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/car.png", width=70)
st.sidebar.title("🚗 Car Details")
st.sidebar.markdown("Configure the vehicle and click **Predict** to get a market value estimate.")

make         = st.sidebar.selectbox("Make / Brand", MAKES,
                                    index=MAKES.index("Toyota") if "Toyota" in MAKES else 0)
year         = st.sidebar.slider("Model Year", 2000, 2022, 2018)
miles        = st.sidebar.slider("Mileage (km)", 0, 500_000, 60_000, step=1_000)
body_type    = st.sidebar.selectbox("Body Type", BODY_TYPES,
                                    index=BODY_TYPES.index("Sedan") if "Sedan" in BODY_TYPES else 0)
drivetrain   = st.sidebar.selectbox("Drivetrain", DRIVETRAINS)
transmission = st.sidebar.selectbox("Transmission", TRANSMISSIONS)
fuel_type    = st.sidebar.selectbox("Fuel Type", FUEL_TYPES,
                                    index=FUEL_TYPES.index("Gasoline") if "Gasoline" in FUEL_TYPES else 0)
engine_size  = st.sidebar.slider("Engine Size (L)", 1.0, 8.0, 2.5, step=0.1)
engine_block = st.sidebar.selectbox("Engine Block", ENG_BLOCKS)
state        = st.sidebar.selectbox("Province / State", STATES,
                                    index=STATES.index("On") if "On" in STATES else 0)

predict_btn  = st.sidebar.button("🔍 Predict Price", use_container_width=True)


# ─── Main layout ─────────────────────────────────────────────────────────────
st.title("🚗 Car Cost Predictor")
st.markdown(
    f"Trained on **{len(df):,} real Canadian used-car listings** using a "
    "Gradient Boosting model. Prices in **CAD**."
)
st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model Accuracy (R²)", f"{r2:.3f}")
col2.metric("Mean Abs. Error",     f"${mae:,.0f}")
col3.metric("Real Listings",       f"{len(df):,}")
col4.metric("Features Used",       str(len(FEATURES)))
st.divider()

# ── Prediction result ─────────────────────────────────────────────────────────
if predict_btn or "last_pred" in st.session_state:
    if predict_btn:
        price = predict_price(make, year, miles, body_type, drivetrain,
                              transmission, fuel_type, engine_size, engine_block, state)
        st.session_state["last_pred"] = price

    price = st.session_state["last_pred"]
    low   = price * 0.92
    high  = price * 1.08

    st.subheader("💰 Estimated Market Value (CAD)")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,#1e3a5f,#2d6a9f);
                        border-radius:16px;padding:28px;text-align:center;color:white;">
                <p style="font-size:16px;margin:0;opacity:.8;">Predicted Price</p>
                <h1 style="font-size:52px;margin:8px 0;">${price:,.0f}</h1>
                <p style="font-size:14px;opacity:.75;">Typical range: ${low:,.0f} – ${high:,.0f} CAD</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    avg_price = df["price"].median()
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=price,
        delta={"reference": avg_price, "valueformat": "$,.0f"},
        number={"prefix": "$", "valueformat": ",.0f"},
        gauge={
            "axis": {"range": [0, 120_000], "tickformat": "$,.0f"},
            "bar":  {"color": "#2d6a9f"},
            "steps": [
                {"range": [0,      20_000], "color": "#e8f4f8"},
                {"range": [20_000, 45_000], "color": "#b8d9ea"},
                {"range": [45_000, 75_000], "color": "#7fb5d5"},
                {"range": [75_000, 120_000],"color": "#4a8fbd"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 3},
                "thickness": 0.8,
                "value": avg_price,
            },
        },
        title={"text": f"vs. Median Listing (${avg_price:,.0f} CAD)"},
    ))
    fig_gauge.update_layout(height=280, margin=dict(t=40, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Price Distribution", "🔑 Feature Importance", "📈 Model Performance", "🔎 Dataset Explorer"]
)

# Tab 1 – Price Distribution
with tab1:
    st.subheader("Price Distribution by Category")
    cat_opt = st.selectbox(
        "Group by",
        ["make", "body_type", "fuel_type", "drivetrain", "transmission", "state"],
        key="dist_cat",
        format_func=lambda x: x.replace("_", " ").title(),
    )

    # Only show groups with enough listings
    top_groups = df[cat_opt].value_counts().head(15).index
    plot_df    = df[df[cat_opt].isin(top_groups)]
    fig_box = px.box(
        plot_df, x=cat_opt, y="price", color=cat_opt,
        title=f"Price Distribution by {cat_opt.replace('_',' ').title()} (top 15)",
        labels={"price": "Price (CAD)"},
    )
    fig_box.update_layout(showlegend=False, xaxis_tickangle=-30)
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Price vs. Mileage")
    sample = df.sample(min(800, len(df)), random_state=1)
    fig_scatter = px.scatter(
        sample, x="miles", y="price", color="body_type",
        opacity=0.5, trendline="ols",
        title="Mileage vs. Price (sample of 800)",
        labels={"miles": "Mileage (km)", "price": "Price (CAD)", "body_type": "Body"},
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Tab 2 – Feature Importance
with tab2:
    st.subheader("What Drives the Price?")
    fi_df = feat_imp.reset_index()
    fi_df.columns = ["Feature", "Importance"]
    fi_df["Feature"] = fi_df["Feature"].str.replace("_", " ").str.title()
    fig_fi = px.bar(
        fi_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Blues",
        title="Feature Importance (Gradient Boosting)",
    )
    fig_fi.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.subheader("Median Price by Model Year")
    year_avg = df.groupby("year")["price"].median().reset_index()
    fig_year = px.line(
        year_avg, x="year", y="price", markers=True,
        title="Median Listing Price by Model Year",
        labels={"price": "Median Price (CAD)", "year": "Year"},
    )
    st.plotly_chart(fig_year, use_container_width=True)

# Tab 3 – Model Performance
with tab3:
    st.subheader("Model Performance on Hold-out Test Set")
    n_sample = min(500, len(y_test))
    idx      = np.random.default_rng(42).choice(len(y_test), n_sample, replace=False)
    perf_df  = pd.DataFrame({
        "Actual":    y_test.values[idx],
        "Predicted": preds[idx],
    })

    max_v = perf_df[["Actual", "Predicted"]].max().max()
    fig_act = px.scatter(
        perf_df, x="Actual", y="Predicted", opacity=0.6,
        title="Actual vs. Predicted Prices",
        labels={"Actual": "Actual Price (CAD)", "Predicted": "Predicted Price (CAD)"},
    )
    fig_act.add_shape(type="line", x0=0, x1=max_v, y0=0, y1=max_v,
                      line=dict(color="red", dash="dash"))
    st.plotly_chart(fig_act, use_container_width=True)

    perf_df["Residual"] = perf_df["Predicted"] - perf_df["Actual"]
    fig_res = px.histogram(
        perf_df, x="Residual", nbins=60,
        title="Residual Distribution (Predicted − Actual)",
        labels={"Residual": "Residual (CAD)"},
    )
    st.plotly_chart(fig_res, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    rmse = float(np.sqrt((perf_df["Residual"] ** 2).mean()))
    m1.metric("R² Score", f"{r2:.4f}")
    m2.metric("MAE",       f"${mae:,.0f} CAD")
    m3.metric("RMSE",      f"${rmse:,.0f} CAD")

# Tab 4 – Dataset Explorer
with tab4:
    st.subheader("Explore the Dataset")
    c1, c2 = st.columns(2)
    filter_make = c1.multiselect("Filter by Make",  MAKES,   default=[])
    filter_body = c2.multiselect("Filter by Body",  BODY_TYPES, default=[])

    show_df = df.copy()
    if filter_make:
        show_df = show_df[show_df["make"].isin(filter_make)]
    if filter_body:
        show_df = show_df[show_df["body_type"].isin(filter_body)]

    display_cols = ["make", "year", "miles", "body_type", "drivetrain",
                    "transmission", "fuel_type", "engine_size", "state", "price"]
    st.dataframe(
        show_df[display_cols].sort_values("price", ascending=False).reset_index(drop=True),
        use_container_width=True, height=420,
    )
    st.caption(f"Showing {len(show_df):,} of {len(df):,} records")
    st.download_button(
        "📥 Download filtered CSV",
        show_df[display_cols].to_csv(index=False),
        "car_listings.csv", "text/csv",
    )
