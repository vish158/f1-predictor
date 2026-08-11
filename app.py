"""
F1 Grand Prix Winner Predictor — Streamlit App
===============================================
Run locally:
    streamlit run app.py

Deploy free:
    https://share.streamlit.io  (connect your GitHub repo, point to app.py)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import os
import pickle

from src.pipeline import (
    fetch_race_results,
    build_features,
    encode_features,
    split_data,
    train_model,
    evaluate_model,
    predict_next_race,
    FEATURES,
)

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background-color: #0f0f13; color: #e8e8f0; }

    .hero {
        background: linear-gradient(135deg, #1a0505 0%, #0f0f13 70%);
        border: 1px solid #ff1801;
        border-radius: 12px;
        padding: 32px 36px;
        margin-bottom: 28px;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 900;
        color: #ffffff;
        margin: 0 0 6px;
        letter-spacing: -1px;
    }
    .hero p { color: #888; font-size: 1rem; margin: 0; }

    .metric-card {
        background: #1a1a24;
        border: 1px solid #2a2a3a;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-label { font-size: 0.75rem; letter-spacing: 2px; color: #ff1801;
                    text-transform: uppercase; font-weight: 700; }
    .metric-value { font-size: 2rem; font-weight: 900; color: #fff; margin-top: 4px; }
    .metric-sub   { font-size: 0.8rem; color: #666; margin-top: 2px; }

    .driver-card {
        background: #13131e;
        border: 1px solid #2a2a3a;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .podium-1 { border-left: 4px solid #FFD700; }
    .podium-2 { border-left: 4px solid #C0C0C0; }
    .podium-3 { border-left: 4px solid #CD7F32; }
    .podium-n { border-left: 4px solid #2a2a3a; }

    .badge {
        display: inline-block;
        background: #ff1801;
        color: white;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    section[data-testid="stSidebar"] {
        background: #0a0a0e;
        border-right: 1px solid #1e1e2e;
    }

    div[data-testid="stSelectbox"] > div,
    div[data-testid="stNumberInput"] > div {
        background: #1a1a24 !important;
        border-color: #2a2a3a !important;
        color: #e8e8f0 !important;
    }

    .stButton > button {
        background: #ff1801 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        padding: 0.5rem 2rem !important;
        width: 100%;
    }

    .info-box {
        background: #131320;
        border: 1px solid #2a2a42;
        border-radius: 8px;
        padding: 16px 20px;
        font-size: 0.88rem;
        color: #9090b0;
        line-height: 1.7;
        margin-bottom: 16px;
    }
    .info-box b { color: #e0e0f0; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────
CIRCUITS = [
    "bahrain", "jeddah", "albert_park", "suzuka", "shanghai", "miami",
    "imola", "monaco", "villeneuve", "catalunya", "red_bull_ring",
    "silverstone", "hungaroring", "spa", "zandvoort", "monza",
    "baku", "marina_bay", "americas", "rodriguez", "interlagos", "yas_marina"
]

DRIVERS = [
    "max_verstappen", "sergio_perez", "lewis_hamilton", "george_russell",
    "charles_leclerc", "carlos_sainz", "lando_norris", "oscar_piastri",
    "fernando_alonso", "lance_stroll", "esteban_ocon", "pierre_gasly",
    "valtteri_bottas", "guanyu_zhou", "yuki_tsunoda", "daniel_ricciardo",
    "kevin_magnussen", "nico_hulkenberg", "logan_sargeant", "alexander_albon",
]

CONSTRUCTORS = {
    "max_verstappen": "red_bull", "sergio_perez": "red_bull",
    "lewis_hamilton": "mercedes", "george_russell": "mercedes",
    "charles_leclerc": "ferrari", "carlos_sainz": "ferrari",
    "lando_norris": "mclaren", "oscar_piastri": "mclaren",
    "fernando_alonso": "alpine", "lance_stroll": "aston_martin",
    "esteban_ocon": "alpine", "pierre_gasly": "alpine",
    "valtteri_bottas": "alfa", "guanyu_zhou": "alfa",
    "yuki_tsunoda": "alphatauri", "daniel_ricciardo": "alphatauri",
    "kevin_magnussen": "haas", "nico_hulkenberg": "haas",
    "logan_sargeant": "williams", "alexander_albon": "williams",
}

MODEL_CACHE = "model_cache.pkl"


# ── Session state helpers ─────────────────────────────────────────────

def save_cache(df, model, encoders):
    with open(MODEL_CACHE, "wb") as f:
        pickle.dump({"df": df, "model": model, "encoders": encoders}, f)


def load_cache():
    if os.path.exists(MODEL_CACHE):
        with open(MODEL_CACHE, "rb") as f:
            return pickle.load(f)
    return None


# ── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Model Settings")
    season_start = st.slider("Training data from", 2005, 2018, 2010)
    season_end   = st.slider("Training data to",   2020, 2024, 2024)
    test_year    = st.slider("Test from year",      2021, 2024, 2023)

    st.markdown("---")
    train_btn = st.button("🧠 Fetch Data & Train Model")
    if os.path.exists(MODEL_CACHE):
        load_btn = st.button("⚡ Load Cached Model")
    else:
        load_btn = False

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <b>Data source:</b> Jolpica F1 API<br>
    <b>Algorithm:</b> XGBoost (gradient boosting)<br>
    <b>Key features:</b> Grid position, rolling driver form, circuit win rate, constructor performance
    </div>
    """, unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div style="font-size:11px;letter-spacing:3px;color:#ff1801;font-weight:700;text-transform:uppercase;margin-bottom:8px">
        Machine Learning · Formula 1 · Prediction
    </div>
    <h1>🏎️ Grand Prix Winner Predictor</h1>
    <p>Train an XGBoost model on 14+ years of F1 race data, then predict win probabilities for any upcoming race lineup.</p>
</div>
""", unsafe_allow_html=True)

# ── Model training ────────────────────────────────────────────────────

if "model" not in st.session_state:
    st.session_state.model    = None
    st.session_state.encoders = None
    st.session_state.df       = None
    st.session_state.metrics  = None

if train_btn:
    with st.spinner("📡 Fetching race data from Jolpica API…"):
        df = fetch_race_results(season_start, season_end)
    with st.spinner("⚙️ Engineering features…"):
        df = build_features(df)
        df, encoders = encode_features(df)
    with st.spinner("🧠 Training XGBoost…"):
        X_train, y_train, X_test, y_test = split_data(df, test_year)
        model   = train_model(X_train, y_train, X_test, y_test)
        metrics = evaluate_model(model, X_test, y_test)

    st.session_state.update(model=model, encoders=encoders, df=df, metrics=metrics)
    save_cache(df, model, encoders)
    st.success("✅ Model trained and cached!")

if load_btn:
    cache = load_cache()
    if cache:
        st.session_state.update(
            model=cache["model"],
            encoders=cache["encoders"],
            df=cache["df"],
        )
        st.success("⚡ Cached model loaded!")

# ── Metrics row ───────────────────────────────────────────────────────

if st.session_state.model:
    df      = st.session_state.df
    metrics = st.session_state.metrics or {}

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Races in dataset</div>
            <div class="metric-value">{df['round'].count() // 20:,}</div>
            <div class="metric-sub">since {int(df['year'].min())}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total driver entries</div>
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-sub">rows of training data</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        auc = metrics.get("roc_auc", 0)
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">ROC-AUC Score</div>
            <div class="metric-value">{auc:.3f}</div>
            <div class="metric-sub">1.0 = perfect · 0.5 = random</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        unique_drivers = df["driver"].nunique()
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Unique drivers</div>
            <div class="metric-value">{unique_drivers}</div>
            <div class="metric-sub">in training data</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature importance plot ────────────────────────────────────────
    st.subheader("📊 Feature Importance")
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0f0f13")
    ax.set_facecolor("#13131e")

    importances = pd.Series(st.session_state.model.feature_importances_, index=FEATURES)
    importances = importances.sort_values()
    bars = ax.barh(importances.index, importances.values, color="#ff1801", alpha=0.85)
    ax.set_xlabel("Importance", color="#888")
    ax.tick_params(colors="#aaa")
    ax.spines[:].set_color("#2a2a3a")
    for bar in bars:
        bar.set_edgecolor("#ff4433")
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    # ── Predict section ────────────────────────────────────────────────
    st.subheader("🏁 Predict an Upcoming Race")

    col_a, col_b = st.columns([1, 2])

    with col_a:
        circuit = st.selectbox("Circuit", CIRCUITS, index=CIRCUITS.index("monaco"))
        round_num = st.number_input("Race round number", 1, 24, 8)
        num_drivers = st.slider("Number of drivers to compare", 2, 10, 5)

    with col_b:
        st.markdown("**Configure the grid:**")
        driver_entries = []
        for i in range(num_drivers):
            c1, c2 = st.columns([2, 1])
            with c1:
                driver = st.selectbox(
                    f"P{i+1} driver", DRIVERS,
                    index=min(i, len(DRIVERS)-1),
                    key=f"drv_{i}"
                )
            with c2:
                grid = st.number_input(f"Grid", 1, 20, i+1, key=f"grid_{i}")
            driver_entries.append({
                "driver": driver,
                "grid": grid,
                "constructor": CONSTRUCTORS.get(driver, "unknown")
            })

    predict_btn = st.button("🏎️ Run Prediction")

    if predict_btn:
        results = predict_next_race(
            st.session_state.model,
            st.session_state.encoders,
            circuit,
            driver_entries,
            df,
            round_num=round_num,
        )

        st.markdown(f"### Predicted Win Probabilities — {circuit.replace('_', ' ').title()} GP")

        for idx, row in results.iterrows():
            pos_class = {0: "podium-1", 1: "podium-2", 2: "podium-3"}.get(idx, "podium-n")
            medal     = {0: "🥇", 1: "🥈", 2: "🥉"}.get(idx, f"P{idx+1}")
            bar_width = int(row["win_pct"] / results["win_pct"].max() * 100)

            st.markdown(f"""
            <div class="driver-card {pos_class}">
                <span style="font-size:1.4rem">{medal}</span>
                <div style="flex:1">
                    <div style="font-weight:700;color:#fff;font-size:1rem">
                        {row['driver'].replace('_',' ').title()}
                        <span style="font-size:0.75rem;color:#666;margin-left:8px">
                            {row['constructor'].replace('_',' ').title()} · Grid P{int(row['grid'])}
                        </span>
                    </div>
                    <div style="background:#1a1a2e;border-radius:4px;height:6px;margin-top:6px;width:100%">
                        <div style="background:#ff1801;height:6px;border-radius:4px;width:{bar_width}%"></div>
                    </div>
                </div>
                <div style="text-align:right;min-width:60px">
                    <div style="font-size:1.4rem;font-weight:900;color:#fff">{row['win_pct']:.1f}%</div>
                    <div style="font-size:0.7rem;color:#555">win chance</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Bar chart
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        fig2.patch.set_facecolor("#0f0f13")
        ax2.set_facecolor("#13131e")
        colors = ["#FFD700", "#C0C0C0", "#CD7F32"] + ["#ff1801"] * len(results)
        ax2.bar(
            [d.replace("_", " ").title() for d in results["driver"]],
            results["win_pct"],
            color=colors[:len(results)],
            edgecolor="#2a2a3a"
        )
        ax2.set_ylabel("Win Probability (%)", color="#888")
        ax2.tick_params(colors="#aaa", axis="both")
        ax2.spines[:].set_color("#2a2a3a")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig2, use_container_width=True)

else:
    st.info("👈 Use the sidebar to **fetch data & train the model** to get started. Training takes 2–4 minutes as it downloads 14 seasons of race data.")

    st.markdown("""
    <div class="info-box">
    <b>How it works:</b><br>
    1. Downloads F1 race results (2010–2024) from the Jolpica API<br>
    2. Engineers features: rolling driver form, circuit win rate, grid advantage, championship points<br>
    3. Trains an XGBoost model on races up to 2022 and validates on 2023–2024<br>
    4. Outputs per-driver win probabilities for any race lineup you configure
    </div>
    """, unsafe_allow_html=True)
