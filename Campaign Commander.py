# app.py — Marketoria — Campaign Commander
# Streamlit uplift targeting app (two-model approach)

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Marketoria — Campaign Commander", layout="centered")
st.title("🎯 Marketoria — Campaign Commander")

# ---------------------------
# Load models (trained in Colab and saved via joblib)
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_models(path="uplift_two_model.joblib"):
    return joblib.load(path)  # {"t": clf_t, "c": clf_c}

try:
    models = load_models()
except Exception as e:
    st.error(f"Could not load 'uplift_two_model.joblib'. Place it next to app.py. Details: {e}")
    st.stop()

# columns the preprocessors expect
FEATURE_COLS = ["age", "income", "web_visits", "prior_purch", "segment", "treatment"]

def ensure_feature_df(d):
    """Build a one-row DataFrame with proper column names and types."""
    X = pd.DataFrame([d], columns=FEATURE_COLS)
    # basic type coercions
    X["age"] = X["age"].astype(int)
    X["income"] = X["income"].astype(float)
    X["web_visits"] = X["web_visits"].astype(int)
    X["prior_purch"] = X["prior_purch"].astype(int)
    X["segment"] = X["segment"].astype(str)
    X["treatment"] = X["treatment"].astype(int)
    return X

def predict_uplift_batch(df_untreated):
    """
    Given a DataFrame WITHOUT 'treatment', score uplift by
    scoring treated and control views through the two pipelines.
    Expected columns: age, income, web_visits, prior_purch, segment
    """
    req = ["age", "income", "web_visits", "prior_purch", "segment"]
    missing = [c for c in req if c not in df_untreated.columns]
    if missing:
        raise ValueError(f"Holdout CSV missing columns: {missing}")

    Ht = df_untreated.copy()
    Ht["treatment"] = 1
    Hc = df_untreated.copy()
    Hc["treatment"] = 0

    p_t = models["t"].predict_proba(Ht[FEATURE_COLS])[:, 1]
    p_c = models["c"].predict_proba(Hc[FEATURE_COLS])[:, 1]
    return p_t - p_c

# ---------------------------
# Customer input UI
# ---------------------------
st.subheader("Customer features")
c1, c2 = st.columns(2)

with c1:
    age = st.number_input("Age", min_value=18, max_value=90, value=45, step=1)
    income = st.number_input("Income", min_value=5_000, max_value=300_000, value=60_000, step=1_000)
    web_visits = st.number_input("Web visits (30d)", min_value=0, max_value=100, value=5, step=1)

with c2:
    prior_purch = st.selectbox("Prior purchaser?", options=[0, 1], index=0)
    segment = st.selectbox("Segment", options=list("ABCDE"), index=0)

# Score uplift for this single customer (treated vs control copies)
Xt = ensure_feature_df({
    "age": age, "income": income, "web_visits": web_visits,
    "prior_purch": prior_purch, "segment": segment, "treatment": 1
})
Xc = Xt.copy()
Xc["treatment"] = 0

p_t_single = models["t"].predict_proba(Xt[FEATURE_COLS])[:, 1][0]
p_c_single = models["c"].predict_proba(Xc[FEATURE_COLS])[:, 1][0]
uplift_single = float(p_t_single - p_c_single)
st.metric("Predicted uplift (Δ conversion prob.)", f"{uplift_single:.4f}")

# ---------------------------
# Targeting policy UI
# ---------------------------
st.subheader("Targeting policy")
k = st.slider("Target top-k% of customers by uplift", min_value=1, max_value=50, value=10, step=1)
cost = st.number_input("Cost per contact ($)", min_value=0.0, max_value=50.0, value=1.0, step=0.1)
margin = st.number_input("Margin per conversion ($)", min_value=0.0, max_value=1000.0, value=50.0, step=1.0)

st.caption("Run the policy on a holdout set to estimate incremental conversions and ROI.")

# ---------------------------
# Holdout data source
# 1) Try bundled CSV alongside app.py
# 2) If not present, offer upload
# ---------------------------
holdout_path = "marketoria_holdout.csv"
holdout_df = None
bundled_ok = False

if os.path.exists(holdout_path):
    try:
        holdout_df = pd.read_csv(holdout_path)
        bundled_ok = True
        st.success("Using bundled holdout: marketoria_holdout.csv")
    except Exception as e:
        st.warning(f"Found {holdout_path} but could not load it: {e}")

if not bundled_ok:
    uploaded = st.file_uploader(
        "Upload holdout CSV with columns: age, income, web_visits, prior_purch, segment",
        type="csv"
    )
    if uploaded is not None:
        try:
            holdout_df = pd.read_csv(uploaded)
            st.success("Holdout CSV uploaded.")
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")

# ---------------------------
# Policy simulation
# ---------------------------
if holdout_df is not None:
    try:
        uplift_all = predict_uplift_batch(holdout_df)
        n = len(uplift_all)
        top_n = int(np.ceil(k / 100 * n))
        idx = np.argsort(-uplift_all)[:top_n]

        # expected incremental conversions = sum(uplift of targeted)
        inc_conv = float(uplift_all[idx].sum())

        # simple ROI: (inc_conv * margin - top_n * cost) / (top_n * cost + ε)
        spend = top_n * cost
        roi = (inc_conv * margin - spend) / (spend + 1e-9)

        st.write(f"**Targeted customers:** {top_n:,} / {n:,}")
        st.write(f"**Expected incremental conversions:** {inc_conv:.2f}")
        st.write(f"**Estimated ROI:** {roi:.2f}")

        # Optional: show a small distribution preview
        with st.expander("See uplift distribution (sample)"):
            st.write(pd.Series(uplift_all).describe().to_frame("uplift").T)

    except Exception as e:
        st.error(f"Policy simulation failed: {e}")
else:
    st.info("Provide a holdout dataset (bundled or upload) to run the targeting simulation.")

# ==== Add under policy simulation, after ROI is computed ====

# Build exportable table
export_df = holdout_df.copy()
export_df = export_df.reset_index(drop=True)
export_df["uplift_score"] = uplift_all
export_df["policy_targeted"] = 0
export_df.loc[idx, "policy_targeted"] = 1  # mark top-k%

# Let user choose to download all rows or just targeted
scope = st.radio(
    "Download which rows?",
    options=("Targeted only", "All rows"),
    horizontal=True,
)

if scope == "Targeted only":
    to_dl = export_df.loc[export_df["policy_targeted"] == 1].copy()
else:
    to_dl = export_df.copy()

csv_bytes = to_dl.to_csv(index=False).encode("utf-8")
st.download_button(
    label=f"⬇️ Download {scope.lower()} as CSV",
    data=csv_bytes,
    file_name="marketoria_target_list.csv" if scope=="Targeted only" else "marketoria_scored_holdout.csv",
    mime="text/csv",
)
