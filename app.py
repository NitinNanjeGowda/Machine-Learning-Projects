import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Healith — Meditara Rescue", layout="centered")
st.title("🩺 Healith — Meditara Rescue")

# --- Load calibrated model (pipeline includes preprocessing) ---
model = joblib.load("healith_calibrated_lr.joblib")

st.subheader("Patient inputs")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 55)
    sex = st.selectbox("Sex", ["F","M"])
    bmi = st.number_input("BMI", 10.0, 60.0, 27.0)
    systolic_bp = st.number_input("Systolic BP", 80.0, 230.0, 125.0)
    cholesterol = st.number_input("Cholesterol", 100.0, 400.0, 195.0)
with col2:
    glucose = st.number_input("Glucose", 60.0, 300.0, 100.0)
    smoker = st.selectbox("Smoker", [0,1])
    activity_days = st.number_input("Active days/week", 0, 7, 3)
    family_history = st.selectbox("Family history", [0,1])
    med_adherence = st.slider("Medication adherence", 0.0, 1.0, 0.7, 0.01)
region = st.selectbox("Region", ["North","South","East","West"])

X = pd.DataFrame([{
    "age": age, "sex": sex, "bmi": bmi, "systolic_bp": systolic_bp,
    "cholesterol": cholesterol, "glucose": glucose, "smoker": smoker,
    "activity_days": activity_days, "family_history": family_history,
    "med_adherence": med_adherence, "region": region
}])

# Risk score
risk = float(model.predict_proba(X)[:,1])
st.metric("Calibrated 1-year risk", f"{risk:.3f}")

# Policy threshold (use your chosen one from Step 1 as default)
thr_default = 0.25  # <- replace with your best threshold
thr = st.slider("Intervention threshold", 0.05, 0.60, thr_default, 0.01)

# Decision
flag = risk >= thr
action = "🔶 Flag for outreach" if flag else "🟢 Routine follow-up"
st.subheader("Decision")
st.write(f"**{action}**  — threshold = {thr:.2f}")

# Explain drivers (optional quick peek using LR coefficients)
with st.expander("Why this risk? (quick view)"):
    st.caption("Approximate contribution via logistic regression coef × value (not SHAP).")
    try:
        # best-effort: pull internal steps (works if model is Calibrated LR on a Pipeline)
        enc = model.base_estimator_.named_estimators_["pre"]
        clf = model.base_estimator_.named_estimators_["clf"]
        # Not robust across versions, but often works; for full explainability use SHAP.
        st.write("Model: Calibrated LR (isotonic)")
    except Exception as e:
        st.write("Explanation preview unavailable in this build.")
