import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Parkinson's Disease Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Space+Grotesk:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main { background-color: #f5f7ff; }
    
    .login-container {
        max-width: 420px;
        margin: 80px auto;
        background: #ffffff;
        border: 1px solid #dde1f0;
        border-radius: 16px;
        padding: 48px 40px;
        text-align: center;
        box-shadow: 0 8px 40px rgba(0,0,0,0.08);
    }
    .login-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 28px;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 6px;
    }
    .login-sub {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 32px;
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #dde1f0;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 12px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 32px;
        font-weight: 700;
        color: #6d28d9;
    }
    .metric-delta {
        font-size: 13px;
        color: #059669;
        margin-top: 4px;
    }
    
    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 20px;
        font-weight: 700;
        color: #1a1a2e;
        margin: 28px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #dde1f0;
    }
    
    .stSelectbox > div > div { background-color: #ffffff !important; }
    .stSlider > div { color: #6d28d9; }

    div[data-testid="stSidebarNav"] { background: #ffffff; }
    section[data-testid="stSidebar"] { background: #f0f2fb; border-right: 1px solid #dde1f0; }
    section[data-testid="stSidebar"] .css-1d391kg { padding: 24px 16px; }
    
    .pred-result-pos {
        background: linear-gradient(135deg, #faf5ff, #ede9fe);
        border: 1px solid #7c3aed;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 16px;
    }
    .pred-result-neg {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border: 1px solid #059669;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 16px;
    }
    .pred-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 22px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PASSWORD PROTECTION
# ─────────────────────────────────────────────
CORRECT_PASSWORD = "parkinsons2026"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("""
    <div class="login-container">
        <div style="font-size:48px; margin-bottom:12px;">🧠</div>
        <div class="login-title">PD Analytics Dashboard</div>
        <div class="login-sub">Healthcare Analytics · MSBA382</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        password_input = st.text_input("Enter password", type="password", placeholder="Password")
        if st.button("Access Dashboard", use_container_width=True):
            if password_input == CORRECT_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.")
    st.stop()

# ─────────────────────────────────────────────
# DATA LOADING & GENERATION
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("parkinsons_disease_data.csv")
    # Map numeric Gender to labels
    df["Gender"] = df["Gender"].map({0: "Male", 1: "Female"})
    # Map numeric Ethnicity to labels
    df["Ethnicity"] = df["Ethnicity"].map({0: "Caucasian", 1: "African American", 2: "Asian", 3: "Other"})
    # Map numeric EducationLevel to labels
    df["EducationLevel"] = df["EducationLevel"].map({0: "None", 1: "High School", 2: "Bachelor's", 3: "Higher"})
    return df

df = load_data()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 PD Analytics")
    st.markdown("---")
    
    page = st.radio("Navigate", [
        "Overview",
        "Demographics",
        "Clinical Analysis",
        "Risk Factors",
        "Global Map",
        "Smoking & PD",
        "Pesticide Exposure",
        "ML Prediction"
    ])
    
    st.markdown("---")
    st.markdown("**Filters**")
    gender_filter = st.multiselect("Gender", ["Male", "Female"], default=["Male", "Female"])
    age_filter = st.slider("Age Range", 50, 90, (50, 90))
    diagnosis_filter = st.multiselect("Diagnosis", ["PD Positive", "PD Negative"], default=["PD Positive", "PD Negative"])
    
    st.markdown("---")
    if st.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.rerun()

# Apply filters
diag_map = {"PD Positive": 1, "PD Negative": 0}
diag_vals = [diag_map[d] for d in diagnosis_filter]
filtered_df = df[
    (df["Gender"].isin(gender_filter)) &
    (df["Age"].between(age_filter[0], age_filter[1])) &
    (df["Diagnosis"].isin(diag_vals))
]

# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
COLORS = {
    "primary": "#6d28d9",
    "secondary": "#059669",
    "danger": "#dc2626",
    "warning": "#d97706",
    "bg": "#ffffff",
    "grid": "#e2e6f0",
    "text": "#1a1a2e",
    "muted": "#6b7280"
}
PD_COLORS = ["#6d28d9", "#059669"]

def apply_theme(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family="Inter"),
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# ─────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────
if page == "Overview":
    st.markdown("## Parkinson's Disease Analytics Dashboard")
    st.markdown(f"<span style='color:{COLORS['muted']}'>MSBA382 · Healthcare Analytics · {len(filtered_df):,} patients shown</span>", unsafe_allow_html=True)
    st.markdown("")
    
    pd_pos = filtered_df[filtered_df["Diagnosis"] == 1]
    pd_neg = filtered_df[filtered_df["Diagnosis"] == 0]
    prevalence = len(pd_pos) / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
    avg_age_pd = pd_pos["Age"].mean() if len(pd_pos) > 0 else 0
    avg_updrs = pd_pos["UPDRS"].mean() if len(pd_pos) > 0 else 0
    avg_moca = pd_pos["MoCA"].mean() if len(pd_pos) > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, delta in [
        (c1, "Total Patients", f"{len(filtered_df):,}", "In filtered view"),
        (c2, "PD Prevalence", f"{prevalence:.1f}%", f"{len(pd_pos):,} diagnosed"),
        (c3, "Avg Age (PD+)", f"{avg_age_pd:.1f} yrs", "At diagnosis"),
        (c4, "Avg UPDRS (PD+)", f"{avg_updrs:.1f}", "Disease severity score"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-delta">{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Diagnosis Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        diag_counts = filtered_df["Diagnosis"].map({1: "PD Positive", 0: "PD Negative"}).value_counts().reset_index()
        diag_counts.columns = ["Diagnosis", "Count"]
        fig = px.pie(diag_counts, names="Diagnosis", values="Count",
                     color="Diagnosis", color_discrete_map={"PD Positive": COLORS["primary"], "PD Negative": COLORS["secondary"]},
                     hole=0.55, title="PD vs Non-PD Patients")
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        age_diag = filtered_df.copy()
        age_diag["AgeGroup"] = pd.cut(age_diag["Age"], bins=[50,60,70,80,91], labels=["50-59","60-69","70-79","80+"])
        age_rate = age_diag.groupby("AgeGroup")["Diagnosis"].mean().reset_index()
        age_rate["PD Rate (%)"] = age_rate["Diagnosis"] * 100
        fig = px.bar(age_rate, x="AgeGroup", y="PD Rate (%)",
                     color="PD Rate (%)", color_continuous_scale=["#5b21b6", "#6d28d9"],
                     title="PD Diagnosis Rate by Age Group (%)")
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: DEMOGRAPHICS
# ─────────────────────────────────────────────
elif page == "Demographics":
    st.markdown("## Demographics Analysis")
    st.markdown(f"<span style='color:{COLORS['muted']}'>Real-world Parkinson's demographics, grounded in published research</span>", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin:16px 0; padding:16px; background:#1a1d27; border-left:4px solid #d97706; border-radius:8px; color:#e8e8f0; font-size:14px;">
    ⚠️ <b>Note on data source.</b> The Kaggle dataset used for this dashboard is synthetic (generated for educational purposes), so it does not 
    reliably reflect real-world demographic patterns — for example, it shows roughly equal PD rates between males and females, while real 
    epidemiological research shows men are about 1.5 times more likely to develop PD than women. The charts below use real published data instead.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Real-World Gender Gap (GBD 2023 — Prevalent Cases per 100,000)</div>', unsafe_allow_html=True)

    gbd_gender_data = pd.DataFrame({
        "Region": ["Sub-Saharan Africa", "MENA", "Europe", "Americas", "South Asia", "East Asia & Pacific"],
        "Males": [62, 215, 290, 240, 95, 130],
        "Females": [48, 165, 230, 195, 80, 115]
    })
    gbd_gender_data["Male_to_Female_Ratio"] = (gbd_gender_data["Males"] / gbd_gender_data["Females"]).round(2)

    col1, col2 = st.columns(2)
    with col1:
        fig_gbd = go.Figure()
        fig_gbd.add_trace(go.Bar(x=gbd_gender_data["Region"], y=gbd_gender_data["Males"],
                                  name="Males", marker_color=COLORS["primary"]))
        fig_gbd.add_trace(go.Bar(x=gbd_gender_data["Region"], y=gbd_gender_data["Females"],
                                  name="Females", marker_color=COLORS["warning"]))
        fig_gbd.update_layout(barmode="group", title="PD Prevalence by Sex & Region (GBD 2023)",
                               yaxis_title="Cases per 100,000")
        fig_gbd = apply_theme(fig_gbd)
        st.plotly_chart(fig_gbd, use_container_width=True)

    with col2:
        fig_ratio = px.bar(gbd_gender_data.sort_values("Male_to_Female_Ratio"),
                            x="Male_to_Female_Ratio", y="Region", orientation="h",
                            title="Male-to-Female Prevalence Ratio by Region",
                            color="Male_to_Female_Ratio",
                            color_continuous_scale=["#fbbf24", "#dc2626"],
                            labels={"Male_to_Female_Ratio": "Ratio (Male:Female)"})
        fig_ratio.add_vline(x=1, line_dash="dash", line_color=COLORS["muted"], annotation_text="Equal (1.0)")
        fig_ratio = apply_theme(fig_ratio)
        st.plotly_chart(fig_ratio, use_container_width=True)

    st.markdown('<div class="section-header">Age Distribution</div>', unsafe_allow_html=True)
    fig = px.box(filtered_df, x=filtered_df["Diagnosis"].map({1:"PD+", 0:"PD-"}), y="Age",
                 color=filtered_df["Diagnosis"].map({1:"PD+", 0:"PD-"}),
                 color_discrete_map={"PD+": COLORS["primary"], "PD-": COLORS["secondary"]},
                 title="Age Distribution by Diagnosis")
    fig = apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div style="margin-top:12px; padding:16px; background:#1a1d27; border:1px solid #2d3147; border-radius:10px; color:#7a7f9a; font-size:13px;">
    📖 <b>Data Source:</b> Global Burden of Disease Study 2023 (IHME) — GBD Compare tool, age-standardized prevalent cases per 100,000 by sex and region.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: CLINICAL ANALYSIS
# ─────────────────────────────────────────────
elif page == "Clinical Analysis":
    st.markdown("## Clinical Analysis — Early Symptom Detection")
    st.markdown(f"<span style='color:{COLORS['muted']}'>Symptoms and cognitive scores that can help flag PD before formal diagnosis</span>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Motor Symptoms Prevalence</div>', unsafe_allow_html=True)
    symptoms = ["Tremor", "Rigidity", "Bradykinesia", "PosturalInstability"]
    symptom_labels = ["Tremor", "Rigidity", "Bradykinesia", "Postural Instability"]
    
    pd_pos = filtered_df[filtered_df["Diagnosis"] == 1]
    pd_neg = filtered_df[filtered_df["Diagnosis"] == 0]
    
    sym_data = pd.DataFrame({
        "Symptom": symptom_labels,
        "PD Positive (%)": [pd_pos[s].mean() * 100 for s in symptoms],
        "PD Negative (%)": [pd_neg[s].mean() * 100 for s in symptoms]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name="PD Positive", x=sym_data["Symptom"], y=sym_data["PD Positive (%)"],
                         marker_color=COLORS["primary"]))
    fig.add_trace(go.Bar(name="PD Negative", x=sym_data["Symptom"], y=sym_data["PD Negative (%)"],
                         marker_color=COLORS["secondary"]))
    fig.update_layout(barmode="group", title="Motor Symptom Prevalence (%)")
    fig = apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"<span style='color:{COLORS['muted']}; font-size:13px;'>Note: Speech Problems was excluded — it showed no meaningful difference between PD+ and PD- patients (~29-30% both groups).</span>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Cognitive Score (MoCA)</div>', unsafe_allow_html=True)
    fig = px.box(filtered_df, x=filtered_df["Diagnosis"].map({1:"PD+", 0:"PD-"}), y="MoCA",
                 color=filtered_df["Diagnosis"].map({1:"PD+", 0:"PD-"}),
                 color_discrete_map={"PD+": COLORS["primary"], "PD-": COLORS["secondary"]},
                 title="MoCA Cognitive Score by Diagnosis")
    fig.add_hline(y=26, line_dash="dash", line_color=COLORS["warning"],
                  annotation_text="Normal cutoff (26)", annotation_position="top right")
    fig = apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: RISK FACTORS
# ─────────────────────────────────────────────
elif page == "Risk Factors":
    st.markdown("## Risk Factor Analysis")
    st.markdown(f"<span style='color:{COLORS['muted']}'>Identifying key contributors to Parkinson's disease</span>", unsafe_allow_html=True)

    risk_factors = {
        "FamilyHistoryParkinsons": "Family History",
        "TraumaticBrainInjury": "Traumatic Brain Injury",
        "Depression": "Depression",
        "Hypertension": "Hypertension",
        "Diabetes": "Diabetes",
        "Smoking": "Smoking",
        "Stroke": "Stroke",
    }
    
    risk_data = []
    for col, label in risk_factors.items():
        total_with = filtered_df[filtered_df[col] == 1]
        total_without = filtered_df[filtered_df[col] == 0]
        rate_with = total_with["Diagnosis"].mean() * 100 if len(total_with) > 0 else 0
        rate_without = total_without["Diagnosis"].mean() * 100 if len(total_without) > 0 else 0
        risk_data.append({"Factor": label, "With Factor (%)": rate_with, "Without Factor (%)": rate_without,
                           "Difference": rate_with - rate_without})

    # AlcoholConsumption is a continuous 0-10 scale in this dataset, not binary — split by median instead
    alcohol_median = filtered_df["AlcoholConsumption"].median()
    high_alcohol = filtered_df[filtered_df["AlcoholConsumption"] >= alcohol_median]
    low_alcohol = filtered_df[filtered_df["AlcoholConsumption"] < alcohol_median]
    rate_high = high_alcohol["Diagnosis"].mean() * 100 if len(high_alcohol) > 0 else 0
    rate_low = low_alcohol["Diagnosis"].mean() * 100 if len(low_alcohol) > 0 else 0
    risk_data.append({"Factor": "Alcohol Consumption (Above Median)", "With Factor (%)": rate_high,
                       "Without Factor (%)": rate_low, "Difference": rate_high - rate_low})
    
    risk_df = pd.DataFrame(risk_data).sort_values("Difference", ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(y=risk_df["Factor"], x=risk_df["With Factor (%)"], name="With Risk Factor",
                         orientation="h", marker_color=COLORS["primary"]))
    fig.add_trace(go.Bar(y=risk_df["Factor"], x=risk_df["Without Factor (%)"], name="Without Risk Factor",
                         orientation="h", marker_color=COLORS["secondary"]))
    fig.update_layout(barmode="group", title="PD Diagnosis Rate With vs Without Risk Factor (%)", height=450)
    fig = apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"""
    <span style='color:{COLORS['muted']}; font-size:13px;'>
    ⚠️ Note: Several results here (Traumatic Brain Injury, Family History, Smoking, Hypertension) don't show the strong associations 
    seen in established research — likely a limitation of this sample dataset rather than a contradiction of the literature. 
    See the Smoking & PD and Pesticide Exposure pages for findings grounded in published, peer-reviewed research.
    </span>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Correlation Matrix: Clinical Variables</div>', unsafe_allow_html=True)
    corr_cols = ["Age", "BMI", "UPDRS", "MoCA", "FunctionalAssessment", "SleepQuality", "PhysicalActivity", "DietQuality"]
    corr_matrix = filtered_df[corr_cols + ["Diagnosis"]].corr()
    fig = px.imshow(corr_matrix, color_continuous_scale=["#059669", "#f5f7ff", "#6d28d9"],
                    title="Correlation Matrix: Clinical Variables", text_auto=".2f", aspect="auto")
    fig = apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"""
    <span style='color:{COLORS['muted']}; font-size:13px;'>
    This is the most reliable chart on this page — UPDRS (0.40), Functional Assessment (-0.23), and MoCA (-0.17) show the strongest 
    correlation with diagnosis, while lifestyle factors (Age, BMI, Sleep, Activity, Diet) show near-zero correlation in this dataset.
    </span>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: ML PREDICTION
# ─────────────────────────────────────────────
elif page == "ML Prediction":
    st.markdown("## ML-Powered Diagnosis Prediction")
    st.markdown(f"<span style='color:{COLORS['muted']}'>Random Forest classifier trained on clinical features</span>", unsafe_allow_html=True)

    @st.cache_resource
    def train_model(df):
        features = ["Age", "BMI", "UPDRS", "MoCA", "FunctionalAssessment",
                    "Tremor", "Rigidity", "Bradykinesia", "PosturalInstability",
                    "SpeechProblems", "FamilyHistoryParkinsons", "TraumaticBrainInjury",
                    "Depression", "SleepQuality", "PhysicalActivity"]
        X = df[features]
        y = df["Diagnosis"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        importance = pd.DataFrame({"Feature": features, "Importance": clf.feature_importances_}).sort_values("Importance", ascending=False)
        cm = confusion_matrix(y_test, y_pred)
        return clf, acc, report, importance, cm, features

    clf, acc, report, importance, cm, features = train_model(df)

    col1, col2, col3 = st.columns(3)
    for col, label, val in [
        (col1, "Model Accuracy", f"{acc*100:.1f}%"),
        (col2, "Precision (PD+)", f"{report['1']['precision']*100:.1f}%"),
        (col3, "Recall (PD+)", f"{report['1']['recall']*100:.1f}%"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-delta">Random Forest · 150 trees</div>
            </div>
            """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(importance.head(10), x="Importance", y="Feature", orientation="h",
                     title="Top 10 Feature Importances",
                     color="Importance", color_continuous_scale=["#5b21b6", "#6d28d9"])
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.imshow(cm, text_auto=True,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=["PD Negative", "PD Positive"], y=["PD Negative", "PD Positive"],
                        color_continuous_scale=["#ede9fe", "#6d28d9"],
                        title="Confusion Matrix")
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">🔍 Predict for a New Patient</div>', unsafe_allow_html=True)
    st.markdown("Enter patient details to get a predicted diagnosis:")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        p_age = st.slider("Age", 50, 90, 65)
        p_bmi = st.slider("BMI", 15.0, 45.0, 27.0)
        p_updrs = st.slider("UPDRS Score", 0, 199, 50)
        p_moca = st.slider("MoCA Score", 0, 30, 22)
        p_functional = st.slider("Functional Assessment", 0.0, 10.0, 5.0)
    with c2:
        p_tremor = st.selectbox("Tremor", ["No", "Yes"])
        p_rigidity = st.selectbox("Rigidity", ["No", "Yes"])
        p_brady = st.selectbox("Bradykinesia", ["No", "Yes"])
        p_postural = st.selectbox("Postural Instability", ["No", "Yes"])
        p_speech = st.selectbox("Speech Problems", ["No", "Yes"])
    with c3:
        p_family = st.selectbox("Family History of PD", ["No", "Yes"])
        p_tbi = st.selectbox("Traumatic Brain Injury", ["No", "Yes"])
        p_depression = st.selectbox("Depression", ["No", "Yes"])
        p_sleep = st.slider("Sleep Quality", 4.0, 10.0, 7.0)
        p_activity = st.slider("Physical Activity", 0.0, 10.0, 5.0)
    
    def yn(v): return 1 if v == "Yes" else 0
    
    if st.button("🔍 Run Prediction", use_container_width=True):
        patient = pd.DataFrame([[
            p_age, p_bmi, p_updrs, p_moca, p_functional,
            yn(p_tremor), yn(p_rigidity), yn(p_brady), yn(p_postural), yn(p_speech),
            yn(p_family), yn(p_tbi), yn(p_depression), p_sleep, p_activity
        ]], columns=features)
        
        prediction = clf.predict(patient)[0]
        probability = clf.predict_proba(patient)[0]
        
        if prediction == 1:
            st.markdown(f"""
            <div class="pred-result-pos">
                <div class="pred-label" style="color:#6d28d9;">⚠️ PD Positive — High Risk</div>
                <div style="color:#6d28d9; margin-top:8px; font-size:15px;">
                    Confidence: <b>{probability[1]*100:.1f}%</b> · Clinical review recommended
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pred-result-neg">
                <div class="pred-label" style="color:#059669;">✅ PD Negative — Low Risk</div>
                <div style="color:#059669; margin-top:8px; font-size:15px;">
                    Confidence: <b>{probability[0]*100:.1f}%</b> · Continue routine monitoring
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        prob_fig = go.Figure(go.Bar(
            x=["PD Negative", "PD Positive"],
            y=[probability[0]*100, probability[1]*100],
            marker_color=[COLORS["secondary"], COLORS["primary"]]
        ))
        prob_fig.update_layout(title="Prediction Probability (%)", yaxis_range=[0, 100])
        prob_fig = apply_theme(prob_fig)
        st.plotly_chart(prob_fig, use_container_width=True)

    st.markdown("""
    <div style="margin-top:20px; padding:16px; background:#1a1d27; border:1px solid #2d3147; border-radius:10px; color:#7a7f9a; font-size:13px;">
    ⚠️ <b>Disclaimer:</b> This predictive model is built for educational purposes as part of MSBA382. 
    It should not be used for actual clinical diagnosis. Always consult a licensed medical professional.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: GLOBAL MAP
# ─────────────────────────────────────────────
elif page == "Global Map":
    st.markdown("## Global Parkinson's Disease Burden")
    st.markdown(f"<span style='color:{COLORS['muted']}'>Prevalence per 100,000 people · Both sexes, Age-standardized, 2023 · Source: GBD Compare (IHME)</span>", unsafe_allow_html=True)

    # GBD 2023 data — Both sexes, Age-standardized, prevalent cases per 100,000
    # Extracted directly from GBD Compare tool screenshots
    global_data = {
        "Country": [
            # MENA
            "Algeria", "Bahrain", "Comoros", "Djibouti", "Egypt", "Iraq", "Jordan", "Kuwait",
            "Lebanon", "Libya", "Mauritania", "Morocco", "Oman", "Palestine", "Qatar",
            "Saudi Arabia", "Somalia", "Sudan", "Syria", "Tunisia", "UAE", "Yemen",
            "Afghanistan", "Iran", "Pakistan",
            # Sub-Saharan Africa
            "Botswana", "Mauritius", "Seychelles", "South Africa", "Nigeria", "Kenya",
            "Ethiopia", "Rwanda",
            # Europe
            "Italy", "Israel", "Spain", "Turkey", "Greece", "Germany", "France",
            "United Kingdom", "Sweden", "Switzerland", "Hungary", "Kazakhstan",
            # Americas
            "Brazil", "Paraguay", "United States", "Canada", "Argentina", "Mexico",
            "Bolivia", "Panama",
            # South Asia
            "India", "Nepal", "Bangladesh", "Sri Lanka", "Maldives", "Bhutan", "N Korea",
            # East Asia & Pacific
            "Japan", "South Korea", "China", "Australia", "New Zealand", "Indonesia",
            "Malaysia", "Philippines", "Vietnam", "Singapore",
        ],
        "Prevalence_per_100k": [
            158, 180, 65, 55, 183, 197, 125, 181,
            148, 162, 53, 150, 215, 154, 177,
            255, 41, 142, 168, 148, 130, 115,
            139, 162, 86,
            55, 119, 138, 54, 52, 54,
            55, 41,
            198, 217, 162, 310, 198, 113, 113,
            113, 120, 165, 113, 118,
            282, 290, 156, 119, 144, 167,
            173, 162,
            67, 75, 83, 113, 111, 83, 132,
            89, 121, 148, 156, 109, 88,
            101, 109, 102, 158,
        ],
        "Region": [
            "MENA"]*25 + ["Sub-Saharan Africa"]*8 + ["Europe"]*12 + ["Americas"]*8 + ["South Asia"]*7 + ["East Asia & Pacific"]*10
    }

    gdf = pd.DataFrame(global_data)

    # KPI row
    top_country = gdf.loc[gdf["Prevalence_per_100k"].idxmax()]
    low_country = gdf.loc[gdf["Prevalence_per_100k"].idxmin()]
    col1, col2, col3, col4 = st.columns(4)
    for col, label, val in [
        (col1, "Highest Prevalence", f"{top_country['Country']} · {top_country['Prevalence_per_100k']}/100k"),
        (col2, "Lowest Prevalence", f"{low_country['Country']} · {low_country['Prevalence_per_100k']}/100k"),
        (col3, "Global Average", f"{int(gdf['Prevalence_per_100k'].mean())}/100k"),
        (col4, "Countries Tracked", f"{len(gdf)}"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:22px;">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Top 20 Countries by PD Prevalence per 100,000</div>', unsafe_allow_html=True)

    top20 = gdf.nlargest(20, "Prevalence_per_100k").sort_values("Prevalence_per_100k", ascending=True)
    fig = px.bar(top20, x="Prevalence_per_100k", y="Country", orientation="h",
                 color="Prevalence_per_100k",
                 color_continuous_scale=["#5b21b6", "#7c3aed", "#6d28d9"],
                 labels={"Prevalence_per_100k": "Prevalence per 100,000"},
                 title="Top 20 Countries by PD Prevalence per 100,000 (GBD 2023)")
    fig = apply_theme(fig)
    fig.update_layout(height=550, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        region_avg = gdf.groupby("Region")["Prevalence_per_100k"].mean().reset_index().sort_values("Prevalence_per_100k", ascending=True)
        fig2 = px.bar(region_avg, x="Prevalence_per_100k", y="Region", orientation="h",
                      title="Average Prevalence by Region",
                      color="Prevalence_per_100k",
                      color_continuous_scale=["#5b21b6", "#6d28d9"])
        fig2 = apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        top10 = gdf.nlargest(10, "Prevalence_per_100k")
        fig3 = px.bar(top10, x="Prevalence_per_100k", y="Country", orientation="h",
                      title="Top 10 Countries by PD Prevalence",
                      color="Prevalence_per_100k",
                      color_continuous_scale=["#5b21b6", "#6d28d9"])
        fig3 = apply_theme(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    <div style="margin-top:12px; padding:16px; background:#1a1d27; border:1px solid #2d3147; border-radius:10px; color:#7a7f9a; font-size:13px;">
    📖 <b>Data Source:</b> Global Burden of Disease Study 2023 (IHME) — GBD Compare tool, age-standardized prevalent cases per 100,000 population (both sexes). 
    Since these figures are age-standardized, differences between countries are not simply explained by older populations — they more likely reflect 
    diagnostic access, healthcare infrastructure, and reporting differences, alongside any genuine regional variation.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: SMOKING & PD
# ─────────────────────────────────────────────
elif page == "Smoking & PD":
    st.markdown("## Smoking & Parkinson's Disease")
    st.markdown(f"<span style='color:{COLORS['muted']}'>DALYs attributed to smoking · GBD Compare 2023 · Age-standardized</span>", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin:16px 0; padding:16px; background:#1a1d27; border-left:4px solid #a78bfa; border-radius:8px; color:#e8e8f0; font-size:14px;">
    ⚠️ <b>Negative DALYs = Protective Effect.</b> Across all regions, smoking is associated with <i>fewer</i> DALYs from Parkinson's disease —
    meaning smokers appear less likely to develop PD. This is one of the most well-documented paradoxes in neurology, 
    thought to be linked to nicotine's effect on dopaminergic pathways. Males consistently show a stronger protective association than females.
    </div>
    """, unsafe_allow_html=True)

    # GBD 2023 data extracted from screenshots — DALYs per 100,000 (negative = protective)
    smoking_data = {
        "Region": [
            "MENA", "MENA", "MENA", "MENA", "MENA", "MENA", "MENA", "MENA",
            "Europe", "Europe", "Europe", "Europe", "Europe", "Europe", "Europe", "Europe",
            "Latin America", "Latin America", "Latin America", "Latin America", "Latin America", "Latin America",
            "South Asia", "South Asia", "South Asia", "South Asia", "South Asia",
            "East Asia & Pacific", "East Asia & Pacific", "East Asia & Pacific", "East Asia & Pacific", "East Asia & Pacific",
            "Sub-Saharan Africa", "Sub-Saharan Africa", "Sub-Saharan Africa", "Sub-Saharan Africa", "Sub-Saharan Africa",
        ],
        "Country": [
            "Iraq", "Jordan", "Egypt", "Tunisia", "Lebanon", "Libya", "Syria", "Bahrain",
            "Albania", "Greece", "Montenegro", "Turkey", "Croatia", "Bosnia & Herz.", "Spain", "Germany",
            "Paraguay", "United States", "Cuba", "Uruguay", "Canada", "Brazil",
            "Maldives", "Bangladesh", "Thailand", "N Korea", "Nepal",
            "Solomon Is.", "Nauru", "Tuvalu", "Viet Nam", "China",
            "Seychelles", "Rwanda", "Algeria", "Mauritius", "Botswana",
        ],
        "Males_DALYs": [
            -38, -22, -23, -21, -14, -18, -16, -13,
            -27, -26, -22, -20, -19, -18, -16, -13,
            -20, -16, -15, -13, -13, -11,
            -30, -22, -16, -14, -12,
            -27, -25, -25, -23, -22,
            -22, -18, -17, -16, -14,
        ],
        "Females_DALYs": [
            -2, -1, -1, -1, -5, -1, -1, -1,
            -5, -4, -4, -3, -4, -3, -3, -3,
            -7, -6, -3, -3, -6, -3,
            -1, -1, -1, -1, -8,
            -1, -10, -2, -2, -2,
            -2, -1, -2, -1, -1,
        ]
    }

    sdf = pd.DataFrame(smoking_data)

    # KPI cards
    col1, col2, col3 = st.columns(3)
    for col, label, val, delta in [
        (col1, "Strongest Protection (Males)", "Maldives · -30 DALYs", "South Asia"),
        (col2, "Strongest Protection (Females)", "Nepal · -8 DALYs", "South Asia"),
        (col3, "Gender Gap", "Males 5-10x stronger", "Consistent across all regions"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:18px;">{val}</div>
                <div class="metric-delta">{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Tobacco: The Only GBD-Tracked Risk Factor for PD</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <span style='color:{COLORS['muted']}; font-size:14px;'>
    Across GBD's entire risk factor framework — covering air pollution, high BMI, blood pressure, alcohol, diet, occupational risks, and more — 
    <b>tobacco is the only risk factor with measurable DALYs for Parkinson's disease</b>, in every single WHO region. This holds true globally, 
    confirming smoking is uniquely significant in PD research (even though the direction is protective, not harmful).
    </span>
    """, unsafe_allow_html=True)

    who_region_data = pd.DataFrame({
        "WHO Region": ["Western Pacific", "Region of the Americas", "European Region", "Eastern Mediterranean",
                        "South-East Asia", "African Region"],
        "Tobacco_DALYs": [-10.3, -6.8, -6.5, -6.3, -5.0, -2.0]
    }).sort_values("Tobacco_DALYs")

    fig_who = px.bar(who_region_data, x="Tobacco_DALYs", y="WHO Region", orientation="h",
                      title="Tobacco-Attributable DALYs for PD by WHO Region (GBD 2023) — All Other Risk Factors Show Zero",
                      color="Tobacco_DALYs",
                      color_continuous_scale=["#5b21b6", "#a78bfa"],
                      labels={"Tobacco_DALYs": "DALYs per 100,000 (negative = protective effect)"})
    fig_who.add_vline(x=0, line_dash="dash", line_color=COLORS["muted"])
    fig_who = apply_theme(fig_who)
    fig_who.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_who, use_container_width=True)
    st.markdown(f"<span style='color:{COLORS['muted']}; font-size:13px;'>This is also why pesticide exposure — a well-documented PD risk factor in peer-reviewed literature — does not yet appear in GBD's standard risk factor model, which is why we sourced it separately on the Pesticide Exposure page.</span>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Smoking-Attributable DALYs by Region (Males vs Females)</div>', unsafe_allow_html=True)

    region_avg = sdf.groupby("Region")[["Males_DALYs", "Females_DALYs"]].mean().reset_index()
    region_avg = region_avg.sort_values("Males_DALYs")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=region_avg["Region"], x=region_avg["Males_DALYs"],
        name="Males", orientation="h", marker_color=COLORS["primary"]
    ))
    fig.add_trace(go.Bar(
        y=region_avg["Region"], x=region_avg["Females_DALYs"],
        name="Females", orientation="h", marker_color=COLORS["warning"]
    ))
    fig.update_layout(
        barmode="group",
        title="Average Smoking-Attributable DALYs per 100,000 by Region (GBD 2023)",
        xaxis_title="DALYs per 100,000 (negative = protective effect)",
        height=400
    )
    fig = apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Country-Level Detail</div>', unsafe_allow_html=True)

    region_select = st.selectbox("Select Region", sdf["Region"].unique())
    region_df = sdf[sdf["Region"] == region_select].sort_values("Males_DALYs")

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.bar(region_df, x="Males_DALYs", y="Country", orientation="h",
                      title=f"Males — {region_select}",
                      color="Males_DALYs",
                      color_continuous_scale=["#6d28d9", "#a78bfa"],
                      labels={"Males_DALYs": "DALYs per 100,000"})
        fig2 = apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        fig3 = px.bar(region_df, x="Females_DALYs", y="Country", orientation="h",
                      title=f"Females — {region_select}",
                      color="Females_DALYs",
                      color_continuous_scale=["#d97706", "#fbbf24"],
                      labels={"Females_DALYs": "DALYs per 100,000"})
        fig3 = apply_theme(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    <div style="margin-top:12px; padding:16px; background:#1a1d27; border:1px solid #2d3147; border-radius:10px; color:#7a7f9a; font-size:13px;">
    📖 <b>Data Source:</b> Global Burden of Disease Study 2023 (IHME) — GBD Compare tool, age-standardized DALYs attributed to smoking for Parkinson's disease.
    Note: The protective association of smoking with PD does not imply smoking is beneficial — its harms far outweigh any neuroprotective effect.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: PESTICIDE EXPOSURE
# ─────────────────────────────────────────────
elif page == "Pesticide Exposure":
    st.markdown("## Pesticide Exposure & Parkinson's Disease")
    st.markdown(f"<span style='color:{COLORS['muted']}'>Odds ratios from peer-reviewed epidemiological studies</span>", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin:16px 0; padding:16px; background:#1a1d27; border-left:4px solid #dc2626; border-radius:8px; color:#e8e8f0; font-size:14px;">
    ⚠️ <b>Pesticide exposure is the strongest documented environmental risk factor for Parkinson's disease.</b>
    Unlike smoking, which shows a protective association, every major study below confirms pesticides significantly 
    <i>increase</i> PD risk — with effects compounding by exposure duration, frequency, and proximity (occupational vs household).
    </div>
    """, unsafe_allow_html=True)

    pest_data = pd.DataFrame({
        "Study": [
            "Direct Pesticide Application\n(Family-based, Tanner Lab)",
            "Household Pesticides – Males\n(Payami et al.)",
            "Household Pesticides – Females\n(Payami et al.)",
            "Household Pesticides – All\n(Moura et al.)",
            "Household Pesticides – Males\n(Moura et al.)",
            "Household Pesticides – Females\n(Moura et al.)",
            "Frequent Household Use\n(California study, Narayan et al.)",
            "Occupational Exposure – Mortality Risk\n(Brazil cohort, HR)"
        ],
        "Odds_Ratio": [1.61, 2.52, 2.85, 2.27, 2.19, 2.43, 1.37, 2.32],
        "CI_Lower": [1.13, 1.37, 1.87, 1.46, 1.18, 1.28, 1.13, 1.15],
        "CI_Upper": [2.29, 4.0, 4.0, 3.52, 4.04, 4.6, 1.92, 4.66],
        "Type": ["Case-Control", "Case-Control", "Case-Control", "Case-Control",
                 "Case-Control", "Case-Control", "Case-Control", "Mortality (HR)"]
    })

    col1, col2, col3 = st.columns(3)
    for col, label, val, delta in [
        (col1, "Highest Risk Increase", "2.85x", "Females, household exposure"),
        (col2, "Mortality Risk (PD+)", "2.32x", "Occupational exposure"),
        (col3, "Population Impact", "10-20%", "Of new PD cases potentially preventable"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:26px;">{val}</div>
                <div class="metric-delta">{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Odds Ratios Across Studies (with 95% Confidence Intervals)</div>', unsafe_allow_html=True)

    pest_sorted = pest_data.sort_values("Odds_Ratio")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pest_sorted["Odds_Ratio"], y=pest_sorted["Study"],
        error_x=dict(
            type="data",
            symmetric=False,
            array=pest_sorted["CI_Upper"] - pest_sorted["Odds_Ratio"],
            arrayminus=pest_sorted["Odds_Ratio"] - pest_sorted["CI_Lower"]
        ),
        mode="markers",
        marker=dict(size=14, color=COLORS["danger"]),
        name="Odds Ratio"
    ))
    fig.add_vline(x=1, line_dash="dash", line_color=COLORS["muted"], annotation_text="No effect (OR=1)")
    fig.update_layout(
        title="Pesticide Exposure & PD Risk — Odds Ratios by Study",
        xaxis_title="Odds Ratio (>1 = increased risk)",
        height=450
    )
    fig = apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Pesticide Risk by Gender</div>', unsafe_allow_html=True)
    gender_pest = pd.DataFrame({
        "Gender": ["Males", "Females", "Males", "Females"],
        "Study": ["Payami", "Payami", "Moura", "Moura"],
        "Odds_Ratio": [2.52, 2.85, 2.19, 2.43]
    })
    fig3 = px.bar(gender_pest, x="Study", y="Odds_Ratio", color="Gender",
                  color_discrete_map={"Males": COLORS["primary"], "Females": COLORS["warning"]},
                  barmode="group", title="Pesticide Risk by Gender — Females Consistently Higher")
    fig3.add_hline(y=1, line_dash="dash", line_color=COLORS["muted"])
    fig3 = apply_theme(fig3)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown(f"<span style='color:{COLORS['muted']}; font-size:13px;'>Note: This is the opposite pattern from smoking, where males showed a stronger (protective) effect — suggesting these risk factors interact differently with biological sex.</span>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Key Findings from Literature</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Dose-response relationship**: Frequency, duration, and cumulative pesticide exposure are all significantly associated with PD risk in a dose-response pattern.
    - **Chemical classes implicated**: Organochlorines and organophosphorus insecticides showed the strongest associations; herbicides also significantly increase risk.
    - **Family history interaction**: Pesticide-PD associations were strongest in individuals with *no* family history, suggesting pesticides may act as an independent trigger for sporadic PD.
    - **Mortality**: PD patients with occupational pesticide exposure are more than twice as likely to die compared to unexposed PD patients (HR = 2.32).
    - **Prevention potential**: Reducing excessive household pesticide use could prevent an estimated 10–20% of new PD cases.
    """)

    st.markdown("""
    <div style="margin-top:12px; padding:16px; background:#1a1d27; border:1px solid #2d3147; border-radius:10px; color:#7a7f9a; font-size:13px;">
    📖 <b>Data Sources:</b> Tanner et al., family-based case-control study (PMC2323015); Payami et al. and Moura et al., household pesticide studies 
    (PMC11024193); Brazilian occupational cohort study (PMC7298782). Odds ratios and hazard ratios extracted directly from published, peer-reviewed findings.
    </div>
    """, unsafe_allow_html=True)
