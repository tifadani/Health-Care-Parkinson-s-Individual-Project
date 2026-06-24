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
    .main { background-color: #0f1117; }
    
    .login-container {
        max-width: 420px;
        margin: 80px auto;
        background: #1a1d27;
        border: 1px solid #2d3147;
        border-radius: 16px;
        padding: 48px 40px;
        text-align: center;
        box-shadow: 0 8px 40px rgba(0,0,0,0.4);
    }
    .login-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 28px;
        font-weight: 700;
        color: #e8e8f0;
        margin-bottom: 6px;
    }
    .login-sub {
        font-size: 14px;
        color: #7a7f9a;
        margin-bottom: 32px;
    }
    
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2d3147;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 12px;
        color: #7a7f9a;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 32px;
        font-weight: 700;
        color: #a78bfa;
    }
    .metric-delta {
        font-size: 13px;
        color: #34d399;
        margin-top: 4px;
    }
    
    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 20px;
        font-weight: 700;
        color: #e8e8f0;
        margin: 28px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #2d3147;
    }
    
    .stSelectbox > div > div { background-color: #1a1d27 !important; }
    .stSlider > div { color: #a78bfa; }

    div[data-testid="stSidebarNav"] { background: #1a1d27; }
    section[data-testid="stSidebar"] { background: #13151f; border-right: 1px solid #2d3147; }
    section[data-testid="stSidebar"] .css-1d391kg { padding: 24px 16px; }
    
    .pred-result-pos {
        background: linear-gradient(135deg, #1a1d27, #2d1f3d);
        border: 1px solid #7c3aed;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 16px;
    }
    .pred-result-neg {
        background: linear-gradient(135deg, #1a1d27, #1a2d2a);
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
    """
    Load from uploaded CSV or generate synthetic data matching Kaggle dataset schema.
    To use real data: replace this function with pd.read_csv("parkinsons_disease_data.csv")
    """
    np.random.seed(42)
    n = 2105

    age = np.random.randint(50, 91, n)
    gender = np.random.choice(["Male", "Female"], n, p=[0.55, 0.45])
    ethnicity = np.random.choice(["Caucasian", "African American", "Asian", "Other"], n, p=[0.6, 0.15, 0.15, 0.1])
    education = np.random.choice(["None", "High School", "Bachelor's", "Higher"], n, p=[0.1, 0.35, 0.35, 0.2])
    bmi = np.round(np.random.normal(27, 5, n), 1).clip(15, 45)
    smoking = np.random.choice([0, 1], n, p=[0.7, 0.3])
    alcohol = np.random.choice([0, 1], n, p=[0.6, 0.4])
    physical_activity = np.round(np.random.uniform(0, 10, n), 1)
    diet_quality = np.round(np.random.uniform(0, 10, n), 1)
    sleep_quality = np.round(np.random.uniform(4, 10, n), 1)
    family_history = np.random.choice([0, 1], n, p=[0.75, 0.25])
    traumatic_brain = np.random.choice([0, 1], n, p=[0.85, 0.15])
    hypertension = np.random.choice([0, 1], n, p=[0.65, 0.35])
    diabetes = np.random.choice([0, 1], n, p=[0.8, 0.2])
    depression = np.random.choice([0, 1], n, p=[0.7, 0.3])
    stroke = np.random.choice([0, 1], n, p=[0.9, 0.1])

    cholesterol_total = np.round(np.random.normal(200, 30, n), 1).clip(100, 300)
    cholesterol_ldl = np.round(np.random.normal(120, 25, n), 1).clip(50, 200)
    cholesterol_hdl = np.round(np.random.normal(55, 15, n), 1).clip(20, 100)
    cholesterol_trig = np.round(np.random.normal(150, 50, n), 1).clip(50, 400)

    updrs = np.round(np.random.normal(50, 30, n), 1).clip(0, 199)
    moca = np.round(np.random.normal(22, 5, n), 1).clip(0, 30)
    functional = np.round(np.random.uniform(0, 10, n), 1)
    tremor = np.random.choice([0, 1], n, p=[0.45, 0.55])
    rigidity = np.random.choice([0, 1], n, p=[0.5, 0.5])
    bradykinesia = np.random.choice([0, 1], n, p=[0.48, 0.52])
    postural = np.random.choice([0, 1], n, p=[0.55, 0.45])
    speech = np.random.choice([0, 1], n, p=[0.6, 0.4])
    sleep_disorders = np.random.choice([0, 1], n, p=[0.65, 0.35])
    constipation = np.random.choice([0, 1], n, p=[0.7, 0.3])

    # Diagnosis influenced by risk factors
    risk_score = (
        (age > 65).astype(int) * 2 +
        family_history * 3 +
        traumatic_brain * 2 +
        (updrs > 60).astype(int) * 2 +
        tremor + rigidity + bradykinesia + postural +
        depression + (moca < 20).astype(int)
    )
    diagnosis_prob = 1 / (1 + np.exp(-(risk_score - 8) * 0.3))
    diagnosis = (np.random.random(n) < diagnosis_prob).astype(int)

    df = pd.DataFrame({
        "PatientID": range(3058, 3058 + n),
        "Age": age, "Gender": gender, "Ethnicity": ethnicity,
        "EducationLevel": education, "BMI": bmi,
        "Smoking": smoking, "AlcoholConsumption": alcohol,
        "PhysicalActivity": physical_activity, "DietQuality": diet_quality,
        "SleepQuality": sleep_quality, "FamilyHistoryParkinsons": family_history,
        "TraumaticBrainInjury": traumatic_brain, "Hypertension": hypertension,
        "Diabetes": diabetes, "Depression": depression, "Stroke": stroke,
        "CholesterolTotal": cholesterol_total, "CholesterolLDL": cholesterol_ldl,
        "CholesterolHDL": cholesterol_hdl, "CholesterolTriglycerides": cholesterol_trig,
        "UPDRS": updrs, "MoCA": moca, "FunctionalAssessment": functional,
        "Tremor": tremor, "Rigidity": rigidity, "Bradykinesia": bradykinesia,
        "PosturalInstability": postural, "SpeechProblems": speech,
        "SleepDisorders": sleep_disorders, "Constipation": constipation,
        "Diagnosis": diagnosis
    })
    return df

df = load_data()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 PD Analytics")
    st.markdown("---")
    
    page = st.radio("Navigate", [
        "📊 Overview",
        "👥 Demographics",
        "🩺 Clinical Analysis",
        "🗺️ Risk Factors",
        "🤖 ML Prediction"
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
    "primary": "#a78bfa",
    "secondary": "#34d399",
    "danger": "#f87171",
    "warning": "#fbbf24",
    "bg": "#1a1d27",
    "grid": "#2d3147",
    "text": "#e8e8f0",
    "muted": "#7a7f9a"
}
PD_COLORS = ["#a78bfa", "#34d399"]

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
if page == "📊 Overview":
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
        age_diag["DiagnosisLabel"] = age_diag["Diagnosis"].map({1:"PD Positive", 0:"PD Negative"})
        grouped = age_diag.groupby(["AgeGroup","DiagnosisLabel"]).size().reset_index(name="Count")
        fig = px.bar(grouped, x="AgeGroup", y="Count", color="DiagnosisLabel",
                     color_discrete_map={"PD Positive": COLORS["primary"], "PD Negative": COLORS["secondary"]},
                     barmode="group", title="Diagnosis by Age Group")
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">UPDRS Severity Distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(filtered_df, x="UPDRS", color=filtered_df["Diagnosis"].map({1:"PD Positive", 0:"PD Negative"}),
                       nbins=40, opacity=0.75, barmode="overlay",
                       color_discrete_map={"PD Positive": COLORS["primary"], "PD Negative": COLORS["secondary"]},
                       title="UPDRS Score Distribution (Higher = More Severe)")
    fig = apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: DEMOGRAPHICS
# ─────────────────────────────────────────────
elif page == "👥 Demographics":
    st.markdown("## Demographics Analysis")
    st.markdown(f"<span style='color:{COLORS['muted']}'>Distribution of Parkinson's disease across patient groups</span>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        gender_diag = filtered_df.groupby(["Gender", filtered_df["Diagnosis"].map({1:"PD+", 0:"PD-"})]).size().reset_index(name="Count")
        gender_diag.columns = ["Gender", "Diagnosis", "Count"]
        fig = px.bar(gender_diag, x="Gender", y="Count", color="Diagnosis",
                     color_discrete_map={"PD+": COLORS["primary"], "PD-": COLORS["secondary"]},
                     barmode="group", title="Diagnosis by Gender")
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        eth_diag = filtered_df[filtered_df["Diagnosis"] == 1].groupby("Ethnicity").size().reset_index(name="Count")
        fig = px.bar(eth_diag, x="Ethnicity", y="Count", title="PD Cases by Ethnicity",
                     color="Count", color_continuous_scale=["#4c1d95", "#a78bfa"])
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(filtered_df, x=filtered_df["Diagnosis"].map({1:"PD+", 0:"PD-"}), y="Age",
                     color=filtered_df["Diagnosis"].map({1:"PD+", 0:"PD-"}),
                     color_discrete_map={"PD+": COLORS["primary"], "PD-": COLORS["secondary"]},
                     title="Age Distribution by Diagnosis")
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        edu_diag = filtered_df.groupby(["EducationLevel", filtered_df["Diagnosis"].map({1:"PD+", 0:"PD-"})]).size().reset_index(name="Count")
        edu_diag.columns = ["Education", "Diagnosis", "Count"]
        fig = px.bar(edu_diag, x="Education", y="Count", color="Diagnosis",
                     color_discrete_map={"PD+": COLORS["primary"], "PD-": COLORS["secondary"]},
                     barmode="stack", title="Diagnosis by Education Level")
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">BMI vs Age (PD Patients)</div>', unsafe_allow_html=True)
    pd_only = filtered_df[filtered_df["Diagnosis"] == 1]
    fig = px.scatter(pd_only, x="Age", y="BMI", color="Gender",
                     color_discrete_map={"Male": COLORS["primary"], "Female": COLORS["warning"]},
                     opacity=0.6, title="BMI vs Age in PD-Positive Patients",
                     trendline="lowess")
    fig = apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: CLINICAL ANALYSIS
# ─────────────────────────────────────────────
elif page == "🩺 Clinical Analysis":
    st.markdown("## Clinical Analysis")
    st.markdown(f"<span style='color:{COLORS['muted']}'>Symptoms, cognitive scores, and clinical measurements</span>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Motor Symptoms Prevalence</div>', unsafe_allow_html=True)
    symptoms = ["Tremor", "Rigidity", "Bradykinesia", "PosturalInstability", "SpeechProblems"]
    symptom_labels = ["Tremor", "Rigidity", "Bradykinesia", "Postural Instability", "Speech Problems"]
    
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

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(filtered_df, x=filtered_df["Diagnosis"].map({1:"PD+", 0:"PD-"}), y="MoCA",
                     color=filtered_df["Diagnosis"].map({1:"PD+", 0:"PD-"}),
                     color_discrete_map={"PD+": COLORS["primary"], "PD-": COLORS["secondary"]},
                     title="MoCA Cognitive Score by Diagnosis")
        fig.add_hline(y=26, line_dash="dash", line_color=COLORS["warning"],
                      annotation_text="Normal cutoff (26)", annotation_position="top right")
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(filtered_df[filtered_df["Diagnosis"]==1], x="UPDRS", y="MoCA",
                         color="Age", color_continuous_scale=["#4c1d95", "#a78bfa", "#f87171"],
                         title="UPDRS vs MoCA (PD Patients)", opacity=0.6)
        fig.update_layout(coloraxis_colorbar=dict(title="Age"))
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Sleep & Lifestyle Factors</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    for col, metric, label in [
        (col1, "SleepQuality", "Sleep Quality"),
        (col2, "PhysicalActivity", "Physical Activity"),
        (col3, "DietQuality", "Diet Quality"),
    ]:
        with col:
            fig = px.violin(filtered_df, x=filtered_df["Diagnosis"].map({1:"PD+", 0:"PD-"}), y=metric,
                            color=filtered_df["Diagnosis"].map({1:"PD+", 0:"PD-"}),
                            color_discrete_map={"PD+": COLORS["primary"], "PD-": COLORS["secondary"]},
                            box=True, title=f"{label} by Diagnosis")
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: RISK FACTORS
# ─────────────────────────────────────────────
elif page == "🗺️ Risk Factors":
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
        "AlcoholConsumption": "Alcohol Consumption"
    }
    
    risk_data = []
    for col, label in risk_factors.items():
        total_with = filtered_df[filtered_df[col] == 1]
        total_without = filtered_df[filtered_df[col] == 0]
        rate_with = total_with["Diagnosis"].mean() * 100 if len(total_with) > 0 else 0
        rate_without = total_without["Diagnosis"].mean() * 100 if len(total_without) > 0 else 0
        risk_data.append({"Factor": label, "With Factor (%)": rate_with, "Without Factor (%)": rate_without,
                           "Difference": rate_with - rate_without})
    
    risk_df = pd.DataFrame(risk_data).sort_values("Difference", ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(y=risk_df["Factor"], x=risk_df["With Factor (%)"], name="With Risk Factor",
                         orientation="h", marker_color=COLORS["primary"]))
    fig.add_trace(go.Bar(y=risk_df["Factor"], x=risk_df["Without Factor (%)"], name="Without Risk Factor",
                         orientation="h", marker_color=COLORS["secondary"]))
    fig.update_layout(barmode="group", title="PD Diagnosis Rate With vs Without Risk Factor (%)", height=450)
    fig = apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Comorbidity Impact on UPDRS Severity</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        pd_data = filtered_df[filtered_df["Diagnosis"] == 1]
        comorbidity_score = pd_data[["Hypertension","Diabetes","Depression","Stroke"]].sum(axis=1)
        pd_data = pd_data.copy()
        pd_data["ComorbidityCount"] = comorbidity_score
        fig = px.box(pd_data, x="ComorbidityCount", y="UPDRS",
                     title="UPDRS by Number of Comorbidities (PD+)",
                     color="ComorbidityCount",
                     color_discrete_sequence=[COLORS["primary"], COLORS["warning"], COLORS["danger"], "#60a5fa", "#f472b6"])
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        corr_cols = ["Age", "BMI", "UPDRS", "MoCA", "FunctionalAssessment", "SleepQuality", "PhysicalActivity", "DietQuality"]
        corr_matrix = filtered_df[corr_cols + ["Diagnosis"]].corr()
        fig = px.imshow(corr_matrix, color_continuous_scale=["#059669", "#1a1d27", "#7c3aed"],
                        title="Correlation Matrix: Clinical Variables", text_auto=".2f", aspect="auto")
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Cholesterol Profile in PD Patients</div>', unsafe_allow_html=True)
    chol_data = filtered_df[filtered_df["Diagnosis"] == 1][["CholesterolTotal", "CholesterolLDL", "CholesterolHDL", "Gender"]]
    fig = px.scatter(chol_data, x="CholesterolLDL", y="CholesterolHDL",
                     color="Gender", color_discrete_map={"Male": COLORS["primary"], "Female": COLORS["warning"]},
                     opacity=0.5, title="LDL vs HDL Cholesterol in PD Patients", trendline="ols")
    fig = apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: ML PREDICTION
# ─────────────────────────────────────────────
elif page == "🤖 ML Prediction":
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
                     color="Importance", color_continuous_scale=["#4c1d95", "#a78bfa"])
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.imshow(cm, text_auto=True,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=["PD Negative", "PD Positive"], y=["PD Negative", "PD Positive"],
                        color_continuous_scale=["#1a1d27", "#7c3aed"],
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
                <div class="pred-label" style="color:#a78bfa;">⚠️ PD Positive — High Risk</div>
                <div style="color:#c4b5fd; margin-top:8px; font-size:15px;">
                    Confidence: <b>{probability[1]*100:.1f}%</b> · Clinical review recommended
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pred-result-neg">
                <div class="pred-label" style="color:#34d399;">✅ PD Negative — Low Risk</div>
                <div style="color:#6ee7b7; margin-top:8px; font-size:15px;">
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
