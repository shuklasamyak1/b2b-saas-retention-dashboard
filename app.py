import streamlit as st
import pandas as pd
import numpy as np
import pulp
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="B2B SaaS Customer Retention & CLV Decision Engine",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- BESPOKE THEME & TYPOGRAPHY STYLING ---
# Palette:
# #2C5554 (Dark Slate / Deep Forest Teal)
# #574951 (Dim Slate Gray)
# #6DD5ED (Vibrant Sky Blue)
# #CDB4DB (Soft Thistle / Lavender)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

    /* Global Base */
    .stApp {
        background-color: #1e3d3c;
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #2C5554 !important;
        border-right: 1px solid rgba(109, 213, 237, 0.2) !important;
        font-family: 'Inter', sans-serif !important;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #FFFFFF !important;
    }

    /* Executive Glass Metric Containers */
    .metric-card {
        background: #364449;
        border: 1px solid rgba(109, 213, 237, 0.25);
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .metric-sub {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        color: #CDB4DB;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
    }
    .metric-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.65rem;
        font-weight: 700;
        color: #6DD5ED;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    .metric-caption {
        font-family: 'Inter', sans-serif;
        font-size: 0.74rem;
        color: #E2E8F0;
        margin-top: 4px;
    }

    /* Action Badges */
    .badge-treat {
        background: rgba(109, 213, 237, 0.15);
        color: #6DD5ED;
        border: 1px solid #6DD5ED;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .badge-pass {
        background: rgba(205, 180, 219, 0.15);
        color: #CDB4DB;
        border: 1px solid #CDB4DB;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(109, 213, 237, 0.25);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #2C5554 !important;
        border-radius: 6px 6px 0 0 !important;
        color: #E2E8F0 !important;
        padding: 8px 16px !important;
        border: 1px solid rgba(109, 213, 237, 0.2) !important;
        border-bottom: none !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #364449 !important;
        color: #6DD5ED !important;
        border: 1px solid #6DD5ED !important;
        border-bottom: 2px solid #6DD5ED !important;
        font-weight: 700 !important;
    }

    /* Headings & Text */
    h1, h2, h3, h4 {
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.4px;
    }
    p, span, label {
        color: #F1F5F9;
        font-family: 'Inter', sans-serif;
    }

    /* Sliders & Interactive Elements */
    .stSlider > div {
        color: #6DD5ED !important;
    }
</style>
""", unsafe_allow_html=True)

# --- SYNTHETIC B2B SAAS COHORT ENGINE ---
@st.cache_data
def generate_saas_data(n_accounts=300):
    np.random.seed(42)
    account_ids = [f"ACC-{1000 + i}" for i in range(n_accounts)]
    tiers = np.random.choice(["Starter", "Professional", "Enterprise"], size=n_accounts, p=[0.5, 0.35, 0.15])
    
    mrr_map = {"Starter": 150, "Professional": 600, "Enterprise": 2200}
    mrr = np.array([mrr_map[t] * np.random.uniform(0.85, 1.25) for t in tiers])
    
    tenure_months = np.random.randint(2, 48, size=n_accounts)
    logins_weekly = np.random.poisson(lam=18, size=n_accounts)
    support_tickets = np.random.poisson(lam=3, size=n_accounts)
    license_utilization = np.random.uniform(0.2, 0.98, size=n_accounts)
    
    # Ground truth logistic probability
    logits = (
        -0.05 * tenure_months 
        - 0.08 * logins_weekly 
        + 0.45 * support_tickets 
        - 2.5 * license_utilization 
        + np.where(tiers == "Enterprise", -0.6, 0.3)
    )
    churn_prob_true = 1 / (1 + np.exp(-logits))
    churned = np.random.binomial(1, churn_prob_true)
    
    df = pd.DataFrame({
        "Account_ID": account_ids,
        "Tier": tiers,
        "MRR": np.round(mrr, 2),
        "ARR": np.round(mrr * 12, 2),
        "Tenure_Months": tenure_months,
        "Logins_Weekly": logins_weekly,
        "Support_Tickets": support_tickets,
        "License_Utilization": np.round(license_utilization, 2),
        "Churned": churned
    })
    return df

df_cohort = generate_saas_data()

# --- PREDICTIVE ML ENGINE ---
features = ["MRR", "Tenure_Months", "Logins_Weekly", "Support_Tickets", "License_Utilization"]
X = df_cohort[features]
y = df_cohort["Churned"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = GradientBoostingClassifier(n_estimators=50, random_state=42)
clf.fit(X_scaled, y)

df_cohort["Predicted_Churn_Prob"] = clf.predict_proba(X_scaled)[:, 1]

# CLV Formulation (Assuming 80% Gross Margin with 24-month cap)
df_cohort["CLV"] = np.round(df_cohort["MRR"] * 0.80 * (1 / np.clip(df_cohort["Predicted_Churn_Prob"] / 12, 0.05, 1.0)), 2)

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("<h3 style='color: #CDB4DB;'>⚙️ Retention Budget & Interventions</h3>", unsafe_allow_html=True)
retention_budget = st.sidebar.slider("Total Retention Budget (€)", min_value=1000, max_value=25000, value=7500, step=500)
intervention_cost = st.sidebar.slider("Cost per Account Offer (€)", min_value=50, max_value=800, value=250, step=25)
expected_uplift = st.sidebar.slider("Expected Retention Uplift (%)", min_value=5, max_value=50, value=20, step=1) / 100.0

# --- PRESCRIPTIVE OPTIMIZATION (PuLP Knapsack MILP) ---
def optimize_retention(df, budget, cost_per_offer, uplift):
    prob = pulp.LpProblem("Retention_Budget_Optimization", pulp.LpMaximize)
    accounts = df["Account_ID"].tolist()
    
    # Binary treatment decisions
    x = {a: pulp.LpVariable(f"Treat_{a}", cat="Binary") for a in accounts}
    
    # Net Expected Saved Value
    value_map = {}
    for _, row in df.iterrows():
        a = row["Account_ID"]
        expected_saved = (row["Predicted_Churn_Prob"] * uplift * row["CLV"]) - cost_per_offer
        value_map[a] = expected_saved
        
    prob += pulp.lpSum([x[a] * value_map[a] for a in accounts])
    prob += pulp.lpSum([x[a] * cost_per_offer for a in accounts]) <= budget
    
    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)
    
    treated = {a: int(x[a].varValue) if x[a].varValue is not None else 0 for a in accounts}
    return treated

treatment_dict = optimize_retention(df_cohort, retention_budget, intervention_cost, expected_uplift)
df_cohort["Target_Intervention"] = df_cohort["Account_ID"].map(treatment_dict)
df_cohort["Expected_Saved_ARR"] = np.where(
    df_cohort["Target_Intervention"] == 1,
    df_cohort["Predicted_Churn_Prob"] * expected_uplift * df_cohort["ARR"],
    0.0
)

# Executive Metrics
total_accounts_targeted = df_cohort["Target_Intervention"].sum()
total_spend_allocated = total_accounts_targeted * intervention_cost
total_arr_saved = df_cohort["Expected_Saved_ARR"].sum()
net_roi = ((total_arr_saved - total_spend_allocated) / total_spend_allocated) * 100 if total_spend_allocated > 0 else 0

# --- UI HEADER ---
st.markdown("<h1 style='margin-bottom: 2px;'>💎 B2B SaaS Customer Retention & CLV Decision Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #6DD5ED; font-size: 0.95rem; margin-top: 0px;'>Predictive ML Churn Risk Scoring, Calibrated CLV, and Prescriptive MILP Budget Optimization</p>", unsafe_allow_html=True)

# --- KPI METRICS STRIP ---
st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""<div class='metric-card'><div class='metric-sub'>Prescribed Targets</div><div class='metric-val'>{total_accounts_targeted} Accounts</div><div class='metric-caption'>Allocated €{total_spend_allocated:,.0f} of €{retention_budget:,.0f}</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class='metric-card'><div class='metric-sub'>Expected Saved ARR</div><div class='metric-val'>€{total_arr_saved:,.0f}</div><div class='metric-caption'>Protected recurring revenue</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class='metric-card'><div class='metric-sub'>Net Campaign ROI</div><div class='metric-val' style='color: #CDB4DB;'>{net_roi:.1f}%</div><div class='metric-caption'>Net protected ARR / spend</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class='metric-card'><div class='metric-sub'>Cohort At-Risk ARR</div><div class='metric-val' style='color: #FFAAA6;'>€{(df_cohort['Predicted_Churn_Prob'] * df_cohort['ARR']).sum():,.0f}</div><div class='metric-caption'>Gross churn exposure</div></div>""", unsafe_allow_html=True)

# Custom Plotly Palette Template
PLOTLY_THEME = {
    "layout": {
        "paper_bgcolor": "#364449",
        "plot_bgcolor": "#253337",
        "font": {"color": "#FFFFFF", "family": "Inter, sans-serif"},
        "xaxis": {
            "gridcolor": "rgba(109, 213, 237, 0.15)",
            "zerolinecolor": "rgba(109, 213, 237, 0.2)",
            "tickfont": {"family": "JetBrains Mono, monospace", "size": 11}
        },
        "yaxis": {
            "gridcolor": "rgba(109, 213, 237, 0.15)",
            "zerolinecolor": "rgba(109, 213, 237, 0.2)",
            "tickfont": {"family": "JetBrains Mono, monospace", "size": 11}
        }
    }
}

# --- ANALYTICS WORKBENCH ---
tab1, tab2 = st.tabs([" Prescriptive Retention Ledger", " Churn Probability vs. CLV Matrix"])

with tab1:
    st.markdown("###  High-Priority Account Intervention Ledger")
    display_df = df_cohort[df_cohort["Target_Intervention"] == 1].sort_values(by="Expected_Saved_ARR", ascending=False)[
        ["Account_ID", "Tier", "MRR", "Predicted_Churn_Prob", "CLV", "Expected_Saved_ARR"]
    ].copy()
    display_df["Predicted_Churn_Prob"] = (display_df["Predicted_Churn_Prob"] * 100).map("{:.1f}%".format)
    display_df["MRR"] = display_df["MRR"].map("€{:,.2f}".format)
    display_df["CLV"] = display_df["CLV"].map("€{:,.0f}".format)
    display_df["Expected_Saved_ARR"] = display_df["Expected_Saved_ARR"].map("€{:,.0f}".format)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

with tab2:
    st.markdown("###  Risk Exposure vs. Lifetime Value Matrix")
    fig_scatter = px.scatter(
        df_cohort,
        x="Predicted_Churn_Prob",
        y="CLV",
        color=df_cohort["Target_Intervention"].map({1: "Target Offer", 0: "Pass / Monitor"}),
        size="MRR",
        hover_data=["Account_ID", "Tier", "ARR"],
        color_discrete_map={"Target Offer": "#6DD5ED", "Pass / Monitor": "#CDB4DB"},
        labels={"Predicted_Churn_Prob": "Predicted Churn Probability", "CLV": "Customer Lifetime Value (€)"}
    )
    fig_scatter.update_layout(
        template=PLOTLY_THEME,
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
