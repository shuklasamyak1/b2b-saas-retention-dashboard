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
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- REFINED EDITORIAL PALETTE & TYPOGRAPHY STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

    /* Global App Canvas */
    .stApp {
        background-color: #FAF2F4 !important;
        color: #1D3635 !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }
    
    /* Strict Color Lock for ALL Headings & Paragraphs in Main Body */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp [data-testid="stMarkdownContainer"] h1,
    .stApp [data-testid="stMarkdownContainer"] h2,
    .stApp [data-testid="stMarkdownContainer"] h3,
    .stApp [data-testid="stMarkdownContainer"] h4,
    .stApp [data-testid="stMarkdownContainer"] h5,
    .stApp [data-testid="stMarkdownContainer"] h6 {
        color: #1D3635 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.4px !important;
    }

    .stApp p, .stApp span, .stApp div {
        color: #1D3635;
    }

    /* Sidebar Styling & High Contrast */
    section[data-testid="stSidebar"] {
        background-color: #1D3635 !important;
        border-right: 1px solid rgba(205, 180, 219, 0.3) !important;
    }

    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] label span,
    section[data-testid="stSidebar"] label p,
    section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p {
        color: #FFFFFF !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stThumbValue"] {
        color: #000000 !important;
        background-color: #6DD5ED !important;
        font-family: 'Space Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        border-radius: 4px !important;
        padding: 3px 8px !important;
    }

    section[data-testid="stSidebar"] [data-baseweb="slider"] div div {
        background-color: rgba(255, 255, 255, 0.3) !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="slider"] div div div {
        background-color: #6DD5ED !important;
    }

    /* Executive KPI Metric Cards */
    .metric-card {
        background: #2C5554;
        border: 1px solid #CDB4DB;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 14px rgba(44, 85, 84, 0.12);
    }
    .metric-sub {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.74rem;
        color: #CDB4DB !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
    }
    .metric-val {
        font-family: 'Space Mono', monospace;
        font-size: 1.65rem;
        font-weight: 700;
        color: #6DD5ED !important;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    .metric-caption {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.75rem;
        color: #F1F5F9 !important;
        margin-top: 4px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #2C5554;
        background-color: transparent !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #E6D8DC !important;
        border-radius: 6px 6px 0 0 !important;
        padding: 10px 20px !important;
        border: 1px solid #CDB4DB !important;
        border-bottom: none !important;
    }
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span {
        color: #1D3635 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2C5554 !important;
        border: 1px solid #2C5554 !important;
        border-bottom: 3px solid #6DD5ED !important;
    }
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span {
        color: #6DD5ED !important;
        font-weight: 800 !important;
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

# CLV Formulation
df_cohort["CLV"] = np.round(df_cohort["MRR"] * 0.80 * (1 / np.clip(df_cohort["Predicted_Churn_Prob"] / 12, 0.05, 1.0)), 2)

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("<h3 style='color: #FFFFFF !important; margin-bottom: 12px; font-weight: 800;'>Retention Parameters</h3>", unsafe_allow_html=True)
retention_budget = st.sidebar.slider("Total Retention Budget (€)", min_value=1000, max_value=25000, value=7500, step=500)
intervention_cost = st.sidebar.slider("Cost per Account Offer (€)", min_value=50, max_value=800, value=250, step=25)
expected_uplift = st.sidebar.slider("Expected Retention Uplift (%)", min_value=5, max_value=50, value=20, step=1) / 100.0

# --- PRESCRIPTIVE OPTIMIZATION (PuLP Knapsack MILP) ---
def optimize_retention(df, budget, cost_per_offer, uplift):
    prob = pulp.LpProblem("Retention_Budget_Optimization", pulp.LpMaximize)
    accounts = df["Account_ID"].tolist()
    
    x = {a: pulp.LpVariable(f"Treat_{a}", cat="Binary") for a in accounts}
    
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
df_cohort["Expected_Net_ROI"] = np.where(
    df_cohort["Target_Intervention"] == 1,
    ((df_cohort["Expected_Saved_ARR"] - intervention_cost) / intervention_cost) * 100,
    0.0
)

# Executive Metrics
total_accounts_targeted = df_cohort["Target_Intervention"].sum()
total_spend_allocated = total_accounts_targeted * intervention_cost
total_arr_saved = df_cohort["Expected_Saved_ARR"].sum()
net_roi = ((total_arr_saved - total_spend_allocated) / total_spend_allocated) * 100 if total_spend_allocated > 0 else 0

# --- UI HEADER ---
st.markdown("<h1 style='margin-bottom: 2px; color: #1D3635;'>B2B SaaS Customer Retention & CLV Decision Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #574951; font-size: 0.95rem; margin-top: 0px;'>Predictive ML Churn Risk Scoring, Calibrated CLV, and Prescriptive MILP Budget Optimization</p>", unsafe_allow_html=True)

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

# Plotly Canvas Theme
PLOTLY_THEME = {
    "layout": {
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#FAF2F4",
        "font": {"color": "#1D3635", "family": "Plus Jakarta Sans, sans-serif"},
        "xaxis": {
            "gridcolor": "#E2E8F0",
            "zerolinecolor": "#CBD5E1",
            "tickfont": {"family": "Space Mono, monospace", "size": 11, "color": "#1D3635"}
        },
        "yaxis": {
            "gridcolor": "#E2E8F0",
            "zerolinecolor": "#CBD5E1",
            "tickfont": {"family": "Space Mono, monospace", "size": 11, "color": "#1D3635"}
        }
    }
}

# --- ANALYTICS WORKBENCH ---
tab1, tab2 = st.tabs(["Prescriptive Retention Ledger", "Risk Exposure vs. Lifetime Value Matrix"])

with tab1:
    st.markdown("<h3 style='color: #1D3635;'>High-Priority Account Intervention Ledger</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #574951; font-size: 0.85rem;'>Accounts selected by the MILP solver to maximize preserved ARR within the allocated retention capital.</p>", unsafe_allow_html=True)
    
    ledger_data = df_cohort[df_cohort["Target_Intervention"] == 1].sort_values(by="Expected_Saved_ARR", ascending=False).copy()
    
    col_t1, col_t2 = st.columns([3, 1])
    with col_t2:
        st.markdown(f"""
        <div style='background: #FFFFFF; border: 1px solid #CDB4DB; border-radius: 8px; padding: 14px; margin-bottom: 10px;'>
            <div style='font-size: 0.72rem; color: #574951; font-weight: 700; text-transform: uppercase;'>Intervention Density</div>
            <div style='font-family: "Space Mono", monospace; font-size: 1.4rem; font-weight: 700; color: #2C5554;'>{(total_accounts_targeted/len(df_cohort))*100:.1f}%</div>
            <div style='font-size: 0.72rem; color: #64748B;'>{total_accounts_targeted} of {len(df_cohort)} Total Accounts</div>
        </div>
        """, unsafe_allow_html=True)

    with col_t1:
        st.dataframe(
            ledger_data[[
                "Account_ID", "Tier", "MRR", "Predicted_Churn_Prob", "CLV", "Expected_Saved_ARR", "Expected_Net_ROI"
            ]],
            column_config={
                "Account_ID": st.column_config.TextColumn("Account ID"),
                "Tier": st.column_config.TextColumn("Subscription Tier"),
                "MRR": st.column_config.NumberColumn("Monthly Recurring Revenue", format="€%.2f"),
                "Predicted_Churn_Prob": st.column_config.ProgressColumn(
                    "Churn Hazard Rate",
                    format="%.1f%%",
                    min_value=0.0,
                    max_value=1.0
                ),
                "CLV": st.column_config.NumberColumn("Customer Lifetime Value", format="€%d"),
                "Expected_Saved_ARR": st.column_config.NumberColumn("Protected ARR", format="€%d"),
                "Expected_Net_ROI": st.column_config.NumberColumn("Expected Net ROI", format="%.0f%%")
            },
            use_container_width=True,
            hide_index=True
        )

with tab2:
    st.markdown("<h3 style='color: #1D3635;'>Risk Exposure vs. Lifetime Value Matrix</h3>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color: #574951; font-size: 0.85rem; margin-bottom: 12px;'>"
        "Visualizes individual accounts by Churn Hazard vs. Projected CLV. "
        "The solver prioritizes the <b>High Value & High Risk (Upper Right)</b> quadrant to maximize preserved recurring revenue."
        "</p>", 
        unsafe_allow_html=True
    )
    
    col_chart, col_tier = st.columns([3, 2])
    
    with col_chart:
        fig_scatter = go.Figure()

        pass_df = df_cohort[df_cohort["Target_Intervention"] == 0]
        fig_scatter.add_trace(go.Scatter(
            x=pass_df["Predicted_Churn_Prob"] * 100,
            y=pass_df["CLV"],
            mode="markers",
            name="Pass / Monitor",
            marker=dict(
                color="rgba(87, 73, 81, 0.35)",
                size=7,
                line=dict(color="#574951", width=0.5)
            ),
            hovertemplate="<b>%{text}</b><br>Churn Prob: %{x:.1f}%<br>CLV: €%{y:,.0f}<extra></extra>",
            text=pass_df["Account_ID"]
        ))

        treat_df = df_cohort[df_cohort["Target_Intervention"] == 1]
        fig_scatter.add_trace(go.Scatter(
            x=treat_df["Predicted_Churn_Prob"] * 100,
            y=treat_df["CLV"],
            mode="markers",
            name="Target Retention Offer",
            marker=dict(
                color="#2C5554",
                size=10,
                line=dict(color="#6DD5ED", width=1.5),
                symbol="circle"
            ),
            hovertemplate="<b>%{text}</b> (Targeted)<br>Churn Prob: %{x:.1f}%<br>CLV: €%{y:,.0f}<extra></extra>",
            text=treat_df["Account_ID"]
        ))

        avg_churn = (df_cohort["Predicted_Churn_Prob"].mean()) * 100
        med_clv = df_cohort["CLV"].median()
        
        fig_scatter.add_vline(x=avg_churn, line_dash="dash", line_color="rgba(44, 85, 84, 0.4)", line_width=1.5)
        fig_scatter.add_hline(y=med_clv, line_dash="dash", line_color="rgba(44, 85, 84, 0.4)", line_width=1.5)

        fig_scatter.update_layout(
            template=PLOTLY_THEME,
            height=360,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(title="Predicted Churn Probability (%)", ticksuffix="%"),
            yaxis=dict(title="Customer Lifetime Value (€)", tickprefix="€"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_tier:
        st.markdown("<h5 style='color: #1D3635;'>Budget Allocation by Tier</h5>", unsafe_allow_html=True)
        tier_summary = df_cohort.groupby("Tier").agg(
            Total_Accounts=("Account_ID", "count"),
            Targeted_Accounts=("Target_Intervention", "sum"),
            Saved_ARR=("Expected_Saved_ARR", "sum")
        ).reset_index()

        fig_tier = go.Figure()
        fig_tier.add_trace(go.Bar(
            x=tier_summary["Tier"],
            y=tier_summary["Targeted_Accounts"],
            name="Targeted Accounts",
            marker_color="#2C5554"
        ))
        fig_tier.add_trace(go.Bar(
            x=tier_summary["Tier"],
            y=tier_summary["Total_Accounts"] - tier_summary["Targeted_Accounts"],
            name="Untargeted",
            marker_color="rgba(205, 180, 219, 0.5)"
        ))
        
        fig_tier.update_layout(
            barmode="stack",
            template=PLOTLY_THEME,
            height=320,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(title="Subscription Tier"),
            yaxis=dict(title="Number of Accounts"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_tier, use_container_width=True)
