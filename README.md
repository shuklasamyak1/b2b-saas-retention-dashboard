# B2B SaaS Revenue & Customer Retention Intelligence Dashboard

An end-to-end business intelligence engine designed to analyze subscription lifecycles, monitor Monthly Recurring Revenue (MRR), and diagnose customer churn dynamics across a multi-tier B2B SaaS portfolio.

---

## 📌 Executive Summary
* **Total Tracked Portfolio:** €49.00K Monthly Recurring Revenue (MRR)
* **Active Retained MRR:** €17.93K across enterprise and mid-market accounts
* **Lost / Churned MRR:** €13.67K isolated for targeted retention intervention
* **Architecture:** PostgreSQL Star Schema $\rightarrow$ Power BI Data Engine $\rightarrow$ DAX Metric Layer

---

## 🛠️ Architecture & Data Modeling

### 1. Relational Star Schema (PostgreSQL $\rightarrow$ Power BI)
* **Fact Table (`Subscriptions Fact`):** Captures atomic subscription events, billing amounts, contract start/end dates, and account status.
* **Dimension Tables:**
  * `Customers`: Company profiles, country codes, and industry sector metadata.
  * `Plans`: Tier structures, baseline pricing, and feature access tiers.
  * `Calendar`: Dynamically generated DAX date table supporting chronological time-series analysis and month-over-month comparisons.

### 2. Advanced DAX Business Logic
* **Active MRR:** Isolates ongoing active subscription cash flow while accounting for global and visual filter contexts:
  $$\text{Active MRR} = \sum(\text{Amount}) \quad \text{where Status} = \text{"Active"}$$
* **Cancelled MRR:** Tracks aggregate revenue leakage from churned accounts.
* **Portfolio Churn Rate %:** Evaluates health via dynamic baseline comparison:
  $$\text{Churn Rate} = \frac{\text{Cancelled MRR}}{\text{Active MRR} + \text{Cancelled MRR}}$$

---

## 📊 Dashboard Modules

* **Page 1: Executive Portfolio Overview**
  * High-level KPI summary cards for instant ARR/MRR visibility.
  * Geographic revenue distribution by market (Germany, UK, France, Netherlands, Sweden).
  * Sector concentration breakdown across Fintech, Healthtech, EdTech, Logistics, and E-Commerce.

* **Page 2: Retention & Growth Trends**
  * Time-series smooth area retention curves tracking status evolution.
  * Account-level cross-tabulated revenue matrix.
  * Interactive cross-filtering, dynamic page navigation, and instant filter-reset controls.

* **Diagnostic Layer:**
  * Report-page hover tooltips providing granular account previews.
  * Granular account identification for customer success outreach.

---

## 📁 Repository Structure
* `/B2B_SaaS_Analytics.pbix` — Full interactive Power BI report file.
* `/B2B_SaaS_Revenue_Retention_Dashboard.pdf` — High-resolution executive slide deck.
* `/dashboard_demo.mp4` — 60-second interactive feature walkthrough.
* `/schema.sql` — PostgreSQL schema definitions and relationship constraints.

---

## 💻 Tech Stack
* **Database & Querying:** PostgreSQL, SQL
* **Data Modeling & BI:** Microsoft Power BI Desktop
* **Calculations & Analytics:** DAX (Data Analysis Expressions)
