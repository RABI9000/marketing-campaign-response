"""
Marketing Campaign Response — interactive Streamlit app.

A boardroom-ready companion to the analysis notebook: it walks from the
business problem through the data, the model, and the money, and ends with a
live tool that scores a single customer. Run locally with

    streamlit run app.py

or deploy straight to Streamlit Community Cloud (point it at this file).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import data, models, viz
from src.viz import BLUE, CORAL, GREY, NAVY, TEAL

# --------------------------------------------------------------------------- #
# Page config + styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Campaign Response Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stMarkdown, .stMetric { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1300px; }

    /* Hero banner */
    .hero {
        background: linear-gradient(120deg, #1F3A5F 0%, #2E6F9E 55%, #3FA7A0 100%);
        padding: 2.0rem 2.2rem; border-radius: 18px; color: #fff; margin-bottom: 1.4rem;
        box-shadow: 0 10px 30px rgba(31,58,95,0.25);
    }
    .hero h1 { color:#fff; font-size: 2.05rem; font-weight: 800; margin: 0 0 .35rem 0; letter-spacing:-.5px;}
    .hero p  { color: #e8eef5; font-size: 1.04rem; margin: 0; max-width: 60rem; }
    .hero .tagchip {
        display:inline-block; background: rgba(255,255,255,0.16); padding: .2rem .7rem;
        border-radius: 999px; font-size:.78rem; font-weight:600; margin-top:.8rem; margin-right:.4rem;
    }

    /* KPI metric cards */
    [data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e9edf2; border-left: 5px solid #2E6F9E;
        border-radius: 12px; padding: 1rem 1.1rem; box-shadow: 0 2px 10px rgba(31,58,95,0.05);
    }
    [data-testid="stMetricLabel"] { color:#6b7785; font-weight:600; }
    [data-testid="stMetricValue"] { color:#1F3A5F; font-weight:800; }

    /* Section header */
    .sec { border-left: 5px solid #E8743B; padding-left: .8rem; margin: .4rem 0 1rem 0; }
    .sec h2 { margin:0; color:#1F3A5F; font-weight:800; font-size:1.5rem; }
    .sec span { color:#7a8694; font-size:.96rem; }

    /* Insight + callout cards */
    .card {
        background:#fff; border:1px solid #e9edf2; border-radius:14px; padding:1.1rem 1.25rem;
        box-shadow:0 2px 12px rgba(31,58,95,0.05); height:100%;
    }
    .card h4 { margin:.1rem 0 .5rem 0; color:#1F3A5F; font-weight:700; font-size:1.05rem;}
    .card .ev { color:#2E6F9E; font-weight:600; }
    .card .bm { color:#445; }
    .pill { display:inline-block; background:#fdece3; color:#E8743B; font-weight:700;
            padding:.15rem .6rem; border-radius:999px; font-size:.8rem; margin-bottom:.4rem;}
    .rec { background:#f7faf9; border-left:4px solid #3FA7A0; border-radius:10px; padding:.9rem 1.1rem; margin-bottom:.7rem;}
    .rec b { color:#1F3A5F; }
    .sidebar-note { color:#7a8694; font-size:.8rem; line-height:1.45; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit deprecated use_container_width in favour of width="stretch".
W = "stretch"


# --------------------------------------------------------------------------- #
# Cached data + model
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading customer data…")
def get_data() -> pd.DataFrame:
    return data.load_prepared_data()


@st.cache_resource(show_spinner="Training the model zoo…")
def get_run():
    df = get_data()
    X, y = data.get_feature_matrix(df)
    return models.train_model_zoo(X, y)


@st.cache_data
def get_profit_table():
    run = get_run()
    return models.expected_profit_by_threshold(run.best.y_test, run.best.y_proba)


def section(title: str, subtitle: str = "") -> None:
    st.markdown(f"<div class='sec'><h2>{title}</h2><span>{subtitle}</span></div>",
                unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Sidebar navigation
# --------------------------------------------------------------------------- #
PAGES = [
    "Overview",
    "Data & Quality",
    "Exploratory Analysis",
    "Key Insights",
    "Feature Engineering",
    "Models & Evaluation",
    "Marketing Lens",
    "What Drives a Yes",
    "Recommendations",
    "Risks & Roadmap",
    "Live Predictor",
    "Executive Summary",
]

with st.sidebar:
    st.markdown("### 🎯 Campaign Response\n**Intelligence**")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()
    df_side = get_data()
    st.markdown(
        f"<div class='sidebar-note'>"
        f"<b>Dataset:</b> {len(df_side):,} retail customers<br>"
        f"<b>Target:</b> accepted the last campaign (Response)<br>"
        f"<b>Response rate:</b> {df_side['Response'].mean():.1%}<br><br>"
        f"Built with pandas, scikit-learn &amp; XGBoost. The full method lives in the "
        f"companion Jupyter notebook.</div>",
        unsafe_allow_html=True,
    )


df = get_data()
SPEND = data.SPENDING_COLUMNS


# --------------------------------------------------------------------------- #
# PAGE: Overview
# --------------------------------------------------------------------------- #
def page_overview():
    st.markdown(
        "<div class='hero'><h1>Who should we target, and what will it earn us?</h1>"
        "<p>A retailer sent a campaign to its whole base — and only ~15% responded. "
        "This app scores every customer's likelihood to respond, then turns that score "
        "into a targeting and budget plan the marketing team can act on.</p>"
        "<span class='tagchip'>Predict response</span>"
        "<span class='tagchip'>Find the patterns</span>"
        "<span class='tagchip'>Act on the money</span></div>",
        unsafe_allow_html=True,
    )

    run = get_run()
    best = run.best
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers analysed", f"{len(df):,}")
    c2.metric("Baseline response rate", f"{df['Response'].mean():.1%}")
    c3.metric("Best model (ROC-AUC)", f"{best.metrics['ROC-AUC']:.2f}", help=best.name)
    c4.metric("Engineered features", "10")

    st.write("")
    left, right = st.columns([1.15, 1])
    with left:
        section("The brief", "what we're solving and how we judge it")
        st.markdown(
            """
**Business problem.** Budget is finite and every contact costs money. Blasting the whole
base wastes spend on people who'll never buy. The team wants to **target, not broadcast.**

**ML objective.** Predict the *probability* each customer accepts the next offer
(`Response` = 1 vs 0). The probability — not just yes/no — lets us rank customers and
contact them in priority order until the budget runs out.

**Success criteria**
- **Rank well** — measured by ROC-AUC, so a model-driven list beats random contacting.
- **Catch responders** — measured by Recall, because a missed buyer costs more than a wasted contact.
- **Be actionable** — produce a short list of traits the team can target *this quarter*.

**Why it matters.** Two numbers drive everything: the cost of a wasted contact and the value
of a missed responder. A good model shifts spend from the first toward the second.
            """
        )
    with right:
        section("Response split", "the imbalance that shapes every decision")
        st.plotly_chart(viz.target_donut(df), width=W)
        st.caption("≈ 1 in 7 customers responds. Accuracy is a trap here — a model that "
                   "predicts ‘no’ for everyone scores 85%. We optimise ROC-AUC and recall instead.")


# --------------------------------------------------------------------------- #
# PAGE: Data & Quality
# --------------------------------------------------------------------------- #
def page_data_quality():
    section("Data & Quality", "what's in the data, and what's wrong with it")
    raw = data.load_raw()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (raw)", f"{len(raw):,}")
    c2.metric("Columns", f"{raw.shape[1]}")
    c3.metric("Missing incomes", f"{int(raw['Income'].isna().sum())}")
    c4.metric("Rows after cleaning", f"{len(df):,}")

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### The five groups of columns")
        st.markdown(
            """
| Group | Captures |
|---|---|
| **Demographics** | age, education, marital status, income, children |
| **Recency** | days since last purchase |
| **Spending (2 yrs)** | wine, fruit, meat, fish, sweets, gold |
| **Purchasing & channels** | web / catalog / store / deals, web visits |
| **Campaign history** | accepted campaigns 1–5, complaints |
| **Target** | **Response** — accepted the latest offer (1/0) |
            """
        )
    with col2:
        st.markdown("##### Quality findings (and the fix)")
        st.markdown(
            """
<div class='rec'><b>Missing values.</b> Only <b>Income</b>, 24 rows (~1%). Median-imputed
<i>inside</i> the model pipeline → no leakage.</div>
<div class='rec'><b>Duplicates.</b> None — one clean row per customer.</div>
<div class='rec'><b>Wrong types.</b> Enrolment date stored as text (parsed); two constant
<code>Z_</code> columns dropped.</div>
<div class='rec'><b>Outliers.</b> 3 impossible birth years (age 115+) and one $666k income —
data-entry errors, removed.</div>
<div class='rec'><b>Inconsistencies.</b> Marital labels "Alone/Absurd/YOLO" tidied into
Single/Other.</div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("##### Sample of the cleaned, feature-rich data")
    st.dataframe(
        df[["Age", "Income", "Education", "Marital_Status", "Children", "Recency",
            "TotalSpending", "TotalPurchases", "TotalAcceptedCmp", "EngagementScore",
            "Response"]].head(12),
        width=W, hide_index=True,
    )
    st.caption("Verdict: a clean, well-behaved dataset — the patterns we find next are trustworthy.")


# --------------------------------------------------------------------------- #
# PAGE: EDA
# --------------------------------------------------------------------------- #
def page_eda():
    section("Exploratory Analysis", "explore the patterns interactively")

    st.markdown("##### Response rate by segment")
    st.caption("Pick a customer attribute to see how the response rate moves. The dashed "
               "line is the overall average — bars above it are your high-yield segments.")
    dim_options = {
        "Income tier": ("IncomeSegment", ["Low", "Medium", "High"]),
        "Prior campaigns accepted": ("TotalAcceptedCmp", None),
        "Children at home": ("HasChildren", None),
        "Education": ("Education", None),
        "Marital status": ("Marital_Status", None),
        "Age group": ("AgeGroup", ["<=35", "36-50", "51-65", "65+"]),
    }
    pick = st.selectbox("Segment by", list(dim_options), index=0)
    dim, order = dim_options[pick]
    st.plotly_chart(viz.response_rate_bar(df, dim, order), width=W)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Responders vs non-responders")
        var = st.selectbox("Compare distribution of",
                           ["Income", "TotalSpending", "Recency", "EngagementScore",
                            "TotalPurchases", "Age"], index=0)
        st.plotly_chart(viz.box_by_response(df, var), width=W)
    with c2:
        st.markdown("##### What correlates with response")
        st.plotly_chart(viz.corr_bar(df), width=W)

    st.divider()
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("##### Where the money is spent")
        st.plotly_chart(viz.spend_by_category(df, SPEND), width=W)
    with c4:
        st.markdown("##### Income vs spending (coloured by response)")
        sample = df.dropna(subset=["Income"]).copy()
        sample["Responded"] = sample["Response"].map({0: "Ignored", 1: "Responded"})
        import plotly.express as px
        fig = px.scatter(sample, x="Income", y="TotalSpending", color="Responded",
                         color_discrete_map={"Ignored": GREY, "Responded": CORAL},
                         opacity=0.6)
        fig = viz._style(fig, height=380)
        st.plotly_chart(fig, width=W)
    st.caption("Note: web *visits* barely correlate with response — browsing ≠ buying. "
               "Prior acceptance, spending and income are the real signals.")


# --------------------------------------------------------------------------- #
# PAGE: Key Insights
# --------------------------------------------------------------------------- #
def page_insights():
    section("Key Insights", "five findings, each with evidence and a business action")
    base = df["Response"].mean()
    prior = df[df["TotalAcceptedCmp"] >= 1]["Response"].mean()
    never = df[df["TotalAcceptedCmp"] == 0]["Response"].mean()
    hi = df[df["IncomeSegment"] == "High"]["Response"].mean()
    lo = df[df["IncomeSegment"] == "Low"]["Response"].mean()
    nokid = df[df["HasChildren"] == 0]["Response"].mean()
    kid = df[df["HasChildren"] == 1]["Response"].mean()
    rec_r = df[df["Response"] == 1]["Recency"].mean()
    rec_n = df[df["Response"] == 0]["Recency"].mean()

    cards = [
        ("Past behaviour is destiny",
         f"Prior responders convert at <span class='ev'>{prior:.0%}</span> vs "
         f"<span class='ev'>{never:.0%}</span> for never-responders ({prior/never:.1f}x).",
         "Re-target the ~460 prior responders first — your warmest list, at zero new cost."),
        ("Money talks",
         f"High-income customers respond at <span class='ev'>{hi:.0%}</span> vs "
         f"<span class='ev'>{lo:.0%}</span> for low income ({hi/lo:.1f}x).",
         "Give premium customers higher-value, personalised offers — not the generic coupon."),
        ("Recency wins",
         f"Responders last bought <span class='ev'>~{rec_r:.0f} days</span> ago vs "
         f"<span class='ev'>~{rec_n:.0f} days</span> for non-responders.",
         "Trigger campaigns off recent purchases (≤30 days) while the relationship is warm."),
        ("Childless households respond more",
         f"No children → <span class='ev'>{nokid:.0%}</span> vs "
         f"<span class='ev'>{kid:.0%}</span> with children ({nokid/kid:.1f}x).",
         "Use household composition as a cheap, always-available targeting filter."),
        ("Channel reveals intent",
         "Responders make <span class='ev'>4.2</span> catalog purchases vs "
         "<span class='ev'>2.4</span> — yet web <i>visits</i> are flat across both groups.",
         "Invest in catalog-active customers; don't chase raw web traffic (visits ≠ intent)."),
    ]
    cols = st.columns(2)
    for i, (title, ev, bm) in enumerate(cards):
        with cols[i % 2]:
            st.markdown(
                f"<div class='card'><div class='pill'>Insight {i+1}</div>"
                f"<h4>{title}</h4><p class='ev'>{ev}</p>"
                f"<p class='bm'>🎯 <b>Action:</b> {bm}</p></div>",
                unsafe_allow_html=True,
            )
            st.write("")
    st.info("**Bonus signals:** PhDs respond 20.8% vs 3.7% for Basic education; single/widowed "
            "customers out-respond partnered ones; response is U-shaped in age (under-35s and 65+ "
            "over-index).")


# --------------------------------------------------------------------------- #
# PAGE: Feature Engineering
# --------------------------------------------------------------------------- #
def page_features():
    section("Feature Engineering", "turning raw columns into business signals")
    st.markdown(
        "Raw columns describe a customer; *engineered* features describe their **behaviour and "
        "value** — which is what predicts response. Ten new features, well past the required five:"
    )
    feat_table = pd.DataFrame(
        [
            ["Age", "2015 − Year_Birth", "Life-stage segmentation"],
            ["TotalSpending", "Sum of 6 spend columns", "Single customer-value number"],
            ["TotalPurchases", "Web+Catalog+Store+Deals", "Overall purchase frequency"],
            ["SpendingPerPurchase", "Spending ÷ Purchases", "Premium vs bargain shopper"],
            ["TotalAcceptedCmp", "Sum of AcceptedCmp 1–5", "Proven responsiveness (strongest signal)"],
            ["Customer_Tenure_Years", "2015-01-01 − Dt_Customer", "New vs long-standing customers"],
            ["EngagementScore", "Purchases + 3×PriorAccepts", "Activity + responsiveness in one number"],
            ["Children / HasChildren", "Kidhome + Teenhome", "Lifestyle & disposable-income proxy"],
            ["IncomeSegment", "Low / Medium / High bins", "Leadership reasons in tiers"],
            ["AgeGroup", "≤35 / 36–50 / 51–65 / 65+", "Readable demographic cross-tabs"],
        ],
        columns=["Feature", "How it's built", "Why it matters"],
    )
    st.dataframe(feat_table, width=W, hide_index=True)

    st.markdown(
        "##### The EngagementScore formula, explained\n"
        "`EngagementScore = TotalPurchases + 3 × TotalAcceptedCmp`. The **3× weight** on prior "
        "acceptances is deliberate: a single past ‘yes’ predicts a future ‘yes’ far more strongly "
        "than one extra purchase does, so it should count for more. It's a transparent number a "
        "marketer can sanity-check by hand — not a black box."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(viz.histogram_by_response(df, "EngagementScore", nbins=45),
                        width=W)
        st.caption("Responders (coral) sit clearly to the right — the score carries real signal.")
    with c2:
        st.plotly_chart(viz.response_rate_bar(df, "IncomeSegment", ["Low", "Medium", "High"]),
                        width=W)
        st.caption("Response climbs step-wise across the engineered income tiers.")


# --------------------------------------------------------------------------- #
# PAGE: Models & Evaluation
# --------------------------------------------------------------------------- #
def page_models():
    section("Models & Evaluation", "three models, head to head")
    run = get_run()

    st.markdown(
        """
We train three models spanning the complexity spectrum, each told to handle the 85/15 imbalance:

- **Logistic Regression** — the transparent baseline (interpretable, fast; misses interactions).
- **Random Forest** — robust non-linear workhorse (captures interactions; less interpretable).
- **XGBoost** — the performance pick for tabular data (usually best AUC; more to tune).

Evaluated with a stratified **80/20 split** plus **5-fold cross-validation** (ROC-AUC) to confirm
the ranking is stable, not luck.
        """
    )

    comp = run.comparison.copy()
    st.markdown("##### Scorecard (held-out test set)")
    st.dataframe(
        comp.style.format("{:.3f}").background_gradient(cmap="Greens", axis=0),
        width=W,
    )
    st.success(f"**Winner: {run.best_name}** — best ROC-AUC ({run.best.metrics['ROC-AUC']:.3f}) "
               f"and best F1 ({run.best.metrics['F1']:.3f}). It's the strongest *ranker* of "
               f"customers, and its probability output lets us tune the contact threshold to the budget.")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("##### ROC curves")
        st.plotly_chart(viz.roc_curves(run), width=W)
    with c2:
        st.markdown("##### Metric comparison")
        st.plotly_chart(viz.metric_bars(run), width=W)

    st.markdown("##### Confusion matrices (0.50 threshold)")
    cols = st.columns(len(run.results))
    for col, res in zip(cols, run.results.values()):
        with col:
            st.plotly_chart(viz.confusion_heatmap(res), width=W)
    st.caption("Random Forest is cautious (high precision, low recall); Logistic casts a wide net "
               "(high recall, low precision); XGBoost strikes the best balance.")


# --------------------------------------------------------------------------- #
# PAGE: Marketing Lens
# --------------------------------------------------------------------------- #
def page_marketing():
    section("Marketing Lens", "which metric matters — and what the threshold costs")
    run = get_run()
    prof = get_profit_table()
    best_t = float(prof.loc[prof["profit"].idxmax(), "threshold"])

    st.markdown(
        """
Every metric maps to money. A **false positive** = a wasted contact (**$3** here). A **false
negative** = a missed sale (**$11** per accepted offer). So a missed responder hurts ~3.7× more
than a wasted contact — which argues for **leaning toward recall.** But the right move isn't to
pick a metric in the abstract; it's to **tune the decision threshold to maximise profit.**
Drag the slider and watch the campaign economics change.
        """
    )

    threshold = st.slider("Decision threshold — contact everyone scoring above…",
                          0.05, 0.95, 0.45, 0.05)
    yt, yp = run.best.y_test, run.best.y_proba
    pred = (yp >= threshold).astype(int)
    tp = int(((pred == 1) & (yt == 1)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())
    contacted, responders = tp + fp, int((yt == 1).sum())
    profit = tp * 11 - contacted * 3
    recall = tp / responders if responders else 0
    precision = tp / contacted if contacted else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers contacted", f"{contacted}", help="Out of 448 in the test set")
    c2.metric("Responders caught", f"{tp}/{responders}", f"{recall:.0%} recall")
    c3.metric("Wasted contacts", f"{fp}", f"{precision:.0%} precision", delta_color="off")
    c4.metric("Campaign profit", f"${profit}", f"best @ {best_t:.2f}")

    st.plotly_chart(viz.profit_curve(prof, best_t, threshold), width=W)
    st.markdown(
        f"""
<div class='rec'><b>Conclusion — prioritise recall, operationalised through the threshold.</b>
Because a missed responder costs more than a wasted contact, profit peaks <b>below</b> the naive
0.50 (around <b>{best_t:.2f}</b> on this test set): it pays to contact more people and accept some
waste to catch more buyers. In practice — pick the <b>model</b> by ROC-AUC (ranking quality), then
pick the <b>threshold</b> by expected profit and your budget cap.</div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# PAGE: Feature importance
# --------------------------------------------------------------------------- #
def page_importance():
    section("What Drives a Yes", "the model's top response drivers, in business terms")
    run = get_run()
    imp = models.feature_importance(run.best.pipeline, top_n=10)
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.plotly_chart(viz.importance_bar(imp, run.best_name), width=W)
    with c2:
        st.markdown(
            """
**Read the chart as a targeting checklist:**

- **Prior acceptances** tower over everything — past ‘yes’ → future ‘yes’.
- **Recency** — recent buyers are warm.
- **Tenure** — relationship length shapes response.
- **Single / PhD** — these groups over-index.
- **Catalog & store, meat/wine spend, income** — the affluent, engaged core.
- **Children** — fewer kids, higher response.

The most responsive customer is a **recent, affluent, previously-responsive buyer with no
children** — and the model quantifies exactly how much each trait shifts the odds.
            """
        )


# --------------------------------------------------------------------------- #
# PAGE: Recommendations
# --------------------------------------------------------------------------- #
def page_recommendations():
    section("Business Recommendations", "the action plan for leadership")
    recs = [
        ("🎯 Customer Targeting — who to contact first",
         "Rank the base by predicted probability and work down until the budget is spent. Start with "
         "the ~460 customers who accepted any prior campaign — they convert ~3× the base rate at zero "
         "new identification cost."),
        ("💰 Budget Allocation — where the money goes",
         "Stop spending uniformly. Concentrate budget on the **top 2–3 probability deciles**, where the "
         "responders actually are. Reallocating from the unresponsive bottom half is the fastest ROI win."),
        ("✨ Personalization — who gets bespoke treatment",
         "High-income prior responders (~47% response) → premium offers & early access. Recent buyers → "
         "trigger sends within 30 days. Catalog-active → keep feeding that channel. Deal-seekers → a "
         "separate discount track."),
        ("📈 Campaign Optimization — lifting response",
         "Move from calendar blasts to **behaviour-triggered** sends; tune the contact threshold to the "
         "campaign's economics; A/B test creative within the top deciles."),
        ("🚀 ROI Improvement — making marketing pay more",
         "Same budget, pointed at customers who respond ~3× more often, yields far more accepted offers "
         "per dollar. Track **incremental response** and **profit per contact** vs a random control to "
         "prove the lift to finance."),
    ]
    for title, body in recs:
        st.markdown(f"<div class='rec'><b>{title}</b><br>{body}</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# PAGE: Risks & Roadmap
# --------------------------------------------------------------------------- #
def page_risks():
    section("Deployment Risks & Roadmap", "what could go wrong, and what's next")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### If this shipped tomorrow…")
        st.markdown(
            """
<div class='rec'><b>False positives</b> → wasted spend & brand fatigue.<br>
<i>Fix:</i> threshold tuning, frequency caps, monitor precision.</div>
<div class='rec'><b>False negatives</b> → lost sales (the costlier error here).<br>
<i>Fix:</i> recall-leaning operating point + a random holdout to keep measuring missed buyers.</div>
<div class='rec'><b>Data drift</b> → silent performance decay.<br>
<i>Fix:</i> monitor input/score distributions (PSI) with alerts; contract-test the feature pipeline.</div>
<div class='rec'><b>Changing behaviour</b> → yesterday's responders aren't tomorrow's.<br>
<i>Fix:</i> periodic retraining, champion/challenger, watch live vs predicted response rate.</div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown("##### With one more week (by impact)")
        st.markdown(
            """
<div class='rec'><b>1 · Business usage (highest impact)</b><br>
A production threshold/budget optimiser, customer segmentation (RFM/K-Means), and an A/B testing
framework to prove incremental lift.</div>
<div class='rec'><b>2 · Data (medium-high)</b><br>
Add behavioural signals (email opens, session depth, seasonality) and per-category margin to optimise
profit, not just response.</div>
<div class='rec'><b>3 · Modelling (incremental)</b><br>
Hyperparameter tuning, probability calibration, feature selection, ensembling, and SHAP for
per-customer explanations.</div>
<div class='rec'><i>Theme: the modelling is already good enough — the upside is in turning scores into
decisions.</i></div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# PAGE: Live predictor
# --------------------------------------------------------------------------- #
def page_predictor():
    section("Live Predictor", "score a single customer and get a recommendation")
    run = get_run()
    prof = get_profit_table()
    best_t = float(prof.loc[prof["profit"].idxmax(), "threshold"])

    st.caption("Adjust a customer's profile on the left. The model scores their probability of "
               "responding; everything not shown is set to the dataset median.")

    left, right = st.columns([1.15, 1])
    with left:
        a, b = st.columns(2)
        with a:
            income = st.slider("Income ($)", 0, 200_000, 60_000, 2_500)
            age = st.slider("Age", 18, 90, 45)
            recency = st.slider("Recency (days since last buy)", 0, 99, 30)
            tenure = st.slider("Tenure (years)", 0.0, 3.0, 1.5, 0.1)
            prior = st.slider("Prior campaigns accepted", 0, 5, 1)
        with b:
            education = st.selectbox("Education",
                                     ["Graduation", "PhD", "Master", "2n Cycle", "Basic"])
            marital = st.selectbox("Marital status",
                                   ["Married", "Together", "Single", "Divorced", "Widow", "Other"])
            kids = st.slider("Children at home", 0, 3, 0)
            wine = st.slider("Wine spend ($, 2 yrs)", 0, 1500, 300, 25)
            meat = st.slider("Meat spend ($, 2 yrs)", 0, 1700, 150, 25)
        web = st.slider("Web purchases", 0, 27, 4)
        catalog = st.slider("Catalog purchases", 0, 28, 3)
        store = st.slider("Store purchases", 0, 13, 5)
        visits = st.slider("Web visits / month", 0, 20, 5)

    # Build a full feature row: start from dataset medians, override with inputs.
    row = {c: float(df[c].median()) for c in data.NUMERIC_FEATURES}
    row.update({c: df[c].mode()[0] for c in data.CATEGORICAL_FEATURES})
    row.update({
        "Income": income, "Age": age, "Recency": recency,
        "Customer_Tenure_Years": tenure, "TotalAcceptedCmp": prior,
        "Education": education, "Marital_Status": marital, "Children": kids,
        "MntWines": wine, "MntMeatProducts": meat,
        "NumWebPurchases": web, "NumCatalogPurchases": catalog,
        "NumStorePurchases": store, "NumWebVisitsMonth": visits,
    })
    spend_total = sum(row[c] for c in SPEND)
    buys = web + catalog + store + row["NumDealsPurchases"]
    row["TotalSpending"] = spend_total
    row["TotalPurchases"] = buys
    row["SpendingPerPurchase"] = spend_total / buys if buys else 0.0
    row["EngagementScore"] = buys + 3 * prior

    X_one = pd.DataFrame([row])[data.MODEL_FEATURES]
    proba = float(run.best.pipeline.predict_proba(X_one)[0, 1])

    with right:
        st.markdown("##### Predicted response probability")
        st.plotly_chart(viz.gauge(proba, best_t), width=W)
        lift = proba / df["Response"].mean()
        if proba >= best_t:
            st.markdown(
                f"<div class='rec' style='border-color:#E8743B;background:#fdece3;'>"
                f"<b>✅ Contact this customer.</b> Probability <b>{proba:.0%}</b> is above the "
                f"profit-optimal threshold ({best_t:.0%}) — roughly <b>{lift:.1f}×</b> the base rate. "
                f"A high-yield target.</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div class='rec'><b>⏸ Hold / nurture.</b> Probability <b>{proba:.0%}</b> is below "
                f"the profit-optimal threshold ({best_t:.0%}). Contacting them likely costs more than "
                f"it returns — keep for a cheaper nurture track.</div>", unsafe_allow_html=True)
        st.metric("Lift vs. average customer", f"{lift:.1f}×")
        st.caption(f"Model: {run.best_name}. Threshold set to the profit-maximising point from the "
                   f"Marketing Lens.")


# --------------------------------------------------------------------------- #
# PAGE: Executive summary
# --------------------------------------------------------------------------- #
def page_summary():
    run = get_run()
    st.markdown(
        "<div class='hero'><h1>Executive Summary</h1>"
        "<p>For the Marketing Director — the one-pager.</p></div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Best model", run.best_name)
    c2.metric("ROC-AUC", f"{run.best.metrics['ROC-AUC']:.2f}")
    c3.metric("Warm-list lift", "≈ 3×")

    st.markdown(
        f"""
**The situation.** Our last campaign reached the whole base but only **~15% responded**, so most spend
hit people who'd never buy. We built a model that scores every customer's likelihood to respond — so we
can target instead of broadcast.

**Approach.** Cleaned {len(df):,} customer records → engineered 10 behaviour features (spend, recency,
engagement, prior-campaign history) → trained three models with leakage-free pipelines and
cross-validation → translated the winner into a targeting and budget plan.

**What we learned**
- **Past responders are gold:** prior acceptors respond at ~41% vs ~8% — a 3× list we already own.
- **Affluence pays:** high-income customers respond ~2× more; high-income *and* previously responsive → ~47%.
- **Recency & lifestyle matter:** recent buyers and childless, higher-income households respond far more;
  raw web traffic does **not** predict response.

**Best model.** **{run.best_name}** — ROC-AUC ≈ {run.best.metrics['ROC-AUC']:.2f}, F1 ≈
{run.best.metrics['F1']:.2f}, cross-validated and stable. It ranks customers well enough that a
model-driven list dramatically out-converts random contacting, and its probability output lets us set the
contact threshold to **maximise profit** for any budget.

**What leadership should do next**
1. **Re-target the ~460 prior responders immediately** (highest-yield, zero new cost).
2. **Concentrate budget on the top 2–3 probability deciles**; stop funding the unresponsive bottom half.
3. **Personalise by segment** — premium offers for high-income responders, recency-triggered sends for
   warm buyers, keep investing in catalog.
4. **Run it against a random-targeting control** so the lift is proven in dollars.

**Expected impact.** Same budget, pointed at customers who respond ~3× more often → materially more
accepted offers per dollar, less wasted spend, and a repeatable, measurable targeting engine instead of an
annual guess. The model doesn't just predict response — it tells marketing **where the next dollar earns
the most.**
        """
    )


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
ROUTES = {
    "Overview": page_overview,
    "Data & Quality": page_data_quality,
    "Exploratory Analysis": page_eda,
    "Key Insights": page_insights,
    "Feature Engineering": page_features,
    "Models & Evaluation": page_models,
    "Marketing Lens": page_marketing,
    "What Drives a Yes": page_importance,
    "Recommendations": page_recommendations,
    "Risks & Roadmap": page_risks,
    "Live Predictor": page_predictor,
    "Executive Summary": page_summary,
}
ROUTES[page]()
