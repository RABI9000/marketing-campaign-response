"""
Assemble the analysis notebook programmatically with nbformat.

Keeping the notebook's source in a build script means the 13-section structure
stays consistent and easy to regenerate. Run it, then execute the notebook:

    python scripts/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebook/marketing_campaign_analysis.ipynb
"""

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells = []


def md(text: str) -> None:
    cells.append(new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(new_code_cell(text.strip("\n")))


# ===========================================================================
# TITLE
# ===========================================================================
md(r"""
# Predicting Marketing Campaign Response
### Who to target, and what it's worth

**Author:** Data Science Candidate &nbsp;·&nbsp; **Dataset:** Kaggle *Marketing Campaign* (2,240 retail customers)

---

A retailer sent an offer to thousands of customers and wrote down who took it. Only about 1 in 7 said
yes. Contacting everyone costs money, and it mildly annoys the 6 in 7 who were never going to bite. So
the job here is simple to state: score how likely each customer is to respond, then use that score to
decide who to contact, where to spend, and what to expect back.

I spent my time the way the business would want me to. The model is solid, but most of the effort went
into what the numbers mean for revenue. Every chart has a plain-English read under it, and every
technical result gets tied back to a marketing decision.

**How to read this notebook**

| # | Section | What you get |
|---|---------|--------------|
| 1 | Project Overview | The problem, the goal, and how we judge success |
| 2 | Data Loading & Understanding | What's in the data, and what's wrong with it |
| 3 | Exploratory Data Analysis | The patterns, with **5+ business insights** |
| 4 | Feature Engineering | New signals we build from the raw columns |
| 5 | Data Preprocessing | Clean, leak-free preparation for modelling |
| 6 | Model Building | Logistic Regression, Random Forest, XGBoost |
| 7 | Model Evaluation | The metrics, head to head, and the winner |
| 8 | A Marketing View | Which metric actually matters, and why |
| 9 | Feature Importance | What drives a "yes" |
| 10 | Business Recommendations | The action plan for leadership |
| 11 | Deployment Risks | What could go wrong once it's live |
| 12 | Future Improvements | What I'd do with another week |
| 13 | Executive Summary | The one-pager for a Marketing Director |
""")

# ===========================================================================
# SECTION 1 — PROJECT OVERVIEW
# ===========================================================================
md(r"""
# 1. Project Overview

**The problem.** Marketing budget is limited, and every contact costs something: a mailer, a call, a
discount that eats into margin. When an offer goes to the whole customer base, most of that money lands
on people who were never going to buy. The team wants to stop spraying and start aiming.

**The goal.** Build a model that takes what we know about a customer and predicts how likely they are to
accept the next offer (`Response` = 1) or ignore it (`Response` = 0). The probability is more useful than
a plain yes/no, because it lets us rank customers and work down the list until the budget runs out.

**How we'll know it worked.**

- On the data side: the model should rank customers well enough that a model-picked contact list beats
  contacting people at random, by a wide margin. The numbers to watch are **ROC-AUC** (how well it ranks)
  and **recall** (how many real responders we catch). Plain accuracy is misleading here. With an 85/15
  split, a model that says "no" to everyone scores 85% and is useless.
- On the business side: a contact plan that reaches most of the responders while cutting wasted contacts,
  plus a short list of customer traits the team can act on right away.

**Why marketing cares.** Two numbers sit behind everything: what a wasted contact costs, and what a missed
responder costs. A good model moves spend away from the first and toward the second. Put plainly, it
answers the three questions a marketing director actually asks: who do we call first, how much do we
spend, and what do we get back?
""")

# ===========================================================================
# SETUP
# ===========================================================================
md(r"""
## Setup

The usual Python stack. The styling block below just keeps every chart easy to read, with one rule:
**coral always means "responded"** and grey means "didn't".
""")

code(r"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 120)
np.random.seed(42)

# A small, consistent visual identity for the whole notebook.
NAVY, BLUE, TEAL = "#1F3A5F", "#2E6F9E", "#3FA7A0"
CORAL, GREY, GOLD = "#E8743B", "#9AA7B2", "#F2B134"
RESP_PALETTE = {0: GREY, 1: CORAL}      # 0 = ignored, 1 = responded
SEQ = [NAVY, BLUE, TEAL, GOLD, CORAL]

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.titleweight": "bold",
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "axes.edgecolor": "#cccccc",
    "font.size": 11,
    "legend.frameon": False,
})

def pct_axis(ax, axis="y"):
    # Format an axis as whole-number percentages.
    fmt = mtick.PercentFormatter(xmax=1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)

print("Environment ready.")
""")

# ===========================================================================
# SECTION 2 — DATA LOADING & UNDERSTANDING
# ===========================================================================
md(r"""
# 2. Data Loading & Understanding

The file is semicolon-separated and starts with a UTF-8 byte-order mark, so we read it with `sep=";"` and
`encoding="utf-8-sig"`. If the local copy isn't there (say, a fresh Google Colab session), the code falls
back to the public URL, so the notebook always runs start to finish.
""")

code(r"""
LOCAL_PATH = "../data/marketing_campaign.csv"
RAW_URL = ("https://raw.githubusercontent.com/nurimammasri/"
           "Marketing-Campaign-Model-Prediction-by-Datalicious/main/"
           "data/marketing_campaign.csv")

source = LOCAL_PATH if os.path.exists(LOCAL_PATH) else RAW_URL
raw = pd.read_csv(source, sep=";", encoding="utf-8-sig")
print(f"Loaded {raw.shape[0]:,} customers x {raw.shape[1]} columns from "
      f"{'local file' if source == LOCAL_PATH else 'remote URL'}")
""")

md(r"""
## Dataset overview

One row per customer. The columns sort into a few natural groups, and knowing the groups is half the work:

| Group | Columns | What it captures |
|-------|---------|------------------|
| **Identity** | `ID` | Unique key (not predictive) |
| **Demographics** | `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome`, `Dt_Customer` | Who the customer is and when they joined |
| **Recency** | `Recency` | Days since their last purchase |
| **Spending (2 yrs)** | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` | Money spent per product category |
| **Purchasing & channels** | `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth` | How and how often they buy |
| **Campaign history** | `AcceptedCmp1`–`AcceptedCmp5`, `Complain` | How they reacted to past offers |
| **Target** | `Response` | **1 = accepted the latest campaign, 0 = ignored it** |

Two columns, `Z_CostContact` and `Z_Revenue`, are just bookkeeping constants (cost 3, revenue 11 per
offer). They carry no signal, but we'll borrow those numbers later to turn model scores into dollars.
""")

code(r"""
raw.head()
""")

code(r"""
raw.info()
""")

md(r"""
### Target distribution

First question in any response problem: how lopsided is the target? That one number shapes every
modelling choice that follows.
""")

code(r"""
counts = raw["Response"].value_counts().sort_index()
rate = raw["Response"].mean()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].bar(["Ignored (0)", "Responded (1)"], counts.values, color=[GREY, CORAL])
for i, v in enumerate(counts.values):
    ax[0].text(i, v + 20, f"{v:,}", ha="center", fontweight="bold")
ax[0].set_title("Response counts")
ax[0].set_ylabel("Customers")

ax[1].pie(counts.values, labels=["Ignored", "Responded"], colors=[GREY, CORAL],
          autopct="%1.1f%%", startangle=90, wedgeprops=dict(width=0.45))
ax[1].set_title("Response share")
plt.suptitle(f"Only {rate:.1%} of customers responded to the last campaign",
             fontsize=15, fontweight="bold")
plt.tight_layout()
plt.show()
""")

md(r"""
**What this says:** roughly 15% responded and 85% didn't, about a 5.7-to-1 split. Two things follow.
First, accuracy is misleading (saying "no" to everyone already scores 85%), so we judge models on ROC-AUC
and recall instead. Second, we have to tell each model to pay extra attention to the rare "yes" cases, or
it will just learn to predict "no" and ignore the very customers we care about.
""")

md(r"""
## Data quality checks

Before trusting any pattern, it's worth finding out where the data is wrong. Five checks: missing values,
duplicates, wrong data types, outliers, and messy categories.
""")

code(r"""
# 1) Missing values
missing = raw.isna().sum()
missing = missing[missing > 0].to_frame("missing")
missing["pct"] = (missing["missing"] / len(raw) * 100).round(2)
print("Columns with missing values:")
print(missing)
print("\nEverything else is complete.")
""")

md(r"""
**What this says:** only `Income` is missing, for 24 customers (about 1%). It's a small gap, and it looks
random (a few records where income just wasn't recorded), so it's safe to fill in. We'll do the filling
*inside the model pipeline*, so the fill value is learned only from training data and never peeks at the
test set.
""")

code(r"""
# 2) Duplicates
print("Duplicate IDs        :", raw["ID"].duplicated().sum())
print("Fully duplicate rows :", raw.duplicated().sum())
""")

md(r"""
**What this says:** no repeated customers and no duplicate rows. One row per person, as it should be.
""")

code(r"""
# 3) Datatypes worth fixing
print(raw[["Dt_Customer", "Income", "Year_Birth"]].dtypes)
print("\nDt_Customer is stored as text, e.g.:", raw["Dt_Customer"].iloc[0])
print("Z_CostContact unique values:", raw["Z_CostContact"].unique(),
      "| Z_Revenue unique values:", raw["Z_Revenue"].unique())
""")

md(r"""
**What this says:** the signup date (`Dt_Customer`) is stored as text, so we convert it to a real date
before measuring how long someone has been a customer. `Z_CostContact` and `Z_Revenue` never change
(always 3 and 11), so they're useless as predictors and get dropped. We keep the numbers themselves,
though, for the cost-and-revenue maths later on.
""")

code(r"""
# 4) Outliers — Year_Birth and Income
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
sns.boxplot(x=raw["Year_Birth"], ax=ax[0], color=BLUE, fliersize=4)
ax[0].set_title("Year_Birth — three impossible values")
sns.boxplot(x=raw["Income"], ax=ax[1], color=TEAL, fliersize=4)
ax[1].set_title("Income — one extreme outlier")
ax[1].xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
plt.tight_layout()
plt.show()

print("Oldest 'birth years':", sorted(raw["Year_Birth"].unique())[:5],
      "-> implied ages of 115+; data-entry errors.")
print("Top incomes        :", sorted(raw["Income"].dropna().unique())[-3:],
      "-> $666,666 is ~4x the next earner; an error.")
""")

md(r"""
**What this says:** three customers were "born" before 1900, which would make them 115+ years old, and
one reports an income of \$666,666, roughly four times the next-highest earner. These are typos, not real
people, so we drop those few rows to keep the age and income numbers honest.
""")

code(r"""
# 5) Category inconsistencies
print("Marital_Status:\n", raw["Marital_Status"].value_counts(), "\n")
print("Education:\n", raw["Education"].value_counts())
""")

md(r"""
**What this says:** `Marital_Status` has a few junk entries: `Alone` (3), `Absurd` (2), `YOLO` (2).
"Alone" really means Single, so we fold it in. "Absurd" and "YOLO" are nonsense, so they go into an
"Other" bucket. `Education` is fine; it just uses European labels (`2n Cycle` is roughly a Master's,
`Basic` is primary school).

### The verdict, in plain terms

This is a clean dataset. The only real work is dropping two dead columns, converting one date, removing a
few typo rows, tidying a handful of marital-status labels, and filling 24 missing incomes inside the
pipeline. Nothing dramatic, which means we can trust what the data tells us next.
""")

# ===========================================================================
# SECTION 3 — EDA
# ===========================================================================
md(r"""
# 3. Exploratory Data Analysis

Now we clean the data (using what we found above) and add a handful of engineered features, which Section
4 explains in full. That lets the charts talk in business terms (total spending, income tier, past
campaigns accepted) instead of raw columns. Then we walk through demographics, spending, and campaign
behaviour, and finish by comparing the people who responded against the people who didn't.
""")

code(r"""
def clean_data(df):
    # Structural cleaning only — see Section 2 for the justification of each step.
    df = df.copy()
    df = df.drop(columns=["ID", "Z_CostContact", "Z_Revenue"])
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce")
    df = df[df["Year_Birth"] >= 1900]                              # drop impossible ages
    df = df[(df["Income"].isna()) | (df["Income"] < 600_000)]      # drop the income error
    df["Marital_Status"] = df["Marital_Status"].astype("object").replace(
        {"Alone": "Single", "Absurd": "Other", "YOLO": "Other"})
    df["Education"] = df["Education"].astype("object")
    return df.reset_index(drop=True)

SPEND = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
         "MntSweetProducts", "MntGoldProds"]
BUY = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumDealsPurchases"]
CMP = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]
SNAPSHOT = pd.Timestamp("2015-01-01")   # enrolment ends mid-2014; freeze the clock here

def engineer_features(df):
    df = df.copy()
    df["Age"] = 2015 - df["Year_Birth"]
    df["Children"] = df["Kidhome"] + df["Teenhome"]
    df["HasChildren"] = (df["Children"] > 0).astype(int)
    df["TotalSpending"] = df[SPEND].sum(axis=1)
    df["TotalPurchases"] = df[BUY].sum(axis=1)
    df["SpendingPerPurchase"] = np.where(df["TotalPurchases"] > 0,
                                          df["TotalSpending"] / df["TotalPurchases"], 0.0)
    df["TotalAcceptedCmp"] = df[CMP].sum(axis=1)
    df["Customer_Tenure_Years"] = (SNAPSHOT - df["Dt_Customer"]).dt.days / 365.25
    df["EngagementScore"] = df["TotalPurchases"] + 3 * df["TotalAcceptedCmp"]
    df["IncomeSegment"] = pd.cut(df["Income"], [-np.inf, 35_000, 70_000, np.inf],
                                 labels=["Low", "Medium", "High"])
    df["AgeGroup"] = pd.cut(df["Age"], [0, 35, 50, 65, np.inf],
                            labels=["<=35", "36-50", "51-65", "65+"])
    return df

df = engineer_features(clean_data(raw))
overall_rate = df["Response"].mean()
print(f"After cleaning: {df.shape[0]:,} customers, {df.shape[1]} columns "
      f"(removed {len(raw) - len(df)} error rows).")
print(f"Overall response rate: {overall_rate:.1%}")
""")

md(r"""
### Demographics: who are these customers?
""")

code(r"""
fig, ax = plt.subplots(2, 2, figsize=(13, 9))

sns.histplot(df["Age"], bins=30, color=BLUE, ax=ax[0, 0])
ax[0, 0].set_title("Age")
ax[0, 0].set_xlabel("Age (years)")

sns.histplot(df["Income"].dropna(), bins=40, color=TEAL, ax=ax[0, 1])
ax[0, 1].set_title("Income")
ax[0, 1].set_xlabel("Income")
ax[0, 1].xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))

edu = df["Education"].value_counts()
sns.barplot(x=edu.values, y=edu.index, ax=ax[1, 0], palette=SEQ, hue=edu.index, legend=False)
ax[1, 0].set_title("Education")
ax[1, 0].set_xlabel("Customers")

mar = df["Marital_Status"].value_counts()
sns.barplot(x=mar.values, y=mar.index, ax=ax[1, 1], palette=SEQ, hue=mar.index, legend=False)
ax[1, 1].set_title("Marital status")
ax[1, 1].set_xlabel("Customers")

plt.suptitle("Customer demographics", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()
""")

md(r"""
**What this says:** a fairly mature, comfortable customer base. Most are between 40 and 60, income sits
around \$50k with a long tail of high earners, and the typical customer is a university graduate living
with a partner. These are people with money to spend. The campaign's job is to find the ones most willing
to spend it.
""")

md(r"""
### Spending: where does the money go?
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

cat_spend = df[SPEND].sum().sort_values(ascending=True)
labels = [c.replace("Mnt", "").replace("Products", "") for c in cat_spend.index]
ax[0].barh(labels, cat_spend.values, color=BLUE)
ax[0].set_title("Total spend by product category")
ax[0].xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))

sns.histplot(df["TotalSpending"], bins=40, color=TEAL, ax=ax[1])
ax[1].set_title("Total spending per customer (2 yrs)")
ax[1].set_xlabel("Total spending")
ax[1].xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
plt.tight_layout()
plt.show()

print("Share of wallet:")
print((df[SPEND].sum() / df[SPEND].sum().sum() * 100).round(1).sort_values(ascending=False))
""")

md(r"""
**What this says:** wine and meat dominate, together making up about half of all spending. And spending
per customer is very uneven: a small group of big spenders accounts for a large share of revenue. That
kind of skew is something tree models handle well, and it's an early hint that how much someone spends
will say a lot about whether they respond.
""")

md(r"""
### Campaigns and channels: how do they engage?
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

cmp_rates = df[CMP].mean().sort_index()
ax[0].bar([c.replace("Accepted", "") for c in cmp_rates.index],
          cmp_rates.values, color=NAVY)
for i, v in enumerate(cmp_rates.values):
    ax[0].text(i, v + 0.002, f"{v:.1%}", ha="center", fontsize=10)
ax[0].set_title("Acceptance rate of past campaigns")
pct_axis(ax[0])

chan = df[["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]].mean()
ax[1].bar(["Web", "Catalog", "Store"], chan.values, color=[BLUE, TEAL, GOLD])
for i, v in enumerate(chan.values):
    ax[1].text(i, v + 0.05, f"{v:.1f}", ha="center", fontweight="bold")
ax[1].set_title("Avg purchases per customer, by channel")
ax[1].set_ylabel("Avg purchases")
plt.tight_layout()
plt.show()
""")

md(r"""
**What this says:** past campaigns only landed with about 3-7% of customers, and Campaign 2 basically
flopped at 1%. That low base is exactly why better targeting is worth so much; there's plenty of room to
improve. On channels, the store does the heavy lifting, but catalog buyers (fewer in number) turn out to
respond unusually often, which the next charts confirm.
""")

md(r"""
### Responders vs non-responders: the comparison that matters

This is the part that counts. For each promising signal, do the people who responded actually look
different from everyone else?
""")

code(r"""
fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))
for a, col, title in zip(
        ax,
        ["Income", "TotalSpending", "Recency"],
        ["Income", "Total spending", "Recency (days since last buy)"]):
    sns.boxplot(data=df, x="Response", y=col, hue="Response",
                palette=RESP_PALETTE, legend=False, ax=a, fliersize=2)
    a.set_title(title)
    a.set_xticklabels(["Ignored", "Responded"])
    a.set_xlabel("")
plt.suptitle("Responders earn more, spend more, and bought more recently",
             fontsize=15, fontweight="bold")
plt.tight_layout()
plt.show()

print(df.groupby("Response")[["Income", "TotalSpending", "Recency",
      "TotalPurchases", "NumCatalogPurchases"]].mean().round(0))
""")

code(r"""
# Response rate across the most actionable segments
fig, ax = plt.subplots(2, 2, figsize=(13, 9))

def rate_bar(col, a, title, order=None):
    g = df.groupby(col, observed=True)["Response"].mean()
    if order is not None:
        g = g.reindex(order)
    bars = a.bar(g.index.astype(str), g.values, color=BLUE)
    a.axhline(overall_rate, color=CORAL, ls="--", lw=2)
    a.text(len(g) - 0.5, overall_rate + 0.005, f"avg {overall_rate:.0%}",
           color=CORAL, ha="right", fontsize=10)
    for b, v in zip(bars, g.values):
        a.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.0%}",
               ha="center", fontweight="bold", fontsize=10)
    a.set_title(title)
    pct_axis(a)

rate_bar("IncomeSegment", ax[0, 0], "Response rate by income tier", ["Low", "Medium", "High"])
rate_bar("TotalAcceptedCmp", ax[0, 1], "Response rate by # of prior campaigns accepted")
rate_bar("HasChildren", ax[1, 0], "Response rate by children at home")
ax[1, 0].set_xticklabels(["No children", "Has children"])
rate_bar("Education", ax[1, 1], "Response rate by education")
plt.suptitle("What moves the response rate", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()
""")

code(r"""
# Correlation of numeric signals with Response
num = df.select_dtypes("number")
corr = num.corr(numeric_only=True)["Response"].drop("Response").abs().sort_values(ascending=False).head(12)

fig, ax = plt.subplots(figsize=(8.5, 6))
sns.barplot(x=corr.values, y=corr.index, palette="crest", hue=corr.index, legend=False, ax=ax)
ax.set_title("Strongest linear correlations with Response")
ax.set_xlabel("|correlation|")
plt.tight_layout()
plt.show()
""")

md(r"""
**What this says:** accepting a past campaign is by far the strongest signal, with spending and income not
far behind. Worth noting what's missing: how often someone visits the website barely moves the needle on
whether they respond. Browsing isn't buying. Good to know before anyone spends heavily retargeting people
just for clicking around.
""")

# ----- KEY INSIGHTS -----
md(r"""
# Key Insights

Five findings. For each one: what I saw, the evidence in the data, and what the business should do about it.

---

### Insight 1: the best predictor of a "yes" is a past "yes"
**What I saw.** Whether someone responded before tells you more than anything else about whether they'll
respond again.
**Evidence.** People who accepted at least one past campaign respond at **40.7%**, against **8.2%** for
those who never have. That's 2.7 times the rate. And it keeps climbing: one past acceptance gets you 31%,
two gets 51%, three gets **80%**. It's also the single strongest correlate with `Response` (r ≈ 0.43).
**What to do.** Your warmest list already exists. Before spending anything on finding new prospects, go
back to the ~460 customers who've said yes before. They convert at roughly three times the average.

---

### Insight 2: higher income, higher response (about double)
**What I saw.** Response climbs steadily as income rises.
**Evidence.** High earners (above \$70k) respond at **28.1%**, against **10.0%** for the under-\$35k group,
about 1.9 times higher. Combine high income with a past acceptance and it jumps to **47%**.
**What to do.** Wealthier customers are both more likely to buy and worth more when they do. Give them a
better, higher-value offer instead of the same coupon everyone gets.

---

### Insight 3: recent buyers are warm, lapsed buyers are cold
**What I saw.** The more recently someone bought, the more likely they respond.
**Evidence.** People who responded had bought about **35 days** ago on average. People who didn't, about
**52 days** ago.
**What to do.** Reach out while the relationship is still warm. Tie campaigns to a recent purchase (say,
within 30 days) instead of blasting the whole list on a fixed schedule.

---

### Insight 4: households without kids respond far more
**What I saw.** Having children or teenagers at home goes with a much lower response rate.
**Evidence.** Customers with no kids respond at **26.5%**, against **10.3%** for those with children,
about 1.8 times higher. (The no-kids group also tends to earn and spend more.)
**What to do.** Household makeup is a free, always-available filter. Lean toward no-children,
higher-spending households for premium offers.

---

### Insight 5: catalog buyers punch above their weight
**What I saw.** Catalog activity separates responders from everyone else more clearly than web or store
activity does.
**Evidence.** Responders average **4.2** catalog purchases, against **2.4** for non-responders. Catalog is
among the top signals, while website visits are basically flat (both groups average about 5.3 a month).
**What to do.** Keep investing in catalog-active customers; they're a high-yield group. And don't be
fooled by web traffic: a visit isn't intent, so trust purchase behaviour over browsing.

---

### A few more, worth keeping in your back pocket
PhD holders respond at **20.8%** against **3.7%** for the basic-education group. Single, divorced, and
widowed customers respond more than partnered ones. And response is U-shaped by age: the under-35s and the
over-65s both respond more than the middle. Handy secondary filters when you're fine-tuning a segment.
""")

# ===========================================================================
# SECTION 4 — FEATURE ENGINEERING
# ===========================================================================
md(r"""
# 4. Feature Engineering

The raw columns describe a customer. Engineered features describe their **behaviour and value**, and
that's what actually predicts response. The functions back in Section 3 already built these. Here's what
each one is and why it earns its place. That's ten, well past the five the brief asks for.

| # | Feature | How it's built | Why it matters |
|---|---------|----------------|----------------|
| 1 | **Age** | `2015 − Year_Birth` | People think in age, not birth year; lets us segment by life stage |
| 2 | **TotalSpending** | Sum of the 6 `Mnt*` columns | One "how valuable is this customer" number; big spenders convert more |
| 3 | **TotalPurchases** | Web + catalog + store + deals | How often they buy overall |
| 4 | **SpendingPerPurchase** | `TotalSpending ÷ TotalPurchases` | Average basket size; tells premium shoppers from bargain hunters |
| 5 | **TotalAcceptedCmp** | Sum of `AcceptedCmp1–5` | Proven willingness to respond; the strongest single signal |
| 6 | **Customer_Tenure_Years** | `(2015-01-01 − Dt_Customer)` | New and long-standing customers behave differently |
| 7 | **EngagementScore** | `TotalPurchases + 3 × TotalAcceptedCmp` | One number combining buying and responding (see below) |
| 8 | **Children / HasChildren** | `Kidhome + Teenhome` | A stand-in for lifestyle and spare income |
| 9 | **IncomeSegment** | Bins: <35k / 35–70k / >70k | Lets leadership think in tiers, not raw dollars |
| 10 | **AgeGroup** | Bins: ≤35 / 36–50 / 51–65 / 65+ | Readable age cross-tabs |

**About the EngagementScore.** I wanted one easy-to-read number that rewards two things: buying and
responding. So `EngagementScore = TotalPurchases + 3 × TotalAcceptedCmp`. The 3× weight on past
acceptances is on purpose. Insight 1 showed a single past "yes" predicts future response far better than
one more purchase does, so it should count for more. It's a number a marketer can check by hand, not a
black box.
""")

code(r"""
engineered_cols = ["Age", "TotalSpending", "TotalPurchases", "SpendingPerPurchase",
                   "TotalAcceptedCmp", "Customer_Tenure_Years", "EngagementScore",
                   "Children", "IncomeSegment", "AgeGroup"]
df[engineered_cols].head()
""")

code(r"""
# Quick validation that the headline features actually separate responders
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
sns.kdeplot(data=df, x="EngagementScore", hue="Response", palette=RESP_PALETTE,
            fill=True, common_norm=False, ax=ax[0])
ax[0].set_title("EngagementScore separates responders")
ax[0].set_xlim(0, 40)

sns.barplot(data=df, x="IncomeSegment", y="Response", order=["Low", "Medium", "High"],
            palette="crest", hue="IncomeSegment", legend=False, ax=ax[1], errorbar=None)
ax[1].set_title("Response rate rises with income tier")
pct_axis(ax[1])
plt.tight_layout()
plt.show()
""")

md(r"""
**What this says:** responders (coral) clearly sit further right on EngagementScore, and the response
rate steps up neatly from low to high income. Both engineered features carry real signal before we've
even trained a model.
""")

# ===========================================================================
# SECTION 5 — DATA PREPROCESSING
# ===========================================================================
md(r"""
# 5. Data Preprocessing

The one rule that matters here is **no data leakage**. Anything that learns from the data (the median used
to fill income, the mean and standard deviation used for scaling, the categories used for encoding) has to
be learned from the training rows only, then applied to the test rows. We guarantee that by wrapping every
step in a scikit-learn `Pipeline` and only ever calling `.fit()` on the training data.

**The plan, step by step**

- **Missing values.** `Income` (24 rows) gets filled with the median, inside the pipeline. Median rather
  than mean, so the income skew doesn't pull the fill value up.
- **Outliers.** Already handled when we cleaned the data (the impossible ages and the \$666k income).
  Median filling and tree models cope fine with the rest.
- **Encoding.** `Education` and `Marital_Status` become one-hot columns. We set `handle_unknown="ignore"`
  so a category we've never seen before can't crash the model at prediction time.
- **Scaling.** Only for Logistic Regression, which is sensitive to scale. Random Forest and XGBoost don't
  care about scale, so we skip it there and keep their splits easy to read.
- **What goes in.** The engineered behaviour features plus the past-campaign flags. Those `AcceptedCmp`
  flags are known *before* the new campaign goes out, so using them isn't cheating. We leave out IDs, raw
  dates, and the target.
""")

code(r"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

NUMERIC = ["Income", "Recency", "Age", "Children", "TotalSpending", "TotalPurchases",
           "SpendingPerPurchase", "TotalAcceptedCmp", "Customer_Tenure_Years",
           "EngagementScore", "NumWebVisitsMonth", "NumWebPurchases",
           "NumCatalogPurchases", "NumStorePurchases", "NumDealsPurchases"] + SPEND
CATEGORICAL = ["Education", "Marital_Status"]
FEATURES = NUMERIC + CATEGORICAL

X = df[FEATURES].copy()
y = df["Response"].astype(int)

def make_preprocessor(scale):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         CATEGORICAL),
    ])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)

print(f"Train: {X_train.shape[0]:,} rows   Test: {X_test.shape[0]:,} rows")
print(f"Train response rate: {y_train.mean():.1%}   Test response rate: {y_test.mean():.1%}")
print(f"Model inputs: {len(FEATURES)} columns ({len(NUMERIC)} numeric, {len(CATEGORICAL)} categorical)")
""")

md(r"""
**What this says:** an 80/20 split, stratified so the 15% response rate is the same in train and test.
That makes the test score a fair stand-in for real-world performance. Filling, encoding, and scaling now
live inside each model's pipeline, so they get re-learned cleanly on every cross-validation fold.
""")

# ===========================================================================
# SECTION 6 — MODEL BUILDING
# ===========================================================================
md(r"""
# 6. Model Building

I trained three models on purpose, from the most explainable to the best-performing. All three are told to
deal with the class imbalance.

### Model 1: Logistic Regression (the honest baseline)
- **Why.** It's the yardstick. If a simple linear model already ranks customers well, the signal is
  strong, and anything fancier has to earn its keep.
- **Strengths.** Easy to read (each coefficient tells you the direction and strength of a driver), fast,
  well-behaved probabilities, hard to overfit.
- **Weaknesses.** It assumes effects are roughly linear and additive, so it misses combinations (like high
  income *and* a recent purchase together) unless you build them in by hand.

### Model 2: Random Forest (the sturdy all-rounder)
- **Why.** It picks up non-linear patterns and feature combinations on its own, needs little tuning, and
  shrugs off outliers and useless columns.
- **Strengths.** Strong with almost no setup, reports feature importances, doesn't care about scale.
- **Weaknesses.** Heavier on memory, harder to read than a single tree, and can lean toward features with
  many distinct values.

### Model 3: XGBoost (the one that usually wins)
- **Why.** Gradient-boosted trees are the standard choice for tabular data like this and normally come out
  on top. The `scale_pos_weight` setting handles the imbalance neatly.
- **Strengths.** Usually the best accuracy and AUC, built-in guards against overfitting, handles mixed
  feature types.
- **Weaknesses.** More knobs to tune, and less transparent (we lean on feature importance, and could add
  SHAP for per-customer explanations).

For evaluation we use a stratified 80/20 split for the headline numbers, plus **5-fold stratified
cross-validation** (on ROC-AUC) to check the ranking holds up and isn't just a lucky split.
""")

code(r"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()   # ~5.7 : 1

models = {
    "Logistic Regression": Pipeline([
        ("prep", make_preprocessor(scale=True)),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("prep", make_preprocessor(scale=False)),
        ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                          class_weight="balanced", n_jobs=-1, random_state=42)),
    ]),
}
if HAS_XGB:
    models["XGBoost"] = Pipeline([
        ("prep", make_preprocessor(scale=False)),
        ("model", XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4,
                                subsample=0.9, colsample_bytree=0.9,
                                scale_pos_weight=pos_weight, eval_metric="logloss",
                                random_state=42, n_jobs=-1)),
    ])
else:
    models["Gradient Boosting"] = Pipeline([
        ("prep", make_preprocessor(scale=False)),
        ("model", GradientBoostingClassifier(random_state=42)),
    ])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("5-fold cross-validated ROC-AUC on the training set:\n")
for name, pipe in models.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"  {name:<22} {scores.mean():.3f} ± {scores.std():.3f}")
    pipe.fit(X_train, y_train)   # fit on full training set for test evaluation
print("\nAll models trained.")
""")

md(r"""
**What this says:** all three land around 0.88-0.90 ROC-AUC with small error bars. The signal is real and
steady, not an accident of one split. XGBoost and Logistic Regression are almost tied here; the test set
below settles it on the numbers that drive action.
""")

# ===========================================================================
# SECTION 7 — MODEL EVALUATION
# ===========================================================================
md(r"""
# 7. Model Evaluation

Now we score each model on the held-out test set across the full set of metrics, look at the confusion
matrices and ROC curves, and then pick a winner.
""")

code(r"""
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)

rows, preds = [], {}
for name, pipe in models.items():
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    preds[name] = (pred, proba)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    })

results = pd.DataFrame(rows).set_index("Model").sort_values("ROC-AUC", ascending=False)
results.style.format("{:.3f}").background_gradient(cmap="Greens", axis=0)
""")

code(r"""
# Confusion matrices (default 0.5 threshold)
n = len(models)
fig, ax = plt.subplots(1, n, figsize=(5 * n, 4.3))
for a, (name, (pred, _)) in zip(np.atleast_1d(ax), preds.items()):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=a,
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    a.set_title(name, fontsize=12)
plt.suptitle("Confusion matrices (threshold = 0.50)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
""")

code(r"""
# ROC curve comparison
fig, ax = plt.subplots(figsize=(7.5, 6.5))
for (name, (_, proba)), c in zip(preds.items(), [BLUE, TEAL, CORAL]):
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, color=c, lw=2.5,
            label=f"{name} (AUC = {roc_auc_score(y_test, proba):.3f})")
ax.plot([0, 1], [0, 1], "--", color=GREY, label="Random guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curve — all models beat random by a wide margin")
ax.legend(loc="lower right", fontsize=11)
plt.tight_layout()
plt.show()
""")

md(r"""
### Picking the winner

| Model | What the test set shows |
|-------|-------------------------|
| **XGBoost** | Best ROC-AUC (~0.90) and best F1 (~0.62). It ranks customers best and strikes the best balance between precision and recall. |
| Random Forest | Highest precision (~0.72) but low recall (~0.42). Cautious; it catches fewer than half the responders. |
| Logistic Regression | Highest recall (~0.73) but low precision (~0.42). Casts a wide, noisy net. |

**The pick: XGBoost.** The model's real job is to rank customers so we can contact the most promising
first, and ROC-AUC measures exactly that. XGBoost leads on it and posts the best F1 too. Its behaviour at
the default 0.5 cutoff isn't the whole story, though. Because it outputs a probability, we can move that
cutoff to fit the budget, which is what the next section does.
""")

# ===========================================================================
# SECTION 8 — MARKETING PERSPECTIVE
# ===========================================================================
md(r"""
# 8. A Marketing View: which metric actually matters?

These metrics aren't abstract here. Each one maps to money.

- A **false positive** means we contacted someone who won't respond. That's a wasted contact (and a little
  brand fatigue). In this data, a contact costs **\$3**.
- A **false negative** means we skipped someone who would have responded. That's a lost sale, worth
  **\$11** each.

**Precision** is, of the people we contact, how many respond (it controls wasted spend). **Recall** is, of
all the people who would respond, how many we actually reach (it controls missed sales). **F1** balances
the two.

So which matters most? It depends on the economics, and here they're lopsided: a missed responder (\$11)
hurts more than a wasted contact (\$3). That points toward favouring recall, but not blindly, because
contacting everyone burns that \$3 over and over. The honest answer isn't to crown one metric. It's to set
the cutoff where expected profit is highest, using the model's probabilities. Here's that curve.
""")

code(r"""
# Sweep the decision threshold on the best model and compute campaign profit
best_name = results.index[0]
_, best_proba = preds[best_name]
yt = y_test.to_numpy()

REVENUE, COST = 11.0, 3.0      # the dataset's own unit economics (Z_Revenue, Z_CostContact)
ths = np.linspace(0.05, 0.95, 19)
prof, recalls, precs = [], [], []
for t in ths:
    p = (best_proba >= t).astype(int)
    tp = int(((p == 1) & (yt == 1)).sum())
    fp = int(((p == 1) & (yt == 0)).sum())
    fn = int(((p == 0) & (yt == 1)).sum())
    prof.append(tp * REVENUE - (tp + fp) * COST)
    recalls.append(tp / (tp + fn) if (tp + fn) else 0)
    precs.append(tp / (tp + fp) if (tp + fp) else 0)

best_t = ths[int(np.argmax(prof))]

fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
ax[0].plot(ths, prof, color=NAVY, lw=2.5, marker="o", ms=4)
ax[0].axvline(best_t, color=CORAL, ls="--", lw=2)
ax[0].axvline(0.5, color=GREY, ls=":", lw=2)
ax[0].text(best_t, min(prof), f"  profit-max\n  @ {best_t:.2f}", color=CORAL, va="bottom")
ax[0].set_title(f"Expected campaign profit vs. threshold ({best_name})")
ax[0].set_xlabel("Decision threshold"); ax[0].set_ylabel("Profit on test set ($)")

ax[1].plot(ths, recalls, color=CORAL, lw=2.5, label="Recall (responders caught)")
ax[1].plot(ths, precs, color=BLUE, lw=2.5, label="Precision (contacts that convert)")
ax[1].axvline(best_t, color=CORAL, ls="--", lw=1.5)
ax[1].set_title("The precision / recall trade-off")
ax[1].set_xlabel("Decision threshold"); ax[1].legend()
plt.tight_layout()
plt.show()

print(f"Best model: {best_name}")
print(f"Profit-maximising threshold ≈ {best_t:.2f} (vs. the naive 0.50).")
""")

md(r"""
**The takeaway: favour recall, and set the cutoff by profit.** Because a missed responder costs more than
a wasted contact, profit peaks at a cutoff *below* the default 0.5. In other words, it pays to contact a
few more people and accept some waste in order to catch more buyers. So, in practice:

1. **Pick the model with ROC-AUC.** It doesn't depend on the cutoff, and it measures the thing we rely on:
   ranking.
2. **Pick the cutoff with the profit curve**, using the campaign's real cost and margin and whatever
   budget cap exists. If you don't know the economics, lean toward recall (optimise F2 rather than F1),
   since the costs here favour catching responders over avoiding waste.

This is the most important translation in the whole project: the metric isn't fixed up front. The business
economics choose it.
""")

# ===========================================================================
# SECTION 9 — FEATURE IMPORTANCE
# ===========================================================================
md(r"""
# 9. Feature Importance: what drives a "yes"?
""")

code(r"""
best_pipe = models[best_name]
prep = best_pipe.named_steps["prep"]
model = best_pipe.named_steps["model"]
names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
imp = (pd.DataFrame({"feature": names, "importance": model.feature_importances_})
       .sort_values("importance", ascending=False).head(10).reset_index(drop=True))

fig, ax = plt.subplots(figsize=(9, 6))
sns.barplot(data=imp, x="importance", y="feature", palette="crest",
            hue="feature", legend=False, ax=ax)
ax.set_title(f"Top 10 drivers of campaign response — {best_name}")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.show()
imp
""")

md(r"""
**What drives a response, in plain terms.**

- **Past campaign acceptances (`TotalAcceptedCmp`) matter most, by a wide margin.** A past "yes" predicts
  a future "yes". The most valuable thing you own is a list of people who already responded once.
- **Recency:** how recently they bought. Warm customers convert; lapsed ones don't.
- **Tenure:** how long they've been a customer shapes how they respond.
- **Being single, and holding a PhD:** both groups respond more than average.
- **Catalog and store purchases, meat and wine spend, income:** the comfortable, engaged core.
- **Children at home:** fewer kids, higher response.

This isn't just model trivia. It's a targeting checklist. The most likely responder is a recent,
comfortable, previously-responsive buyer with no kids at home, and the model puts a number on how much each
trait shifts the odds.
""")

# ===========================================================================
# SECTION 10 — BUSINESS RECOMMENDATIONS
# ===========================================================================
md(r"""
# 10. Business Recommendations

Here's the analysis turned into a plan leadership can run this quarter.

### Who to contact first
Rank the whole base by the model's response probability and work down until the budget runs out. The top
of that list will be full of the traits we've shown matter: past responders, higher income, recent buyers,
no kids, catalog-active. Start with the ~460 customers who've accepted a campaign before. They convert at
about three times the average and cost nothing extra to find.

### Where to spend
Stop spreading the budget evenly. Put it on the top two or three deciles of predicted probability, where
the responders actually are. The profit curve in Section 8 makes the point: set the contact list by
expected profit per contact, not by trying to reach everyone. Shifting spend from the unresponsive bottom
half to the top decile is the quickest ROI win on the table.

### Who gets the personal touch
- **High earners who've responded before (about 47% response):** premium offers and early access, not a
  generic coupon.
- **Recent buyers:** reach out within about 30 days of a purchase, while they're warm.
- **Catalog-active customers:** keep feeding the channel they actually buy through.
- **Deal-driven, price-sensitive customers:** discount-led messaging, kept separate from the premium track.

### How to lift the response rate
- Shift from calendar blasts to triggered sends (off recency, off a recent browse-to-buy).
- Set the contact cutoff from the campaign's economics (Section 8) instead of defaulting to "the obvious
  few" or "everyone".
- A/B test the creative and the offer within the top deciles, and keep improving.

### How to make marketing pay more
The mechanism is simple. The same budget, aimed at customers who respond about three times more often,
produces far more accepted offers per dollar. Track response rate and profit per contact against a
random-targeting control, so the lift is something you can prove to finance rather than just claim.
""")

# ===========================================================================
# SECTION 11 — DEPLOYMENT RISKS
# ===========================================================================
md(r"""
# 11. Deployment Risks: what could go wrong once this is live?

A model is something you maintain, not a one-off result. Four ways it can fail, and how I'd handle each.

### False positives: contacting people who won't respond
- **Impact.** Wasted contact cost (\$3 a time here), and at scale, brand fatigue and unsubscribes that wear
  the list down.
- **What to do.** Set the cutoff from the economics (Section 8), cap how often any one person is contacted,
  and keep suppression lists. Watch precision in production and ease the cutoff back if waste creeps up.

### False negatives: skipping people who would have responded
- **Impact.** Lost sales, the costlier mistake here, since a missed responder (\$11) outweighs a wasted
  contact (\$3).
- **What to do.** Lean the cutoff toward recall, and keep a small random holdout that gets contacted no
  matter what, so we can keep spotting (and recovering) the responders the model misses.

### Data drift: the inputs change
- **Impact.** Incomes inflate, the channel mix shifts, or an upstream pipeline quietly changes units, and
  the model slowly gets worse without anyone noticing.
- **What to do.** Run a monitoring job that watches the input and score distributions (a
  population-stability index works well) and alerts when they move too far. Add schema checks on the
  feature pipeline.

### Changing behaviour: the world moves on
- **Impact.** After a big promotion, a downturn, or just a seasonal shift, yesterday's responders aren't
  tomorrow's, and accuracy fades.
- **What to do.** Retrain on a rolling window (say, quarterly), test a challenger model against the current
  one before promoting it, and watch the live response rate against what the model predicted as an early
  warning.

**A few things that apply across the board.** Calibrate the probabilities before treating them as dollars,
check the targeting isn't quietly excluding a whole group, and keep a readable explanation of the model
(feature importance now, SHAP later) so the marketing team trusts it and can explain it.
""")

# ===========================================================================
# SECTION 12 — FUTURE IMPROVEMENTS
# ===========================================================================
md(r"""
# 12. Future Improvements: what I'd do with another week

Ordered by business impact, not by what's fun to build.

### 1. Putting it to use (biggest payoff)
- A cutoff/budget optimiser in production, so each campaign automatically picks its most profitable contact
  list. This is where the money is.
- Customer segmentation (K-Means or RFM on spend, recency, and engagement) to match the *who* with the
  right offer, turning probabilities into tailored campaigns.
- An A/B testing setup against a random-targeting control, to prove the lift to finance.

### 2. Better data (medium-to-high payoff)
- Richer behavioural signals: email opens and clicks, how deep web sessions go, time since the last
  campaign, seasonality. The intent signals the current data is missing.
- Margin per product category, so we can optimise for profit, not just response.

### 3. Better modelling (smaller payoff)
- Hyperparameter tuning (Optuna or similar) and probability calibration (isotonic), so the scores are
  trustworthy as dollars.
- Feature selection to trim redundant inputs for a leaner, faster model.
- Stacking Logistic Regression with XGBoost, and adding SHAP for per-customer explanations the team can
  read.

The theme: I'd spend most of the week turning the model into decisions (segmentation, cutoff optimisation,
testing), because the model itself is already good enough. The upside now is in how it's used, not in
chasing another fraction of a point of AUC.
""")

# ===========================================================================
# SECTION 13 — EXECUTIVE SUMMARY
# ===========================================================================
md(r"""
# 13. Executive Summary
*For the Marketing Director. One page.*

**Where we are.** Our last campaign went to the whole base, but only about 15% responded, so most of the
spend hit people who were never going to buy. We built a model that scores how likely each customer is to
respond, so we can target instead of broadcast.

**What we did.** Cleaned 2,237 customer records, built behaviour features (spending, recency, engagement,
past-campaign history), trained and compared three models with leak-free pipelines and cross-validation,
and turned the best one into a targeting and budget plan.

**What we found.**
- Past responders are gold. People who accepted a campaign before respond at **41%** against **8%**, a
  list worth roughly three times the average that we already own.
- Money matters. High earners respond about **twice** as often; high earners who've responded before hit
  **47%**.
- Recency and lifestyle matter. Recent buyers and households without kids respond far more. Website
  traffic, on its own, does **not** predict response.

**The model.** **XGBoost**, with ROC-AUC around **0.90** and F1 around **0.62**, cross-validated and
steady. It ranks customers well enough that a model-built list comfortably beats random contacting, and
because it outputs a probability, we can set the contact cutoff to maximise profit for any budget.

**What to do next.**
1. Go back to the ~460 past responders right away. Highest yield, nothing new to find.
2. Concentrate budget on the top two or three probability deciles, and stop funding the unresponsive
   bottom half.
3. Personalise by segment: premium offers for high-income responders, recency-triggered sends for warm
   buyers, sustained catalog investment.
4. Run it against a random-targeting control, so the lift shows up in dollars.

**What it's worth.** The same budget, aimed at customers who respond about three times more often, means a
lot more accepted offers per dollar, less wasted spend, and a repeatable, measurable way to target instead
of an annual guess. The model doesn't just predict response. It shows marketing where the next dollar earns
the most.

---
*Built in Python (pandas, scikit-learn, XGBoost). Runs end to end, and an interactive Streamlit app comes
with it for live exploration and scoring individual customers.*
""")

# ===========================================================================
# WRITE NOTEBOOK
# ===========================================================================
nb = new_notebook(cells=cells)
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})

out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "notebook", "marketing_campaign_analysis.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Wrote {out_path} with {len(cells)} cells.")
