"""
Data layer: load, clean, and feature-engineer the marketing campaign dataset.

The raw file is the public "Marketing Campaign" dataset (2,240 customers of a
retailer, used in the Kaggle *marketing-data* competition). Each row is one
customer; the target ``Response`` is 1 if they accepted the most recent
campaign offer and 0 otherwise.

Design choices worth knowing:

* Cleaning here is *structural only* — we drop dead columns, fix data-entry
  errors, and tidy categories. We deliberately do **not** impute ``Income``
  here; that happens inside the modelling pipeline so it is re-fit on every
  cross-validation fold and never leaks test information into training.
* ``ANALYSIS_SNAPSHOT`` is the reference date used to turn enrolment dates and
  birth years into "age" and "tenure". Enrolment in this dataset ends in
  mid-2014, so we freeze the clock at 2015-01-01.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Local copy shipped with the repo; falls back to the public raw URL (handy in
# a fresh Colab runtime that has only cloned the notebook).
LOCAL_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "marketing_campaign.csv",
)
RAW_DATA_URL = (
    "https://raw.githubusercontent.com/nurimammasri/"
    "Marketing-Campaign-Model-Prediction-by-Datalicious/main/"
    "data/marketing_campaign.csv"
)

# The raw file is semicolon-separated and carries a UTF-8 byte-order mark.
RAW_SEP = ";"
RAW_ENCODING = "utf-8-sig"

# We treat the start of 2015 as "today" because the enrolment dates stop in
# mid-2014. Using a fixed snapshot keeps Age / Tenure reproducible over time.
ANALYSIS_SNAPSHOT = pd.Timestamp("2015-01-01")
REFERENCE_YEAR = 2015

# Columns that carry no signal and are dropped up front.
#   ID            -> identifier, not a feature
#   Z_CostContact -> constant (always 3)
#   Z_Revenue     -> constant (always 11)
DROP_COLUMNS = ["ID", "Z_CostContact", "Z_Revenue"]

SPENDING_COLUMNS = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]
PURCHASE_COLUMNS = [
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumDealsPurchases",
]
CAMPAIGN_COLUMNS = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
]
TARGET = "Response"


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_raw(path: str | None = None) -> pd.DataFrame:
    """Return the raw dataset exactly as published (no cleaning).

    Tries the local repo copy first, then the public raw URL so the code also
    runs in a clean Colab/Streamlit Cloud environment.
    """
    path = path or LOCAL_DATA_PATH
    source = path if os.path.exists(path) else RAW_DATA_URL
    df = pd.read_csv(source, sep=RAW_SEP, encoding=RAW_ENCODING)
    return df


# --------------------------------------------------------------------------- #
# Clean
# --------------------------------------------------------------------------- #
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply structural cleaning and return a new, tidy DataFrame.

    Steps (each maps to a known issue found during data-quality checks):

    1. Drop the identifier and the two constant ``Z_*`` columns.
    2. Parse ``Dt_Customer`` (enrolment date) to a real datetime.
    3. Remove 3 impossible birth years (< 1900 -> age 115+); these are
       clearly data-entry errors.
    4. Drop the single absurd income (666,666) that sits ~6x above the next
       highest earner and distorts every income statistic.
    5. Collapse the messy marital-status labels ("Alone", "Absurd", "YOLO")
       into meaningful groups.

    ``Income`` keeps its 24 genuine missing values — they are imputed later
    inside the cross-validated pipeline to avoid leakage.
    """
    df = df.copy()

    # 1. Drop dead columns (only those present, to stay robust).
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    # 2. Enrolment date -> datetime.
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce")

    # 3. Impossible birth years (3 customers "born" in 1893/1899/1900).
    df = df[df["Year_Birth"] >= 1900].copy()

    # 4. Single income outlier (666,666) — a data-entry error.
    df = df[(df["Income"].isna()) | (df["Income"] < 600_000)].copy()

    # 5. Tidy marital status.
    marital_map = {
        "Alone": "Single",
        "Absurd": "Other",
        "YOLO": "Other",
    }
    df["Marital_Status"] = (
        df["Marital_Status"].astype("object").replace(marital_map)
    )
    # Strings come back as the new pandas string dtype; force plain object so
    # scikit-learn's encoders see clean Python strings.
    df["Education"] = df["Education"].astype("object")

    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add business-meaningful features on top of the cleaned data.

    Every feature below answers a marketing question that the raw columns only
    hint at. See the notebook's Feature Engineering section for the full
    rationale; the short version is in the inline comments.
    """
    df = df.copy()

    # Age — customers think in age, not birth year.
    df["Age"] = REFERENCE_YEAR - df["Year_Birth"]

    # Children at home — a strong lifestyle / disposable-income signal.
    df["Children"] = df["Kidhome"] + df["Teenhome"]
    df["HasChildren"] = (df["Children"] > 0).astype(int)

    # Household value: how much each customer spends in total (2-year window).
    df["TotalSpending"] = df[SPENDING_COLUMNS].sum(axis=1)

    # How often they buy, across every channel.
    df["TotalPurchases"] = df[PURCHASE_COLUMNS].sum(axis=1)

    # Average basket value — premium vs. bargain shoppers.
    df["SpendingPerPurchase"] = np.where(
        df["TotalPurchases"] > 0,
        df["TotalSpending"] / df["TotalPurchases"],
        0.0,
    )

    # Loyalty signal: how many of the 5 prior campaigns they already accepted.
    df["TotalAcceptedCmp"] = df[CAMPAIGN_COLUMNS].sum(axis=1)

    # Tenure — long-standing customers behave differently from newcomers.
    df["Customer_Tenure_Days"] = (
        ANALYSIS_SNAPSHOT - df["Dt_Customer"]
    ).dt.days
    df["Customer_Tenure_Years"] = df["Customer_Tenure_Days"] / 365.25

    # Engagement score — a single transparent number combining how much a
    # customer transacts with how responsive they have been to past offers.
    # Prior campaign acceptance is the single strongest predictor of future
    # response, so it is weighted x3 relative to a raw purchase.
    df["EngagementScore"] = df["TotalPurchases"] + 3 * df["TotalAcceptedCmp"]

    # Income segments — leadership reasons in tiers, not raw dollars.
    df["IncomeSegment"] = pd.cut(
        df["Income"],
        bins=[-np.inf, 35_000, 70_000, np.inf],
        labels=["Low", "Medium", "High"],
    )

    # Age groups — for readable demographic cross-tabs.
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 35, 50, 65, np.inf],
        labels=["<=35", "36-50", "51-65", "65+"],
    )

    return df


def load_prepared_data(path: str | None = None) -> pd.DataFrame:
    """One-call helper: load -> clean -> engineer features."""
    return engineer_features(clean_data(load_raw(path)))


# --------------------------------------------------------------------------- #
# Model input definition
# --------------------------------------------------------------------------- #
# Features fed to the models. We use engineered Age/Tenure instead of the raw
# Year_Birth/Dt_Customer, keep the legitimate prior-campaign signals (they are
# known *before* the new campaign, so no leakage), and exclude IDs/dates.
NUMERIC_FEATURES: List[str] = [
    "Income",
    "Recency",
    "Age",
    "Children",
    "TotalSpending",
    "TotalPurchases",
    "SpendingPerPurchase",
    "TotalAcceptedCmp",
    "Customer_Tenure_Years",
    "EngagementScore",
    "NumWebVisitsMonth",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumDealsPurchases",
] + SPENDING_COLUMNS
CATEGORICAL_FEATURES: List[str] = ["Education", "Marital_Status"]
MODEL_FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def get_feature_matrix(df: pd.DataFrame):
    """Split a prepared DataFrame into (X, y) for modelling."""
    X = df[MODEL_FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    return X, y
