"""
Modelling layer: leakage-free preprocessing + the three-model zoo.

We compare a transparent linear baseline (Logistic Regression) against two
non-linear ensembles (Random Forest, XGBoost). Every model is wrapped in a
scikit-learn ``Pipeline`` so that imputation, scaling, and encoding are re-fit
on the training rows of every cross-validation fold — the test fold never
influences them. The target is imbalanced (~15% responders), so each model is
told to weight the minority class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

try:  # XGBoost is preferred but optional; gracefully degrade if absent.
    from xgboost import XGBClassifier

    _HAS_XGB = True
except Exception:  # pragma: no cover
    from sklearn.ensemble import GradientBoostingClassifier

    _HAS_XGB = False

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def build_preprocessor(scale: bool) -> ColumnTransformer:
    """Return a ColumnTransformer for the model inputs.

    Numeric columns are median-imputed (robust to the income outlier/missing)
    and optionally standard-scaled — scaling matters for Logistic Regression
    but is irrelevant for tree ensembles, so we switch it off there to keep
    the fitted trees interpretable. Categoricals are most-frequent imputed and
    one-hot encoded, ignoring any unseen category at predict time.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


# --------------------------------------------------------------------------- #
# Model zoo
# --------------------------------------------------------------------------- #
def build_model_zoo(scale_pos_weight: float) -> Dict[str, Pipeline]:
    """Build the three competing pipelines, keyed by display name."""
    zoo: Dict[str, Pipeline] = {}

    zoo["Logistic Regression"] = Pipeline(
        [
            ("prep", build_preprocessor(scale=True)),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    zoo["Random Forest"] = Pipeline(
        [
            ("prep", build_preprocessor(scale=False)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    if _HAS_XGB:
        booster = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        zoo["XGBoost"] = Pipeline(
            [("prep", build_preprocessor(scale=False)), ("model", booster)]
        )
    else:  # pragma: no cover - fallback when xgboost is unavailable
        from sklearn.ensemble import GradientBoostingClassifier

        zoo["Gradient Boosting"] = Pipeline(
            [
                ("prep", build_preprocessor(scale=False)),
                (
                    "model",
                    GradientBoostingClassifier(random_state=RANDOM_STATE),
                ),
            ]
        )

    return zoo


# --------------------------------------------------------------------------- #
# Train + evaluate
# --------------------------------------------------------------------------- #
@dataclass
class ModelResult:
    """Everything we need to report and plot for one fitted model."""

    name: str
    pipeline: Pipeline
    metrics: Dict[str, float]
    y_test: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    cv_auc_mean: float
    cv_auc_std: float

    @property
    def confusion(self) -> np.ndarray:
        return confusion_matrix(self.y_test, self.y_pred)

    @property
    def roc(self):
        fpr, tpr, _ = roc_curve(self.y_test, self.y_proba)
        return fpr, tpr


@dataclass
class TrainingRun:
    """Container for a full train/evaluate pass over the model zoo."""

    results: Dict[str, ModelResult] = field(default_factory=dict)
    X_train: pd.DataFrame = None
    X_test: pd.DataFrame = None
    y_train: pd.Series = None
    y_test: pd.Series = None

    @property
    def comparison(self) -> pd.DataFrame:
        """Tidy metric table, best ROC-AUC first."""
        rows = []
        for res in self.results.values():
            row = {"Model": res.name, **res.metrics}
            row["CV ROC-AUC"] = res.cv_auc_mean
            rows.append(row)
        out = pd.DataFrame(rows).set_index("Model")
        return out.sort_values("ROC-AUC", ascending=False)

    @property
    def best_name(self) -> str:
        return self.comparison.index[0]

    @property
    def best(self) -> ModelResult:
        return self.results[self.best_name]


def _metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
    }


def train_model_zoo(X: pd.DataFrame, y: pd.Series) -> TrainingRun:
    """Stratified split, cross-validate, fit, and score every model.

    Returns a :class:`TrainingRun` holding fitted pipelines, a comparison
    table, and the per-model predictions needed for ROC / confusion plots.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    zoo = build_model_zoo(scale_pos_weight=pos_weight)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    run = TrainingRun(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    for name, pipe in zoo.items():
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        run.results[name] = ModelResult(
            name=name,
            pipeline=pipe,
            metrics=_metrics(y_test, y_pred, y_proba),
            y_test=y_test.to_numpy(),
            y_pred=y_pred,
            y_proba=y_proba,
            cv_auc_mean=float(cv_auc.mean()),
            cv_auc_std=float(cv_auc.std()),
        )
    return run


# --------------------------------------------------------------------------- #
# Feature importance
# --------------------------------------------------------------------------- #
def feature_importance(pipeline: Pipeline, top_n: int = 10) -> pd.DataFrame:
    """Return the top-N features for a fitted pipeline.

    Works for tree models (``feature_importances_``) and linear models
    (absolute standardized coefficients), mapping the one-hot-encoded columns
    back to readable names.
    """
    prep: ColumnTransformer = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    names = prep.get_feature_names_out()
    names = [n.split("__", 1)[-1] for n in names]  # strip "num__"/"cat__"

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:  # linear model -> absolute coefficient as importance
        importances = np.abs(np.ravel(model.coef_))

    imp = (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return imp


def expected_profit_by_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    revenue_per_responder: float = 11.0,
    cost_per_contact: float = 3.0,
):
    """Sweep the decision threshold and return campaign profit at each point.

    Uses the dataset's own economics (Z_Revenue = 11 per accepted offer,
    Z_CostContact = 3 per contact) to show the money consequence of choosing a
    threshold — the bridge between an ML metric and a marketing decision.
    """
    thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        contacted = tp + fp
        profit = tp * revenue_per_responder - contacted * cost_per_contact
        rows.append(
            {
                "threshold": round(float(t), 2),
                "contacted": contacted,
                "responders_reached": tp,
                "wasted_contacts": fp,
                "profit": profit,
            }
        )
    return pd.DataFrame(rows)
