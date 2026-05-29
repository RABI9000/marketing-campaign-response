"""
Marketing Campaign Response Prediction — shared production code.

This package holds the clean, reusable logic that powers both the analysis
notebook and the Streamlit app:

- ``data``   : load the raw Kaggle file, clean it, and engineer features.
- ``models`` : build leakage-free preprocessing pipelines, train the model
               zoo (Logistic Regression, Random Forest, XGBoost), evaluate
               them, and extract feature importances.

The two entry points most callers need are :func:`data.load_prepared_data`
and :func:`models.train_model_zoo`.
"""

from . import data, models

__all__ = ["data", "models"]
