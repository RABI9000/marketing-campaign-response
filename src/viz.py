"""
Plotly figure builders for the Streamlit app.

Every chart shares one visual identity: **coral = responded, grey = ignored**,
a clean white template, and a brand palette that matches the notebook. Keeping
the figures here (separate from layout/copy) keeps ``app.py`` readable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve

# Brand palette (shared with the notebook).
NAVY, BLUE, TEAL = "#1F3A5F", "#2E6F9E", "#3FA7A0"
CORAL, GREY, GOLD = "#E8743B", "#9AA7B2", "#F2B134"
SEQ = [NAVY, BLUE, TEAL, GOLD, CORAL]
RESP_COLORS = {0: GREY, 1: CORAL}

FONT = "Inter, -apple-system, Segoe UI, sans-serif"


def _style(fig: go.Figure, height: int = 360, title: str | None = None,
           showlegend: bool = True) -> go.Figure:
    layout = dict(
        template="plotly_white",
        height=height,
        font=dict(family=FONT, size=13, color="#33404f"),
        margin=dict(l=10, r=10, t=50 if title else 24, b=10),
        # An explicit empty legend title avoids a stray "undefined" label that
        # plotly.js otherwise renders for horizontal legends inside Streamlit.
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        hoverlabel=dict(font_size=12, font_family=FONT),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=showlegend,
    )
    if title:
        layout["title"] = dict(text=title, font=dict(size=17, family=FONT, color=NAVY))
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#eef1f4", zeroline=False)
    return fig


# --------------------------------------------------------------------------- #
# Overview / EDA
# --------------------------------------------------------------------------- #
def target_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["Response"].value_counts().sort_index()
    fig = go.Figure(
        go.Pie(
            labels=["Ignored", "Responded"],
            values=counts.values,
            hole=0.62,
            marker=dict(colors=[GREY, CORAL]),
            textinfo="label+percent",
            sort=False,
        )
    )
    fig.add_annotation(text=f"<b>{df['Response'].mean():.0%}</b><br>responded",
                       showarrow=False, font=dict(size=18, color=NAVY))
    return _style(fig, height=320, showlegend=False)


def response_rate_bar(df: pd.DataFrame, dim: str, order=None) -> go.Figure:
    """Response rate by any categorical dimension, with the base rate marked."""
    base = df["Response"].mean()
    g = df.groupby(dim, observed=True)["Response"].agg(["mean", "count"])
    if order is not None:
        g = g.reindex([o for o in order if o in g.index])
    fig = go.Figure(
        go.Bar(
            x=g.index.astype(str),
            y=g["mean"],
            marker_color=BLUE,
            text=[f"{v:.0%}" for v in g["mean"]],
            textposition="outside",
            customdata=g["count"],
            hovertemplate="%{x}<br>Response: %{y:.1%}<br>Customers: %{customdata:,}<extra></extra>",
        )
    )
    fig.add_hline(y=base, line_dash="dash", line_color=CORAL,
                  annotation_text=f"base {base:.0%}", annotation_font_color=CORAL)
    fig.update_yaxes(tickformat=".0%")
    return _style(fig, height=380, showlegend=False)


def box_by_response(df: pd.DataFrame, column: str) -> go.Figure:
    fig = go.Figure()
    for resp, name in [(0, "Ignored"), (1, "Responded")]:
        sub = df[df["Response"] == resp][column].dropna()
        fig.add_trace(go.Box(y=sub, name=name, marker_color=RESP_COLORS[resp],
                             boxmean=True))
    return _style(fig, height=380)


def histogram_by_response(df: pd.DataFrame, column: str, nbins: int = 40) -> go.Figure:
    fig = go.Figure()
    for resp, name in [(0, "Ignored"), (1, "Responded")]:
        sub = df[df["Response"] == resp][column].dropna()
        fig.add_trace(go.Histogram(x=sub, name=name, opacity=0.75,
                                   marker_color=RESP_COLORS[resp], nbinsx=nbins))
    fig.update_layout(barmode="overlay")
    return _style(fig, height=380)


def corr_bar(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    num = df.select_dtypes("number")
    corr = (num.corr(numeric_only=True)["Response"].drop("Response")
            .abs().sort_values().tail(top_n))
    fig = go.Figure(go.Bar(x=corr.values, y=corr.index, orientation="h",
                           marker_color=TEAL,
                           hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
    return _style(fig, height=420, showlegend=False)


def spend_by_category(df: pd.DataFrame, spend_cols) -> go.Figure:
    s = df[spend_cols].sum().sort_values()
    labels = [c.replace("Mnt", "").replace("Products", "") for c in s.index]
    fig = go.Figure(go.Bar(x=s.values, y=labels, orientation="h", marker_color=BLUE,
                           text=[f"${v/1000:.0f}k" for v in s.values],
                           textposition="outside"))
    return _style(fig, height=360, showlegend=False)


# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
def roc_curves(run) -> go.Figure:
    fig = go.Figure()
    colors = [BLUE, TEAL, CORAL, GOLD]
    for (name, res), c in zip(run.results.items(), colors):
        fpr, tpr, _ = roc_curve(res.y_test, res.y_proba)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} ({res.metrics['ROC-AUC']:.3f})",
                                 line=dict(color=c, width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line=dict(color=GREY, dash="dash")))
    fig.update_xaxes(title="False Positive Rate")
    fig.update_yaxes(title="True Positive Rate")
    return _style(fig, height=420)


def metric_bars(run) -> go.Figure:
    comp = run.comparison.reset_index()
    metrics = ["Precision", "Recall", "F1", "ROC-AUC"]
    fig = go.Figure()
    for m, c in zip(metrics, [BLUE, CORAL, TEAL, NAVY]):
        fig.add_trace(go.Bar(name=m, x=comp["Model"], y=comp[m],
                             marker_color=c, text=comp[m].round(2), textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 1])
    return _style(fig, height=420)


def confusion_heatmap(res) -> go.Figure:
    cm = confusion_matrix(res.y_test, res.y_pred)
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Pred: Ignore", "Pred: Respond"], y=["True: Ignore", "True: Respond"],
        text=cm, texttemplate="%{text:,}", colorscale="Blues", showscale=False,
        hovertemplate="%{y} / %{x}: %{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _style(fig, height=340, title=res.name, showlegend=False)


def importance_bar(imp: pd.DataFrame, model_name: str) -> go.Figure:
    imp = imp.sort_values("importance")
    fig = go.Figure(go.Bar(x=imp["importance"], y=imp["feature"], orientation="h",
                           marker_color=NAVY,
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    return _style(fig, height=460, title=f"Top drivers — {model_name}", showlegend=False)


def profit_curve(prof: pd.DataFrame, best_t: float, current_t: float) -> go.Figure:
    fig = go.Figure(go.Scatter(x=prof["threshold"], y=prof["profit"], mode="lines+markers",
                               line=dict(color=NAVY, width=3), name="Profit"))
    fig.add_vline(x=best_t, line_dash="dash", line_color=CORAL,
                  annotation_text=f"profit-max {best_t:.2f}", annotation_font_color=CORAL)
    fig.add_vline(x=current_t, line_color=TEAL,
                  annotation_text=f"you: {current_t:.2f}", annotation_font_color=TEAL)
    fig.update_xaxes(title="Decision threshold")
    fig.update_yaxes(title="Campaign profit ($)")
    return _style(fig, height=380, showlegend=False)


def gauge(proba: float, threshold: float) -> go.Figure:
    color = CORAL if proba >= threshold else GREY
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        title={"text": ""},
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, threshold * 100], "color": "#f1f3f5"},
                {"range": [threshold * 100, 100], "color": "#fdece3"},
            ],
            "threshold": {"line": {"color": NAVY, "width": 3}, "value": threshold * 100},
        },
    ))
    return _style(fig, height=300, showlegend=False)
